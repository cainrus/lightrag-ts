// Port of lightrag/operate.py: chunking, entity/relation extraction, merge and
// summary (B2); query lands in B3.
import { type BaseGraphStorage, type BaseVectorStorage, type Clock, systemClock } from './base.ts';
import {
  DEFAULT_EMBEDDING_BATCH_NUM,
  DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
  DEFAULT_ENTITY_NAME_MAX_LENGTH,
  DEFAULT_ENTITY_TYPES,
  DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
  DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
  DEFAULT_MAX_ASYNC,
  DEFAULT_MAX_FILE_PATHS,
  DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
  DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
  DEFAULT_SOURCE_IDS_LIMIT_METHOD,
  DEFAULT_SUMMARY_CONTEXT_SIZE,
  DEFAULT_SUMMARY_LANGUAGE,
  DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
  DEFAULT_SUMMARY_MAX_TOKENS,
  GRAPH_FIELD_SEP,
  SOURCE_IDS_LIMIT_METHOD_FIFO,
  SOURCE_IDS_LIMIT_METHOD_KEEP,
} from './constants.ts';
import { ChunkTokenLimitExceededError, createPrefixedException, PipelineCancelledException, pyError } from './exceptions.ts';
import { pyTruthy } from './kg/json_kv_impl.ts';
import { PROMPTS } from './prompt.ts';
import {
  applySourceIdsLimit,
  type ChunkKv,
  computeMdhashId,
  fixTupleDelimiterCorruption,
  isFloatRegex,
  llmCacheLookup,
  type LlmFunc,
  type LlmResponseCache,
  makeRelationChunkKey,
  mergeSourceIds,
  normalizeSourceIdsLimitMethod,
  packUserAssToOpenaiMessages,
  pyCompare,
  pyFormat,
  pyJsonDumps,
  pyLen,
  priorityLimitAsyncFuncCall,
  pyReprList,
  pyReprStr,
  pySliceTo,
  pyStrip,
  pySumFloats,
  safeVdbOperationWithException,
  sanitizeAndNormalizeExtractedText,
  splitStringByMultiMarkers,
  type Tokenizer,
  truncateListByTokenSize,
  updateChunkCacheList,
  useLlmFuncWithCache,
} from './utils.ts';

export type ExtractionPromptConfig = {
  /** addon_params.language */
  language?: string;
  /** addon_params.entity_types */
  entityTypes?: readonly string[];
};

export type ExtractionPrompts = {
  system: string;
  user: string;
  continueUser: string;
};

/** The three prompts extract_entities sends for one chunk. */
export function buildExtractionPrompts(content: string, cfg: ExtractionPromptConfig = {}): ExtractionPrompts {
  const language = cfg.language ?? DEFAULT_SUMMARY_LANGUAGE;
  const entityTypes = cfg.entityTypes ?? DEFAULT_ENTITY_TYPES;
  const exampleContextBase = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: entityTypes.join(', '),
    language,
  };
  const examples = pyFormat(PROMPTS.entity_extraction_examples.join('\n'), exampleContextBase);
  const contextBase = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: entityTypes.join(','),
    examples,
    language,
  };
  return {
    system: pyFormat(PROMPTS.entity_extraction_system_prompt, contextBase),
    user: pyFormat(PROMPTS.entity_extraction_user_prompt, { ...contextBase, input_text: content }),
    continueUser: pyFormat(PROMPTS.entity_continue_extraction_user_prompt, { ...contextBase, input_text: content }),
  };
}

/**
 * Cache keys of the initial extraction call and of the gleaning call for a
 * chunk. The gleaning key depends on the initial LLM answer (it is part of the
 * history), so it is computed from `initialResult` - typically the cached one.
 */
export function extractionCacheKeys(
  content: string,
  cfg: ExtractionPromptConfig = {},
  initialResult?: string,
): { initial: string; gleaning?: string } {
  const p = buildExtractionPrompts(content, cfg);
  const initial = llmCacheLookup(p.user, { systemPrompt: p.system, cacheType: 'extract' }).cacheKey;
  if (initialResult === undefined) return { initial };
  const history = packUserAssToOpenaiMessages(p.user, initialResult);
  const gleaning = llmCacheLookup(p.continueUser, {
    systemPrompt: p.system,
    historyMessages: history,
    cacheType: 'extract',
  }).cacheKey;
  return { initial, gleaning };
}

export type TextChunk = { tokens: number; content: string; chunk_order_index: number };

/** operate.chunking_by_token_size */
export function chunkingByTokenSize(
  tokenizer: Tokenizer,
  content: string,
  splitByCharacter: string | null = null,
  splitByCharacterOnly = false,
  chunkOverlapTokenSize = 100,
  chunkTokenSize = 1200,
): TextChunk[] {
  const step = chunkTokenSize - chunkOverlapTokenSize;
  // Python's range(0, n, step): step 0 raises, a negative step yields nothing.
  if (step === 0) throw pyError('ValueError', 'range() arg 3 must not be zero');
  const starts = (n: number) => (step > 0 ? Array.from({ length: Math.ceil(n / step) }, (_, i) => i * step) : []);
  const results: TextChunk[] = [];
  if (splitByCharacter) {
    const newChunks: [number, string][] = [];
    for (const chunk of content.split(splitByCharacter)) {
      const tokens = tokenizer.encode(chunk);
      if (tokens.length <= chunkTokenSize) {
        newChunks.push([tokens.length, chunk]);
        continue;
      }
      if (splitByCharacterOnly) {
        throw new ChunkTokenLimitExceededError(tokens.length, chunkTokenSize, Array.from(chunk).slice(0, 120).join(''));
      }
      for (const start of starts(tokens.length)) {
        newChunks.push([Math.min(chunkTokenSize, tokens.length - start), tokenizer.decode(tokens.slice(start, start + chunkTokenSize))]);
      }
    }
    newChunks.forEach(([len, chunk], index) => results.push({ tokens: len, content: pyStrip(chunk), chunk_order_index: index }));
    return results;
  }
  const tokens = tokenizer.encode(content);
  for (const [index, start] of starts(tokens.length).entries()) {
    results.push({
      tokens: Math.min(chunkTokenSize, tokens.length - start),
      content: pyStrip(tokenizer.decode(tokens.slice(start, start + chunkTokenSize))),
      chunk_order_index: index,
    });
  }
  return results;
}

// ---------------------------------------------------------------- extraction

export type ExtractedEntity = {
  entity_name: string;
  entity_type: string;
  description: string;
  source_id: string;
  file_path: string;
  timestamp: number;
};
export type ExtractedRelation = {
  src_id: string;
  tgt_id: string;
  weight: number;
  description: string;
  keywords: string;
  source_id: string;
  file_path: string;
  timestamp: number;
};
/** One chunk's extraction: entity name -> records, `[src, tgt]` -> records (insertion order kept). */
export type ChunkExtraction = {
  nodes: Map<string, ExtractedEntity[]>;
  edges: Map<string, ExtractedRelation[]>;
};
/** Map key for an edge tuple; split with edgeKeyParts. */
export const edgeKey = (src: string, tgt: string): string => JSON.stringify([src, tgt]);
export const edgeKeyParts = (k: string): [string, string] => JSON.parse(k) as [string, string];

/** operate._truncate_entity_identifier (Python slices code points) */
export function truncateEntityIdentifier(identifier: string, limit: number): string {
  return pyLen(identifier) <= limit ? identifier : pySliceTo(identifier, limit);
}

const BAD_ENTITY_TYPE_CHARS = ["'", '(', ')', '<', '>', '|', '/', '\\'];

/** operate._handle_single_entity_extraction */
export function handleSingleEntityExtraction(
  attrs: string[],
  chunkKey: string,
  timestamp: number,
  filePath = 'unknown_source',
): ExtractedEntity | null {
  if (attrs.length !== 4 || !attrs[0]!.includes('entity')) return null;
  try {
    const entityName = sanitizeAndNormalizeExtractedText(attrs[1]!, true);
    if (!entityName || !pyStrip(entityName)) return null;
    let entityType = sanitizeAndNormalizeExtractedText(attrs[2]!, true);
    if (!pyStrip(entityType) || BAD_ENTITY_TYPE_CHARS.some((c) => entityType.includes(c))) return null;
    if (entityType.includes(',')) {
      const nonEmpty = entityType.split(',').map(pyStrip).filter((t) => t);
      if (!nonEmpty.length) return null;
      entityType = nonEmpty[0]!;
    }
    entityType = pyLower(entityType.replaceAll(' ', ''));
    const description = sanitizeAndNormalizeExtractedText(attrs[3]!);
    if (!pyStrip(description)) return null;
    return { entity_name: entityName, entity_type: entityType, description, source_id: chunkKey, file_path: filePath, timestamp };
  } catch {
    return null;
  }
}

/** operate._handle_single_relationship_extraction */
export function handleSingleRelationshipExtraction(
  attrs: string[],
  chunkKey: string,
  timestamp: number,
  filePath = 'unknown_source',
): ExtractedRelation | null {
  if (attrs.length !== 5 || !attrs[0]!.includes('relation')) return null;
  try {
    const source = sanitizeAndNormalizeExtractedText(attrs[1]!, true);
    const target = sanitizeAndNormalizeExtractedText(attrs[2]!, true);
    if (!source || !target || source === target) return null;
    const keywords = sanitizeAndNormalizeExtractedText(attrs[3]!, true).replaceAll('，', ',');
    const description = sanitizeAndNormalizeExtractedText(attrs[4]!);
    if (!pyStrip(description)) return null;
    // upstream reads the weight from the last field, which is the description
    const last = pyStripChars(pyStripChars(attrs[attrs.length - 1]!, '"'), "'");
    const weight = isFloatRegex(last) ? Number.parseFloat(last) : 1.0;
    return { src_id: source, tgt_id: target, weight, description, keywords, source_id: chunkKey, file_path: filePath, timestamp };
  } catch {
    return null;
  }
}

/** Python `str.strip(chars)` for a single character. */
function pyStripChars(s: string, ch: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && s[a] === ch) a++;
  while (b > a && s[b - 1] === ch) b--;
  return s.slice(a, b);
}

/** Python `str.lower()`; JS toLowerCase applies the same full mappings incl. final sigma. */
const pyLower = (s: string) => s.toLowerCase();

/** operate._process_extraction_result */
export function processExtractionResult(
  result: string,
  chunkKey: string,
  timestamp: number,
  filePath = 'unknown_source',
  tupleDelimiter = '<|#|>',
  completionDelimiter = '<|COMPLETE|>',
): ChunkExtraction {
  const nodes = new Map<string, ExtractedEntity[]>();
  const edges = new Map<string, ExtractedRelation[]>();
  const records = splitStringByMultiMarkers(result, ['\n', completionDelimiter, pyLower(completionDelimiter)]);
  const fixed: string[] = [];
  for (let record of records) {
    record = pyStrip(record);
    for (let entityRecord of splitStringByMultiMarkers(record, [`${tupleDelimiter}entity${tupleDelimiter}`])) {
      if (!entityRecord.startsWith('entity') && !entityRecord.startsWith('relation')) entityRecord = `entity<|${entityRecord}`;
      const parts = splitStringByMultiMarkers(entityRecord, [`${tupleDelimiter}relationship${tupleDelimiter}`, `${tupleDelimiter}relation${tupleDelimiter}`]);
      for (let r of parts) {
        if (!r.startsWith('entity') && !r.startsWith('relation')) r = `relation${tupleDelimiter}${r}`;
        fixed.push(r);
      }
    }
  }
  const core = tupleDelimiter.slice(2, -2);
  const coreLower = pyLower(core);
  for (let record of fixed) {
    record = pyStrip(record);
    record = fixTupleDelimiterCorruption(record, core, tupleDelimiter);
    if (core !== coreLower) record = fixTupleDelimiterCorruption(record, coreLower, tupleDelimiter);
    const attrs = splitStringByMultiMarkers(record, [tupleDelimiter]);
    const entity = handleSingleEntityExtraction(attrs, chunkKey, timestamp, filePath);
    if (entity) {
      entity.entity_name = truncateEntityIdentifier(entity.entity_name, DEFAULT_ENTITY_NAME_MAX_LENGTH);
      const list = nodes.get(entity.entity_name);
      if (list) list.push(entity);
      else nodes.set(entity.entity_name, [entity]);
      continue;
    }
    const rel = handleSingleRelationshipExtraction(attrs, chunkKey, timestamp, filePath);
    if (rel) {
      rel.src_id = truncateEntityIdentifier(rel.src_id, DEFAULT_ENTITY_NAME_MAX_LENGTH);
      rel.tgt_id = truncateEntityIdentifier(rel.tgt_id, DEFAULT_ENTITY_NAME_MAX_LENGTH);
      const k = edgeKey(rel.src_id, rel.tgt_id);
      const list = edges.get(k);
      if (list) list.push(rel);
      else edges.set(k, [rel]);
    }
  }
  return { nodes, edges };
}

export type ExtractChunk = { content: string; file_path?: string; [k: string]: unknown };
export type ExtractConfig = {
  llm_model_func: LlmFunc;
  entity_extract_max_gleaning: number;
  addon_params: { language?: string; entity_types?: readonly string[] };
  tokenizer: Tokenizer;
  max_extract_input_tokens: number;
  llm_model_max_async?: number;
};
export type PipelineStatus = { cancellation_requested?: boolean; latest_message?: string; history_messages?: string[] };

const descLen = (records: { description?: string }[]) => pyLen(records[0]?.description ?? '');

/**
 * operate.extract_entities. Results come back in chunk order; upstream
 * collects them from an asyncio `done` set, whose order is not defined.
 */
export async function extractEntities(
  chunks: ReadonlyMap<string, ExtractChunk> | Record<string, ExtractChunk>,
  globalConfig: ExtractConfig,
  opts: { pipelineStatus?: PipelineStatus; llmResponseCache?: LlmResponseCache | null; textChunksStorage?: ChunkKv | null; log?: MergeLog } = {},
): Promise<ChunkExtraction[]> {
  const status = opts.pipelineStatus;
  const log = opts.log ?? (() => {});
  if (status?.cancellation_requested) throw new PipelineCancelledException('User cancelled during entity extraction');
  const ordered = chunks instanceof Map ? [...chunks] : Object.entries(chunks);
  const prompts = (content: string) => buildExtractionPrompts(content, globalConfig.addon_params);
  const tupleDelimiter = PROMPTS.DEFAULT_TUPLE_DELIMITER;
  const completionDelimiter = PROMPTS.DEFAULT_COMPLETION_DELIMITER;
  let processed = 0;

  const one = async ([chunkKey, dp]: [string, ExtractChunk]): Promise<ChunkExtraction> => {
    const filePath = dp.file_path ?? 'unknown_source';
    const collector: string[] = [];
    const p = prompts(dp.content);
    const [finalResult, ts] = await useLlmFuncWithCache(p.user, globalConfig.llm_model_func, {
      systemPrompt: p.system,
      llmResponseCache: opts.llmResponseCache ?? null,
      cacheType: 'extract',
      chunkId: chunkKey,
      cacheKeysCollector: collector,
    });
    const history = packUserAssToOpenaiMessages(p.user, finalResult);
    const out = processExtractionResult(finalResult, chunkKey, ts, filePath, tupleDelimiter, completionDelimiter);
    if (globalConfig.entity_extract_max_gleaning > 0) {
      const full = p.system + pyJsonDumps(history) + p.continueUser;
      const tokenCount = globalConfig.tokenizer.encode(full).length;
      if (tokenCount > globalConfig.max_extract_input_tokens) {
        log('warning', `Gleaning stopped for chunk ${chunkKey}: Input tokens (${tokenCount}) exceeded limit (${globalConfig.max_extract_input_tokens}).`);
      } else {
        const [gleanResult, gts] = await useLlmFuncWithCache(p.continueUser, globalConfig.llm_model_func, {
          systemPrompt: p.system,
          llmResponseCache: opts.llmResponseCache ?? null,
          historyMessages: history,
          cacheType: 'extract',
          chunkId: chunkKey,
          cacheKeysCollector: collector,
        });
        const glean = processExtractionResult(gleanResult, chunkKey, gts, filePath, tupleDelimiter, completionDelimiter);
        for (const [name, recs] of glean.nodes) {
          const have = out.nodes.get(name);
          if (!have || descLen(recs) > descLen(have)) out.nodes.set(name, [...recs]);
        }
        for (const [k, recs] of glean.edges) {
          const have = out.edges.get(k);
          if (!have || descLen(recs) > descLen(have)) out.edges.set(k, [...recs]);
        }
      }
    }
    if (collector.length && opts.textChunksStorage) await updateChunkCacheList(chunkKey, opts.textChunksStorage, collector);
    processed++;
    const msg = `Chunk ${processed} of ${ordered.length} extracted ${out.nodes.size} Ent + ${out.edges.size} Rel ${chunkKey}`;
    if (status) {
      status.latest_message = msg;
      status.history_messages?.push(msg);
    }
    return out;
  };

  const limit = Math.max(1, globalConfig.llm_model_max_async ?? 4);
  const results: ChunkExtraction[] = new Array(ordered.length);
  let next = 0;
  let failed: unknown;
  const worker = async () => {
    while (failed === undefined && next < ordered.length) {
      const i = next++;
      const c = ordered[i]!;
      if (status?.cancellation_requested) throw new PipelineCancelledException('User cancelled during chunk processing');
      try {
        results[i] = await one(c);
      } catch (e) {
        throw createPrefixedException(e, c[0]);
      }
    }
  };
  const settled = await Promise.allSettled(Array.from({ length: Math.min(limit, ordered.length) }, worker));
  const firstErr = settled.find((s) => s.status === 'rejected') as PromiseRejectedResult | undefined;
  if (firstErr) throw createPrefixedException(firstErr.reason, `C[${processed + 1}/${ordered.length}]`);
  return results;
}

// ---------------------------------------------------------------- merge

/** The keys of upstream's `global_config` the merge reads. */
export type MergeConfig = {
  tokenizer: Tokenizer;
  /**
   * Called with `_priority` in its options, like upstream's global_config
   * function, and expected to be limited to llm_model_max_async calls at once
   * (priorityLimitAsyncFuncCall). defaultMergeConfig wraps a raw function; a
   * config built by hand passes a limited one (the LightRAG class does it once).
   */
  llm_model_func: LlmFunc;
  addon_params: { language?: string; [k: string]: unknown };
  summary_context_size: number;
  summary_max_tokens: number;
  summary_length_recommended: number;
  force_llm_summary_on_merge: number;
  max_source_ids_per_entity: number;
  max_source_ids_per_relation: number;
  /** read raw, as upstream does: only the exact strings 'KEEP' / 'FIFO' select a branch */
  source_ids_limit_method?: string | null;
  max_file_paths?: number;
  file_path_more_placeholder?: string;
  embedding_token_limit?: number | null;
  llm_model_max_async?: number;
  embedding_batch_num?: number;
  /** embedding calls at once while a merge writes its vectors (upstream's embedding_func_max_async) */
  embedding_func_max_async?: number;
};

/**
 * MergeConfig with LightRAG's defaults (constants.py; the env overrides are
 * LightRAG's job). The LLM function is wrapped in priorityLimitAsyncFuncCall
 * (llm_model_max_async), unless it already is.
 */
export function defaultMergeConfig(tokenizer: Tokenizer, llmModelFunc: LlmFunc, overrides: Partial<MergeConfig> = {}): MergeConfig {
  const gc: MergeConfig = {
    tokenizer,
    llm_model_func: llmModelFunc,
    addon_params: { language: DEFAULT_SUMMARY_LANGUAGE },
    summary_context_size: DEFAULT_SUMMARY_CONTEXT_SIZE,
    summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
    summary_length_recommended: DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    force_llm_summary_on_merge: DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    max_source_ids_per_entity: DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    max_source_ids_per_relation: DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    source_ids_limit_method: normalizeSourceIdsLimitMethod(DEFAULT_SOURCE_IDS_LIMIT_METHOD),
    max_file_paths: DEFAULT_MAX_FILE_PATHS,
    file_path_more_placeholder: DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    embedding_token_limit: null,
    llm_model_max_async: DEFAULT_MAX_ASYNC,
    embedding_batch_num: DEFAULT_EMBEDDING_BATCH_NUM,
    embedding_func_max_async: DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    ...overrides,
  };
  gc.llm_model_func = priorityLimitAsyncFuncCall(gc.llm_model_max_async ?? DEFAULT_MAX_ASYNC)(gc.llm_model_func);
  return gc;
}

export type MergeLog = (level: 'debug' | 'info' | 'warning' | 'error', msg: string) => void;

/** operate._summarize_descriptions */
export async function summarizeDescriptions(
  descriptionType: string,
  descriptionName: string,
  descriptionList: readonly string[],
  globalConfig: MergeConfig,
  llmResponseCache: LlmResponseCache | null = null,
  log: MergeLog = () => {},
): Promise<string> {
  const useLlmFunc: LlmFunc = (p, o) => globalConfig.llm_model_func(p, { ...o, _priority: 8 });
  const addon = globalConfig.addon_params;
  const language = 'language' in addon ? addon.language : DEFAULT_SUMMARY_LANGUAGE;
  const jsonDescriptions = descriptionList.map((desc) => ({ Description: desc }));
  const truncated = truncateListByTokenSize(jsonDescriptions, (x) => pyJsonDumps(x), globalConfig.summary_context_size, globalConfig.tokenizer);
  const usePrompt = pyFormat(PROMPTS.summarize_entity_descriptions, {
    description_type: descriptionType,
    description_name: descriptionName,
    description_list: truncated.map((d) => pyJsonDumps(d)).join('\n'),
    summary_length: globalConfig.summary_length_recommended,
    language,
  });
  const [summary] = await useLlmFuncWithCache(usePrompt, useLlmFunc, { llmResponseCache, cacheType: 'summary' });
  const limit = globalConfig.embedding_token_limit;
  if (limit !== null && limit !== undefined && summary) {
    const n = globalConfig.tokenizer.encode(summary).length;
    if (n > Math.trunc(limit)) {
      log('warning', `Summary tokens(${n}) exceeds embedding_token_limit(${limit})  for ${descriptionType}: ${descriptionName}`);
    }
  }
  return summary;
}

/**
 * operate._handle_entity_relation_summary: join when small, otherwise
 * map-reduce LLM summaries until the list fits. Returns [description, llmWasUsed].
 * `checkStop` (port-only) runs before every LLM call and throws to abandon the
 * summary - a cancelled pipeline, or a merge that already failed elsewhere.
 */
export async function handleEntityRelationSummary(
  descriptionType: string,
  entityOrRelationName: string,
  descriptionList: readonly string[],
  separator: string,
  globalConfig: MergeConfig,
  llmResponseCache: LlmResponseCache | null = null,
  log: MergeLog = () => {},
  checkStop: () => void = () => {},
): Promise<[string, boolean]> {
  if (!descriptionList.length) return ['', false];
  if (descriptionList.length === 1) return [descriptionList[0]!, false];
  const tokenizer = globalConfig.tokenizer;
  const contextSize = globalConfig.summary_context_size;
  const maxTokens = globalConfig.summary_max_tokens;
  const force = globalConfig.force_llm_summary_on_merge;
  const summarize = (list: readonly string[]) => {
    checkStop();
    return summarizeDescriptions(descriptionType, entityOrRelationName, list, globalConfig, llmResponseCache, log);
  };
  let current = [...descriptionList];
  let llmWasUsed = false;
  for (;;) {
    let totalTokens = 0;
    for (const desc of current) totalTokens += tokenizer.encode(desc).length;
    if (totalTokens <= contextSize || current.length <= 2) {
      if (current.length < force && totalTokens < maxTokens) return [current.join(separator), llmWasUsed];
      if (totalTokens > contextSize && current.length <= 2) log('warning', `Summarizing ${entityOrRelationName}: Oversize description found`);
      return [await summarize(current), true];
    }
    // map: groups within the context size, at least two descriptions each when possible
    const groups: string[][] = [];
    let group: string[] = [];
    let groupTokens = 0;
    for (const desc of current) {
      const descTokens = tokenizer.encode(desc).length;
      if (groupTokens + descTokens > contextSize && group.length) {
        if (group.length === 1) {
          group.push(desc);
          groups.push(group);
          log('warning', `Summarizing ${entityOrRelationName}: Oversize description found`);
          group = [];
          groupTokens = 0;
        } else {
          groups.push(group);
          group = [desc];
          groupTokens = descTokens;
        }
      } else {
        group.push(desc);
        groupTokens += descTokens;
      }
    }
    if (group.length) groups.push(group);
    log('info', `   Summarizing ${entityOrRelationName}: Map ${current.length} descriptions into ${groups.length} groups`);
    // reduce, one group after another as upstream
    const next: string[] = [];
    for (const g of groups) {
      if (g.length === 1) next.push(g[0]!);
      else {
        next.push(await summarize(g));
        llmWasUsed = true;
      }
    }
    current = next;
  }
}

/** The slice of BaseKVStorage the merge reads and writes (entity/relation chunk lists, full_entities/relations). */
export type MergeKv = {
  get_by_id(id: string): Promise<Record<string, unknown> | null>;
  upsert(data: Record<string, Record<string, unknown>>): Promise<void>;
};

type VdbOp =
  | { kind: 'upsert'; vdb: BaseVectorStorage; data: Record<string, Record<string, unknown>>; operation: string; name: string; retryDelay: number }
  | { kind: 'delete'; vdb: BaseVectorStorage; ids: string[] };

/**
 * Vector writes of one merge phase, kept in the order upstream issues them.
 * Nothing in the merge reads a vector store, so they are applied when the phase
 * ends: every queued content is embedded in batches (at most
 * embedding_func_max_async at once), then the upserts and deletes replay in
 * order with those vectors. The stores end up as after upstream's
 * one-record-at-a-time writes, at a fraction of the embedding round trips.
 */
class VdbWriteQueue {
  private ops: VdbOp[] = [];
  private readonly batchNum: number;
  private readonly maxAsync: number;
  private readonly log: MergeLog;
  constructor(gc: MergeConfig, log: MergeLog) {
    this.batchNum = gc.embedding_batch_num ?? DEFAULT_EMBEDDING_BATCH_NUM;
    this.maxAsync = Math.max(1, gc.embedding_func_max_async ?? DEFAULT_EMBEDDING_FUNC_MAX_ASYNC);
    this.log = log;
  }

  upsert(vdb: BaseVectorStorage, data: Record<string, Record<string, unknown>>, operation: string, name: string, retryDelay: number): void {
    this.ops.push({ kind: 'upsert', vdb, data, operation, name, retryDelay });
  }
  delete(vdb: BaseVectorStorage, ids: string[]): void {
    this.ops.push({ kind: 'delete', vdb, ids });
  }

  async flush(): Promise<void> {
    const ops = this.ops;
    this.ops = [];
    if (!ops.length) return;
    const vectors = new Map<VdbOp, ArrayLike<number>[]>();
    const byVdb = new Map<BaseVectorStorage, Extract<VdbOp, { kind: 'upsert' }>[]>();
    for (const op of ops) {
      if (op.kind !== 'upsert') continue;
      const list = byVdb.get(op.vdb) ?? [];
      list.push(op);
      byVdb.set(op.vdb, list);
    }
    for (const [vdb, list] of byVdb) {
      // a storage without an embedding function embeds inside its own upsert(data)
      if (typeof vdb.embedding_func !== 'function') continue;
      const contents = list.flatMap((op) => Object.values(op.data).map((v) => v.content as string));
      const embedded: ArrayLike<number>[] = new Array(contents.length);
      const batches: number[] = [];
      for (let i = 0; i < contents.length; i += this.batchNum) batches.push(i);
      const embedFn = vdb.embedding_func;
      await runLimited(batches, this.maxAsync, (start) =>
          safeVdbOperationWithException(
            async () => {
              const slice = contents.slice(start, start + this.batchNum);
              const r = await embedFn(slice, { context: 'document' });
              if (r.length !== slice.length) throw new Error(`embedding is not 1-1 with data, ${r.length} != ${slice.length}`);
              for (let i = 0; i < r.length; i++) embedded[start + i] = r[i]!;
            },
            'batch_embed',
            `${vdb.namespace} [${start}..${start + this.batchNum})`,
            3,
            list[0]!.retryDelay,
            (m) => this.log('warning', m),
          ),
      );
      let at = 0;
      for (const op of list) {
        const n = Object.keys(op.data).length;
        vectors.set(op, embedded.slice(at, at + n));
        at += n;
      }
    }
    for (const op of ops) {
      if (op.kind === 'delete') {
        try {
          await op.vdb.delete(op.ids);
        } catch (e) {
          this.log('debug', `Could not delete old relationship vector records ${op.ids.join(', ')}: ${e}`);
        }
        continue;
      }
      const vecs = vectors.get(op);
      await safeVdbOperationWithException(
        () => (vecs ? op.vdb.upsert(op.data, { embeddings: vecs }) : op.vdb.upsert(op.data)),
        op.operation,
        op.name,
        3,
        op.retryDelay,
        (m) => this.log('warning', m),
      );
    }
  }
}

/** Run fn over items with at most `limit` at once; the first rejection is thrown once all started ones settle. */
async function runLimited<T>(items: readonly T[], limit: number, fn: (x: T) => Promise<unknown>): Promise<void> {
  let next = 0;
  let failure: { e: unknown } | undefined;
  const worker = async () => {
    while (!failure && next < items.length) {
      const x = items[next++]!;
      try {
        await fn(x);
      } catch (e) {
        failure ??= { e };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  if (failure) throw failure.e;
}

type MergeCtx = {
  kg: BaseGraphStorage;
  entityVdb: BaseVectorStorage | null;
  relationshipsVdb: BaseVectorStorage | null;
  gc: MergeConfig;
  status: PipelineStatus | undefined;
  llmResponseCache: LlmResponseCache | null;
  entityChunks: MergeKv | null;
  relationChunks: MergeKv | null;
  vdbQueue: VdbWriteQueue;
  clock: Clock;
  log: MergeLog;
};

function pushStatus(status: PipelineStatus | undefined, msg: string): void {
  if (!status) return;
  status.latest_message = msg;
  status.history_messages?.push(msg);
}

function checkCancelled(status: PipelineStatus | undefined, message: string): void {
  if (status?.cancellation_requested) throw new PipelineCancelledException(message);
}

/** `x or default` with Python truthiness. */
const pyOr = <T>(v: unknown, d: T): unknown => (pyTruthy(v) ? v : d);
/** `d.get(k, default)` */
const pyGet = (d: Record<string, unknown>, k: string, dflt: unknown): unknown => (k in d ? d[k] : dflt);

async function storedChunkIds(kv: MergeKv | null, key: string): Promise<string[]> {
  if (!kv) return [];
  const stored = await kv.get_by_id(key);
  if (!stored || !pyTruthy(stored) || typeof stored !== 'object' || Array.isArray(stored)) return [];
  return ((stored.chunk_ids as string[] | undefined) ?? []).filter((c) => c);
}

/**
 * Step 9 of both merges: the file_path list within max_file_paths.
 * Returns the joined value and the limit method the step re-read (upstream
 * rebinds `limit_method` there, and the status message prints it).
 */
function buildFilePath(
  alreadyFilePaths: readonly string[],
  data: readonly { file_path?: string }[],
  gc: MergeConfig,
  logName: string,
  log: MergeLog,
): { filePath: string; limitMethod?: unknown } {
  const out: string[] = [];
  const seen = new Set<string>();
  let hasPlaceholder = false;
  const maxFilePaths = gc.max_file_paths ?? DEFAULT_MAX_FILE_PATHS;
  const placeholder = gc.file_path_more_placeholder ?? DEFAULT_FILE_PATH_MORE_PLACEHOLDER;
  for (const fp of alreadyFilePaths) {
    if (fp && fp.startsWith(`...${placeholder}`)) {
      hasPlaceholder = true;
      continue;
    }
    if (fp && !seen.has(fp)) {
      out.push(fp);
      seen.add(fp);
    }
  }
  for (const dp of data) {
    const fp = dp.file_path;
    if (fp && !seen.has(fp)) {
      out.push(fp);
      seen.add(fp);
    }
  }
  if (out.length <= maxFilePaths) return { filePath: out.join(GRAPH_FIELD_SEP) };
  const limitMethod = 'source_ids_limit_method' in gc ? gc.source_ids_limit_method : SOURCE_IDS_LIMIT_METHOD_KEEP;
  const originalCount = hasPlaceholder ? `${out.length}+` : String(out.length);
  let kept: string[];
  if (limitMethod === SOURCE_IDS_LIMIT_METHOD_FIFO) {
    kept = out.slice(-maxFilePaths);
    kept.push(`...${placeholder}...(FIFO)`);
  } else {
    kept = out.slice(0, maxFilePaths);
    kept.push(`...${placeholder}...(KEEP Old)`);
  }
  log('info', `Limited ${logName}: file_path ${originalCount} -> ${maxFilePaths} (${limitMethod})`);
  return { filePath: kept.join(GRAPH_FIELD_SEP), limitMethod };
}

/** Python sort key `(x.get("timestamp", 0), -len(x.get("description", "")))`, stable. */
function byTimestampThenLongest<T extends { timestamp?: number; description?: string }>(a: T, b: T): number {
  return (a.timestamp ?? 0) - (b.timestamp ?? 0) || pyLen(b.description ?? '') - pyLen(a.description ?? '');
}

/** The status line steps 10 of both merges build. */
function mergeStatusMessage(
  label: string,
  llmWasUsed: boolean,
  alreadyFragment: number,
  numFragment: number,
  dataCount: number,
  sourceCount: number,
  fullCount: number,
  limitMethod: unknown,
): { message: string; truncation: string } {
  let message = `${llmWasUsed ? 'LLMmrg' : 'Merged'}: ${label} | ${alreadyFragment}+${numFragment - alreadyFragment}`;
  let truncation = '';
  let truncationLog = '';
  if (sourceCount < fullCount) {
    truncationLog = `${limitMethod ?? 'None'} ${sourceCount}/${fullCount}`;
    truncation = limitMethod === SOURCE_IDS_LIMIT_METHOD_FIFO ? truncationLog : 'KEEP Old';
  }
  const deduplicated = alreadyFragment + dataCount - numFragment;
  const dd = deduplicated > 0 ? `dd ${deduplicated}` : '';
  if (dd || truncationLog) message += ` (${[truncationLog, dd].filter((s) => s).join(', ')})`;
  return { message, truncation };
}

type Commit<R> = () => Promise<R>;

/**
 * operate._merge_nodes_then_upsert, split in two. The returned promise does the
 * reads and the LLM summary (nothing another entity's merge writes); the
 * function it resolves to does every write, in upstream's order.
 */
async function prepareNodeMerge(
  entityName: string,
  nodesDataIn: readonly ExtractedEntity[],
  ctx: MergeCtx,
  checkStop: () => void = () => {},
): Promise<Commit<Record<string, unknown>>> {
  const { kg, gc, log } = ctx;
  const alreadyEntityTypes: string[] = [];
  const alreadySourceIds: string[] = [];
  const alreadyDescription: string[] = [];
  const alreadyFilePaths: string[] = [];

  // 1. existing node
  const alreadyNode = await kg.get_node(entityName);
  const nodeExists = pyTruthy(alreadyNode);
  if (alreadyNode && nodeExists) {
    let t = alreadyNode.entity_type as unknown;
    if (typeof t !== 'string' || !pyStrip(t)) t = 'UNKNOWN';
    let type = t as string;
    if (type.includes(',')) {
      const nonEmpty = type.split(',').map(pyStrip).filter((x) => x);
      const original = type;
      type = nonEmpty[0] ?? 'UNKNOWN';
      log('warning', `Entity type read from DB contains comma, taking first non-empty token: '${original}' -> '${type}'`);
    }
    alreadyEntityTypes.push(type);
    alreadySourceIds.push(...String(pyOr(alreadyNode.source_id, '')).split(GRAPH_FIELD_SEP));
    alreadyFilePaths.push(...String(pyOr(alreadyNode.file_path, 'unknown_source')).split(GRAPH_FIELD_SEP));
    const existingDesc = pyStrip(String(pyOr(alreadyNode.description, '')));
    if (existingDesc) alreadyDescription.push(...existingDesc.split(GRAPH_FIELD_SEP));
  }

  const newSourceIds = nodesDataIn.filter((dp) => dp.source_id).map((dp) => dp.source_id);
  let existingFullSourceIds = await storedChunkIds(ctx.entityChunks, entityName);
  if (!existingFullSourceIds.length) existingFullSourceIds = alreadySourceIds.filter((c) => c);

  // 2. merged chunk list (written in the commit)
  const fullSourceIds = mergeSourceIds(existingFullSourceIds, newSourceIds);
  const writeChunkList = async () => {
    if (ctx.entityChunks && fullSourceIds.length) {
      await ctx.entityChunks.upsert({ [entityName]: { chunk_ids: fullSourceIds, count: fullSourceIds.length } });
    }
  };

  // 3. source_id limit
  let limitMethod: unknown = gc.source_ids_limit_method;
  const maxSourceLimit = gc.max_source_ids_per_entity;
  const sourceIds = applySourceIdsLimit(fullSourceIds, maxSourceLimit, limitMethod as string);

  // 4. KEEP drops descriptions from chunks the cap cut off
  let nodesData: ExtractedEntity[];
  if (limitMethod === SOURCE_IDS_LIMIT_METHOD_KEEP) {
    const allowed = new Set(sourceIds);
    const existing = new Set(existingFullSourceIds);
    nodesData = nodesDataIn.filter((dp) => !(dp.source_id && !allowed.has(dp.source_id) && !existing.has(dp.source_id)));
  } else nodesData = [...nodesDataIn];

  // 5. nothing new under KEEP: keep the node as it is
  if (limitMethod === SOURCE_IDS_LIMIT_METHOD_KEEP && existingFullSourceIds.length >= maxSourceLimit && !nodesData.length) {
    return async () => {
      await writeChunkList();
      if (alreadyNode && nodeExists) {
        log('info', `Skipped \`${entityName}\`: KEEP old chunks ${pyReprList(alreadySourceIds)}/${fullSourceIds.length}`);
        return { ...alreadyNode };
      }
      throw pyError('ValueError', `Internal Error: already_node missing for \`${entityName}\``);
    };
  }

  // 6. source_id, entity type by count (first seen wins a tie: stable sort)
  const sourceId = sourceIds.join(GRAPH_FIELD_SEP);
  const counts = new Map<string, number>();
  for (const t of [...nodesData.map((dp) => dp.entity_type), ...alreadyEntityTypes]) counts.set(t, (counts.get(t) ?? 0) + 1);
  if (!counts.size) throw pyError('IndexError', 'list index out of range');
  let entityType = '';
  let best = -1;
  for (const [t, n] of counts) {
    if (n > best) {
      best = n;
      entityType = t;
    }
  }

  // 7. dedupe new descriptions, oldest first, longest first within a timestamp
  const unique = new Map<string, ExtractedEntity>();
  for (const dp of nodesData) {
    if (!dp.description) continue;
    if (!unique.has(dp.description)) unique.set(dp.description, dp);
  }
  const sortedDescriptions = [...unique.values()].sort(byTimestampThenLongest).map((dp) => dp.description);
  let descriptionList = [...alreadyDescription, ...sortedDescriptions];
  if (!descriptionList.length) {
    const fallback = `Entity ${entityName}`;
    log('warning', `Entity \`${entityName}\` has no description; fallback to \`${fallback}\``);
    descriptionList = [fallback];
  }

  const stopOrCancel = () => {
    checkStop();
    checkCancelled(ctx.status, 'User cancelled during entity summary');
  };
  stopOrCancel();

  // 8. summary
  const [description, llmWasUsed] = await handleEntityRelationSummary(
    'Entity',
    entityName,
    descriptionList,
    GRAPH_FIELD_SEP,
    gc,
    ctx.llmResponseCache,
    log,
    stopOrCancel,
  );

  // 9. file_path
  const fp = buildFilePath(alreadyFilePaths, nodesData, gc, `\`${entityName}\``, log);
  if ('limitMethod' in fp) limitMethod = fp.limitMethod;

  // 10. status line
  const { message, truncation } = mergeStatusMessage(
    `\`${entityName}\``,
    llmWasUsed,
    alreadyDescription.length,
    descriptionList.length,
    nodesData.length,
    sourceIds.length,
    fullSourceIds.length,
    limitMethod,
  );

  // 11. writes
  return async () => {
    await writeChunkList();
    if (alreadyDescription.length > 0 || llmWasUsed) {
      log('info', message);
      pushStatus(ctx.status, message);
    } else log('debug', message);
    const nodeData = {
      entity_id: entityName,
      entity_type: entityType,
      description,
      source_id: sourceId,
      file_path: fp.filePath,
      created_at: ctx.clock(),
      truncate: truncation,
    };
    await kg.upsert_node(entityName, { ...nodeData });
    if (ctx.entityVdb) {
      ctx.vdbQueue.upsert(
        ctx.entityVdb,
        {
          [computeMdhashId(entityName, 'ent-')]: {
            entity_name: entityName,
            entity_type: entityType,
            content: `${entityName}\n${description}`,
            source_id: sourceId,
            file_path: fp.filePath,
          },
        },
        'entity_upsert',
        entityName,
        0.1,
      );
    }
    return { ...nodeData, entity_name: entityName };
  };
}

type EdgeResult = [Record<string, unknown> | null, Record<string, unknown>[]];

/** operate._merge_edges_then_upsert, split like prepareNodeMerge. */
async function prepareEdgeMerge(
  srcId: string,
  tgtId: string,
  edgesDataIn: readonly ExtractedRelation[],
  ctx: MergeCtx,
  checkStop: () => void = () => {},
): Promise<Commit<EdgeResult>> {
  const { kg, gc, log } = ctx;
  if (srcId === tgtId) return async () => [null, []];
  let alreadyEdge: Record<string, unknown> | null = null;
  const alreadyWeights: number[] = [];
  const alreadySourceIds: string[] = [];
  const alreadyDescription: string[] = [];
  const alreadyKeywords: string[] = [];
  const alreadyFilePaths: string[] = [];

  // 1. existing edge
  if (await kg.has_edge(srcId, tgtId)) {
    alreadyEdge = await kg.get_edge(srcId, tgtId);
    if (alreadyEdge && pyTruthy(alreadyEdge)) {
      alreadyWeights.push(pyGet(alreadyEdge, 'weight', 1.0) as number);
      if (alreadyEdge.source_id !== null && alreadyEdge.source_id !== undefined) alreadySourceIds.push(...String(alreadyEdge.source_id).split(GRAPH_FIELD_SEP));
      if (alreadyEdge.file_path !== null && alreadyEdge.file_path !== undefined) alreadyFilePaths.push(...String(alreadyEdge.file_path).split(GRAPH_FIELD_SEP));
      if (alreadyEdge.description !== null && alreadyEdge.description !== undefined) alreadyDescription.push(...String(alreadyEdge.description).split(GRAPH_FIELD_SEP));
      if (alreadyEdge.keywords !== null && alreadyEdge.keywords !== undefined) alreadyKeywords.push(...splitStringByMultiMarkers(String(alreadyEdge.keywords), [GRAPH_FIELD_SEP]));
    }
  }

  const newSourceIds = edgesDataIn.filter((dp) => dp.source_id).map((dp) => dp.source_id);
  const storageKey = makeRelationChunkKey(srcId, tgtId);
  let existingFullSourceIds = await storedChunkIds(ctx.relationChunks, storageKey);
  if (!existingFullSourceIds.length) existingFullSourceIds = alreadySourceIds.filter((c) => c);

  // 2. merged chunk list (written in the commit)
  const fullSourceIds = mergeSourceIds(existingFullSourceIds, newSourceIds);
  const writeChunkList = async () => {
    if (ctx.relationChunks && fullSourceIds.length) {
      await ctx.relationChunks.upsert({ [storageKey]: { chunk_ids: fullSourceIds, count: fullSourceIds.length } });
    }
  };

  // 3. source_id limit; edges default an unset method to KEEP from here on
  const maxSourceLimit = gc.max_source_ids_per_relation;
  const sourceIds = applySourceIdsLimit(fullSourceIds, maxSourceLimit, gc.source_ids_limit_method);
  let limitMethod: unknown = pyOr(gc.source_ids_limit_method, SOURCE_IDS_LIMIT_METHOD_KEEP);

  // 4.
  let edgesData: ExtractedRelation[];
  if (limitMethod === SOURCE_IDS_LIMIT_METHOD_KEEP) {
    const allowed = new Set(sourceIds);
    const existing = new Set(existingFullSourceIds);
    edgesData = edgesDataIn.filter((dp) => !(dp.source_id && !allowed.has(dp.source_id) && !existing.has(dp.source_id)));
  } else edgesData = [...edgesDataIn];

  // 5.
  if (limitMethod === SOURCE_IDS_LIMIT_METHOD_KEEP && existingFullSourceIds.length >= maxSourceLimit && !edgesData.length) {
    return async () => {
      await writeChunkList();
      if (alreadyEdge && pyTruthy(alreadyEdge)) {
        log('info', `Skipped \`${srcId}\`~\`${tgtId}\`: KEEP old chunks  ${pyReprList(alreadySourceIds)}/${fullSourceIds.length}`);
        return [{ ...alreadyEdge }, []];
      }
      throw pyError('ValueError', `Internal Error: already_node missing for \`${srcId}\`~\`${tgtId}\``);
    };
  }

  // 6. source_id, weight (Python float sum), keywords (set, code point order)
  const sourceId = sourceIds.join(GRAPH_FIELD_SEP);
  const weight = pySumFloats([...edgesData.map((dp) => dp.weight), ...alreadyWeights]);
  const allKeywords = new Set<string>();
  const addKeywords = (s: string) => {
    for (const k of s.split(',')) {
      const t = pyStrip(k);
      if (t) allKeywords.add(t);
    }
  };
  for (const s of alreadyKeywords) if (s) addKeywords(s);
  for (const e of edgesData) if (e.keywords) addKeywords(e.keywords);
  const keywords = [...allKeywords].sort(pyCompare).join(',');

  // 7.
  const unique = new Map<string, ExtractedRelation>();
  for (const dp of edgesData) {
    if (!dp.description) continue;
    if (!unique.has(dp.description)) unique.set(dp.description, dp);
  }
  const sortedDescriptions = [...unique.values()].sort(byTimestampThenLongest).map((dp) => dp.description);
  const descriptionList = [...alreadyDescription, ...sortedDescriptions];
  if (!descriptionList.length) {
    log('error', `Relation ${srcId}~${tgtId} has no description`);
    throw pyError('ValueError', `Relation ${srcId}~${tgtId} has no description`);
  }

  const stopOrCancel = () => {
    checkStop();
    checkCancelled(ctx.status, 'User cancelled during relation summary');
  };
  stopOrCancel();

  // 8.
  const [description, llmWasUsed] = await handleEntityRelationSummary(
    'Relation',
    `(${srcId}, ${tgtId})`,
    descriptionList,
    GRAPH_FIELD_SEP,
    gc,
    ctx.llmResponseCache,
    log,
    stopOrCancel,
  );

  // 9.
  const fp = buildFilePath(alreadyFilePaths, edgesData, gc, `\`${srcId}\`~\`${tgtId}\``, log);
  if ('limitMethod' in fp) limitMethod = fp.limitMethod;
  const filePath = fp.filePath;

  // 10.
  const { message, truncation } = mergeStatusMessage(
    `\`${srcId}\`~\`${tgtId}\``,
    llmWasUsed,
    alreadyDescription.length,
    descriptionList.length,
    edgesData.length,
    sourceIds.length,
    fullSourceIds.length,
    limitMethod,
  );

  // 11. writes: endpoints first (they may not exist yet), then the edge
  return async () => {
    await writeChunkList();
    if (alreadyDescription.length > 0 || llmWasUsed) {
      log('info', message);
      pushStatus(ctx.status, message);
    } else log('debug', message);

    const addedEntities: Record<string, unknown>[] = [];
    for (const needId of [srcId, tgtId]) {
      const existingNode = await kg.get_node(needId);
      if (existingNode === null) {
        const nodeCreatedAt = ctx.clock();
        await kg.upsert_node(needId, {
          entity_id: needId,
          source_id: sourceId,
          description,
          entity_type: 'UNKNOWN',
          file_path: filePath,
          created_at: nodeCreatedAt,
          truncate: '',
        });
        if (ctx.entityChunks) {
          const chunkIds = fullSourceIds.filter((c) => c);
          if (chunkIds.length) await ctx.entityChunks.upsert({ [needId]: { chunk_ids: chunkIds, count: chunkIds.length } });
        }
        if (ctx.entityVdb) {
          ctx.vdbQueue.upsert(
            ctx.entityVdb,
            {
              [computeMdhashId(needId, 'ent-')]: {
                content: `${needId}\n${description}`,
                entity_name: needId,
                source_id: sourceId,
                entity_type: 'UNKNOWN',
                file_path: filePath,
              },
            },
            'added_entity_upsert',
            needId,
            0.1,
          );
        }
        addedEntities.push({ entity_name: needId, entity_type: 'UNKNOWN', description, source_id: sourceId, file_path: filePath, created_at: nodeCreatedAt });
        continue;
      }
      // the node exists: append this relation's chunks to it
      let updated = false;
      let existingFull = await storedChunkIds(ctx.entityChunks, needId);
      if (!existingFull.length && pyTruthy(existingNode.source_id)) existingFull = String(existingNode.source_id).split(GRAPH_FIELD_SEP);
      const merged = mergeSourceIds(existingFull, sourceIds.filter((c) => c));
      if (ctx.entityChunks && !sameList(merged, existingFull)) {
        updated = true;
        await ctx.entityChunks.upsert({ [needId]: { chunk_ids: merged, count: merged.length } });
      }
      const nodeLimitMethod = pyGet(gc, 'source_ids_limit_method', SOURCE_IDS_LIMIT_METHOD_KEEP) as string;
      const limited = applySourceIdsLimit(merged, gc.max_source_ids_per_entity, nodeLimitMethod).join(GRAPH_FIELD_SEP);
      if (limited !== pyGet(existingNode, 'source_id', '')) {
        updated = true;
        await kg.upsert_node(needId, { ...existingNode, source_id: limited });
        if (ctx.entityVdb) {
          ctx.vdbQueue.upsert(
            ctx.entityVdb,
            {
              [computeMdhashId(needId, 'ent-')]: {
                content: `${needId}\n${String(pyGet(existingNode, 'description', ''))}`,
                entity_name: needId,
                source_id: limited,
                entity_type: pyGet(existingNode, 'entity_type', 'UNKNOWN'),
                file_path: pyGet(existingNode, 'file_path', 'unknown_source'),
              },
            },
            'existing_entity_update',
            needId,
            0.1,
          );
        }
      }
      if (updated) {
        const msg = `Chunks appended from relation: \`${needId}\``;
        log('info', msg);
        pushStatus(ctx.status, msg);
      }
    }

    const edgeCreatedAt = ctx.clock();
    await kg.upsert_edge(srcId, tgtId, {
      weight,
      description,
      keywords,
      source_id: sourceId,
      file_path: filePath,
      created_at: edgeCreatedAt,
      truncate: truncation,
    });
    const edgeData = {
      src_id: srcId,
      tgt_id: tgtId,
      description,
      keywords,
      source_id: sourceId,
      file_path: filePath,
      created_at: edgeCreatedAt,
      truncate: truncation,
      weight,
    };
    if (ctx.relationshipsVdb) {
      const [a, b] = pyCompare(srcId, tgtId) > 0 ? [tgtId, srcId] : [srcId, tgtId];
      const relVdbId = computeMdhashId(a + b, 'rel-');
      ctx.vdbQueue.delete(ctx.relationshipsVdb, [relVdbId, computeMdhashId(b + a, 'rel-')]);
      ctx.vdbQueue.upsert(
        ctx.relationshipsVdb,
        {
          [relVdbId]: {
            src_id: a,
            tgt_id: b,
            source_id: sourceId,
            content: `${keywords}\t${a}\n${b}\n${description}`,
            keywords,
            description,
            weight,
            file_path: filePath,
          },
        },
        'relationship_upsert',
        `${a}-${b}`,
        0.2,
      );
    }
    return [edgeData, addedEntities];
  };
}

type SingleMergeOptions = {
  pipelineStatus?: PipelineStatus;
  llmResponseCache?: LlmResponseCache | null;
  entityChunksStorage?: MergeKv | null;
  relationChunksStorage?: MergeKv | null;
  clock?: Clock;
  log?: MergeLog;
};

function singleCtx(kg: BaseGraphStorage, entityVdb: BaseVectorStorage | null, relationshipsVdb: BaseVectorStorage | null, gc: MergeConfig, o: SingleMergeOptions): MergeCtx {
  const log = o.log ?? (() => {});
  return {
    kg,
    entityVdb,
    relationshipsVdb,
    gc,
    status: o.pipelineStatus,
    llmResponseCache: o.llmResponseCache ?? null,
    entityChunks: o.entityChunksStorage ?? null,
    relationChunks: o.relationChunksStorage ?? null,
    vdbQueue: new VdbWriteQueue(gc, log),
    clock: o.clock ?? systemClock,
    log,
  };
}

const graphLocks = new WeakMap<object, Promise<void>>();

/**
 * Runs `fn` as the only merge writing to `graph` (and the KV/vector stores that
 * go with it) at that time; callers queue in arrival order. Upstream takes a
 * keyed lock per entity / relation name inside merge_nodes_and_edges, so two
 * documents merged at once interleave name by name; here a whole merge waits
 * for the previous one, and the result is the documents merged in call order.
 * Anything else that rewrites graph nodes or edges (B4's edit/delete paths)
 * should take the same lock.
 */
export async function withGraphLock<T>(graph: BaseGraphStorage, fn: () => Promise<T>): Promise<T> {
  const previous = graphLocks.get(graph) ?? Promise.resolve();
  let release!: () => void;
  const mine = new Promise<void>((r) => (release = r));
  const tail = previous.then(() => mine);
  graphLocks.set(graph, tail);
  await previous;
  try {
    return await fn();
  } finally {
    release();
    if (graphLocks.get(graph) === tail) graphLocks.delete(graph);
  }
}

/** operate._merge_nodes_then_upsert for one entity (mergeNodesAndEdges batches these). */
export async function mergeNodesThenUpsert(
  entityName: string,
  nodesData: readonly ExtractedEntity[],
  knowledgeGraphInst: BaseGraphStorage,
  entityVdb: BaseVectorStorage | null,
  globalConfig: MergeConfig,
  opts: SingleMergeOptions = {},
): Promise<Record<string, unknown>> {
  return withGraphLock(knowledgeGraphInst, async () => {
    const ctx = singleCtx(knowledgeGraphInst, entityVdb, null, globalConfig, opts);
    const result = await (await prepareNodeMerge(entityName, nodesData, ctx))();
    await ctx.vdbQueue.flush();
    return result;
  });
}

/**
 * operate._merge_edges_then_upsert for one relation. Entities it had to create
 * are appended to `opts.addedEntities` (upstream's `added_entities` list).
 */
export async function mergeEdgesThenUpsert(
  srcId: string,
  tgtId: string,
  edgesData: readonly ExtractedRelation[],
  knowledgeGraphInst: BaseGraphStorage,
  relationshipsVdb: BaseVectorStorage | null,
  entityVdb: BaseVectorStorage | null,
  globalConfig: MergeConfig,
  opts: SingleMergeOptions & { addedEntities?: Record<string, unknown>[] | null } = {},
): Promise<Record<string, unknown> | null> {
  return withGraphLock(knowledgeGraphInst, async () => {
    const ctx = singleCtx(knowledgeGraphInst, entityVdb, relationshipsVdb, globalConfig, opts);
    const [edgeData, added] = await (await prepareEdgeMerge(srcId, tgtId, edgesData, ctx))();
    await ctx.vdbQueue.flush();
    opts.addedEntities?.push(...added);
    return edgeData;
  });
}

const sameList = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i]);

/** Thrown by an item's checkStop once an earlier item failed; never reported. */
class MergeAborted extends Error {
  override name = 'MergeAborted';
}

/**
 * Prepare up to `limit` items at once, commit them strictly in index order.
 * When item k fails, items before it still prepare and commit, items after it
 * are not started, and the ones in flight abandon their work at their next
 * checkStop (between LLM calls); `fail` reports and throws only k's error.
 */
async function orderedPool<P, R>(
  n: number,
  limit: number,
  prepare: (i: number, checkStop: () => void) => Promise<P>,
  commit: (i: number, p: P) => Promise<R>,
  fail: (i: number, e: unknown) => never,
): Promise<R[]> {
  const prepared = Array.from({ length: n }, () => {
    let resolve!: (v: P) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<P>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    promise.catch(() => {});
    return { promise, resolve, reject };
  });
  let next = 0;
  let stopAt = Infinity;
  const worker = async () => {
    while (next < n && next <= stopAt) {
      const i = next++;
      const checkStop = () => {
        if (i > stopAt) throw new MergeAborted();
      };
      try {
        prepared[i]!.resolve(await prepare(i, checkStop));
      } catch (e) {
        if (!(e instanceof MergeAborted)) stopAt = Math.min(stopAt, i);
        prepared[i]!.reject(e);
      }
    }
  };
  const workers = Array.from({ length: Math.max(1, Math.min(limit, n)) }, worker);
  const results: R[] = [];
  try {
    for (let i = 0; i < n; i++) {
      let p: P;
      try {
        p = await prepared[i]!.promise;
      } catch (e) {
        stopAt = Math.min(stopAt, i);
        fail(i, e);
      }
      try {
        results.push(await commit(i, p));
      } catch (e) {
        stopAt = Math.min(stopAt, i);
        fail(i, e);
      }
    }
  } finally {
    await Promise.allSettled(workers);
  }
  return results;
}

export type MergeOptions = {
  fullEntitiesStorage?: MergeKv | null;
  fullRelationsStorage?: MergeKv | null;
  docId?: string | null;
  pipelineStatus?: PipelineStatus;
  llmResponseCache?: LlmResponseCache | null;
  entityChunksStorage?: MergeKv | null;
  relationChunksStorage?: MergeKv | null;
  currentFileNumber?: number;
  totalFiles?: number;
  filePath?: string;
  /** port-only: `int(time.time())` for created_at */
  clock?: Clock;
  /** port-only: how many entity / relation merges prepare at once; default llm_model_max_async * 2 */
  maxConcurrency?: number;
  log?: MergeLog;
};

/**
 * operate.merge_nodes_and_edges: entities first, then relations (which may
 * create missing entities), then the per-document entity/relation index.
 *
 * Upstream runs the per-entity and per-relation merges as concurrent tasks; the
 * order their writes land in follows asyncio scheduling and, in production,
 * LLM and embedding latency. Here the LLM work still runs concurrently, but
 * writes happen in input order, so the result is upstream's with the merges run
 * one after another - deterministic, independent of latency. Concurrent calls
 * on the same graph run one after another (withGraphLock).
 */
export async function mergeNodesAndEdges(
  chunkResults: readonly ChunkExtraction[],
  knowledgeGraphInst: BaseGraphStorage,
  entityVdb: BaseVectorStorage | null,
  relationshipsVdb: BaseVectorStorage | null,
  globalConfig: MergeConfig,
  opts: MergeOptions = {},
): Promise<void> {
  return withGraphLock(knowledgeGraphInst, () => mergeLocked(chunkResults, knowledgeGraphInst, entityVdb, relationshipsVdb, globalConfig, opts));
}

async function mergeLocked(
  chunkResults: readonly ChunkExtraction[],
  knowledgeGraphInst: BaseGraphStorage,
  entityVdb: BaseVectorStorage | null,
  relationshipsVdb: BaseVectorStorage | null,
  globalConfig: MergeConfig,
  opts: MergeOptions,
): Promise<void> {
  const status = opts.pipelineStatus;
  const log = opts.log ?? (() => {});
  checkCancelled(status, 'User cancelled during merge phase');

  const allNodes = new Map<string, ExtractedEntity[]>();
  const allEdges = new Map<string, { src: string; tgt: string; edges: ExtractedRelation[] }>();
  for (const { nodes, edges } of chunkResults) {
    for (const [name, entities] of nodes) {
      const list = allNodes.get(name);
      if (list) list.push(...entities);
      else allNodes.set(name, [...entities]);
    }
    for (const [k, rels] of edges) {
      const [src, tgt] = edgeKeyParts(k).sort(pyCompare) as [string, string];
      const sk = edgeKey(src, tgt);
      const e = allEdges.get(sk);
      if (e) e.edges.push(...rels);
      else allEdges.set(sk, { src, tgt, edges: [...rels] });
    }
  }

  const docId = opts.docId ?? null;
  const docLabel = docId ?? 'None';
  const say = (m: string) => {
    log('info', m);
    pushStatus(status, m);
  };
  say(`Merging stage ${opts.currentFileNumber ?? 0}/${opts.totalFiles ?? 0}: ${opts.filePath ?? 'unknown_source'}`);
  const graphMaxAsync = (globalConfig.llm_model_max_async ?? 4) * 2;
  const limit = opts.maxConcurrency ?? graphMaxAsync;
  const ctx: MergeCtx = {
    kg: knowledgeGraphInst,
    entityVdb,
    relationshipsVdb,
    gc: globalConfig,
    status,
    llmResponseCache: opts.llmResponseCache ?? null,
    entityChunks: opts.entityChunksStorage ?? null,
    relationChunks: opts.relationChunksStorage ?? null,
    vdbQueue: new VdbWriteQueue(globalConfig, log),
    clock: opts.clock ?? systemClock,
    log,
  };
  const failed = (label: string, e: unknown, prefix: string): never => {
    const msg = `Error processing ${label}: ${(e as Error)?.message ?? e}`;
    log('error', msg);
    pushStatus(status, msg);
    throw createPrefixedException(e, prefix);
  };
  // Each phase's vectors are written before the next phase starts, also when
  // one of its merges failed (then the merges before it keep their vectors).
  const thenFlush = async <T>(phase: () => Promise<T>): Promise<T> => {
    let result: T | undefined;
    let error: { e: unknown } | undefined;
    try {
      result = await phase();
    } catch (e) {
      error = { e };
    }
    try {
      await ctx.vdbQueue.flush();
    } catch (e) {
      error ??= { e };
    }
    if (error) throw error.e;
    return result as T;
  };

  // ===== Phase 1: entities
  say(`Phase 1: Processing ${allNodes.size} entities from ${docLabel} (async: ${graphMaxAsync})`);
  const nodeItems = [...allNodes];
  const entityLabel = (i: number) => `\`${nodeItems[i]![0]}\``;
  const processedEntities = await thenFlush(() =>
    orderedPool(
      nodeItems.length,
      limit,
      async (i, checkStop) => {
        checkCancelled(status, 'User cancelled during entity merge');
        const [name, entities] = nodeItems[i]!;
        return prepareNodeMerge(name, entities, ctx, checkStop);
      },
      (_i, commit) => commit(),
      (i, e) => failed(`entity ${entityLabel(i)}`, e, entityLabel(i)),
    ),
  );

  // ===== Phase 2: relations
  say(`Phase 2: Processing ${allEdges.size} relations from ${docLabel} (async: ${graphMaxAsync})`);
  const edgeItems = [...allEdges.values()];
  const edgeLabel = (i: number) => `[${pyReprStr(edgeItems[i]!.src)}, ${pyReprStr(edgeItems[i]!.tgt)}]`;
  const edgeResults = await thenFlush(() =>
    orderedPool(
      edgeItems.length,
      limit,
      async (i, checkStop) => {
        checkCancelled(status, 'User cancelled during relation merge');
        const { src, tgt, edges } = edgeItems[i]!;
        return prepareEdgeMerge(src, tgt, edges, ctx, checkStop);
      },
      (_i, commit) => commit(),
      (i, e) => failed(`relation \`${edgeLabel(i)}\``, e, edgeLabel(i)),
    ),
  );
  const processedEdges: Record<string, unknown>[] = [];
  const allAddedEntities: Record<string, unknown>[] = [];
  for (const [edgeData, added] of edgeResults) {
    if (edgeData !== null) processedEdges.push(edgeData);
    allAddedEntities.push(...added);
  }

  // ===== Phase 3: the document's entity and relation index
  if (opts.fullEntitiesStorage && opts.fullRelationsStorage && docId) {
    try {
      const finalEntityNames = new Set<string>();
      for (const e of processedEntities) if (pyTruthy(e) && e.entity_name) finalEntityNames.add(e.entity_name as string);
      for (const e of allAddedEntities) if (pyTruthy(e) && e.entity_name) finalEntityNames.add(e.entity_name as string);
      const pairs = new Map<string, [string, string]>();
      for (const e of processedEdges) {
        if (!pyTruthy(e)) continue;
        const s = e.src_id as string | undefined;
        const t = e.tgt_id as string | undefined;
        if (s && t) {
          const p = [s, t].sort(pyCompare) as [string, string];
          pairs.set(edgeKey(...p), p);
        }
      }
      say(
        `Phase 3: Updating final ${finalEntityNames.size}(${processedEntities.length}+${allAddedEntities.length}) entities and  ${pairs.size} relations from ${docId}`,
      );
      if (finalEntityNames.size) {
        await opts.fullEntitiesStorage.upsert({ [docId]: { entity_names: [...finalEntityNames], count: finalEntityNames.size } });
      }
      if (pairs.size) {
        await opts.fullRelationsStorage.upsert({ [docId]: { relation_pairs: [...pairs.values()].map((p) => [...p]), count: pairs.size } });
      }
    } catch (e) {
      log('error', `Failed to update entity-relation index for document ${docId}: ${(e as Error)?.message ?? e}`);
    }
  }
  say(`Completed merging: ${processedEntities.length} entities, ${allAddedEntities.length} extra entities, ${processedEdges.length} relations`);
}
