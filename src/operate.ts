// Port of lightrag/operate.py: chunking, entity/relation extraction; merge
// lands in B2, query in B3.
import { DEFAULT_ENTITY_NAME_MAX_LENGTH, DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE } from './constants.ts';
import { ChunkTokenLimitExceededError, createPrefixedException, PipelineCancelledException } from './exceptions.ts';
import { PROMPTS } from './prompt.ts';
import {
  type ChunkKv,
  fixTupleDelimiterCorruption,
  isFloatRegex,
  llmCacheLookup,
  type LlmFunc,
  type LlmResponseCache,
  packUserAssToOpenaiMessages,
  pyFormat,
  pyJsonDumps,
  pyLen,
  pySliceTo,
  pyStrip,
  sanitizeAndNormalizeExtractedText,
  splitStringByMultiMarkers,
  type Tokenizer,
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
  if (step === 0) throw new Error('range() arg 3 must not be zero');
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
  opts: { pipelineStatus?: PipelineStatus; llmResponseCache?: LlmResponseCache | null; textChunksStorage?: ChunkKv | null } = {},
): Promise<ChunkExtraction[]> {
  const status = opts.pipelineStatus;
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
      if (globalConfig.tokenizer.encode(full).length <= globalConfig.max_extract_input_tokens) {
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
