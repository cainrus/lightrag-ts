// Port of the query half of lightrag/operate.py (1.4.16): keyword extraction,
// kg_query (local / global / hybrid / mix), naive_query and the four-stage
// context builder (search -> truncate -> merge chunks -> context string).
// Functions keep upstream's names in camelCase and upstream's argument order,
// so each one diffs against its Python counterpart. Lives outside
// src/operate.ts while B2 edits that file.
import type { BaseGraphStorage, EdgeData, NodeData, VectorQueryResult } from './base.ts';
import { edgePairKey } from './base.ts';
import { DEFAULT_KG_CHUNK_PICK_METHOD, DEFAULT_RELATED_CHUNK_NUMBER, DEFAULT_SUMMARY_LANGUAGE, GRAPH_FIELD_SEP } from './constants.ts';
import { PROMPTS } from './prompt.ts';
import {
  generateCacheKey,
  handleCache,
  type LlmResponseCache,
  pyCompare,
  pyFormat,
  pyLen,
  pyStrip,
  removeThinkTags,
  saveToCache,
  splitStringByMultiMarkers,
} from './utils.ts';
import { makeQueryParam, QueryContextResult, type QueryLlmFunc, type QueryLlmOptions, type QueryParam, type QueryRawData, QueryResult } from './query/base.ts';
import {
  type ChunkGlobalConfig,
  computeArgsHashPy,
  convertToUserFormat,
  type Dict,
  generateReferenceListFromChunks,
  jsonLineKey,
  jsonRepairLoads,
  noLog,
  pairKey,
  pickByVectorSimilarity,
  pickByWeightedPolling,
  processChunksUnified,
  pyGet,
  pyJsonDumpsAny,
  pyStr,
  pyTruthy,
  type QueryEmbeddingFunc,
  type QueryLog,
  truncateListByTokenSize,
} from './query/utils.ts';

export * from './query/base.ts';
export {
  applyRerankIfEnabled,
  computeArgsHashPy,
  convertToUserFormat,
  cosineSimilarity,
  generateReferenceListFromChunks,
  jsonRepairLoads,
  pickByVectorSimilarity,
  pickByWeightedPolling,
  processChunksUnified,
  pyFloatRepr,
  pyJsonDumpsAny,
  pyRound,
  pyStr,
  truncateListByTokenSize,
} from './query/utils.ts';
export type { Dict, QueryEmbeddingFunc, QueryLog, RerankFunc, RerankResult } from './query/utils.ts';

// ------------------------------------------------------------------ contracts

/** The `global_config` keys the query path reads (upstream passes `asdict(LightRAG)`). */
export type QueryGlobalConfig = ChunkGlobalConfig & {
  llm_model_func?: QueryLlmFunc;
  addon_params?: { language?: string; [k: string]: unknown };
  kg_chunk_pick_method?: string;
  related_chunk_number?: number;
  system_prompt_template?: string;
  enable_llm_cache?: boolean;
};

export type QueryGraphStorage = Pick<BaseGraphStorage, 'get_nodes_batch' | 'node_degrees_batch' | 'get_nodes_edges_batch' | 'get_edges_batch' | 'edge_degrees_batch'>;

export type QueryVectorStorage = {
  readonly cosine_better_than_threshold: number;
  query(query: string, top_k: number, query_embedding?: ArrayLike<number> | null): Promise<VectorQueryResult[]>;
  get_vectors_by_ids(ids: string[]): Promise<Record<string, ArrayLike<number>>>;
};

/** text_chunks_db: upstream reads both its records and its `global_config` / `embedding_func`. */
export type QueryTextChunksStorage = {
  readonly global_config: QueryGlobalConfig;
  readonly embedding_func?: QueryEmbeddingFunc | null;
  get_by_ids(ids: string[]): Promise<(Dict | null)[]>;
};

/** chunk_id -> where it came from (C vector, E entity, R relation), occurrence count, 1-based order. */
export type ChunkTracking = Map<string, { source: string; frequency: number; order: number }>;

const logOf = (c: { log?: QueryLog } | null | undefined): QueryLog => c?.log ?? noLog;

/** `query_param.model_func` or `partial(global_config["llm_model_func"], _priority=5)`. */
function queryModelFunc(qp: QueryParam, gc: QueryGlobalConfig): QueryLlmFunc {
  if (qp.model_func) return qp.model_func;
  return (prompt: string, opts: QueryLlmOptions = {}) => {
    const f = gc.llm_model_func;
    if (!f) throw new Error("KeyError: 'llm_model_func'");
    return f(prompt, { _priority: 5, ...opts });
  };
}

const pyTypeName = (v: unknown): string =>
  v === null || v === undefined ? 'NoneType' : Array.isArray(v) ? 'list' : typeof v === 'string' ? 'str' : typeof v === 'number' ? (Number.isInteger(v) ? 'int' : 'float') : typeof v === 'boolean' ? 'bool' : 'dict';

/** `obj.get(key, [])` on what json_repair returned; a non-dict raises like Python. */
function dictGet(obj: unknown, key: string): unknown {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    throw Object.assign(new Error(`'${pyTypeName(obj)}' object has no attribute 'get'`), { name: 'AttributeError' });
  }
  return pyGet(obj as Dict, key, []);
}

/** `", ".join(x) if x else ""` for whatever the keyword JSON held. */
function joinKeywords(x: unknown): string {
  if (!pyTruthy(x)) return '';
  const items: unknown[] = typeof x === 'string' ? Array.from(x) : Array.isArray(x) ? x : typeof x === 'object' ? Object.keys(x as Dict) : [x];
  if (typeof x !== 'string' && !Array.isArray(x) && (typeof x !== 'object' || x === null)) throw new TypeError(`can only join an iterable`);
  items.forEach((it, i) => {
    if (typeof it !== 'string') throw new TypeError(`sequence item ${i}: expected str instance, ${pyTypeName(it)} found`);
  });
  return items.join(', ');
}

const isEmptyList = (x: unknown): boolean => Array.isArray(x) && x.length === 0;

// ------------------------------------------------------------- cache keys

/** compute_args_hash arguments of extract_keywords_only. */
export function keywordsArgsHash(mode: string, text: string, language: string): string {
  return computeArgsHashPy(mode, text, language);
}

/** compute_args_hash arguments of kg_query (keywords already joined with ", "). */
export function kgQueryArgsHash(qp: QueryParam, query: string, hlKeywordsStr: string, llKeywordsStr: string): string {
  return computeArgsHashPy(
    qp.mode,
    query,
    qp.response_type,
    qp.top_k,
    qp.chunk_top_k,
    qp.max_entity_tokens,
    qp.max_relation_tokens,
    qp.max_total_tokens,
    hlKeywordsStr,
    llKeywordsStr,
    qp.user_prompt || '',
    qp.enable_rerank,
  );
}

/** compute_args_hash arguments of naive_query. */
export function naiveQueryArgsHash(qp: QueryParam, query: string): string {
  return computeArgsHashPy(
    qp.mode,
    query,
    qp.response_type,
    qp.top_k,
    qp.chunk_top_k,
    qp.max_entity_tokens,
    qp.max_relation_tokens,
    qp.max_total_tokens,
    qp.user_prompt || '',
    qp.enable_rerank,
  );
}

/** The `queryparam` dict upstream stores next to a keywords / naive cache entry. */
export function queryParamDict(qp: QueryParam): Dict {
  return {
    mode: qp.mode,
    response_type: qp.response_type,
    top_k: qp.top_k,
    chunk_top_k: qp.chunk_top_k,
    max_entity_tokens: qp.max_entity_tokens,
    max_relation_tokens: qp.max_relation_tokens,
    max_total_tokens: qp.max_total_tokens,
    user_prompt: qp.user_prompt || '',
    enable_rerank: qp.enable_rerank,
  };
}

/** The `queryparam` dict of a kg_query cache entry (adds the joined keywords). */
export function kgQueryParamDict(qp: QueryParam, hlKeywordsStr: string, llKeywordsStr: string): Dict {
  const { user_prompt, enable_rerank, ...head } = queryParamDict(qp);
  return { ...head, hl_keywords: hlKeywordsStr, ll_keywords: llKeywordsStr, user_prompt, enable_rerank };
}

/** Cache keys a query would read, without retrieval: `{mode}:keywords:` and `{mode}:query:`. */
export function queryCacheKeys(
  query: string,
  qp: QueryParam,
  o: { language?: string; hlKeywords?: string[]; llKeywords?: string[] } = {},
): { keywords?: string; query: string } {
  if (qp.mode === 'naive') return { query: generateCacheKey(qp.mode, 'query', naiveQueryArgsHash(qp, query)) };
  const keywords = generateCacheKey(qp.mode, 'keywords', keywordsArgsHash(qp.mode, query, o.language ?? DEFAULT_SUMMARY_LANGUAGE));
  const hl = joinKeywords(o.hlKeywords ?? []);
  const ll = joinKeywords(o.llKeywords ?? []);
  return { keywords, query: generateCacheKey(qp.mode, 'query', kgQueryArgsHash(qp, query, hl, ll)) };
}

// ------------------------------------------------------------------ kg_query

/** operate.kg_query */
export async function kgQuery(
  query: string,
  knowledgeGraphInst: QueryGraphStorage,
  entitiesVdb: QueryVectorStorage,
  relationshipsVdb: QueryVectorStorage,
  textChunksDb: QueryTextChunksStorage,
  queryParam: QueryParam,
  globalConfig: QueryGlobalConfig,
  hashingKv: LlmResponseCache | null = null,
  systemPrompt: string | null = null,
  chunksVdb: QueryVectorStorage | null = null,
): Promise<QueryResult | null> {
  const log = logOf(globalConfig);
  if (!query) return new QueryResult({ content: PROMPTS.fail_response });
  const useModelFunc = queryModelFunc(queryParam, globalConfig);

  const keywords = await resolveKgKeywords(query, queryParam, globalConfig, hashingKv);
  if (keywords === null) return new QueryResult({ content: PROMPTS.fail_response });
  const { hl: hlKeywordsStr, ll: llKeywordsStr } = keywords;

  const contextResult = await buildQueryContext(
    query,
    llKeywordsStr,
    hlKeywordsStr,
    knowledgeGraphInst,
    entitiesVdb,
    relationshipsVdb,
    textChunksDb,
    queryParam,
    chunksVdb,
  );
  if (contextResult === null) {
    log('info', '[kg_query] No query context could be built; returning no-result.');
    return null;
  }
  if (queryParam.only_need_context && !queryParam.only_need_prompt) {
    return new QueryResult({ content: contextResult.context, raw_data: contextResult.raw_data });
  }

  const userPrompt = queryParam.user_prompt ? `\n\n${queryParam.user_prompt}` : 'n/a';
  const responseType = queryParam.response_type ? queryParam.response_type : 'Multiple Paragraphs';
  const sysPromptTemp = systemPrompt ? systemPrompt : PROMPTS.rag_response;
  const sysPrompt = pyFormat(sysPromptTemp, { response_type: responseType, user_prompt: userPrompt, context_data: contextResult.context });
  const userQuery = query;
  if (queryParam.only_need_prompt) {
    return new QueryResult({ content: [sysPrompt, '---User Query---', userQuery].join('\n\n'), raw_data: contextResult.raw_data });
  }

  // upstream encodes the prompt for a debug line; kept because tiktoken raises on special-token text
  const lenOfPrompts = globalConfig.tokenizer!.encode(query + sysPrompt).length;
  log('debug', `[kg_query] Sending to LLM: ${lenOfPrompts} tokens`);

  const argsHash = kgQueryArgsHash(queryParam, query, hlKeywordsStr, llKeywordsStr);
  const cached = await handleCache(hashingKv, argsHash, userQuery, queryParam.mode, 'query');
  let response: string | AsyncIterable<string>;
  if (cached !== null) {
    log('info', ' == LLM cache == Query cache hit, using cached response as query result');
    response = cached[0];
  } else {
    response = await useModelFunc(userQuery, {
      system_prompt: sysPrompt,
      history_messages: queryParam.conversation_history,
      enable_cot: true,
      stream: queryParam.stream,
    });
    if (hashingKv && hashingKv.global_config.enable_llm_cache && typeof response === 'string') {
      await saveToCache(hashingKv, {
        argsHash,
        content: response,
        prompt: query,
        mode: queryParam.mode,
        cacheType: 'query',
        queryparam: kgQueryParamDict(queryParam, hlKeywordsStr, llKeywordsStr),
      });
    }
  }

  if (typeof response === 'string') {
    if (pyLen(response) > pyLen(sysPrompt)) response = cleanResponse(response, sysPrompt, query);
    return new QueryResult({ content: response, raw_data: contextResult.raw_data });
  }
  return new QueryResult({ response_iterator: response, raw_data: contextResult.raw_data, is_streaming: true });
}

/**
 * The keyword step of kg_query: get the lists (given, cached or from the LLM),
 * force the query itself as ll keyword when both are empty and it is short,
 * join with ", ". null means kg_query answers fail_response.
 */
export async function resolveKgKeywords(
  query: string,
  queryParam: QueryParam,
  globalConfig: QueryGlobalConfig,
  hashingKv: LlmResponseCache | null = null,
): Promise<{ hl: string; ll: string } | null> {
  const log = logOf(globalConfig);
  const [hlKeywords, kw] = (await getKeywordsFromQuery(query, queryParam, globalConfig, hashingKv)) as [unknown, unknown];
  let llKeywords = kw;
  if (isEmptyList(llKeywords) && ['local', 'hybrid', 'mix'].includes(queryParam.mode)) log('warning', 'low_level_keywords is empty');
  if (isEmptyList(hlKeywords) && ['global', 'hybrid', 'mix'].includes(queryParam.mode)) log('warning', 'high_level_keywords is empty');
  if (isEmptyList(hlKeywords) && isEmptyList(llKeywords)) {
    if (pyLen(query) >= 50) return null;
    log('warning', `Forced low_level_keywords to origin query: ${query}`);
    llKeywords = [query];
  }
  return { hl: joinKeywords(hlKeywords), ll: joinKeywords(llKeywords) };
}

/** The `.replace(...)...strip()` chain both query functions apply to an over-long answer. */
function cleanResponse(response: string, sysPrompt: string, query: string): string {
  return pyStrip(
    response
      .replaceAll(sysPrompt, '')
      .replaceAll('user', '')
      .replaceAll('model', '')
      .replaceAll(query, '')
      .replaceAll('<system>', '')
      .replaceAll('</system>', ''),
  );
}

/** operate.get_keywords_from_query */
export async function getKeywordsFromQuery(
  query: string,
  queryParam: QueryParam,
  globalConfig: QueryGlobalConfig,
  hashingKv: LlmResponseCache | null = null,
): Promise<[string[], string[]]> {
  if (pyTruthy(queryParam.hl_keywords) || pyTruthy(queryParam.ll_keywords)) return [queryParam.hl_keywords, queryParam.ll_keywords];
  return extractKeywordsOnly(query, queryParam, globalConfig, hashingKv);
}

/**
 * operate.extract_keywords_only. Returns whatever lists the keyword JSON held
 * (upstream does not validate their element types either).
 */
export async function extractKeywordsOnly(
  text: string,
  param: QueryParam,
  globalConfig: QueryGlobalConfig,
  hashingKv: LlmResponseCache | null = null,
): Promise<[string[], string[]]> {
  const log = logOf(globalConfig);
  const examples = PROMPTS.keywords_extraction_examples.join('\n');
  const language = pyGet(globalConfig.addon_params ?? {}, 'language', DEFAULT_SUMMARY_LANGUAGE);
  const argsHash = keywordsArgsHash(param.mode, text, language);
  const cached = await handleCache(hashingKv, argsHash, text, param.mode, 'keywords');
  if (cached !== null) {
    const data = jsonRepairLoads(cached[0]);
    return [dictGet(data, 'high_level_keywords'), dictGet(data, 'low_level_keywords')] as [string[], string[]];
  }

  const kwPrompt = pyFormat(PROMPTS.keywords_extraction, { query: text, examples, language });
  const lenOfPrompts = globalConfig.tokenizer!.encode(kwPrompt).length;
  log('debug', `[extract_keywords] Sending to LLM: ${lenOfPrompts} tokens (Prompt: ${lenOfPrompts})`);

  const useModelFunc = queryModelFunc(param, globalConfig);
  const raw = await useModelFunc(kwPrompt, { keyword_extraction: true });
  if (typeof raw !== 'string') throw new TypeError('expected string or bytes-like object, got an async iterator');
  const result = removeThinkTags(raw);
  const keywordsData = jsonRepairLoads(result);
  if (!pyTruthy(keywordsData)) {
    log('error', 'No JSON-like structure found in the LLM respond.');
    return [[], []];
  }
  const hlKeywords = dictGet(keywordsData, 'high_level_keywords');
  const llKeywords = dictGet(keywordsData, 'low_level_keywords');

  if (pyTruthy(hlKeywords) || pyTruthy(llKeywords)) {
    const cacheData = { high_level_keywords: hlKeywords, low_level_keywords: llKeywords };
    // upstream dereferences hashing_kv here unguarded; a missing cache is skipped instead
    if (hashingKv?.global_config.enable_llm_cache) {
      await saveToCache(hashingKv, {
        argsHash,
        content: pyJsonDumpsAny(cacheData, true),
        prompt: text,
        mode: param.mode,
        cacheType: 'keywords',
        queryparam: queryParamDict(param),
      });
    }
  }
  return [hlKeywords, llKeywords] as [string[], string[]];
}

// ----------------------------------------------------------- retrieval stage

/** operate._get_vector_context */
export async function getVectorContext(
  query: string,
  chunksVdb: QueryVectorStorage,
  queryParam: QueryParam,
  queryEmbedding: ArrayLike<number> | null = null,
  log: QueryLog = noLog,
): Promise<Dict[]> {
  try {
    const searchTopK = queryParam.chunk_top_k || queryParam.top_k;
    const results = await chunksVdb.query(query, searchTopK, queryEmbedding);
    if (!results.length) {
      log('info', `Naive query: 0 chunks (chunk_top_k:${searchTopK} cosine:${chunksVdb.cosine_better_than_threshold})`);
      return [];
    }
    const valid: Dict[] = [];
    for (const r of results) {
      if ('content' in r) {
        valid.push({
          content: r.content,
          created_at: pyGet(r, 'created_at', null),
          file_path: pyGet(r, 'file_path', 'unknown_source'),
          source_type: 'vector',
          chunk_id: pyGet(r, 'id', null),
        });
      }
    }
    log('info', `Naive query: ${valid.length} chunks (chunk_top_k:${searchTopK} cosine:${chunksVdb.cosine_better_than_threshold})`);
    return valid;
  } catch (e) {
    log('error', `Error in _get_vector_context: ${e}`);
    return [];
  }
}

export type KgSearchResult = {
  final_entities: Dict[];
  final_relations: Dict[];
  vector_chunks: Dict[];
  chunk_tracking: ChunkTracking;
  query_embedding: ArrayLike<number> | null;
};

const relationKey = (r: Dict): string => {
  const pair = Object.hasOwn(r, 'src_tgt') ? (r.src_tgt as [string, string]) : [pyGet(r, 'src_id', null), pyGet(r, 'tgt_id', null)];
  return JSON.stringify([...pair].sort((a, b) => pyCompare(String(a), String(b))));
};

/** operate._perform_kg_search */
export async function performKgSearch(
  query: string,
  llKeywords: string,
  hlKeywords: string,
  knowledgeGraphInst: QueryGraphStorage,
  entitiesVdb: QueryVectorStorage,
  relationshipsVdb: QueryVectorStorage,
  textChunksDb: QueryTextChunksStorage,
  queryParam: QueryParam,
  chunksVdb: QueryVectorStorage | null = null,
): Promise<KgSearchResult> {
  const log = logOf(textChunksDb.global_config);
  let localEntities: Dict[] = [];
  let localRelations: Dict[] = [];
  let globalEntities: Dict[] = [];
  let globalRelations: Dict[] = [];
  let vectorChunks: Dict[] = [];
  const chunkTracking: ChunkTracking = new Map();

  const kgChunkPickMethod = pyGet(textChunksDb.global_config, 'kg_chunk_pick_method', DEFAULT_KG_CHUNK_PICK_METHOD);
  const embed = textChunksDb.embedding_func ?? null;
  let queryEmbedding: ArrayLike<number> | null = null;
  let llEmbedding: ArrayLike<number> | null = null;
  let hlEmbedding: ArrayLike<number> | null = null;
  const mode = queryParam.mode;
  const needLl = ['local', 'hybrid', 'mix'].includes(mode) && Boolean(llKeywords);
  const needHl = ['global', 'hybrid', 'mix'].includes(mode) && Boolean(hlKeywords);

  if (embed) {
    const texts: string[] = [];
    const purposes: ('query' | 'll' | 'hl')[] = [];
    if (query && (kgChunkPickMethod === 'VECTOR' || chunksVdb)) {
      texts.push(query);
      purposes.push('query');
    }
    if (needLl) {
      texts.push(llKeywords);
      purposes.push('ll');
    }
    if (needHl) {
      texts.push(hlKeywords);
      purposes.push('hl');
    }
    if (texts.length) {
      try {
        const all = await embed(texts, { context: 'query', _priority: 5 });
        // assigned one by one: an IndexError midway keeps the earlier ones, as upstream
        for (let i = 0; i < purposes.length; i++) {
          if (i >= all.length) throw new Error('index out of range');
          const p = purposes[i];
          if (p === 'query') queryEmbedding = all[i]!;
          else if (p === 'll') llEmbedding = all[i]!;
          else hlEmbedding = all[i]!;
        }
      } catch (e) {
        log('warning', `Failed to batch pre-compute embeddings: ${e}`);
      }
    }
  }

  if (mode === 'local' && llKeywords.length > 0) {
    [localEntities, localRelations] = await getNodeData(llKeywords, knowledgeGraphInst, entitiesVdb, queryParam, llEmbedding, log);
  } else if (mode === 'global' && hlKeywords.length > 0) {
    [globalRelations, globalEntities] = await getEdgeData(hlKeywords, knowledgeGraphInst, relationshipsVdb, queryParam, hlEmbedding, log);
  } else {
    if (llKeywords.length > 0) {
      [localEntities, localRelations] = await getNodeData(llKeywords, knowledgeGraphInst, entitiesVdb, queryParam, llEmbedding, log);
    }
    if (hlKeywords.length > 0) {
      [globalRelations, globalEntities] = await getEdgeData(hlKeywords, knowledgeGraphInst, relationshipsVdb, queryParam, hlEmbedding, log);
    }
    if (mode === 'mix' && chunksVdb) {
      vectorChunks = await getVectorContext(query, chunksVdb, queryParam, queryEmbedding, log);
      vectorChunks.forEach((chunk, i) => {
        const chunkId = (pyGet(chunk, 'chunk_id', null) || pyGet(chunk, 'id', null)) as string | null;
        if (chunkId) chunkTracking.set(chunkId, { source: 'C', frequency: 1, order: i + 1 });
        else log('warning', `Vector chunk missing chunk_id: ${pyJsonDumpsAny(chunk)}`);
      });
    }
  }

  const finalEntities: Dict[] = [];
  const seenEntities = new Set<string>();
  for (let i = 0; i < Math.max(localEntities.length, globalEntities.length); i++) {
    for (const list of [localEntities, globalEntities]) {
      if (i < list.length) {
        const entity = list[i]!;
        const name = pyGet(entity, 'entity_name', null) as string | null;
        if (name && !seenEntities.has(name)) {
          finalEntities.push(entity);
          seenEntities.add(name);
        }
      }
    }
  }
  const finalRelations: Dict[] = [];
  const seenRelations = new Set<string>();
  for (let i = 0; i < Math.max(localRelations.length, globalRelations.length); i++) {
    for (const list of [localRelations, globalRelations]) {
      if (i < list.length) {
        const relation = list[i]!;
        const k = relationKey(relation);
        if (!seenRelations.has(k)) {
          finalRelations.push(relation);
          seenRelations.add(k);
        }
      }
    }
  }
  log('info', `Raw search results: ${finalEntities.length} entities, ${finalRelations.length} relations, ${vectorChunks.length} vector chunks`);
  return { final_entities: finalEntities, final_relations: finalRelations, vector_chunks: vectorChunks, chunk_tracking: chunkTracking, query_embedding: queryEmbedding };
}

export type TruncationResult = {
  entities_context: Dict[];
  relations_context: Dict[];
  filtered_entities: Dict[];
  filtered_relations: Dict[];
  entity_id_to_original: Map<string, Dict>;
  /** keyed by pairKey(entity1, entity2) */
  relation_id_to_original: Map<string, Dict>;
};

/**
 * operate._apply_token_truncation. Upstream also formats each `created_at`
 * with time.localtime, then pops the field before truncation; nothing reads
 * it, so it is not computed here.
 */
export async function applyTokenTruncation(
  searchResult: KgSearchResult,
  queryParam: QueryParam,
  globalConfig: QueryGlobalConfig,
): Promise<TruncationResult> {
  const log = logOf(globalConfig);
  const tokenizer = globalConfig.tokenizer;
  if (!tokenizer) {
    log('warning', 'No tokenizer found, skipping truncation');
    return {
      entities_context: [],
      relations_context: [],
      filtered_entities: searchResult.final_entities,
      filtered_relations: searchResult.final_relations,
      entity_id_to_original: new Map(),
      relation_id_to_original: new Map(),
    };
  }
  const finalEntities = searchResult.final_entities;
  const finalRelations = searchResult.final_relations;

  let entitiesContext: Dict[] = finalEntities.map((entity) => ({
    entity: entity.entity_name,
    type: pyGet(entity, 'entity_type', 'UNKNOWN'),
    description: pyGet(entity, 'description', 'UNKNOWN'),
  }));
  let relationsContext: Dict[] = finalRelations.map((relation) => {
    const [entity1, entity2] = Object.hasOwn(relation, 'src_tgt')
      ? (relation.src_tgt as [string, string])
      : [pyGet(relation, 'src_id', null), pyGet(relation, 'tgt_id', null)];
    return { entity1, entity2, description: pyGet(relation, 'description', 'UNKNOWN') };
  });
  log('debug', `Before truncation: ${entitiesContext.length} entities, ${relationsContext.length} relations`);
  if (entitiesContext.length) entitiesContext = truncateListByTokenSize(entitiesContext, jsonLineKey, queryParam.max_entity_tokens, tokenizer);
  if (relationsContext.length) relationsContext = truncateListByTokenSize(relationsContext, jsonLineKey, queryParam.max_relation_tokens, tokenizer);
  log('info', `After truncation: ${entitiesContext.length} entities, ${relationsContext.length} relations`);

  const filteredEntities: Dict[] = [];
  const entityIdToOriginal = new Map<string, Dict>();
  if (entitiesContext.length) {
    const names = new Set(entitiesContext.map((e) => e.entity as string));
    const seen = new Set<string>();
    for (const entity of finalEntities) {
      const name = pyGet<unknown>(entity, 'entity_name', null) as string;
      if (names.has(name) && !seen.has(name)) {
        filteredEntities.push(entity);
        entityIdToOriginal.set(name, entity);
        seen.add(name);
      }
    }
  }
  const filteredRelations: Dict[] = [];
  const relationIdToOriginal = new Map<string, Dict>();
  if (relationsContext.length) {
    const pairs = new Set(relationsContext.map((r) => pairKey(r.entity1, r.entity2)));
    const seen = new Set<string>();
    for (const relation of finalRelations) {
      let src = pyGet<unknown>(relation, 'src_id', null);
      let tgt = pyGet<unknown>(relation, 'tgt_id', null);
      if (src === null || tgt === null) [src, tgt] = pyGet(relation, 'src_tgt', [null, null]) as [unknown, unknown];
      const k = pairKey(src, tgt);
      if (pairs.has(k) && !seen.has(k)) {
        filteredRelations.push(relation);
        relationIdToOriginal.set(k, relation);
        seen.add(k);
      }
    }
  }
  return {
    entities_context: entitiesContext,
    relations_context: relationsContext,
    filtered_entities: filteredEntities,
    filtered_relations: filteredRelations,
    entity_id_to_original: entityIdToOriginal,
    relation_id_to_original: relationIdToOriginal,
  };
}

/** operate._merge_all_chunks: vector, entity and relation chunks, round-robin, deduplicated. */
export async function mergeAllChunks(a: {
  filteredEntities: Dict[];
  filteredRelations: Dict[];
  vectorChunks: Dict[];
  query?: string;
  knowledgeGraphInst?: QueryGraphStorage | null;
  textChunksDb?: QueryTextChunksStorage | null;
  queryParam: QueryParam;
  chunksVdb?: QueryVectorStorage | null;
  chunkTracking?: ChunkTracking | null;
  queryEmbedding?: ArrayLike<number> | null;
}): Promise<Dict[]> {
  const tracking = a.chunkTracking ?? new Map();
  const query = a.query ?? '';
  let entityChunks: Dict[] = [];
  if (a.filteredEntities.length && a.textChunksDb) {
    entityChunks = await findRelatedTextUnitFromEntities(
      a.filteredEntities,
      a.queryParam,
      a.textChunksDb,
      a.knowledgeGraphInst ?? null,
      query,
      a.chunksVdb ?? null,
      tracking,
      a.queryEmbedding ?? null,
    );
  }
  let relationChunks: Dict[] = [];
  if (a.filteredRelations.length && a.textChunksDb) {
    relationChunks = await findRelatedTextUnitFromRelations(
      a.filteredRelations,
      a.queryParam,
      a.textChunksDb,
      entityChunks,
      query,
      a.chunksVdb ?? null,
      tracking,
      a.queryEmbedding ?? null,
    );
  }
  const merged: Dict[] = [];
  const seen = new Set<string>();
  const maxLen = Math.max(a.vectorChunks.length, entityChunks.length, relationChunks.length);
  for (let i = 0; i < maxLen; i++) {
    for (const list of [a.vectorChunks, entityChunks, relationChunks]) {
      if (i >= list.length) continue;
      const chunk = list[i]!;
      const chunkId = (pyGet(chunk, 'chunk_id', null) || pyGet(chunk, 'id', null)) as string | null;
      if (chunkId && !seen.has(chunkId)) {
        seen.add(chunkId);
        merged.push({ content: chunk.content, file_path: pyGet(chunk, 'file_path', 'unknown_source'), chunk_id: chunkId });
      }
    }
  }
  const origin = a.vectorChunks.length + entityChunks.length + relationChunks.length;
  logOf(a.textChunksDb?.global_config)('info', `Round-robin merged chunks: ${origin} -> ${merged.length} (deduplicated ${origin - merged.length})`);
  return merged;
}

const emptyRawData = (mode: string, message: string): QueryRawData => {
  const raw = convertToUserFormat([], [], [], [], mode);
  raw.status = 'failure';
  raw.message = message;
  return raw;
};

/** operate._build_context_str */
export async function buildContextStr(a: {
  entitiesContext: Dict[];
  relationsContext: Dict[];
  mergedChunks: Dict[];
  query: string;
  queryParam: QueryParam;
  globalConfig: QueryGlobalConfig;
  chunkTracking?: ChunkTracking | null;
  entityIdToOriginal?: Map<string, Dict> | null;
  relationIdToOriginal?: Map<string, Dict> | null;
}): Promise<[string, QueryRawData]> {
  const qp = a.queryParam;
  const log = logOf(a.globalConfig);
  const tokenizer = a.globalConfig.tokenizer;
  if (!tokenizer) {
    log('error', 'Missing tokenizer, cannot build LLM context');
    return ['', emptyRawData(qp.mode, 'Missing tokenizer, cannot build LLM context.')];
  }
  const maxTotalTokens = qp.max_total_tokens;
  const sysPromptTemplate = pyGet(a.globalConfig, 'system_prompt_template', PROMPTS.rag_response);
  const kgContextTemplate = PROMPTS.kg_query_context;
  const userPrompt = qp.user_prompt ? qp.user_prompt : '';
  const responseType = qp.response_type ? qp.response_type : 'Multiple Paragraphs';

  const entitiesStr = a.entitiesContext.map(jsonLineKey).join('\n');
  const relationsStr = a.relationsContext.map(jsonLineKey).join('\n');
  const preKgContext = pyFormat(kgContextTemplate, { entities_str: entitiesStr, relations_str: relationsStr, text_chunks_str: '', reference_list_str: '' });
  const kgContextTokens = tokenizer.encode(preKgContext).length;
  const preSysPrompt = pyFormat(sysPromptTemplate, { context_data: '', response_type: responseType, user_prompt: userPrompt });
  const sysPromptTokens = tokenizer.encode(preSysPrompt).length;
  const queryTokens = tokenizer.encode(a.query).length;
  const bufferTokens = 200;
  const availableChunkTokens = maxTotalTokens - (sysPromptTokens + kgContextTokens + queryTokens + bufferTokens);
  log(
    'debug',
    `Token allocation - Total: ${maxTotalTokens}, SysPrompt: ${sysPromptTokens}, Query: ${queryTokens}, KG: ${kgContextTokens}, Buffer: ${bufferTokens}, Available for chunks: ${availableChunkTokens}`,
  );

  const truncated = await processChunksUnified({
    query: a.query,
    uniqueChunks: a.mergedChunks,
    queryParam: qp,
    globalConfig: a.globalConfig,
    sourceType: qp.mode,
    chunkTokenLimit: availableChunkTokens,
  });
  const [referenceList, truncatedChunks] = generateReferenceListFromChunks(truncated);
  const chunksContext = truncatedChunks.map((c) => ({ reference_id: c.reference_id, content: c.content }));
  const textUnitsStr = chunksContext.map(jsonLineKey).join('\n');
  const referenceListStr = referenceList
    .filter((r) => pyTruthy(r.reference_id))
    .map((r) => `[${r.reference_id}] ${r.file_path}`)
    .join('\n');
  log('info', `Final context: ${a.entitiesContext.length} entities, ${a.relationsContext.length} relations, ${chunksContext.length} chunks`);

  if (!a.entitiesContext.length && !a.relationsContext.length && !chunksContext.length) {
    return ['', emptyRawData(qp.mode, 'Query returned empty dataset.')];
  }
  if (truncatedChunks.length && a.chunkTracking?.size) {
    const parts = truncatedChunks.map((c) => {
      const t = c.chunk_id ? a.chunkTracking!.get(c.chunk_id as string) : undefined;
      return t ? `${t.source}${t.frequency}/${t.order}` : '?0/0';
    });
    log('info', `Final chunks S+F/O: ${parts.join(' ')}`);
  }
  const result = pyFormat(kgContextTemplate, {
    entities_str: entitiesStr,
    relations_str: relationsStr,
    text_chunks_str: textUnitsStr,
    reference_list_str: referenceListStr,
  });
  const finalData = convertToUserFormat(
    a.entitiesContext,
    a.relationsContext,
    truncatedChunks,
    referenceList,
    qp.mode,
    a.entityIdToOriginal ?? null,
    a.relationIdToOriginal ?? null,
  );
  return [result, finalData];
}

/** operate._build_query_context: search -> truncate -> merge chunks -> context string. */
export async function buildQueryContext(
  query: string,
  llKeywords: string,
  hlKeywords: string,
  knowledgeGraphInst: QueryGraphStorage,
  entitiesVdb: QueryVectorStorage,
  relationshipsVdb: QueryVectorStorage,
  textChunksDb: QueryTextChunksStorage,
  queryParam: QueryParam,
  chunksVdb: QueryVectorStorage | null = null,
): Promise<QueryContextResult | null> {
  if (!query) {
    logOf(textChunksDb.global_config)('warning', 'Query is empty, skipping context building');
    return null;
  }
  const searchResult = await performKgSearch(
    query,
    llKeywords,
    hlKeywords,
    knowledgeGraphInst,
    entitiesVdb,
    relationshipsVdb,
    textChunksDb,
    queryParam,
    chunksVdb,
  );
  if (!searchResult.final_entities.length && !searchResult.final_relations.length) {
    if (queryParam.mode !== 'mix') return null;
    if (!searchResult.chunk_tracking.size) return null;
  }
  const truncation = await applyTokenTruncation(searchResult, queryParam, textChunksDb.global_config);
  const mergedChunks = await mergeAllChunks({
    filteredEntities: truncation.filtered_entities,
    filteredRelations: truncation.filtered_relations,
    vectorChunks: searchResult.vector_chunks,
    query,
    knowledgeGraphInst,
    textChunksDb,
    queryParam,
    chunksVdb,
    chunkTracking: searchResult.chunk_tracking,
    queryEmbedding: searchResult.query_embedding,
  });
  if (!mergedChunks.length && !truncation.entities_context.length && !truncation.relations_context.length) return null;

  const [context, rawData] = await buildContextStr({
    entitiesContext: truncation.entities_context,
    relationsContext: truncation.relations_context,
    mergedChunks,
    query,
    queryParam,
    globalConfig: textChunksDb.global_config,
    chunkTracking: searchResult.chunk_tracking,
    entityIdToOriginal: truncation.entity_id_to_original,
    relationIdToOriginal: truncation.relation_id_to_original,
  });
  rawData.metadata ??= {};
  rawData.metadata.keywords = {
    high_level: hlKeywords ? hlKeywords.split(', ') : [],
    low_level: llKeywords ? llKeywords.split(', ') : [],
  };
  rawData.metadata.processing_info = {
    total_entities_found: searchResult.final_entities.length,
    total_relations_found: searchResult.final_relations.length,
    entities_after_truncation: truncation.filtered_entities.length,
    relations_after_truncation: truncation.filtered_relations.length,
    merged_chunks_count: mergedChunks.length,
    final_chunks_count: rawData.data?.chunks?.length ?? 0,
  };
  return new QueryContextResult(context, rawData);
}

/** operate._get_node_data (local retrieval: entities by ll_keywords, then their edges). */
export async function getNodeData(
  query: string,
  knowledgeGraphInst: QueryGraphStorage,
  entitiesVdb: QueryVectorStorage,
  queryParam: QueryParam,
  queryEmbedding: ArrayLike<number> | null = null,
  log: QueryLog = noLog,
): Promise<[Dict[], Dict[]]> {
  log('info', `Query nodes: ${query} (top_k:${queryParam.top_k}, cosine:${entitiesVdb.cosine_better_than_threshold})`);
  const results = await entitiesVdb.query(query, queryParam.top_k, queryEmbedding);
  if (!results.length) return [[], []];
  const nodeIds = results.map((r) => {
    if (!Object.hasOwn(r, 'entity_name')) throw new Error("KeyError: 'entity_name'");
    return r.entity_name as string;
  });
  const [nodesDict, degreesDict] = await Promise.all([knowledgeGraphInst.get_nodes_batch(nodeIds), knowledgeGraphInst.node_degrees_batch(nodeIds)]);
  const nodeDatas: Dict[] = [];
  results.forEach((k, i) => {
    const n: NodeData | undefined = nodesDict.get(nodeIds[i]!);
    if (n === undefined || n === null) return;
    nodeDatas.push({ ...n, entity_name: k.entity_name, rank: degreesDict.get(nodeIds[i]!) ?? 0, created_at: pyGet(k, 'created_at', null) });
  });
  if (nodeDatas.length !== results.length) log('warning', 'Some nodes are missing, maybe the storage is damaged');
  const useRelations = await findMostRelatedEdgesFromEntities(nodeDatas, queryParam, knowledgeGraphInst, log);
  log('info', `Local query: ${nodeDatas.length} entites, ${useRelations.length} relations`);
  return [nodeDatas, useRelations];
}

/** operate._find_most_related_edges_from_entities */
export async function findMostRelatedEdgesFromEntities(
  nodeDatas: Dict[],
  _queryParam: QueryParam,
  knowledgeGraphInst: QueryGraphStorage,
  log: QueryLog = noLog,
): Promise<Dict[]> {
  const nodeNames = nodeDatas.map((dp) => dp.entity_name as string);
  const batchEdges = await knowledgeGraphInst.get_nodes_edges_batch(nodeNames);
  const allEdges: [string, string][] = [];
  const seen = new Set<string>();
  for (const name of nodeNames) {
    for (const e of batchEdges.get(name) ?? []) {
      const sorted = [...e].sort(pyCompare) as [string, string];
      const k = JSON.stringify(sorted);
      if (!seen.has(k)) {
        seen.add(k);
        allEdges.push(sorted);
      }
    }
  }
  const [edgeDataDict, edgeDegreesDict] = await Promise.all([
    knowledgeGraphInst.get_edges_batch(allEdges.map(([src, tgt]) => ({ src, tgt }))),
    knowledgeGraphInst.edge_degrees_batch(allEdges.map(([s, t]) => [s, t] as [string, string])),
  ]);
  const allEdgesData: Dict[] = [];
  for (const pair of allEdges) {
    const props: EdgeData | undefined = edgeDataDict.get(edgePairKey(pair[0], pair[1]));
    if (props === undefined || props === null) continue;
    if (!Object.hasOwn(props, 'weight')) {
      log('warning', `Edge ${JSON.stringify(pair)} missing 'weight' attribute, using default value 1.0`);
      props.weight = 1.0;
    }
    allEdgesData.push({ src_tgt: pair, rank: edgeDegreesDict.get(edgePairKey(pair[0], pair[1])) ?? 0, ...props });
  }
  // Python's sorted(reverse=True) keeps equal keys in input order, as a stable descending sort does
  allEdgesData.sort((x, y) => (y.rank as number) - (x.rank as number) || (y.weight as number) - (x.weight as number));
  return allEdgesData;
}

type WithChunks = { chunks: string[]; sorted_chunks?: string[]; [k: string]: unknown };

/** Shared steps 4-6 of the two _find_related_text_unit_from_* functions. */
async function selectAndFetchChunks(
  items: WithChunks[],
  totalChunks: number,
  occurrence: Map<string, number>,
  sourceType: 'entity' | 'relationship',
  textChunksDb: QueryTextChunksStorage,
  query: string,
  chunksVdb: QueryVectorStorage | null,
  chunkTracking: ChunkTracking | null,
  queryEmbedding: ArrayLike<number> | null,
): Promise<Dict[]> {
  const gc = textChunksDb.global_config;
  const log = logOf(gc);
  let method = pyGet(gc, 'kg_chunk_pick_method', DEFAULT_KG_CHUNK_PICK_METHOD);
  const maxRelated = pyGet(gc, 'related_chunk_number', DEFAULT_RELATED_CHUNK_NUMBER);
  const label = sourceType === 'entity' ? 'entity' : 'relation';
  let selected: string[] = [];
  if (method === 'VECTOR' && query && chunksVdb) {
    const numOfChunks = Math.trunc((maxRelated * items.length) / 2);
    const embed = textChunksDb.embedding_func ?? null;
    if (!embed) {
      log('warning', 'No embedding function found, falling back to WEIGHT method');
      method = 'WEIGHT';
    } else {
      try {
        selected = await pickByVectorSimilarity({ query, chunksVdb, numOfChunks, entityInfo: items, embeddingFunc: embed, queryEmbedding, log });
        if (!selected.length) {
          method = 'WEIGHT';
          log('warning', `No ${label}-related chunks selected by vector similarity, falling back to WEIGHT method`);
        } else {
          log('info', `Selecting ${selected.length} from ${totalChunks} ${label}-related chunks by vector similarity`);
        }
      } catch (e) {
        log('error', `Error in vector similarity sorting: ${e}, falling back to WEIGHT method`);
        method = 'WEIGHT';
      }
    }
  }
  if (method === 'WEIGHT') {
    selected = pickByWeightedPolling(items, maxRelated, 1);
    log('info', `Selecting ${selected.length} from ${totalChunks} ${label}-related chunks by weighted polling`);
  }
  if (!selected.length) return [];
  const uniqueIds = [...new Set(selected)];
  const dataList = await textChunksDb.get_by_ids(uniqueIds);
  const out: Dict[] = [];
  uniqueIds.forEach((chunkId, i) => {
    const data = dataList[i];
    if (data !== null && data !== undefined && 'content' in data) {
      out.push({ ...data, source_type: sourceType, chunk_id: chunkId });
      chunkTracking?.set(chunkId, { source: sourceType === 'entity' ? 'E' : 'R', frequency: occurrence.get(chunkId) ?? 1, order: i + 1 });
    }
  });
  return out;
}

const sortByOccurrence = (chunks: string[], occurrence: Map<string, number>): string[] =>
  [...chunks].sort((a, b) => (occurrence.get(b) ?? 0) - (occurrence.get(a) ?? 0));

/** operate._find_related_text_unit_from_entities */
export async function findRelatedTextUnitFromEntities(
  nodeDatas: Dict[],
  _queryParam: QueryParam,
  textChunksDb: QueryTextChunksStorage,
  _knowledgeGraphInst: QueryGraphStorage | null,
  query = '',
  chunksVdb: QueryVectorStorage | null = null,
  chunkTracking: ChunkTracking | null = null,
  queryEmbedding: ArrayLike<number> | null = null,
): Promise<Dict[]> {
  if (!nodeDatas.length) return [];
  const items: WithChunks[] = [];
  for (const entity of nodeDatas) {
    if (pyTruthy(pyGet(entity, 'source_id', null))) {
      const chunks = splitStringByMultiMarkers(entity.source_id as string, [GRAPH_FIELD_SEP]);
      if (chunks.length) items.push({ entity_name: entity.entity_name, chunks, entity_data: entity });
    }
  }
  if (!items.length) {
    logOf(textChunksDb.global_config)('warning', 'No entities with text chunks found');
    return [];
  }
  const occurrence = new Map<string, number>();
  for (const info of items) {
    const dedup: string[] = [];
    for (const id of info.chunks) {
      const c = (occurrence.get(id) ?? 0) + 1;
      occurrence.set(id, c);
      if (c === 1) dedup.push(id);
    }
    info.chunks = dedup;
  }
  let total = 0;
  for (const info of items) {
    info.sorted_chunks = sortByOccurrence(info.chunks, occurrence);
    total += info.sorted_chunks.length;
  }
  return selectAndFetchChunks(items, total, occurrence, 'entity', textChunksDb, query, chunksVdb, chunkTracking, queryEmbedding);
}

/** operate._get_edge_data (global retrieval: relations by hl_keywords, then their entities). */
export async function getEdgeData(
  keywords: string,
  knowledgeGraphInst: QueryGraphStorage,
  relationshipsVdb: QueryVectorStorage,
  queryParam: QueryParam,
  queryEmbedding: ArrayLike<number> | null = null,
  log: QueryLog = noLog,
): Promise<[Dict[], Dict[]]> {
  log('info', `Query edges: ${keywords} (top_k:${queryParam.top_k}, cosine:${relationshipsVdb.cosine_better_than_threshold})`);
  const results = await relationshipsVdb.query(keywords, queryParam.top_k, queryEmbedding);
  if (!results.length) return [[], []];
  const edgeDataDict = await knowledgeGraphInst.get_edges_batch(results.map((r) => ({ src: r.src_id as string, tgt: r.tgt_id as string })));
  const edgeDatas: Dict[] = [];
  for (const k of results) {
    const props: EdgeData | undefined = edgeDataDict.get(edgePairKey(k.src_id as string, k.tgt_id as string));
    if (props === undefined || props === null) continue;
    if (!Object.hasOwn(props, 'weight')) {
      log('warning', `Edge ${JSON.stringify([k.src_id, k.tgt_id])} missing 'weight' attribute, using default value 1.0`);
      props.weight = 1.0;
    }
    edgeDatas.push({ src_id: k.src_id, tgt_id: k.tgt_id, created_at: pyGet(k, 'created_at', null), ...props });
  }
  const useEntities = await findMostRelatedEntitiesFromRelationships(edgeDatas, queryParam, knowledgeGraphInst, log);
  log('info', `Global query: ${useEntities.length} entites, ${edgeDatas.length} relations`);
  return [edgeDatas, useEntities];
}

/** operate._find_most_related_entities_from_relationships */
export async function findMostRelatedEntitiesFromRelationships(
  edgeDatas: Dict[],
  _queryParam: QueryParam,
  knowledgeGraphInst: QueryGraphStorage,
  log: QueryLog = noLog,
): Promise<Dict[]> {
  const names: string[] = [];
  const seen = new Set<string>();
  for (const e of edgeDatas) {
    for (const n of [e.src_id as string, e.tgt_id as string]) {
      if (!seen.has(n)) {
        names.push(n);
        seen.add(n);
      }
    }
  }
  const nodesDict = await knowledgeGraphInst.get_nodes_batch(names);
  const out: Dict[] = [];
  for (const name of names) {
    const node = nodesDict.get(name);
    if (node === undefined || node === null) {
      log('warning', `Node '${name}' not found in batch retrieval.`);
      continue;
    }
    out.push({ ...node, entity_name: name });
  }
  return out;
}

/** operate._find_related_text_unit_from_relations */
export async function findRelatedTextUnitFromRelations(
  edgeDatas: Dict[],
  _queryParam: QueryParam,
  textChunksDb: QueryTextChunksStorage,
  entityChunks: Dict[] = [],
  query = '',
  chunksVdb: QueryVectorStorage | null = null,
  chunkTracking: ChunkTracking | null = null,
  queryEmbedding: ArrayLike<number> | null = null,
): Promise<Dict[]> {
  const log = logOf(textChunksDb.global_config);
  if (!edgeDatas.length) return [];
  let items: WithChunks[] = [];
  for (const relation of edgeDatas) {
    if (pyTruthy(pyGet(relation, 'source_id', null))) {
      const chunks = splitStringByMultiMarkers(relation.source_id as string, [GRAPH_FIELD_SEP]);
      if (chunks.length) items.push({ relation_key: relationKey(relation), chunks, relation_data: relation });
    }
  }
  if (!items.length) {
    log('warning', 'No relation-related chunks found');
    return [];
  }
  const entityChunkIds = new Set<string>();
  for (const c of entityChunks) {
    const id = pyGet(c, 'chunk_id', null) as string | null;
    if (id) entityChunkIds.add(id);
  }
  const occurrence = new Map<string, number>();
  const removed = new Set<string>();
  for (const info of items) {
    const dedup: string[] = [];
    for (const id of info.chunks) {
      if (entityChunkIds.has(id)) {
        removed.add(id);
        continue;
      }
      const c = (occurrence.get(id) ?? 0) + 1;
      occurrence.set(id, c);
      if (c === 1) dedup.push(id);
    }
    info.chunks = dedup;
  }
  items = items.filter((i) => i.chunks.length);
  if (!items.length) {
    log('info', `Find no additional relations-related chunks from ${edgeDatas.length} relations`);
    return [];
  }
  let total = 0;
  for (const info of items) {
    info.sorted_chunks = sortByOccurrence(info.chunks, occurrence);
    total += info.sorted_chunks.length;
  }
  log('info', `Find ${total} additional chunks in ${items.length} relations (deduplicated ${removed.size})`);
  return selectAndFetchChunks(items, total, occurrence, 'relationship', textChunksDb, query, chunksVdb, chunkTracking, queryEmbedding);
}

// --------------------------------------------------------------- naive_query

/** operate.naive_query */
export async function naiveQuery(
  query: string,
  chunksVdb: QueryVectorStorage,
  queryParam: QueryParam,
  globalConfig: QueryGlobalConfig,
  hashingKv: LlmResponseCache | null = null,
  systemPrompt: string | null = null,
): Promise<QueryResult | null> {
  const log = logOf(globalConfig);
  if (!query) return new QueryResult({ content: PROMPTS.fail_response });
  const useModelFunc = queryModelFunc(queryParam, globalConfig);
  const tokenizer = globalConfig.tokenizer;
  if (!tokenizer) {
    log('error', 'Tokenizer not found in global configuration.');
    return new QueryResult({ content: PROMPTS.fail_response });
  }
  const chunks = await getVectorContext(query, chunksVdb, queryParam, null, log);
  if (!chunks.length) {
    log('info', '[naive_query] No relevant document chunks found; returning no-result.');
    return null;
  }
  const maxTotalTokens = queryParam.max_total_tokens;
  const userPrompt = queryParam.user_prompt ? `\n\n${queryParam.user_prompt}` : 'n/a';
  const responseType = queryParam.response_type ? queryParam.response_type : 'Multiple Paragraphs';
  const sysPromptTemplate = systemPrompt ? systemPrompt : PROMPTS.naive_rag_response;
  const preSysPrompt = pyFormat(sysPromptTemplate, { response_type: responseType, user_prompt: userPrompt, content_data: '' });
  const sysPromptTokens = tokenizer.encode(preSysPrompt).length;
  const queryTokens = tokenizer.encode(query).length;
  const bufferTokens = 200;
  const availableChunkTokens = maxTotalTokens - (sysPromptTokens + queryTokens + bufferTokens);
  log(
    'debug',
    `Naive query token allocation - Total: ${maxTotalTokens}, SysPrompt: ${sysPromptTokens}, Query: ${queryTokens}, Buffer: ${bufferTokens}, Available for chunks: ${availableChunkTokens}`,
  );
  const processed = await processChunksUnified({
    query,
    uniqueChunks: chunks,
    queryParam,
    globalConfig,
    sourceType: 'vector',
    chunkTokenLimit: availableChunkTokens,
  });
  const [referenceList, withRefs] = generateReferenceListFromChunks(processed);
  log('info', `Final context: ${withRefs.length} chunks`);
  const rawData = convertToUserFormat([], [], withRefs, referenceList, 'naive');
  rawData.metadata ??= {};
  rawData.metadata.keywords = { high_level: [], low_level: [] };
  rawData.metadata.processing_info = { total_chunks_found: chunks.length, final_chunks_count: withRefs.length };

  const textUnitsStr = withRefs.map((c) => jsonLineKey({ reference_id: c.reference_id, content: c.content })).join('\n');
  const referenceListStr = referenceList
    .filter((r) => pyTruthy(r.reference_id))
    .map((r) => `[${r.reference_id}] ${r.file_path}`)
    .join('\n');
  const contextContent = pyFormat(PROMPTS.naive_query_context, { text_chunks_str: textUnitsStr, reference_list_str: referenceListStr });
  if (queryParam.only_need_context && !queryParam.only_need_prompt) return new QueryResult({ content: contextContent, raw_data: rawData });

  // upstream passes the raw response_type here (None prints as "None"), unlike the budget prompt above
  const sysPrompt = pyFormat(sysPromptTemplate, { response_type: pyStr(queryParam.response_type), user_prompt: userPrompt, content_data: contextContent });
  const userQuery = query;
  if (queryParam.only_need_prompt) {
    return new QueryResult({ content: [sysPrompt, '---User Query---', userQuery].join('\n\n'), raw_data: rawData });
  }

  const argsHash = naiveQueryArgsHash(queryParam, query);
  const cached = await handleCache(hashingKv, argsHash, userQuery, queryParam.mode, 'query');
  let response: string | AsyncIterable<string>;
  if (cached !== null) {
    log('info', ' == LLM cache == Query cache hit, using cached response as query result');
    response = cached[0];
  } else {
    response = await useModelFunc(userQuery, {
      system_prompt: sysPrompt,
      history_messages: queryParam.conversation_history,
      enable_cot: true,
      stream: queryParam.stream,
    });
    if (hashingKv && hashingKv.global_config.enable_llm_cache && typeof response === 'string') {
      await saveToCache(hashingKv, {
        argsHash,
        content: response,
        prompt: query,
        mode: queryParam.mode,
        cacheType: 'query',
        queryparam: queryParamDict(queryParam),
      });
    }
  }
  if (typeof response === 'string') {
    if (pyLen(response) > pyLen(sysPrompt)) response = cleanResponse(Array.from(response).slice(pyLen(sysPrompt)).join(''), sysPrompt, query);
    return new QueryResult({ content: response, raw_data: rawData });
  }
  return new QueryResult({ response_iterator: response, raw_data: rawData, is_streaming: true });
}
