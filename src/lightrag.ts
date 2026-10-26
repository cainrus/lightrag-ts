// Port of lightrag/lightrag.py: the LightRAG class - configuration, storages,
// the document pipeline (enqueue, chunk, extract, merge, doc status), deletion
// with rebuild, queries, and the entity/relation editing entry points.
//
// Method and field names stay upstream's (snake_case), so code written against
// Python's LightRAG reads the same. Only the async API is ported: the sync
// wrappers (insert, query, delete_by_doc_id, ...) run an event loop to
// completion, which Node does not do.
import { mkdirSync, existsSync } from 'node:fs';
import {
  type BaseGraphStorage,
  type BaseKVStorage,
  type BaseVectorStorage,
  type DeletionResult,
  type DocProcessingStatus,
  DocStatus,
  type DocStatusValue,
  edgePairKey,
  type EmbeddingFunc,
  type GlobalConfig,
  makeQueryParam,
  type QueryLlmFunc,
  type QueryParam,
  QueryResult,
} from './base.ts';
import {
  DEFAULT_CHUNK_TOP_K,
  DEFAULT_COSINE_THRESHOLD,
  DEFAULT_EMBEDDING_TIMEOUT,
  DEFAULT_ENTITY_TYPES,
  DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
  DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
  DEFAULT_KG_CHUNK_PICK_METHOD,
  DEFAULT_LLM_TIMEOUT,
  DEFAULT_MAX_ASYNC,
  DEFAULT_MAX_ENTITY_TOKENS,
  DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
  DEFAULT_MAX_FILE_PATHS,
  DEFAULT_MAX_GLEANING,
  DEFAULT_MAX_GRAPH_NODES,
  DEFAULT_MAX_PARALLEL_INSERT,
  DEFAULT_MAX_RELATION_TOKENS,
  DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
  DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
  DEFAULT_MAX_TOTAL_TOKENS,
  DEFAULT_MIN_RERANK_SCORE,
  DEFAULT_RELATED_CHUNK_NUMBER,
  DEFAULT_SOURCE_IDS_LIMIT_METHOD,
  DEFAULT_SUMMARY_CONTEXT_SIZE,
  DEFAULT_SUMMARY_LANGUAGE,
  DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
  DEFAULT_SUMMARY_MAX_TOKENS,
  DEFAULT_TOP_K,
  GRAPH_FIELD_SEP,
} from './constants.ts';
import { PipelineCancelledException, pyError } from './exceptions.ts';
import { JsonDocStatusStorage } from './kg/json_doc_status_impl.ts';
import { JsonKVStorage, pyTruthy } from './kg/json_kv_impl.ts';
import { NanoVectorDBStorage } from './kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from './kg/networkx_impl.ts';
import { NameSpace } from './namespace.ts';
import {
  type ChunkExtraction,
  chunkingByTokenSize,
  edgeKey,
  edgeKeyParts,
  extractEntities,
  type ExtractConfig,
  type MergeConfig,
  type MergeLog,
  mergeNodesAndEdges,
  rebuildKnowledgeFromChunks,
  type TextChunk,
} from './operate.ts';
import { PROMPTS } from './prompt.ts';
import { kgQuery, naiveQuery, type QueryGlobalConfig, queryTextChunks } from './query.ts';
import { convertToUserFormat } from './query/utils.ts';
import type { KnowledgeGraph } from './types.ts';
import {
  computeMdhashId,
  generateTrackId,
  getContentSummary,
  getEnvValue,
  type LlmFunc,
  makeRelationChunkKey,
  makeRelationVdbIds,
  normalizeSourceIdsLimitMethod,
  priorityLimitAsyncFuncCall,
  pyCompare,
  pyIntParse,
  pyIsoformatUtc,
  pyLen,
  pyReprStr,
  pySliceTo,
  pyStrip,
  sanitizeTextForEncoding,
  subtractSourceIds,
  TiktokenTokenizer,
  type Tokenizer,
} from './utils.ts';
import * as utilsGraph from './utils_graph.ts';

// ---------------------------------------------------------------- types

export type LightRAGLog = (level: 'debug' | 'info' | 'warning' | 'error', msg: string) => void;

/** LightRAG.chunking_func: sync or async, returns [{tokens, content, chunk_order_index}]. */
export type ChunkingFunc = (
  tokenizer: Tokenizer,
  content: string,
  splitByCharacter: string | null,
  splitByCharacterOnly: boolean,
  chunkOverlapTokenSize: number,
  chunkTokenSize: number,
) => TextChunk[] | Promise<TextChunk[]>;

/** An EmbeddingFunc as LightRAG takes it; `max_token_size` sets embedding_token_limit. */
export type LightRAGEmbeddingFunc = EmbeddingFunc & { max_token_size?: number | null };

/** The LLM function: upstream's keyword arguments arrive as one options object (hashing_kv and llm_model_kwargs included). */
export type LightRAGLlmFunc = (prompt: string, opts?: Record<string, unknown>) => Promise<string | AsyncIterable<string>>;

export const StoragesStatus = { NOT_CREATED: 'not_created', CREATED: 'created', INITIALIZED: 'initialized', FINALIZED: 'finalized' } as const;
export type StoragesStatusValue = (typeof StoragesStatus)[keyof typeof StoragesStatus];

/** shared_storage's pipeline_status namespace for one workspace. */
export type PipelineStatusData = {
  autoscanned: boolean;
  busy: boolean;
  job_name: string;
  job_start: string | null;
  docs: number;
  batchs: number;
  cur_batch: number;
  request_pending: boolean;
  latest_message: string;
  history_messages: string[];
  cancellation_requested?: boolean;
  [k: string]: unknown;
};

/** The dataclass fields of LightRAG. */
export type LightRAGFields = {
  working_dir: string;
  kv_storage: string;
  vector_storage: string;
  graph_storage: string;
  doc_status_storage: string;
  workspace: string;
  top_k: number;
  chunk_top_k: number;
  max_entity_tokens: number;
  max_relation_tokens: number;
  max_total_tokens: number;
  cosine_threshold: number;
  related_chunk_number: number;
  kg_chunk_pick_method: string;
  entity_extract_max_gleaning: number;
  max_extract_input_tokens: number;
  force_llm_summary_on_merge: number;
  chunk_token_size: number;
  chunk_overlap_token_size: number;
  tokenizer: Tokenizer | null;
  tiktoken_model_name: string;
  chunking_func: ChunkingFunc;
  embedding_func: LightRAGEmbeddingFunc | null;
  embedding_batch_num: number;
  embedding_func_max_async: number;
  embedding_cache_config: Record<string, unknown>;
  default_embedding_timeout: number;
  llm_model_func: LightRAGLlmFunc | null;
  llm_model_name: string;
  summary_max_tokens: number;
  summary_context_size: number;
  summary_length_recommended: number;
  llm_model_max_async: number;
  llm_model_kwargs: Record<string, unknown>;
  default_llm_timeout: number;
  rerank_model_func: QueryGlobalConfig['rerank_model_func'];
  min_rerank_score: number;
  vector_db_storage_cls_kwargs: Record<string, unknown>;
  enable_llm_cache: boolean;
  enable_llm_cache_for_entity_extract: boolean;
  max_parallel_insert: number;
  max_graph_nodes: number;
  max_source_ids_per_entity: number;
  max_source_ids_per_relation: number;
  source_ids_limit_method: string;
  max_file_paths: number;
  file_path_more_placeholder: string;
  addon_params: Record<string, unknown>;
  auto_manage_storages_states: boolean;
  cosine_better_than_threshold: number;
  ollama_server_infos: unknown;
};

export type LightRAGInit = Partial<LightRAGFields> & {
  /** deprecated upstream (setup_logger); accepted and dropped with a warning */
  log_level?: number | null;
  log_file_path?: string | null;
  /** port-only: epoch seconds as `time.time()` gives them; every timestamp the pipeline writes comes from it */
  now?: () => number;
  /** port-only: upstream's `lightrag` logger */
  log?: LightRAGLog;
  /**
   * port-only: a document is done once its chunks are stored (status, chunk
   * vectors, chunk records): no entity extraction, no graph, no LLM call.
   * Such a store answers `naive` queries only.
   */
  chunks_only?: boolean;
};

// ---------------------------------------------------------------- field defaults

/** Python `int(os.getenv(name, default))`: raises on a bad value, at import like the class attribute. */
function envIntStrict(name: string, dflt: number): number {
  const v = process.env[name];
  if (v === undefined) return dflt;
  const n = pyIntParse(v);
  if (n === null) throw pyError('ValueError', `invalid literal for int() with base 10: ${pyReprStr(v)}`);
  return n;
}
function envFloatStrict(name: string, dflt: number): number {
  const v = process.env[name];
  if (v === undefined) return dflt;
  const n = getEnvValue<number | null>(name, null, 'float');
  if (n === null) throw pyError('ValueError', `could not convert string to float: ${pyReprStr(v)}`);
  return n;
}

// `field(default=...)`: evaluated once, when the class is defined (module import)
const CLASS_DEFAULTS = {
  top_k: getEnvValue('TOP_K', DEFAULT_TOP_K, 'int')!,
  chunk_top_k: getEnvValue('CHUNK_TOP_K', DEFAULT_CHUNK_TOP_K, 'int')!,
  max_entity_tokens: getEnvValue('MAX_ENTITY_TOKENS', DEFAULT_MAX_ENTITY_TOKENS, 'int')!,
  max_relation_tokens: getEnvValue('MAX_RELATION_TOKENS', DEFAULT_MAX_RELATION_TOKENS, 'int')!,
  max_total_tokens: getEnvValue('MAX_TOTAL_TOKENS', DEFAULT_MAX_TOTAL_TOKENS, 'int')!,
  cosine_threshold: getEnvValue('COSINE_THRESHOLD', DEFAULT_COSINE_THRESHOLD, 'int')!,
  related_chunk_number: getEnvValue('RELATED_CHUNK_NUMBER', DEFAULT_RELATED_CHUNK_NUMBER, 'int')!,
  kg_chunk_pick_method: getEnvValue('KG_CHUNK_PICK_METHOD', DEFAULT_KG_CHUNK_PICK_METHOD, 'str')!,
  entity_extract_max_gleaning: getEnvValue('MAX_GLEANING', DEFAULT_MAX_GLEANING, 'int')!,
  max_extract_input_tokens: getEnvValue('MAX_EXTRACT_INPUT_TOKENS', DEFAULT_MAX_EXTRACT_INPUT_TOKENS, 'int')!,
  force_llm_summary_on_merge: getEnvValue('FORCE_LLM_SUMMARY_ON_MERGE', DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, 'int')!,
  chunk_token_size: envIntStrict('CHUNK_SIZE', 1200),
  chunk_overlap_token_size: envIntStrict('CHUNK_OVERLAP_SIZE', 100),
  embedding_batch_num: envIntStrict('EMBEDDING_BATCH_NUM', 10),
  embedding_func_max_async: envIntStrict('EMBEDDING_FUNC_MAX_ASYNC', 8),
  default_embedding_timeout: envIntStrict('EMBEDDING_TIMEOUT', DEFAULT_EMBEDDING_TIMEOUT),
  summary_max_tokens: envIntStrict('SUMMARY_MAX_TOKENS', DEFAULT_SUMMARY_MAX_TOKENS),
  summary_context_size: envIntStrict('SUMMARY_CONTEXT_SIZE', DEFAULT_SUMMARY_CONTEXT_SIZE),
  summary_length_recommended: envIntStrict('SUMMARY_LENGTH_RECOMMENDED', DEFAULT_SUMMARY_LENGTH_RECOMMENDED),
  llm_model_max_async: envIntStrict('MAX_ASYNC', DEFAULT_MAX_ASYNC),
  default_llm_timeout: envIntStrict('LLM_TIMEOUT', DEFAULT_LLM_TIMEOUT),
  min_rerank_score: getEnvValue('MIN_RERANK_SCORE', DEFAULT_MIN_RERANK_SCORE, 'float')!,
  max_parallel_insert: envIntStrict('MAX_PARALLEL_INSERT', DEFAULT_MAX_PARALLEL_INSERT),
  max_graph_nodes: getEnvValue('MAX_GRAPH_NODES', DEFAULT_MAX_GRAPH_NODES, 'int')!,
  max_source_ids_per_entity: getEnvValue('MAX_SOURCE_IDS_PER_ENTITY', DEFAULT_MAX_SOURCE_IDS_PER_ENTITY, 'int')!,
  max_source_ids_per_relation: getEnvValue('MAX_SOURCE_IDS_PER_RELATION', DEFAULT_MAX_SOURCE_IDS_PER_RELATION, 'int')!,
  max_file_paths: getEnvValue('MAX_FILE_PATHS', DEFAULT_MAX_FILE_PATHS, 'int')!,
  cosine_better_than_threshold: envFloatStrict('COSINE_THRESHOLD', 0.2),
};

/** `default_factory=...`: evaluated for every instance. */
function factoryDefaults(): Pick<LightRAGFields, 'workspace' | 'source_ids_limit_method' | 'addon_params' | 'embedding_cache_config'> {
  return {
    workspace: process.env.WORKSPACE ?? '',
    source_ids_limit_method: normalizeSourceIdsLimitMethod(getEnvValue('SOURCE_IDS_LIMIT_METHOD', DEFAULT_SOURCE_IDS_LIMIT_METHOD, 'str')),
    addon_params: {
      language: getEnvValue('SUMMARY_LANGUAGE', DEFAULT_SUMMARY_LANGUAGE, 'str'),
      entity_types: getEnvValue('ENTITY_TYPES', [...DEFAULT_ENTITY_TYPES], 'list'),
    },
    embedding_cache_config: { enabled: false, similarity_threshold: 0.95, use_llm_check: false },
  };
}

const FIELD_NAMES: readonly (keyof LightRAGFields)[] = [
  'working_dir',
  'kv_storage',
  'vector_storage',
  'graph_storage',
  'doc_status_storage',
  'workspace',
  'top_k',
  'chunk_top_k',
  'max_entity_tokens',
  'max_relation_tokens',
  'max_total_tokens',
  'cosine_threshold',
  'related_chunk_number',
  'kg_chunk_pick_method',
  'entity_extract_max_gleaning',
  'max_extract_input_tokens',
  'force_llm_summary_on_merge',
  'chunk_token_size',
  'chunk_overlap_token_size',
  'tokenizer',
  'tiktoken_model_name',
  'chunking_func',
  'embedding_func',
  'embedding_batch_num',
  'embedding_func_max_async',
  'embedding_cache_config',
  'default_embedding_timeout',
  'llm_model_func',
  'llm_model_name',
  'summary_max_tokens',
  'summary_context_size',
  'summary_length_recommended',
  'llm_model_max_async',
  'llm_model_kwargs',
  'default_llm_timeout',
  'rerank_model_func',
  'min_rerank_score',
  'vector_db_storage_cls_kwargs',
  'enable_llm_cache',
  'enable_llm_cache_for_entity_extract',
  'max_parallel_insert',
  'max_graph_nodes',
  'max_source_ids_per_entity',
  'max_source_ids_per_relation',
  'source_ids_limit_method',
  'max_file_paths',
  'file_path_more_placeholder',
  'addon_params',
  'auto_manage_storages_states',
  'cosine_better_than_threshold',
  'ollama_server_infos',
];

// ---------------------------------------------------------------- shared pipeline status

const PIPELINE_STATUS = new Map<string, PipelineStatusData>();

/**
 * shared_storage's `pipeline_status` for a workspace (get_namespace_data):
 * one object per workspace and process, shared by every LightRAG instance on
 * it. Created by initialize_storages; empty until then, as upstream.
 */
export function getPipelineStatus(workspace = ''): PipelineStatusData {
  let s = PIPELINE_STATUS.get(workspace);
  if (!s) {
    s = {} as PipelineStatusData;
    PIPELINE_STATUS.set(workspace, s);
  }
  return s;
}

/** shared_storage.initialize_pipeline_status */
export function initializePipelineStatus(workspace = ''): PipelineStatusData {
  const s = getPipelineStatus(workspace);
  if ('busy' in s) return s;
  Object.assign(s, {
    autoscanned: false,
    busy: false,
    job_name: '-',
    job_start: null,
    docs: 0,
    batchs: 0,
    cur_batch: 0,
    request_pending: false,
    latest_message: '',
    history_messages: [],
  });
  return s;
}

// ---------------------------------------------------------------- module helpers

/** lightrag._chunk_fields_from_status_doc: (chunks_list, chunks_count) kept from a status document. */
export function chunkFieldsFromStatusDoc(statusDoc: Pick<DocProcessingStatus, 'chunks_list' | 'chunks_count'>): [string[], number] {
  const list = Array.isArray(statusDoc.chunks_list) ? statusDoc.chunks_list.filter((c) => typeof c === 'string' && c) : [];
  const count = statusDoc.chunks_count;
  if (Number.isInteger(count) && typeof count === 'number' && count >= 0) return [list, count];
  return [list, list.length];
}

const PLACEHOLDER_PATHS = new Set(['', 'no-file-path', 'unknown_source']);

/** lightrag._resolve_doc_file_path: a real path from doc status, then full_docs, then a placeholder. */
export function resolveDocFilePath(statusDoc: { file_path?: unknown } | null = null, contentData: Record<string, unknown> | null = null): string {
  const norm = (c: unknown): string | null => {
    if (typeof c !== 'string') return null;
    const s = pyStrip(c);
    return s ? s : null;
  };
  const candidates = [norm(statusDoc?.file_path), norm(contentData && pyTruthy(contentData) ? contentData.file_path : null)];
  for (const c of candidates) if (c && !PLACEHOLDER_PATHS.has(c)) return c;
  for (const c of candidates) if (c) return c === 'no-file-path' ? 'unknown_source' : c;
  return 'unknown_source';
}

/** lightrag._normalize_string_list: the non-empty strings of a list (others dropped with a warning). */
export function normalizeStringList(raw: unknown, context = '', log: LightRAGLog = () => {}): string[] {
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  raw.forEach((v, i) => {
    if (typeof v === 'string' && v) out.push(v);
    else log('warning', `Non-string element dropped from list${context ? ` (${context})` : ''} at index ${i}: ${typeof v === 'string' ? pyReprStr(v) : JSON.stringify(v)}`);
  });
  return out;
}

/** A FIFO semaphore (asyncio.Semaphore wakes waiters in order). */
class Semaphore {
  private waiters: (() => void)[] = [];
  private value: number;
  constructor(value: number) {
    this.value = value;
  }
  async acquire(): Promise<void> {
    if (this.value > 0) {
      this.value--;
      return;
    }
    await new Promise<void>((r) => this.waiters.push(r));
  }
  release(): void {
    const w = this.waiters.shift();
    if (w) w();
    else this.value++;
  }
}

type Deferred = { promise: Promise<void>; resolve: () => void };
const deferred = (): Deferred => {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
};

/** Python's traceback.format_exc() stand-in for history_messages: the JS stack. */
const formatTraceback = (e: unknown): string => (e instanceof Error ? (e.stack ?? `${e.name}: ${e.message}`) : String(e));
const errStr = (e: unknown): string => (e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------- the class

/**
 * LightRAG over the four default storages (JsonKVStorage, NanoVectorDBStorage,
 * NetworkXStorage, JsonDocStatusStorage), reading and writing upstream's files.
 */
export class LightRAG implements LightRAGFields {
  working_dir = './rag_storage';
  kv_storage = 'JsonKVStorage';
  vector_storage = 'NanoVectorDBStorage';
  graph_storage = 'NetworkXStorage';
  doc_status_storage = 'JsonDocStatusStorage';
  workspace: string;
  top_k = CLASS_DEFAULTS.top_k;
  chunk_top_k = CLASS_DEFAULTS.chunk_top_k;
  max_entity_tokens = CLASS_DEFAULTS.max_entity_tokens;
  max_relation_tokens = CLASS_DEFAULTS.max_relation_tokens;
  max_total_tokens = CLASS_DEFAULTS.max_total_tokens;
  cosine_threshold = CLASS_DEFAULTS.cosine_threshold;
  related_chunk_number = CLASS_DEFAULTS.related_chunk_number;
  kg_chunk_pick_method = CLASS_DEFAULTS.kg_chunk_pick_method;
  entity_extract_max_gleaning = CLASS_DEFAULTS.entity_extract_max_gleaning;
  max_extract_input_tokens = CLASS_DEFAULTS.max_extract_input_tokens;
  force_llm_summary_on_merge = CLASS_DEFAULTS.force_llm_summary_on_merge;
  chunk_token_size = CLASS_DEFAULTS.chunk_token_size;
  chunk_overlap_token_size = CLASS_DEFAULTS.chunk_overlap_token_size;
  tokenizer: Tokenizer | null = null;
  tiktoken_model_name = 'gpt-4o-mini';
  chunking_func: ChunkingFunc = chunkingByTokenSize;
  embedding_func: LightRAGEmbeddingFunc | null = null;
  /** set from embedding_func.max_token_size (init=False upstream) */
  embedding_token_limit: number | null = null;
  embedding_batch_num = CLASS_DEFAULTS.embedding_batch_num;
  embedding_func_max_async = CLASS_DEFAULTS.embedding_func_max_async;
  embedding_cache_config: Record<string, unknown>;
  default_embedding_timeout = CLASS_DEFAULTS.default_embedding_timeout;
  llm_model_func: LightRAGLlmFunc | null = null;
  llm_model_name = 'gpt-4o-mini';
  summary_max_tokens = CLASS_DEFAULTS.summary_max_tokens;
  summary_context_size = CLASS_DEFAULTS.summary_context_size;
  summary_length_recommended = CLASS_DEFAULTS.summary_length_recommended;
  llm_model_max_async = CLASS_DEFAULTS.llm_model_max_async;
  llm_model_kwargs: Record<string, unknown> = {};
  default_llm_timeout = CLASS_DEFAULTS.default_llm_timeout;
  rerank_model_func: QueryGlobalConfig['rerank_model_func'] = null;
  min_rerank_score = CLASS_DEFAULTS.min_rerank_score;
  vector_db_storage_cls_kwargs: Record<string, unknown> = {};
  enable_llm_cache = true;
  enable_llm_cache_for_entity_extract = true;
  max_parallel_insert = CLASS_DEFAULTS.max_parallel_insert;
  max_graph_nodes = CLASS_DEFAULTS.max_graph_nodes;
  max_source_ids_per_entity = CLASS_DEFAULTS.max_source_ids_per_entity;
  max_source_ids_per_relation = CLASS_DEFAULTS.max_source_ids_per_relation;
  source_ids_limit_method: string;
  max_file_paths = CLASS_DEFAULTS.max_file_paths;
  file_path_more_placeholder = DEFAULT_FILE_PATH_MORE_PLACEHOLDER;
  addon_params: Record<string, unknown>;
  auto_manage_storages_states = false;
  cosine_better_than_threshold = CLASS_DEFAULTS.cosine_better_than_threshold;
  ollama_server_infos: unknown = null;
  _storages_status: StoragesStatusValue = StoragesStatus.NOT_CREATED;

  // storages (created in the constructor, as __post_init__ does)
  readonly llm_response_cache: JsonKVStorage;
  readonly text_chunks: JsonKVStorage;
  readonly full_docs: JsonKVStorage;
  readonly full_entities: JsonKVStorage;
  readonly full_relations: JsonKVStorage;
  readonly entity_chunks: JsonKVStorage;
  readonly relation_chunks: JsonKVStorage;
  readonly chunk_entity_relation_graph: NetworkXStorage;
  readonly entities_vdb: NanoVectorDBStorage;
  readonly relationships_vdb: NanoVectorDBStorage;
  readonly chunks_vdb: NanoVectorDBStorage;
  readonly doc_status: JsonDocStatusStorage;

  /** port-only: the clock and the logger */
  readonly now: () => number;
  readonly log: LightRAGLog;
  /** port-only: see LightRAGInit.chunks_only */
  readonly chunks_only: boolean;
  /** `asdict(self)` at construction, as the storages received it */
  private readonly initConfig: Record<string, unknown>;

  constructor(init: LightRAGInit = {}) {
    const { log_level, log_file_path, now, log, chunks_only, ...fields } = init;
    const f = factoryDefaults();
    this.workspace = f.workspace;
    this.source_ids_limit_method = f.source_ids_limit_method;
    this.addon_params = f.addon_params;
    this.embedding_cache_config = f.embedding_cache_config;
    for (const [k, v] of Object.entries(fields)) {
      if (!(FIELD_NAMES as readonly string[]).includes(k)) throw new TypeError(`LightRAG.__init__() got an unexpected keyword argument '${k}'`);
      if (v !== undefined) (this as Record<string, unknown>)[k] = v;
    }
    this.now = now ?? (() => Date.now() / 1000);
    this.log = log ?? (() => {});
    this.chunks_only = chunks_only ?? false;

    // ---- __post_init__
    if (log_level !== undefined && log_level !== null) process.emitWarning('WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead');
    if (log_file_path !== undefined && log_file_path !== null) process.emitWarning('WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead');

    if (!existsSync(this.working_dir)) {
      this.log('info', `Creating working directory ${this.working_dir}`);
      mkdirSync(this.working_dir, { recursive: true });
    }
    for (const [type, name, expected] of [
      ['KV_STORAGE', this.kv_storage, 'JsonKVStorage'],
      ['VECTOR_STORAGE', this.vector_storage, 'NanoVectorDBStorage'],
      ['GRAPH_STORAGE', this.graph_storage, 'NetworkXStorage'],
      ['DOC_STATUS_STORAGE', this.doc_status_storage, 'JsonDocStatusStorage'],
    ] as const) {
      if (name !== expected) throw pyError('ValueError', `Storage implementation '${name}' for ${type} is not ported: lightrag-ts ships the default storages only (${expected})`);
    }
    this.vector_db_storage_cls_kwargs = { cosine_better_than_threshold: this.cosine_better_than_threshold, ...this.vector_db_storage_cls_kwargs };
    if (this.tokenizer === null) this.tokenizer = new TiktokenTokenizer(this.tiktoken_model_name || 'gpt-4o-mini');

    if (this.force_llm_summary_on_merge < 3) this.log('warning', `force_llm_summary_on_merge should be at least 3, got ${this.force_llm_summary_on_merge}`);
    if (this.summary_context_size > this.max_total_tokens) {
      this.log('warning', `summary_context_size(${this.summary_context_size}) should no greater than max_total_tokens(${this.max_total_tokens})`);
    }
    if (this.summary_length_recommended > this.summary_max_tokens) {
      this.log('warning', `max_total_tokens(${this.summary_max_tokens}) should greater than summary_length_recommended(${this.summary_length_recommended})`);
    }

    const originalEmbedding = this.embedding_func;
    this.embedding_token_limit = originalEmbedding?.max_token_size ?? null;
    const globalConfig = this.asdict();
    globalConfig.embedding_func = originalEmbedding;
    this.initConfig = globalConfig;

    // a new function object: the caller's EmbeddingFunc stays unwrapped
    if (this.embedding_func) {
      const raw = this.embedding_func;
      const limited = priorityLimitAsyncFuncCall(this.embedding_func_max_async)((texts: string[], o?: Record<string, unknown>) => raw(texts, o));
      this.embedding_func = Object.assign(limited, { embedding_dim: raw.embedding_dim, model_name: raw.model_name, max_token_size: raw.max_token_size }) as LightRAGEmbeddingFunc;
    }

    const clock = () => Math.floor(this.now());
    const gc = globalConfig as GlobalConfig;
    const kv = (namespace: string) => new JsonKVStorage({ namespace, workspace: this.workspace, global_config: gc, clock });
    this.llm_response_cache = kv(NameSpace.KV_STORE_LLM_RESPONSE_CACHE);
    this.text_chunks = kv(NameSpace.KV_STORE_TEXT_CHUNKS);
    this.full_docs = kv(NameSpace.KV_STORE_FULL_DOCS);
    this.full_entities = kv(NameSpace.KV_STORE_FULL_ENTITIES);
    this.full_relations = kv(NameSpace.KV_STORE_FULL_RELATIONS);
    this.entity_chunks = kv(NameSpace.KV_STORE_ENTITY_CHUNKS);
    this.relation_chunks = kv(NameSpace.KV_STORE_RELATION_CHUNKS);
    this.chunk_entity_relation_graph = new NetworkXStorage({ namespace: NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION, workspace: this.workspace, global_config: gc });
    const vdbLog = (level: string, msg: string) => this.log(level === 'warn' ? 'warning' : (level as 'info'), msg);
    const vdb = (namespace: string, meta: string[]) =>
      new NanoVectorDBStorage({
        namespace,
        workspace: this.workspace,
        global_config: gc,
        embedding_func: this.embedding_func as EmbeddingFunc,
        meta_fields: meta,
        clock,
        log: vdbLog,
      });
    this.entities_vdb = vdb(NameSpace.VECTOR_STORE_ENTITIES, ['entity_name', 'source_id', 'content', 'file_path']);
    this.relationships_vdb = vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS, ['src_id', 'tgt_id', 'source_id', 'content', 'file_path']);
    this.chunks_vdb = vdb(NameSpace.VECTOR_STORE_CHUNKS, ['full_doc_id', 'content', 'file_path']);
    this.doc_status = new JsonDocStatusStorage({ namespace: NameSpace.DOC_STATUS, workspace: this.workspace, global_config: gc, clock });

    // partial(llm_model_func, hashing_kv=..., **llm_model_kwargs) behind the priority limiter
    const rawLlm = this.llm_model_func;
    if (typeof rawLlm !== 'function') throw new TypeError('the first argument must be callable');
    const hashingKv = this.llm_response_cache;
    const kwargs = this.llm_model_kwargs;
    this.llm_model_func = priorityLimitAsyncFuncCall(this.llm_model_max_async)((prompt: string, o: Record<string, unknown> = {}) =>
      rawLlm(prompt, { hashing_kv: hashingKv, ...kwargs, ...o }),
    );
    this._storages_status = StoragesStatus.CREATED;
  }

  /** dataclasses.asdict(self): every field, dict fields copied. */
  asdict(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const k of FIELD_NAMES) {
      const v = (this as unknown as Record<string, unknown>)[k];
      out[k] = v && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype ? structuredCloneSafe(v as Record<string, unknown>) : v;
    }
    out.embedding_token_limit = this.embedding_token_limit;
    out._storages_status = this._storages_status;
    // port-only keys the ported functions read
    out.log = this.log;
    return out;
  }

  private get storagesInOrder() {
    return [
      this.full_docs,
      this.text_chunks,
      this.full_entities,
      this.full_relations,
      this.entity_chunks,
      this.relation_chunks,
      this.entities_vdb,
      this.relationships_vdb,
      this.chunks_vdb,
      this.chunk_entity_relation_graph,
      this.llm_response_cache,
      this.doc_status,
    ] as const;
  }

  /** The workspace's shared pipeline status (shared_storage's `pipeline_status`). */
  get pipeline_status(): PipelineStatusData {
    return getPipelineStatus(this.workspace);
  }

  async initialize_storages(): Promise<void> {
    if (this._storages_status !== StoragesStatus.CREATED) return;
    initializePipelineStatus(this.workspace);
    for (const s of this.storagesInOrder) await s.initialize();
    this._storages_status = StoragesStatus.INITIALIZED;
    this.log('debug', 'All storage types initialized');
  }

  async finalize_storages(): Promise<void> {
    if (this._storages_status !== StoragesStatus.INITIALIZED) return;
    const names = ['full_docs', 'text_chunks', 'full_entities', 'full_relations', 'entity_chunks', 'relation_chunks', 'entities_vdb', 'relationships_vdb', 'chunks_vdb', 'chunk_entity_relation_graph', 'llm_response_cache', 'doc_status'];
    const ok: string[] = [];
    const failed: string[] = [];
    for (const [i, s] of this.storagesInOrder.entries()) {
      try {
        await s.finalize();
        ok.push(names[i]!);
      } catch (e) {
        this.log('error', `Failed to finalize ${names[i]}: ${errStr(e)}`);
        failed.push(names[i]!);
      }
    }
    if (ok.length) this.log('info', `Successfully finalized ${ok.length} storages`);
    if (failed.length) this.log('error', `Failed to finalize ${failed.length} storages: ${failed.join(', ')}`);
    this._storages_status = StoragesStatus.FINALIZED;
  }

  // ---------------------------------------------------------------- migration

  /** LightRAG.check_and_migrate_data: seed chunk tracking and full_entities/full_relations from an older store. */
  async check_and_migrate_data(): Promise<void> {
    const labels = await this.chunk_entity_relation_graph.get_all_labels();
    if (!labels.length) {
      this.log('debug', 'No entities found in graph, skipping migration check');
      return;
    }
    await this.migrateChunkTrackingStorage();
    const processed = await this.doc_status.get_docs_by_status(DocStatus.PROCESSED);
    if (!processed.size) {
      this.log('debug', 'No processed documents found, skipping migration');
      return;
    }
    for (const docId of [...processed.keys()].slice(0, Math.min(5, processed.size))) {
      if (pyTruthy(await this.full_entities.get_by_id(docId)) || pyTruthy(await this.full_relations.get_by_id(docId))) {
        this.log('debug', 'Full entities/relations data already exists, no migration needed');
        return;
      }
    }
    this.log('info', `Data migration needed: found ${labels.length} entities in graph but no full_entities/full_relations data`);
    await this.migrateEntityRelationData(processed);
  }

  private async migrateEntityRelationData(processed: Map<string, DocProcessingStatus>): Promise<void> {
    this.log('info', `Starting data migration for ${processed.size} documents`);
    const chunkToDoc = new Map<string, string>();
    for (const [docId, st] of processed) for (const c of st.chunks_list ?? []) chunkToDoc.set(c, docId);
    const docEntities = new Map<string, Set<string>>();
    const docRelations = new Map<string, Map<string, [string, string]>>();
    for (const node of await this.chunk_entity_relation_graph.get_all_nodes()) {
      if (!('source_id' in node)) continue;
      const id = (pyTruthy(node.entity_id) ? node.entity_id : node.id) as string | undefined;
      if (!pyTruthy(id)) continue;
      for (const c of String(node.source_id).split(GRAPH_FIELD_SEP)) {
        const d = chunkToDoc.get(c);
        if (!d) continue;
        if (!docEntities.has(d)) docEntities.set(d, new Set());
        docEntities.get(d)!.add(id!);
      }
    }
    for (const edge of await this.chunk_entity_relation_graph.get_all_edges()) {
      if (!('source_id' in edge)) continue;
      const src = edge.source as string | undefined;
      const tgt = edge.target as string | undefined;
      if (!src || !tgt) continue;
      for (const c of String(edge.source_id).split(GRAPH_FIELD_SEP)) {
        const d = chunkToDoc.get(c);
        if (!d) continue;
        if (!docRelations.has(d)) docRelations.set(d, new Map());
        const pair = [src, tgt].sort(pyCompare) as [string, string];
        docRelations.get(d)!.set(edgeKey(...pair), pair);
      }
    }
    if (docEntities.size) {
      const data: Record<string, Record<string, unknown>> = {};
      for (const [d, set] of docEntities) data[d] = { entity_names: [...set], count: set.size };
      await this.full_entities.upsert(data);
    }
    if (docRelations.size) {
      const data: Record<string, Record<string, unknown>> = {};
      for (const [d, m] of docRelations) data[d] = { relation_pairs: [...m.values()].map((p) => [...p]), count: m.size };
      await this.full_relations.upsert(data);
    }
    const migrated = new Set([...docEntities.keys(), ...docRelations.keys()]).size;
    await this.full_entities.index_done_callback();
    await this.full_relations.index_done_callback();
    this.log('info', `Data migration completed: migrated ${migrated} documents with entities/relations`);
  }

  private async migrateChunkTrackingStorage(): Promise<void> {
    const needEntity = await this.entity_chunks.is_empty();
    const needRelation = await this.relation_chunks.is_empty();
    if (!needEntity && !needRelation) return;
    const BATCH = 500;
    const ids = (raw: unknown) => String(raw || '').split(GRAPH_FIELD_SEP).filter((c) => c);
    if (needEntity) {
      const nodes = await this.chunk_entity_relation_graph.get_all_nodes();
      this.log('info', `Starting chunk_tracking data migration: ${nodes.length} nodes`);
      let total = 0;
      for (let b = 0; b < nodes.length; b += BATCH) {
        const payload: Record<string, Record<string, unknown>> = {};
        for (const node of nodes.slice(b, b + BATCH)) {
          const id = (pyTruthy(node.entity_id) ? node.entity_id : node.id) as string | undefined;
          if (!pyTruthy(id)) continue;
          const c = ids(node.source_id);
          if (c.length) payload[id!] = { chunk_ids: c, count: c.length };
        }
        if (Object.keys(payload).length) {
          await this.entity_chunks.upsert(payload);
          total += Object.keys(payload).length;
        }
      }
      if (total > 0) {
        await this.entity_chunks.index_done_callback();
        this.log('info', `Entity chunk_tracking migration completed: ${total} records persisted`);
      }
    }
    if (needRelation) {
      const edges = await this.chunk_entity_relation_graph.get_all_edges();
      this.log('info', `Starting chunk_tracking data migration: ${edges.length} edges`);
      let total = 0;
      for (let b = 0; b < edges.length; b += BATCH) {
        const payload: Record<string, Record<string, unknown>> = {};
        for (const e of edges.slice(b, b + BATCH)) {
          const src = (e.source || e.src_id || e.src) as string;
          const tgt = (e.target || e.tgt_id || e.tgt) as string;
          if (!src || !tgt) continue;
          const c = ids(e.source_id);
          if (c.length) payload[makeRelationChunkKey(src, tgt)] = { chunk_ids: c, count: c.length };
        }
        if (Object.keys(payload).length) {
          await this.relation_chunks.upsert(payload);
          total += Object.keys(payload).length;
        }
      }
      if (total > 0) {
        await this.relation_chunks.index_done_callback();
        this.log('info', `Relation chunk_tracking migration completed: ${total} records persisted`);
      }
    }
  }

  // ---------------------------------------------------------------- graph access

  async get_graph_labels(): Promise<string[]> {
    return this.chunk_entity_relation_graph.get_all_labels();
  }

  async get_knowledge_graph(nodeLabel: string, maxDepth = 3, maxNodes: number | null = null): Promise<KnowledgeGraph> {
    const limit = maxNodes === null ? this.max_graph_nodes : Math.min(maxNodes, this.max_graph_nodes);
    return this.chunk_entity_relation_graph.get_knowledge_graph(nodeLabel, maxDepth, limit);
  }

  // ---------------------------------------------------------------- insert

  private isoNow(): string {
    return pyIsoformatUtc(this.now());
  }
  private intNow(): number {
    return Math.floor(this.now());
  }

  /** LightRAG.ainsert: enqueue, then process the queue. Returns the track id. */
  async ainsert(
    input: string | string[],
    split_by_character: string | null = null,
    split_by_character_only = false,
    ids: string | string[] | null = null,
    file_paths: string | string[] | null = null,
    track_id: string | null = null,
  ): Promise<string> {
    if (track_id === null) track_id = generateTrackId('insert', this.now);
    await this.apipeline_enqueue_documents(input, ids, file_paths, track_id);
    await this.apipeline_process_enqueue_documents(split_by_character, split_by_character_only);
    return track_id;
  }

  /**
   * LightRAG.ainsert_custom_chunks (deprecated upstream): stores the document and
   * the given chunks and extracts them. As upstream, the extraction results are
   * not merged into the graph.
   */
  async ainsert_custom_chunks(fullText: string, textChunks: string[], docId: string | null = null): Promise<void> {
    let updateStorage = false;
    try {
      fullText = sanitizeTextForEncoding(fullText);
      textChunks = textChunks.map((c) => sanitizeTextForEncoding(c));
      const filePath = '';
      const docKey = docId === null ? computeMdhashId(fullText, 'doc-') : docId;
      const add = await this.full_docs.filter_keys([docKey]);
      if (!add.has(docKey)) {
        this.log('warning', 'This document is already in the storage.');
        return;
      }
      const newDocs = { [docKey]: { content: fullText, file_path: filePath } };
      updateStorage = true;
      this.log('info', 'Inserting 1 docs');
      const inserting = new Map<string, Record<string, unknown>>();
      textChunks.forEach((chunkText, index) => {
        inserting.set(computeMdhashId(chunkText, 'chunk-'), {
          content: chunkText,
          full_doc_id: docKey,
          tokens: this.tokenizer!.encode(chunkText).length,
          chunk_order_index: index,
          file_path: filePath,
        });
      });
      const addChunks = await this.text_chunks.filter_keys(inserting.keys());
      for (const k of [...inserting.keys()]) if (!addChunks.has(k)) inserting.delete(k);
      if (!inserting.size) {
        this.log('warning', 'All chunks are already in the storage.');
        return;
      }
      const obj = Object.fromEntries(inserting);
      await Promise.all([
        this.chunks_vdb.upsert(obj),
        this.processExtractEntities(obj as never, null),
        this.full_docs.upsert(newDocs),
        this.text_chunks.upsert(obj),
      ]);
    } finally {
      if (updateStorage) await this.insertDone();
    }
  }

  /** LightRAG.apipeline_enqueue_documents. Returns the track id, or null when nothing new was enqueued. */
  async apipeline_enqueue_documents(
    input: string | string[],
    ids: string | string[] | null = null,
    file_paths: string | string[] | null = null,
    track_id: string | null = null,
  ): Promise<string | null> {
    if (track_id === null || pyStrip(track_id) === '') track_id = generateTrackId('enqueue', this.now);
    const docs = typeof input === 'string' ? [input] : input;
    const idList = typeof ids === 'string' ? [ids] : ids;
    let paths: string[];
    if (file_paths !== null) {
      const fp = typeof file_paths === 'string' ? [file_paths] : file_paths;
      if (fp.length !== docs.length) throw pyError('ValueError', 'Number of file paths must match the number of documents');
      paths = fp.map((p) => (typeof p === 'string' ? pyStrip(p) : '')).map((p) => (p ? p : 'unknown_source'));
    } else paths = docs.map(() => 'unknown_source');

    const contents = new Map<string, { content: string; file_path: string }>();
    if (idList !== null) {
      if (idList.length !== docs.length) throw pyError('ValueError', 'Number of IDs must match the number of documents');
      if (new Set(idList).size !== idList.length) throw pyError('ValueError', 'IDs must be unique');
      const unique = new Map<string, [string, string]>();
      docs.forEach((doc, i) => {
        const cleaned = sanitizeTextForEncoding(doc);
        if (!unique.has(cleaned)) unique.set(cleaned, [idList[i]!, paths[i]!]);
      });
      for (const [content, [id, fp]] of unique) contents.set(id, { content, file_path: fp });
    } else {
      const unique = new Map<string, string>();
      docs.forEach((doc, i) => {
        const cleaned = sanitizeTextForEncoding(doc);
        if (!unique.has(cleaned)) unique.set(cleaned, paths[i]!);
      });
      for (const [content, fp] of unique) contents.set(computeMdhashId(content, 'doc-'), { content, file_path: fp });
    }

    // 2. initial status, without the content
    const newDocs = new Map<string, Record<string, unknown>>();
    for (const [id, c] of contents) {
      newDocs.set(id, {
        status: DocStatus.PENDING,
        content_summary: getContentSummary(c.content),
        content_length: pyLen(c.content),
        created_at: this.isoNow(),
        updated_at: this.isoNow(),
        file_path: c.file_path,
        track_id,
      });
    }

    // 3. documents already known get a duplicate record of this attempt
    const allNew = [...newDocs.keys()];
    const unique = await this.doc_status.filter_keys(allNew);
    const ignored = allNew.filter((id) => !unique.has(id));
    if (ignored.length) {
      const duplicates: Record<string, Record<string, unknown>> = {};
      for (const docId of ignored) {
        const filePath = (newDocs.get(docId)?.file_path as string) || 'unknown_source';
        this.log('warning', `Duplicate document detected: ${docId} (${filePath})`);
        const existing = await this.doc_status.get_by_id(docId);
        let existingStatus = pyTruthy(existing) ? ('status' in existing! ? existing!.status : 'unknown') : 'unknown';
        // a status this process wrote is a DocStatus member upstream (str() "DocStatus.PROCESSED"); one read from the file is a plain str
        if (pyTruthy(existing) && this.enumStatus.has(docId) && typeof existingStatus === 'string') existingStatus = `DocStatus.${existingStatus.toUpperCase()}`;
        const existingTrack = pyTruthy(existing) ? ('track_id' in existing! ? existing!.track_id : '') : '';
        duplicates[computeMdhashId(`${docId}-${track_id}`, 'dup-')] = {
          status: DocStatus.FAILED,
          content_summary: `[DUPLICATE] Original document: ${docId}`,
          content_length: (newDocs.get(docId)?.content_length as number | undefined) ?? 0,
          chunks_count: 0,
          chunks_list: [],
          created_at: this.isoNow(),
          updated_at: this.isoNow(),
          file_path: filePath,
          track_id,
          error_msg: `Content already exists. Original doc_id: ${docId}, Status: ${String(existingStatus)}`,
          metadata: { is_duplicate: true, original_doc_id: docId, original_track_id: existingTrack },
        };
      }
      if (Object.keys(duplicates).length) {
        await this.upsertStatus(duplicates);
        this.log('info', `Created ${Object.keys(duplicates).length} duplicate document records with track_id: ${track_id}`);
      }
    }
    const toStore = [...unique].filter((id) => newDocs.has(id));
    if (!toStore.length) {
      this.log('warning', 'No new unique documents were found.');
      return null;
    }

    // 4. content to full_docs (persisted now), status to doc_status
    const fullDocs: Record<string, Record<string, unknown>> = {};
    for (const id of toStore) fullDocs[id] = { content: contents.get(id)!.content, file_path: contents.get(id)!.file_path };
    await this.full_docs.upsert(fullDocs);
    await this.full_docs.index_done_callback();
    const statuses: Record<string, Record<string, unknown>> = {};
    for (const id of toStore) statuses[id] = newDocs.get(id)!;
    await this.upsertStatus(statuses);
    this.log('debug', `Stored ${toStore.length} new unique documents`);
    return track_id;
  }

  /** LightRAG.apipeline_enqueue_error_documents: FAILED doc-status records for files that could not be read. */
  async apipeline_enqueue_error_documents(errorFiles: Record<string, unknown>[], track_id: string | null = null): Promise<void> {
    if (!errorFiles.length) {
      this.log('debug', 'No error files to record');
      return;
    }
    if (track_id === null || pyStrip(track_id) === '') track_id = generateTrackId('error', this.now);
    const errorDocs: Record<string, Record<string, unknown>> = {};
    const currentTime = this.isoNow();
    const g = (o: Record<string, unknown>, k: string, d: unknown) => (k in o ? o[k] : d);
    for (const ef of errorFiles) {
      const filePath = g(ef, 'file_path', 'unknown_file');
      const description = g(ef, 'error_description', 'File extraction failed');
      errorDocs[computeMdhashId(`${filePath}-${description}`, 'error-')] = {
        status: DocStatus.FAILED,
        content_summary: description,
        content_length: g(ef, 'file_size', 0),
        error_msg: g(ef, 'original_error', 'Unknown error'),
        chunks_count: 0,
        chunks_list: [],
        created_at: currentTime,
        updated_at: currentTime,
        file_path: filePath,
        track_id,
        metadata: { error_type: 'file_extraction_error' },
      };
    }
    await this.upsertStatus(errorDocs);
    for (const [id, d] of Object.entries(errorDocs)) this.log('error', `File processing error: - ID: ${id} ${String(d.file_path)}`);
  }

  /** Doc ids whose stored status this process wrote as a DocStatus member (see the duplicate record). */
  private readonly enumStatus = new Set<string>();

  /** doc_status.upsert of records whose status is a DocStatus literal. */
  private async upsertStatus(records: Record<string, Record<string, unknown>>): Promise<void> {
    await this.doc_status.upsert(records);
    for (const id of Object.keys(records)) this.enumStatus.add(id);
  }

  private async deleteStatus(ids: string[]): Promise<void> {
    await this.doc_status.delete(ids);
    for (const id of ids) this.enumStatus.delete(id);
  }

  private say(status: PipelineStatusData, msg: string, level: 'info' | 'warning' | 'error' = 'info'): void {
    this.log(level, msg);
    status.latest_message = msg;
    status.history_messages.push(msg);
  }

  /** LightRAG._validate_and_fix_document_consistency */
  private async validateAndFixDocumentConsistency(
    toProcess: Map<string, DocProcessingStatus>,
    status: PipelineStatusData,
  ): Promise<Map<string, DocProcessingStatus>> {
    const inconsistent: string[] = [];
    const failedToPreserve: string[] = [];
    for (const [docId, st] of toProcess) {
      if (!pyTruthy(await this.full_docs.get_by_id(docId))) {
        if (st.status === DocStatus.FAILED) failedToPreserve.push(docId);
        else inconsistent.push(docId);
      }
    }
    if (failedToPreserve.length) {
      this.say(status, `Preserving ${failedToPreserve.length} failed document entries for manual review`);
      for (const id of failedToPreserve) toProcess.delete(id);
    }
    if (inconsistent.length) {
      this.say(status, `Inconsistent document entries found: ${inconsistent.length}`);
      for (const docId of inconsistent) {
        try {
          const filePath = resolveDocFilePath(toProcess.get(docId)!);
          await this.deleteStatus([docId]);
          this.say(status, `Deleted inconsistent entry: ${docId} (${filePath})`);
          toProcess.delete(docId);
        } catch (e) {
          this.say(status, `Failed to delete entry: ${docId} - ${errStr(e)}`, 'error');
        }
      }
    }
    const reset: Record<string, Record<string, unknown>> = {};
    let resetCount = 0;
    for (const [docId, st] of toProcess) {
      const content = await this.full_docs.get_by_id(docId);
      if (!pyTruthy(content)) continue;
      if (st.status !== DocStatus.PROCESSING && st.status !== DocStatus.FAILED) continue;
      const [chunksList, chunksCount] = chunkFieldsFromStatusDoc(st);
      const resolved = resolveDocFilePath(st, content);
      reset[docId] = {
        status: DocStatus.PENDING,
        content_summary: st.content_summary,
        content_length: st.content_length,
        chunks_count: chunksCount,
        chunks_list: chunksList,
        created_at: st.created_at,
        updated_at: this.isoNow(),
        file_path: resolved,
        track_id: st.track_id ?? '',
        error_msg: '',
        metadata: {},
      };
      st.status = DocStatus.PENDING;
      st.file_path = resolved;
      resetCount++;
    }
    if (Object.keys(reset).length) {
      await this.upsertStatus(reset);
      this.say(status, `Reset ${resetCount} documents from PROCESSING/FAILED to PENDING status`);
    }
    return toProcess;
  }

  /**
   * LightRAG.apipeline_process_enqueue_documents: chunk, extract and merge every
   * PENDING / PROCESSING / FAILED document, max_parallel_insert at a time.
   *
   * Upstream lets documents in flight interleave: their chunk writes, and their
   * merges entity by entity, land in whatever order asyncio schedules them.
   * Here chunking and extraction run concurrently, but each document's chunk
   * writes and its merge wait for the previous document's, so the store ends
   * up as upstream's with max_parallel_insert=1, whatever the parallelism.
   */
  async apipeline_process_enqueue_documents(split_by_character: string | null = null, split_by_character_only = false): Promise<void> {
    const status = initializePipelineStatus(this.workspace);
    let toProcess: Map<string, DocProcessingStatus>;
    if (!status.busy) {
      toProcess = await this.doc_status.get_docs_by_statuses([DocStatus.PROCESSING, DocStatus.FAILED, DocStatus.PENDING]);
      if (!toProcess.size) {
        this.log('info', 'No documents to process');
        return;
      }
      Object.assign(status, {
        busy: true,
        job_name: 'Default Job',
        job_start: this.isoNow(),
        docs: 0,
        batchs: 0,
        cur_batch: 0,
        request_pending: false,
        cancellation_requested: false,
        latest_message: '',
      });
      status.history_messages.length = 0;
    } else {
      status.request_pending = true;
      this.log('info', 'Another process is already processing the document queue. Request queued.');
      return;
    }

    try {
      for (;;) {
        if (status.cancellation_requested) {
          status.request_pending = false;
          status.cancellation_requested = false;
          this.say(status, 'Pipeline cancelled by user');
          return;
        }
        if (!toProcess.size) {
          this.say(status, 'All enqueued documents have been processed');
          break;
        }
        toProcess = await this.validateAndFixDocumentConsistency(toProcess, status);
        if (!toProcess.size) {
          this.say(status, 'No valid documents to process after consistency check');
          break;
        }
        const msg = `Processing ${toProcess.size} document(s)`;
        this.log('info', msg);
        status.docs = toProcess.size;
        status.batchs = toProcess.size;
        status.cur_batch = 0;
        status.latest_message = msg;
        status.history_messages.push(msg);

        const firstPath = toProcess.values().next().value!.file_path;
        const prefix = firstPath ? pySliceTo(firstPath, 20) + (pyLen(firstPath) > 20 ? '...' : '') : 'unknown_source';
        const totalFiles = toProcess.size;
        status.job_name = `${prefix}[${totalFiles} files]`;

        const docs = [...toProcess];
        const semaphore = new Semaphore(Math.max(1, this.max_parallel_insert));
        const stage1 = docs.map(deferred);
        const merged = docs.map(deferred);
        const counter = { processed: 0 };
        await Promise.all(
          docs.map(([docId, st], i) =>
            this.processDocument(docId, st, i, { split_by_character, split_by_character_only, status, semaphore, totalFiles, counter, stage1, merged }),
          ),
        );

        if (!status.request_pending) break;
        status.request_pending = false;
        this.say(status, 'Processing additional documents due to pending request');
        toProcess = await this.doc_status.get_docs_by_statuses([DocStatus.PROCESSING, DocStatus.FAILED, DocStatus.PENDING]);
      }
    } finally {
      const m = 'Enqueued document processing pipeline stopped';
      this.log('info', m);
      status.busy = false;
      status.cancellation_requested = false;
      status.latest_message = m;
      status.history_messages.push(m);
    }
  }

  /** process_document of apipeline_process_enqueue_documents, plus the ordering turns (see there). */
  private async processDocument(
    docId: string,
    statusDoc: DocProcessingStatus,
    index: number,
    b: {
      split_by_character: string | null;
      split_by_character_only: boolean;
      status: PipelineStatusData;
      semaphore: Semaphore;
      totalFiles: number;
      counter: { processed: number };
      stage1: Deferred[];
      merged: Deferred[];
    },
  ): Promise<void> {
    const { status, totalFiles } = b;
    let filePath = resolveDocFilePath(statusDoc);
    let currentFileNumber = 0;
    let extractionOk = false;
    let processingStartTime = this.intNow();
    let chunks = new Map<string, Record<string, unknown>>();
    let chunkResults: ChunkExtraction[] = [];
    const failedChunkSnapshot = (): [string[], number] => (chunks.size ? [[...chunks.keys()], chunks.size] : chunkFieldsFromStatusDoc(statusDoc));
    const waitTurn = async (turns: Deferred[]) => {
      if (index > 0) await turns[index - 1]!.promise;
    };
    const writeFailed = async (e: unknown) => {
      try {
        await this.llm_response_cache.index_done_callback();
      } catch (pe) {
        this.log('error', `Failed to persist LLM cache: ${errStr(pe)}`);
      }
      const processingEndTime = this.intNow();
      const [list, count] = failedChunkSnapshot();
      await this.upsertStatus({
        [docId]: {
          status: DocStatus.FAILED,
          error_msg: errStr(e),
          chunks_count: count,
          chunks_list: list,
          content_summary: statusDoc.content_summary,
          content_length: statusDoc.content_length,
          created_at: statusDoc.created_at,
          updated_at: this.isoNow(),
          file_path: filePath,
          track_id: statusDoc.track_id,
          metadata: { processing_start_time: processingStartTime, processing_end_time: processingEndTime },
        },
      });
    };

    await b.semaphore.acquire();
    try {
      try {
        const contentData = await this.full_docs.get_by_id(docId);
        if (pyTruthy(contentData)) {
          filePath = resolveDocFilePath(statusDoc, contentData);
          statusDoc.file_path = filePath;
        }
        if (status.cancellation_requested) throw new PipelineCancelledException('User cancelled');
        b.counter.processed++;
        currentFileNumber = b.counter.processed;
        status.cur_batch = b.counter.processed;
        let m = `Extracting stage ${currentFileNumber}/${totalFiles}: ${filePath}`;
        this.log('info', m);
        status.history_messages.push(m);
        m = `Processing d-id: ${docId}`;
        this.say(status, m);
        if (status.history_messages.length > 10000) {
          this.log('info', `Trimming pipeline history from ${status.history_messages.length} to 5000 messages`);
          status.history_messages.splice(0, status.history_messages.length - 5000);
        }
        if (!pyTruthy(contentData)) throw new Error(`Document content not found in full_docs for doc_id: ${docId}`);
        const content = contentData!.content as string;

        let result: unknown = this.chunking_func(this.tokenizer!, content, b.split_by_character, b.split_by_character_only, this.chunk_overlap_token_size, this.chunk_token_size);
        if (result instanceof Promise) result = await result;
        if (!Array.isArray(result)) throw new TypeError(`chunking_func must return a list or tuple of dicts, got ${typeof result}`);
        chunks = new Map();
        for (const dp of result as TextChunk[]) {
          chunks.set(computeMdhashId(dp.content, 'chunk-'), { ...dp, full_doc_id: docId, file_path: filePath, llm_cache_list: [] });
        }
        if (!chunks.size) this.log('warning', 'No document chunks to process');
        processingStartTime = this.intNow();
        if (status.cancellation_requested) throw new PipelineCancelledException('User cancelled');

        // stage 1: status, chunk vectors, chunk records - after the previous document's
        await waitTurn(b.stage1);
        const chunkObj = Object.fromEntries(chunks);
        const vdbChunks = Object.fromEntries([...chunks].map(([k, v]) => [k, { ...v }]));
        try {
          await Promise.all([
            this.upsertStatus({
              [docId]: {
                status: DocStatus.PROCESSING,
                chunks_count: chunks.size,
                chunks_list: [...chunks.keys()],
                content_summary: statusDoc.content_summary,
                content_length: statusDoc.content_length,
                created_at: statusDoc.created_at,
                updated_at: this.isoNow(),
                file_path: filePath,
                track_id: statusDoc.track_id,
                metadata: { processing_start_time: processingStartTime },
              },
            }),
            this.chunks_vdb.upsert(vdbChunks),
            this.text_chunks.upsert(chunkObj),
          ]);
        } finally {
          b.stage1[index]!.resolve();
        }
        // stage 2: extraction (concurrent with other documents)
        if (!this.chunks_only) chunkResults = await this.processExtractEntities(chunkObj as never, status);
        extractionOk = true;
      } catch (e) {
        b.stage1[index]!.resolve();
        if (e instanceof PipelineCancelledException) {
          this.say(status, `User cancelled ${currentFileNumber}/${totalFiles}: ${filePath}`, 'warning');
        } else {
          this.log('error', formatTraceback(e));
          const m = `Failed to extract document ${currentFileNumber}/${totalFiles}: ${filePath}`;
          this.log('error', m);
          status.latest_message = m;
          status.history_messages.push(formatTraceback(e));
          status.history_messages.push(m);
        }
        await writeFailed(e);
      }

      if (extractionOk) {
        await waitTurn(b.merged);
        try {
          if (status.cancellation_requested) throw new PipelineCancelledException('User cancelled');
          if (!this.chunks_only) await mergeNodesAndEdges(chunkResults, this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, this.mergeConfig(), {
            fullEntitiesStorage: this.full_entities,
            fullRelationsStorage: this.full_relations,
            docId,
            pipelineStatus: status,
            llmResponseCache: this.llm_response_cache,
            entityChunksStorage: this.entity_chunks,
            relationChunksStorage: this.relation_chunks,
            currentFileNumber,
            totalFiles,
            filePath,
            clock: () => this.intNow(),
            log: this.log as MergeLog,
          });
          const processingEndTime = this.intNow();
          await this.upsertStatus({
            [docId]: {
              status: DocStatus.PROCESSED,
              chunks_count: chunks.size,
              chunks_list: [...chunks.keys()],
              content_summary: statusDoc.content_summary,
              content_length: statusDoc.content_length,
              created_at: statusDoc.created_at,
              updated_at: this.isoNow(),
              file_path: filePath,
              track_id: statusDoc.track_id,
              metadata: { processing_start_time: processingStartTime, processing_end_time: processingEndTime },
            },
          });
          await this.insertDone();
          this.say(status, `Completed processing file ${currentFileNumber}/${totalFiles}: ${filePath}`);
        } catch (e) {
          if (e instanceof PipelineCancelledException) {
            this.say(status, `User cancelled during merge ${currentFileNumber}/${totalFiles}: ${filePath}`, 'warning');
          } else {
            this.log('error', formatTraceback(e));
            const m = `Merging stage failed in document ${currentFileNumber}/${totalFiles}: ${filePath}`;
            this.log('error', m);
            status.latest_message = m;
            status.history_messages.push(formatTraceback(e));
            status.history_messages.push(m);
          }
          await writeFailed(e);
        }
      }
    } finally {
      b.stage1[index]!.resolve();
      b.merged[index]!.resolve();
      b.semaphore.release();
    }
  }

  /** The global_config keys extract_entities reads (asdict(self) upstream). */
  private extractConfig(): ExtractConfig {
    return this.asdict() as unknown as ExtractConfig;
  }

  /** The global_config keys the merge and the rebuild read. */
  private mergeConfig(): MergeConfig {
    return this.asdict() as unknown as MergeConfig;
  }

  /** LightRAG._process_extract_entities */
  private async processExtractEntities(chunks: Record<string, { content: string; file_path?: string }>, status: PipelineStatusData | null): Promise<ChunkExtraction[]> {
    try {
      return await extractEntities(chunks, this.extractConfig(), {
        ...(status ? { pipelineStatus: status } : {}),
        llmResponseCache: this.llm_response_cache,
        textChunksStorage: this.text_chunks,
        log: this.log as MergeLog,
      });
    } catch (e) {
      const m = `Failed to extract entities and relationships: ${errStr(e)}`;
      this.log('error', m);
      if (status) {
        status.latest_message = m;
        status.history_messages.push(m);
      }
      throw e;
    }
  }

  /** LightRAG._insert_done: persist every storage. */
  private async insertDone(status: PipelineStatusData | null = null): Promise<void> {
    await Promise.all(
      [
        this.full_docs,
        this.doc_status,
        this.text_chunks,
        this.full_entities,
        this.full_relations,
        this.entity_chunks,
        this.relation_chunks,
        this.llm_response_cache,
        this.entities_vdb,
        this.relationships_vdb,
        this.chunks_vdb,
        this.chunk_entity_relation_graph,
      ].map((s) => s.index_done_callback()),
    );
    const m = 'In memory DB persist to disk';
    this.log('info', m);
    if (status) {
      status.latest_message = m;
      status.history_messages.push(m);
    }
  }

  /** LightRAG.ainsert_custom_kg: chunks, entities and relationships given directly. */
  async ainsert_custom_kg(customKg: Record<string, unknown>, fullDocId: string | null = null): Promise<void> {
    let updateStorage = false;
    try {
      const allChunks = new Map<string, Record<string, unknown>>();
      const chunkToSource = new Map<string, string>();
      const g = (o: Record<string, unknown>, k: string, d: unknown) => (k in o ? o[k] : d);
      for (const cd of (customKg.chunks as Record<string, unknown>[] | undefined) ?? []) {
        const content = sanitizeTextForEncoding(cd.content as string);
        const sourceId = cd.source_id as string;
        const chunkId = computeMdhashId(content, 'chunk-');
        allChunks.set(chunkId, {
          content,
          source_id: sourceId,
          tokens: this.tokenizer!.encode(content).length,
          chunk_order_index: 'chunk_order_index' in cd ? cd.chunk_order_index : 0,
          full_doc_id: fullDocId !== null ? fullDocId : sourceId,
          file_path: g(cd, 'file_path', 'custom_kg'),
          status: DocStatus.PROCESSED,
        });
        chunkToSource.set(sourceId, chunkId);
        updateStorage = true;
      }
      if (allChunks.size) {
        const obj = Object.fromEntries(allChunks);
        await Promise.all([this.chunks_vdb.upsert(Object.fromEntries([...allChunks].map(([k, v]) => [k, { ...v }]))), this.text_chunks.upsert(obj)]);
      }
      const graph = this.chunk_entity_relation_graph;
      const dedupedEntities = new Map<string, Record<string, unknown>>();
      for (const ed of (customKg.entities as Record<string, unknown>[] | undefined) ?? []) {
        dedupedEntities.delete(ed.entity_name as string);
        dedupedEntities.set(ed.entity_name as string, ed);
      }
      const allEntities: Record<string, unknown>[] = [];
      const entityNodes: [string, Record<string, string | number>][] = [];
      for (const ed of dedupedEntities.values()) {
        const name = ed.entity_name as string;
        const sourceId = chunkToSource.get(g(ed, 'source_id', 'UNKNOWN') as string) ?? 'UNKNOWN';
        if (sourceId === 'UNKNOWN') this.log('warning', `Entity '${name}' has an UNKNOWN source_id. Please check the source mapping.`);
        const node = {
          entity_id: name,
          entity_type: g(ed, 'entity_type', 'UNKNOWN') as string,
          description: g(ed, 'description', 'No description provided') as string,
          source_id: sourceId,
          file_path: g(ed, 'file_path', 'custom_kg') as string,
          created_at: this.intNow(),
        };
        entityNodes.push([name, node]);
        allEntities.push({ ...node, entity_name: name });
        updateStorage = true;
      }
      if (entityNodes.length) await graph.upsert_nodes_batch(entityNodes);

      const dedupedRels = new Map<string, Record<string, unknown>>();
      for (const rd of (customKg.relationships as Record<string, unknown>[] | undefined) ?? []) {
        const k = edgeKey(...([rd.src_id as string, rd.tgt_id as string].sort(pyCompare) as [string, string]));
        dedupedRels.delete(k);
        dedupedRels.set(k, rd);
      }
      const needed = new Set<string>();
      for (const rd of dedupedRels.values()) {
        needed.add(rd.src_id as string);
        needed.add(rd.tgt_id as string);
      }
      const existing = await graph.has_nodes_batch([...needed]);
      const missing: [string, Record<string, string | number>][] = [];
      const allRels: Record<string, unknown>[] = [];
      const edgeList: [string, string, Record<string, string | number>][] = [];
      for (const rd of dedupedRels.values()) {
        const src = rd.src_id as string;
        const tgt = rd.tgt_id as string;
        const sourceId = chunkToSource.get(g(rd, 'source_id', 'UNKNOWN') as string) ?? 'UNKNOWN';
        const filePath = g(rd, 'file_path', 'custom_kg') as string;
        if (sourceId === 'UNKNOWN') this.log('warning', `Relationship from '${src}' to '${tgt}' has an UNKNOWN source_id. Please check the source mapping.`);
        for (const id of [src, tgt]) {
          if (existing.has(id)) continue;
          missing.push([id, { entity_id: id, source_id: sourceId, description: 'UNKNOWN', entity_type: 'UNKNOWN', file_path: filePath, created_at: this.intNow() }]);
          existing.add(id);
        }
        const [ns, nt] = [src, tgt].sort(pyCompare) as [string, string];
        const weight = g(rd, 'weight', 1.0) as number;
        edgeList.push([src, tgt, { weight, description: rd.description as string, keywords: rd.keywords as string, source_id: sourceId, file_path: filePath, created_at: this.intNow() }]);
        allRels.push({ src_id: ns, tgt_id: nt, description: rd.description, keywords: rd.keywords, source_id: sourceId, weight, file_path: filePath, created_at: this.intNow() });
        updateStorage = true;
      }
      if (missing.length) await graph.upsert_nodes_batch(missing);
      if (edgeList.length) await graph.upsert_edges_batch(edgeList);

      const entVdb: Record<string, Record<string, unknown>> = {};
      for (const dp of allEntities) {
        entVdb[computeMdhashId(dp.entity_name as string, 'ent-')] = {
          content: `${dp.entity_name as string}\n${dp.description as string}`,
          entity_name: dp.entity_name,
          source_id: dp.source_id,
          description: dp.description,
          entity_type: dp.entity_type,
          file_path: g(dp, 'file_path', 'custom_kg'),
        };
      }
      const relVdb: Record<string, Record<string, unknown>> = {};
      for (const dp of allRels) {
        relVdb[computeMdhashId((dp.src_id as string) + (dp.tgt_id as string), 'rel-')] = {
          src_id: dp.src_id,
          tgt_id: dp.tgt_id,
          source_id: dp.source_id,
          content: `${String(dp.keywords)}\t${dp.src_id as string}\n${dp.tgt_id as string}\n${String(dp.description)}`,
          keywords: dp.keywords,
          description: dp.description,
          weight: dp.weight,
          file_path: g(dp, 'file_path', 'custom_kg'),
        };
      }
      const legacy = [...new Set(allRels.flatMap((dp) => makeRelationVdbIds(dp.src_id as string, dp.tgt_id as string).slice(1)))].sort(pyCompare);
      await Promise.all([this.entities_vdb.upsert(entVdb), this.relationships_vdb.upsert(relVdb)]);
      if (legacy.length) await this.relationships_vdb.delete(legacy);
    } catch (e) {
      this.log('error', `Error in ainsert_custom_kg: ${errStr(e)}`);
      throw e;
    } finally {
      if (updateStorage) await this.insertDone();
    }
  }

  // ---------------------------------------------------------------- query

  /** The global_config the query functions get (asdict(self) upstream). */
  private queryConfig(): QueryGlobalConfig {
    return this.asdict() as unknown as QueryGlobalConfig;
  }

  /** text_chunks_db as upstream's query path reads it: the records plus the storage's config and embedding function. */
  private queryTextChunksDb() {
    return queryTextChunks(this.text_chunks, this.initConfig as unknown as QueryGlobalConfig, this.embedding_func);
  }

  /** LightRAG.aquery: the answer only (a string, or an async iterator when streaming). */
  async aquery(query: string, param: QueryParam = makeQueryParam(), system_prompt: string | null = null): Promise<string | AsyncIterable<string> | null> {
    const result = await this.aquery_llm(query, param, system_prompt);
    const llm = (result.llm_response ?? {}) as { is_streaming?: boolean; response_iterator?: AsyncIterable<string> | null; content?: string | null };
    if (llm.is_streaming) return llm.response_iterator ?? null;
    return 'content' in llm ? (llm.content ?? null) : '';
  }

  /** LightRAG.aquery_data: retrieval only (only_need_context), structured. */
  async aquery_data(query: string, param: QueryParam = makeQueryParam()): Promise<Record<string, unknown>> {
    const gc = this.queryConfig();
    const dataParam = makeQueryParam({
      mode: param.mode,
      only_need_context: true,
      only_need_prompt: false,
      response_type: param.response_type,
      stream: false,
      top_k: param.top_k,
      chunk_top_k: param.chunk_top_k,
      max_entity_tokens: param.max_entity_tokens,
      max_relation_tokens: param.max_relation_tokens,
      max_total_tokens: param.max_total_tokens,
      hl_keywords: param.hl_keywords,
      ll_keywords: param.ll_keywords,
      conversation_history: param.conversation_history,
      history_turns: param.history_turns,
      model_func: param.model_func,
      user_prompt: param.user_prompt,
      enable_rerank: param.enable_rerank,
    });
    let result: QueryResult | null;
    const mode = dataParam.mode as string;
    if (['local', 'global', 'hybrid', 'mix'].includes(mode)) {
      result = await kgQuery(pyStrip(query), this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, this.queryTextChunksDb(), dataParam, gc, this.llm_response_cache, null, this.chunks_vdb);
    } else if (mode === 'naive') {
      result = await naiveQuery(pyStrip(query), this.chunks_vdb, dataParam, gc, this.llm_response_cache, null);
    } else if (mode === 'bypass') {
      result = new QueryResult({ content: '', raw_data: convertToUserFormat([], [], [], [], 'bypass') });
    } else throw pyError('ValueError', `Unknown mode ${mode}`);

    let finalData: Record<string, unknown>;
    if (result === null) {
      finalData = {
        status: 'failure',
        message: mode === 'naive' ? 'No relevant document chunks found.' : 'Query returned no results',
        data: {},
        metadata: { failure_reason: 'no_results', mode },
      };
      this.log('info', '[aquery_data] Query returned no results.');
    } else {
      finalData = (result.raw_data ?? {}) as Record<string, unknown>;
      if (!(finalData && 'data' in finalData)) this.log('warning', '[aquery_data] No data section found in query result');
    }
    await this.queryDone();
    return finalData;
  }

  /** LightRAG.aquery_llm: retrieval plus the answer; failures come back as a "failure" dict. */
  async aquery_llm(query: string, param: QueryParam = makeQueryParam(), system_prompt: string | null = null): Promise<Record<string, unknown>> {
    const gc = this.queryConfig();
    try {
      let result: QueryResult | null;
      const mode = param.mode as string;
      if (['local', 'global', 'hybrid', 'mix'].includes(mode)) {
        result = await kgQuery(pyStrip(query), this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, this.queryTextChunksDb(), param, gc, this.llm_response_cache, system_prompt, this.chunks_vdb);
      } else if (mode === 'naive') {
        result = await naiveQuery(pyStrip(query), this.chunks_vdb, param, gc, this.llm_response_cache, system_prompt);
      } else if (mode === 'bypass') {
        const useLlm = (param.model_func ?? gc.llm_model_func) as QueryLlmFunc | null | undefined;
        if (typeof useLlm !== 'function') throw new TypeError("'NoneType' object is not callable");
        // stream is a bool here; upstream also maps None to True
        if ((param.stream as unknown) === null || param.stream === undefined) param.stream = true;
        const response = await useLlm(pyStrip(query), {
          system_prompt,
          history_messages: param.conversation_history,
          enable_cot: true,
          stream: param.stream,
          _priority: 8,
        });
        if (typeof response === 'string') {
          return {
            status: 'success',
            message: 'Bypass mode LLM non streaming response',
            data: {},
            metadata: {},
            llm_response: { content: response, response_iterator: null, is_streaming: false },
          };
        }
        return {
          status: 'success',
          message: 'Bypass mode LLM streaming response',
          data: {},
          metadata: {},
          llm_response: { content: null, response_iterator: response, is_streaming: true },
        };
      } else throw pyError('ValueError', `Unknown mode ${mode}`);

      await this.queryDone();
      if (result === null) {
        return {
          status: 'failure',
          message: 'Query returned no results',
          data: {},
          metadata: { failure_reason: 'no_results', mode },
          llm_response: { content: PROMPTS.fail_response, response_iterator: null, is_streaming: false },
        };
      }
      const raw = (result.raw_data ?? {}) as Record<string, unknown>;
      raw.llm_response = {
        content: result.is_streaming ? null : result.content,
        response_iterator: result.is_streaming ? result.response_iterator : null,
        is_streaming: result.is_streaming,
      };
      return raw;
    } catch (e) {
      this.log('error', `Query failed: ${errStr(e)}`);
      return {
        status: 'failure',
        message: `Query failed: ${errStr(e)}`,
        data: {},
        metadata: {},
        llm_response: { content: null, response_iterator: null, is_streaming: false },
      };
    }
  }

  private async queryDone(): Promise<void> {
    await this.llm_response_cache.index_done_callback();
  }

  // ---------------------------------------------------------------- status queries

  /** LightRAG.aclear_cache: drop every LLM cache entry. */
  async aclear_cache(): Promise<void> {
    try {
      const r = await this.llm_response_cache.drop();
      if (r.status === 'success') this.log('info', 'Cleared all cache');
      else this.log('warning', 'Failed to clear all cache');
      await this.llm_response_cache.index_done_callback();
    } catch (e) {
      this.log('error', `Error while clearing cache: ${errStr(e)}`);
    }
  }

  async get_docs_by_status(status: DocStatusValue): Promise<Map<string, DocProcessingStatus>> {
    return this.doc_status.get_docs_by_status(status);
  }

  /** LightRAG.aget_docs_by_ids: the statuses found, in request order. */
  async aget_docs_by_ids(ids: string | string[] | null): Promise<Map<string, DocProcessingStatus | Record<string, unknown>>> {
    if (ids === null) {
      this.log('warning', 'aget_docs_by_ids called with None input, returning empty dict.');
      return new Map();
    }
    const list = typeof ids === 'string' ? [ids] : ids;
    if (!list.length) {
      this.log('debug', 'aget_docs_by_ids called with an empty list of IDs.');
      return new Map();
    }
    const results = await Promise.all(list.map((id) => this.doc_status.get_by_id(id)));
    const found = new Map<string, Record<string, unknown>>();
    const missing: string[] = [];
    results.forEach((r, i) => {
      if (pyTruthy(r)) found.set(list[i]!, r!);
      else missing.push(list[i]!);
    });
    if (missing.length) this.log('warning', `Document statuses not found for the following IDs: [${missing.map(pyReprStr).join(', ')}]`);
    return found;
  }

  async get_processing_status(): Promise<Record<string, number>> {
    return this.doc_status.get_status_counts();
  }

  async aget_docs_by_track_id(track_id: string): Promise<Map<string, DocProcessingStatus>> {
    return this.doc_status.get_docs_by_track_id(track_id);
  }

  // ---------------------------------------------------------------- delete

  /** LightRAG._update_delete_retry_state: persist what a retried deletion needs. */
  private async updateDeleteRetryState(
    docId: string,
    docStatusData: Record<string, unknown>,
    o: { deletionStage: string; docLlmCacheIds: string[]; errorMessage?: string | null; failed: boolean },
  ): Promise<Record<string, unknown>> {
    let metadata = 'metadata' in docStatusData ? docStatusData.metadata : {};
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) metadata = {};
    const meta = metadata as Record<string, unknown>;
    const backup = normalizeStringList('deletion_llm_cache_ids' in meta ? meta.deletion_llm_cache_ids : [], `doc ${docId} metadata.deletion_llm_cache_ids`, this.log);
    const retry = o.docLlmCacheIds.length ? o.docLlmCacheIds : backup;
    const updated: Record<string, unknown> = { ...meta };
    if (retry.length) updated.deletion_llm_cache_ids = retry;
    updated.last_deletion_attempt_at = this.isoNow();
    if (o.failed) {
      updated.deletion_failed = true;
      updated.deletion_failure_stage = o.deletionStage;
    } else {
      delete updated.deletion_failed;
      delete updated.deletion_failure_stage;
    }
    const record = { ...docStatusData, updated_at: this.isoNow(), metadata: updated, error_msg: o.failed ? (o.errorMessage ?? null) : '' };
    // a copy of the stored record: its status keeps whatever form it had
    await this.doc_status.upsert({ [docId]: record });
    return record;
  }

  /** LightRAG._get_existing_llm_cache_ids: the ids still present after a delete. */
  private async getExistingLlmCacheIds(cacheIds: string[]): Promise<string[]> {
    if (!cacheIds.length) return [];
    let records: (Record<string, unknown> | null)[];
    try {
      records = await this.llm_response_cache.get_by_ids(cacheIds);
    } catch (e) {
      throw new Error(`Failed to verify LLM cache deletion (delete may have succeeded): ${errStr(e)}`, { cause: e });
    }
    return cacheIds.filter((_, i) => records[i] !== null && records[i] !== undefined);
  }

  /**
   * LightRAG.adelete_by_doc_id: remove a document, its chunks and whatever only
   * it supported; entities and relations other documents still cite are rebuilt
   * from their remaining chunks' cached extractions.
   */
  async adelete_by_doc_id(doc_id: string, delete_llm_cache = false): Promise<DeletionResult> {
    const status = initializePipelineStatus(this.workspace);
    let weAcquired = false;
    if (!status.busy) {
      weAcquired = true;
      Object.assign(status, {
        busy: true,
        job_name: 'Single document deletion',
        job_start: this.isoNow(),
        docs: 1,
        batchs: 1,
        cur_batch: 0,
        request_pending: false,
        cancellation_requested: false,
        latest_message: `Starting deletion for document: ${doc_id}`,
      });
      status.history_messages.splice(0, status.history_messages.length, `Starting deletion for document: ${doc_id}`);
    } else {
      const jobName = String(status.job_name ?? '').toLowerCase();
      if (!jobName.startsWith('deleting') || !jobName.includes('document')) {
        return {
          status: 'not_allowed',
          doc_id,
          message: `Deletion not allowed: current job '${String(status.job_name)}' is not a document deletion job`,
          status_code: 403,
          file_path: null,
        };
      }
    }

    let operationsStarted = false;
    let fullyCompleted = false;
    let inFinalDeleteStage = false;
    let originalException: unknown = null;
    let docLlmCacheIds: string[] = [];
    let stage = 'initializing';
    let docStatusData: Record<string, unknown> | null = null;
    let filePath: string | null = null;
    this.say(status, `Starting deletion process for document ${doc_id}`);

    try {
      try {
        // 1. status
        docStatusData = await this.doc_status.get_by_id(doc_id);
        filePath = docStatusData ? ((docStatusData.file_path as string | undefined) ?? null) : null;
        if (!pyTruthy(docStatusData)) {
          this.log('warning', `Document ${doc_id} not found`);
          return { status: 'not_found', doc_id, message: `Document ${doc_id} not found.`, status_code: 404, file_path: '' };
        }
        const raw = docStatusData!.status;
        if (raw !== DocStatus.PROCESSED) {
          // upstream names the enum member for a known status, str() of the raw value otherwise
          const label = Object.values(DocStatus).includes(raw as DocStatusValue) ? String(raw).toUpperCase() : raw === null || raw === undefined ? 'None' : String(raw);
          const m = `Deleting ${doc_id} ${String(filePath)}(previous status: ${label})`;
          this.say(status, m);
        }

        // 2. chunks of the document
        let metadata = 'metadata' in docStatusData! ? docStatusData!.metadata : {};
        if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) metadata = {};
        const meta = metadata as Record<string, unknown>;
        const metadataCacheIds = normalizeStringList('deletion_llm_cache_ids' in meta ? meta.deletion_llm_cache_ids : [], `doc ${doc_id} metadata.deletion_llm_cache_ids`, this.log);
        const chunkIds = new Set(normalizeStringList('chunks_list' in docStatusData! ? docStatusData!.chunks_list : [], `doc ${doc_id} chunks_list`, this.log));

        if (!chunkIds.size) {
          this.log('warning', `No chunks found for document ${doc_id}`);
          operationsStarted = true;
          if (delete_llm_cache && metadataCacheIds.length) {
            try {
              stage = 'delete_llm_cache';
              await this.llm_response_cache.delete(metadataCacheIds);
              const remaining = await this.getExistingLlmCacheIds(metadataCacheIds);
              if (remaining.length) throw new Error(`${remaining.length} LLM cache entries still exist after delete`);
              this.log('info', `Cleaned up ${metadataCacheIds.length} LLM cache entries from prior attempt for document ${doc_id}`);
            } catch (e) {
              throw new Error(`Failed to delete LLM cache for document ${doc_id}: ${errStr(e)}`, { cause: e });
            }
          }
          try {
            stage = 'delete_doc_entries';
            await this.deleteStatus([doc_id]);
            await this.full_docs.delete([doc_id]);
          } catch (e) {
            this.log('error', `Failed to delete document ${doc_id} with no chunks: ${errStr(e)}`);
            throw new Error(`Failed to delete document entry: ${errStr(e)}`, { cause: e });
          }
          const m = `Document deleted without associated chunks: ${doc_id}`;
          this.say(status, m);
          fullyCompleted = true;
          return { status: 'success', doc_id, message: m, status_code: 200, file_path: filePath };
        }

        operationsStarted = true;
        // 3. cache ids of the chunks, kept in the status for a retried delete
        stage = 'collect_llm_cache';
        docLlmCacheIds = [...metadataCacheIds];
        try {
          const seen = new Set(docLlmCacheIds);
          for (const cd of await this.text_chunks.get_by_ids([...chunkIds])) {
            if (!cd || typeof cd !== 'object' || Array.isArray(cd)) continue;
            const ids = 'llm_cache_list' in cd ? cd.llm_cache_list : [];
            if (!Array.isArray(ids)) continue;
            for (const id of ids) {
              if (typeof id === 'string' && id && !seen.has(id)) {
                docLlmCacheIds.push(id);
                seen.add(id);
              }
            }
          }
        } catch (e) {
          this.log('error', `Failed to collect LLM cache ids for document ${doc_id}: ${errStr(e)}`);
          throw new Error(`Failed to collect LLM cache ids for document ${doc_id}: ${errStr(e)}`, { cause: e });
        }
        if (docLlmCacheIds.length) {
          try {
            docStatusData = await this.updateDeleteRetryState(doc_id, docStatusData!, { deletionStage: stage, docLlmCacheIds, failed: false });
          } catch (e) {
            this.log('error', `Failed to persist LLM cache IDs for document ${doc_id} to retry state: ${errStr(e)}`);
            const ctxt = metadataCacheIds.length ? 'retry — prior partial deletions may exist' : 'deletion not yet started';
            throw new Error(`Failed to persist LLM cache IDs for document ${doc_id} (${ctxt}): ${errStr(e)}`, { cause: e });
          }
          this.log('info', `Collected ${docLlmCacheIds.length} LLM cache entries for document ${doc_id}`);
        } else this.log('info', `No LLM cache entries found for document ${doc_id}`);

        // 4. what the document's entities and relations keep
        const entitiesToDelete = new Set<string>();
        const entitiesToRebuild = new Map<string, string[]>();
        const relationshipsToDelete = new Map<string, [string, string]>();
        const relationshipsToRebuild = new Map<string, string[]>();
        const entityChunkUpdates = new Map<string, string[]>();
        const relationChunkUpdates = new Map<string, [[string, string], string[]]>();
        const graph = this.chunk_entity_relation_graph;
        const affectedNodes: Record<string, unknown>[] = [];
        const affectedEdges: Record<string, unknown>[] = [];
        try {
          stage = 'analyze_graph_dependencies';
          const docEntities = await this.full_entities.get_by_id(doc_id);
          const docRelations = await this.full_relations.get_by_id(doc_id);
          if (pyTruthy(docEntities) && 'entity_names' in docEntities!) {
            const names = docEntities!.entity_names as string[];
            const nodes = await graph.get_nodes_batch(names);
            for (const name of names) {
              const nd = nodes.get(name);
              if (!nd || !pyTruthy(nd)) continue;
              // the graph's live dict, as upstream: the key reaches the file
              if (!('id' in nd)) nd.id = name;
              affectedNodes.push(nd);
            }
          }
          if (pyTruthy(docRelations) && 'relation_pairs' in docRelations!) {
            const pairs = docRelations!.relation_pairs as [string, string][];
            const edges = await graph.get_edges_batch(pairs.map((p) => ({ src: p[0], tgt: p[1] })));
            for (const [src, tgt] of pairs) {
              const ed = edges.get(edgePairKey(src, tgt));
              if (!ed || !pyTruthy(ed)) continue;
              if (!('source' in ed)) ed.source = src;
              if (!('target' in ed)) ed.target = tgt;
              affectedEdges.push(ed);
            }
          }
        } catch (e) {
          this.log('error', `Failed to analyze affected graph elements: ${errStr(e)}`);
          throw new Error(`Failed to analyze graph dependencies: ${errStr(e)}`, { cause: e });
        }

        try {
          for (const nd of affectedNodes) {
            const label = nd.entity_id as string | undefined;
            if (!label) continue;
            let existing: string[] = [];
            let graphSources: string[] = [];
            const stored = await this.entity_chunks.get_by_id(label);
            if (pyTruthy(stored) && typeof stored === 'object') existing = (('chunk_ids' in stored! ? stored!.chunk_ids : []) as string[]).filter((c) => c);
            if (pyTruthy(nd.source_id)) graphSources = String(nd.source_id).split(GRAPH_FIELD_SEP).filter((c) => c);
            if (!existing.length) existing = graphSources;
            if (!existing.length) {
              entitiesToDelete.add(label);
              entityChunkUpdates.set(label, []);
              continue;
            }
            const remaining = subtractSourceIds(existing, chunkIds);
            const graphRefsDeleted = graphSources.length > 0 && graphSources.some((c) => chunkIds.has(c));
            if (!remaining.length) {
              entitiesToDelete.add(label);
              entityChunkUpdates.set(label, []);
            } else if (!sameList(remaining, existing) || graphRefsDeleted) {
              entitiesToRebuild.set(label, remaining);
              entityChunkUpdates.set(label, remaining);
            } else this.log('info', `Untouch entity: ${label}`);
          }
          this.say(status, `Found ${entitiesToRebuild.size} affected entities`);

          for (const ed of affectedEdges) {
            const src = ed.source as string | undefined;
            const tgt = ed.target as string | undefined;
            if (!src || !tgt || !('source_id' in ed)) continue;
            const pair = [src, tgt].sort(pyCompare) as [string, string];
            const k = edgeKey(...pair);
            if (relationshipsToDelete.has(k) || relationshipsToRebuild.has(k)) continue;
            let existing: string[] = [];
            let graphSources: string[] = [];
            const stored = await this.relation_chunks.get_by_id(makeRelationChunkKey(src, tgt));
            if (pyTruthy(stored) && typeof stored === 'object') existing = (('chunk_ids' in stored! ? stored!.chunk_ids : []) as string[]).filter((c) => c);
            if (pyTruthy(ed.source_id)) graphSources = String(ed.source_id).split(GRAPH_FIELD_SEP).filter((c) => c);
            if (!existing.length) existing = graphSources;
            if (!existing.length) {
              relationshipsToDelete.set(k, pair);
              relationChunkUpdates.set(k, [pair, []]);
              continue;
            }
            const remaining = subtractSourceIds(existing, chunkIds);
            const graphRefsDeleted = graphSources.length > 0 && graphSources.some((c) => chunkIds.has(c));
            if (!remaining.length) {
              relationshipsToDelete.set(k, pair);
              relationChunkUpdates.set(k, [pair, []]);
            } else if (!sameList(remaining, existing) || graphRefsDeleted) {
              relationshipsToRebuild.set(k, remaining);
              relationChunkUpdates.set(k, [pair, remaining]);
            } else this.log('info', `Untouch relation: (${pyReprStr(pair[0])}, ${pyReprStr(pair[1])})`);
          }
          this.say(status, `Found ${relationshipsToRebuild.size} affected relations`);

          const currentTime = this.intNow();
          stage = 'update_chunk_tracking';
          if (entityChunkUpdates.size) {
            const payload: Record<string, Record<string, unknown>> = {};
            for (const [name, rem] of entityChunkUpdates) if (rem.length) payload[name] = { chunk_ids: rem, count: rem.length, updated_at: currentTime };
            if (Object.keys(payload).length) await this.entity_chunks.upsert(payload);
          }
          if (relationChunkUpdates.size) {
            const payload: Record<string, Record<string, unknown>> = {};
            for (const [[s, t], rem] of relationChunkUpdates.values()) if (rem.length) payload[makeRelationChunkKey(s, t)] = { chunk_ids: rem, count: rem.length, updated_at: currentTime };
            if (Object.keys(payload).length) await this.relation_chunks.upsert(payload);
          }
        } catch (e) {
          this.log('error', `Failed to process graph analysis results: ${errStr(e)}`);
          throw new Error(`Failed to process graph dependencies: ${errStr(e)}`, { cause: e });
        }

        // 5. chunks
        try {
          stage = 'delete_chunks';
          await this.chunks_vdb.delete([...chunkIds]);
          await this.text_chunks.delete([...chunkIds]);
          this.say(status, `Successfully deleted ${chunkIds.size} chunks from storage`);
        } catch (e) {
          this.log('error', `Failed to delete chunks: ${errStr(e)}`);
          throw new Error(`Failed to delete document chunks: ${errStr(e)}`, { cause: e });
        }

        // 6. relations left without chunks
        if (relationshipsToDelete.size) {
          try {
            stage = 'delete_relationships';
            const ids: string[] = [];
            for (const [s, t] of relationshipsToDelete.values()) ids.push(computeMdhashId(s + t, 'rel-'), computeMdhashId(t + s, 'rel-'));
            await this.relationships_vdb.delete(ids);
            await graph.remove_edges([...relationshipsToDelete.values()]);
            await this.relation_chunks.delete([...relationshipsToDelete.values()].map(([s, t]) => makeRelationChunkKey(s, t)));
            this.say(status, `Successfully deleted ${relationshipsToDelete.size} relations`);
          } catch (e) {
            this.log('error', `Failed to delete relationships: ${errStr(e)}`);
            throw new Error(`Failed to delete relationships: ${errStr(e)}`, { cause: e });
          }
        }

        // 7. entities left without chunks (and edges that still hang on them)
        if (entitiesToDelete.size) {
          try {
            stage = 'delete_entities';
            const nodesEdges = await graph.get_nodes_edges_batch([...entitiesToDelete]);
            const edgesToDelete = new Map<string, [string, string]>();
            let stillExist = 0;
            for (const edges of nodesEdges.values()) {
              if (!edges.length) continue;
              for (const [s, t] of edges) {
                const pair = [s, t].sort(pyCompare) as [string, string];
                edgesToDelete.set(edgeKey(...pair), pair);
                if (entitiesToDelete.has(s) && entitiesToDelete.has(t)) this.log('warning', `Edge still exists: ${s} <-> ${t}`);
                else if (entitiesToDelete.has(s)) this.log('warning', `Edge still exists: ${s} --> ${t}`);
                else this.log('warning', `Edge still exists: ${s} <-- ${t}`);
              }
              stillExist++;
            }
            if (stillExist) this.log('warning', `⚠️ ${stillExist} entities still has edges before deletion`);
            if (edgesToDelete.size) {
              const ids: string[] = [];
              for (const [s, t] of edgesToDelete.values()) ids.push(computeMdhashId(s + t, 'rel-'), computeMdhashId(t + s, 'rel-'));
              await this.relationships_vdb.delete(ids);
              await this.relation_chunks.delete([...edgesToDelete.values()].map(([s, t]) => makeRelationChunkKey(s, t)));
              this.log('info', `Cleaned ${edgesToDelete.size} residual edges from VDB and chunk-tracking storage`);
            }
            await graph.remove_nodes([...entitiesToDelete]);
            await this.entities_vdb.delete([...entitiesToDelete].map((e) => computeMdhashId(e, 'ent-')));
            await this.entity_chunks.delete([...entitiesToDelete]);
            this.say(status, `Successfully deleted ${entitiesToDelete.size} entities`);
          } catch (e) {
            this.log('error', `Failed to delete entities: ${errStr(e)}`);
            throw new Error(`Failed to delete entities: ${errStr(e)}`, { cause: e });
          }
        }

        try {
          stage = 'persist_pre_rebuild_changes';
          await this.insertDone();
        } catch (e) {
          this.log('error', `Failed to persist pre-rebuild changes: ${errStr(e)}`);
          throw new Error(`Failed to persist pre-rebuild changes: ${errStr(e)}`, { cause: e });
        }

        // 8. rebuild from the remaining chunks
        if (entitiesToRebuild.size || relationshipsToRebuild.size) {
          try {
            stage = 'rebuild_knowledge_graph';
            await rebuildKnowledgeFromChunks(
              entitiesToRebuild,
              relationshipsToRebuild,
              graph,
              this.entities_vdb,
              this.relationships_vdb,
              this.text_chunks,
              this.llm_response_cache,
              this.mergeConfig(),
              {
                pipelineStatus: status,
                entityChunksStorage: this.entity_chunks,
                relationChunksStorage: this.relation_chunks,
                clock: () => this.intNow(),
                log: this.log as MergeLog,
              },
            );
          } catch (e) {
            this.log('error', `Failed to rebuild knowledge from chunks: ${errStr(e)}`);
            throw new Error(`Failed to rebuild knowledge graph: ${errStr(e)}`, { cause: e });
          }
        }

        // 9. the document's LLM cache, while its status still exists
        let logMessage = `Document ${doc_id} successfully deleted`;
        if (delete_llm_cache && docLlmCacheIds.length) {
          try {
            stage = 'delete_llm_cache';
            await this.llm_response_cache.delete(docLlmCacheIds);
            const remaining = await this.getExistingLlmCacheIds(docLlmCacheIds);
            if (remaining.length) {
              docLlmCacheIds = remaining;
              throw new Error(`${remaining.length} LLM cache entries still exist after delete`);
            }
            const m = `Successfully deleted ${docLlmCacheIds.length} LLM cache entries for document ${doc_id}`;
            this.say(status, m);
            logMessage = m;
          } catch (e) {
            const m = `Failed to delete LLM cache for document ${doc_id}: ${errStr(e)}`;
            this.log('error', formatTraceback(e));
            this.say(status, m, 'error');
            throw new Error(m, { cause: e });
          }
        }

        // 10. the document's entity / relation index
        try {
          stage = 'delete_doc_graph_metadata';
          await this.full_entities.delete([doc_id]);
          await this.full_relations.delete([doc_id]);
        } catch (e) {
          this.log('error', `Failed to delete from full_entities/full_relations: ${errStr(e)}`);
          throw new Error(`Failed to delete from full_entities/full_relations: ${errStr(e)}`, { cause: e });
        }

        // 11. status first, then the document
        try {
          stage = 'delete_doc_entries';
          inFinalDeleteStage = true;
          await this.deleteStatus([doc_id]);
          await this.full_docs.delete([doc_id]);
        } catch (e) {
          this.log('error', `Failed to delete document and status: ${errStr(e)}`);
          throw new Error(`Failed to delete document and status: ${errStr(e)}`, { cause: e });
        }
        fullyCompleted = true;
        return { status: 'success', doc_id, message: logMessage, status_code: 200, file_path: filePath };
      } catch (e) {
        originalException = e;
        let message = `Error while deleting document ${doc_id}: ${errStr(e)}`;
        this.log('error', message);
        this.log('error', formatTraceback(e));
        try {
          if (docStatusData !== null && !inFinalDeleteStage) {
            docStatusData = await this.updateDeleteRetryState(doc_id, docStatusData, { deletionStage: stage, docLlmCacheIds, errorMessage: message, failed: true });
          }
        } catch (se) {
          this.log('error', `Failed to update deletion retry state for document ${doc_id}: ${errStr(se)}`);
          message = `${message}. Additionally, failed to persist retry state: ${errStr(se)}. Manual cleanup may be required.`;
        }
        return { status: 'fail', doc_id, message, status_code: 500, file_path: filePath };
      }
    } finally {
      if (operationsStarted) {
        try {
          await this.insertDone();
        } catch (pe) {
          this.log('error', `Failed to persist data after deletion attempt for ${doc_id}: ${errStr(pe)}`);
          // upstream overrides the result only when no stage raised and none completed: unreachable
        }
      } else this.log('debug', `No deletion operations were started for document ${doc_id}, skipping persistence`);
      if (weAcquired) {
        status.busy = false;
        status.cancellation_requested = false;
        this.say(status, `Deletion process completed for document: ${doc_id}`);
      }
    }
  }

  /** LightRAG.adelete_by_entity (upstream passes no chunk-tracking stores here). */
  async adelete_by_entity(entity_name: string): Promise<DeletionResult> {
    return utilsGraph.adeleteByEntity(this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, entity_name, null, null, { log: this.log });
  }

  /** LightRAG.adelete_by_relation (upstream passes no chunk-tracking store here). */
  async adelete_by_relation(source_entity: string, target_entity: string): Promise<DeletionResult> {
    return utilsGraph.adeleteByRelation(this.chunk_entity_relation_graph, this.relationships_vdb, source_entity, target_entity, null, { log: this.log });
  }

  // ---------------------------------------------------------------- editing

  async get_entity_info(entity_name: string, include_vector_data = false): Promise<Record<string, unknown>> {
    return utilsGraph.utilsGraphHooks.getEntityInfo(this.chunk_entity_relation_graph, this.entities_vdb, entity_name, include_vector_data);
  }

  async get_relation_info(src_entity: string, tgt_entity: string, include_vector_data = false): Promise<Record<string, unknown>> {
    return utilsGraph.utilsGraphHooks.getRelationInfo(this.chunk_entity_relation_graph, this.relationships_vdb, src_entity, tgt_entity, include_vector_data);
  }

  async aedit_entity(entity_name: string, updated_data: Record<string, unknown>, allow_rename = true, allow_merge = false): Promise<Record<string, unknown>> {
    return utilsGraph.aeditEntity(
      this.chunk_entity_relation_graph,
      this.entities_vdb,
      this.relationships_vdb,
      entity_name,
      updated_data,
      allow_rename,
      allow_merge,
      this.entity_chunks,
      this.relation_chunks,
      { log: this.log },
    );
  }

  async aedit_relation(source_entity: string, target_entity: string, updated_data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return utilsGraph.aeditRelation(this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, source_entity, target_entity, updated_data, this.relation_chunks, {
      log: this.log,
    });
  }

  /** LightRAG.acreate_entity (upstream passes no chunk-tracking stores here). */
  async acreate_entity(entity_name: string, entity_data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return utilsGraph.acreateEntity(this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, entity_name, entity_data, null, null, {
      log: this.log,
      clock: () => this.intNow(),
    });
  }

  /** LightRAG.acreate_relation (upstream passes no chunk-tracking store here). */
  async acreate_relation(source_entity: string, target_entity: string, relation_data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return utilsGraph.acreateRelation(this.chunk_entity_relation_graph, this.entities_vdb, this.relationships_vdb, source_entity, target_entity, relation_data, null, {
      log: this.log,
      clock: () => this.intNow(),
    });
  }

  async amerge_entities(
    source_entities: string[],
    target_entity: string,
    merge_strategy: Record<string, string> | null = null,
    target_entity_data: Record<string, unknown> | null = null,
  ): Promise<Record<string, unknown>> {
    return utilsGraph.amergeEntities(
      this.chunk_entity_relation_graph,
      this.entities_vdb,
      this.relationships_vdb,
      source_entities,
      target_entity,
      merge_strategy,
      target_entity_data,
      this.entity_chunks,
      this.relation_chunks,
      { log: this.log },
    );
  }
}

const sameList = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i]);

/** A deep copy of plain JSON-like data; functions and class instances are kept by reference. */
function structuredCloneSafe<T>(v: T): T {
  if (Array.isArray(v)) return v.map(structuredCloneSafe) as T;
  if (v && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype) {
    const out: Record<string, unknown> = {};
    for (const [k, x] of Object.entries(v)) out[k] = structuredCloneSafe(x);
    return out as T;
  }
  return v;
}

// referenced so the type-only imports stay meaningful for readers of the port
export type { BaseGraphStorage, BaseKVStorage, BaseVectorStorage, LlmFunc };
export { edgeKeyParts };
