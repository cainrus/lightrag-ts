// Port of lightrag/base.py: storage contracts, DocStatus, DocProcessingStatus.
// Abstract dataclasses become interfaces; the default-method bodies upstream
// gives BaseGraphStorage (the *_batch helpers) live in `graphBatchDefaults`.
// QueryParam and the result dataclasses (B3) are at the end.
import { join } from 'node:path';
import {
  DEFAULT_CHUNK_TOP_K,
  DEFAULT_HISTORY_TURNS,
  DEFAULT_MAX_ENTITY_TOKENS,
  DEFAULT_MAX_RELATION_TOKENS,
  DEFAULT_MAX_TOTAL_TOKENS,
  DEFAULT_TOP_K,
} from './constants.ts';
import type { KnowledgeGraph } from './types.ts';
import type { ChatMessage } from './utils.ts';

/** The keys of upstream's `global_config` the storages read. */
export type GlobalConfig = {
  working_dir: string;
  embedding_batch_num?: number;
  max_graph_nodes?: number;
  vector_db_storage_cls_kwargs?: { cosine_better_than_threshold?: number; [k: string]: unknown };
  [k: string]: unknown;
};

/** StorageNameSpace dataclass fields. */
export type StorageOptions = {
  namespace: string;
  workspace?: string;
  global_config: GlobalConfig;
};

export type DropResult = { status: 'success' | 'error'; message: string };

/** upstream utils.EmbeddingFunc, the part storages use. */
export type EmbeddingFunc = {
  embedding_dim: number;
  model_name?: string;
  (texts: string[], opts?: { context?: 'document' | 'query'; _priority?: number }): Promise<ArrayLike<ArrayLike<number>>>;
};

export interface StorageNameSpace {
  readonly namespace: string;
  readonly workspace: string;
  readonly global_config: GlobalConfig;
  initialize(): Promise<void>;
  finalize(): Promise<void>;
  index_done_callback(): Promise<unknown>;
  drop(): Promise<DropResult>;
}

export type KvRecord = Record<string, unknown>;

export interface BaseKVStorage extends StorageNameSpace {
  get_by_id(id: string): Promise<KvRecord | null>;
  get_by_ids(ids: string[]): Promise<(KvRecord | null)[]>;
  filter_keys(keys: Iterable<string>): Promise<Set<string>>;
  upsert(data: Record<string, KvRecord>): Promise<void>;
  delete(ids: string[]): Promise<void>;
  is_empty(): Promise<boolean>;
}

export type VectorQueryResult = Record<string, unknown> & { id: string; distance: number; created_at: number | null };

export interface BaseVectorStorage extends StorageNameSpace {
  /**
   * Upstream's dataclass field. Optional here: the merge embeds queued writes
   * itself through it and falls back to plain upsert(data) without it.
   */
  readonly embedding_func?: EmbeddingFunc;
  readonly cosine_better_than_threshold: number;
  readonly meta_fields: ReadonlySet<string>;
  query(query: string, top_k: number, query_embedding?: ArrayLike<number> | null): Promise<VectorQueryResult[]>;
  /** `opts.embeddings` (port-only): precomputed vectors for the entries, in order. */
  upsert(data: Record<string, KvRecord>, opts?: { embeddings?: ArrayLike<ArrayLike<number>> }): Promise<unknown>;
  delete_entity(entity_name: string): Promise<void>;
  delete_entity_relation(entity_name: string): Promise<void>;
  get_by_id(id: string): Promise<KvRecord | null>;
  get_by_ids(ids: string[]): Promise<(KvRecord | null)[]>;
  delete(ids: string[]): Promise<void>;
  get_vectors_by_ids(ids: string[]): Promise<Record<string, number[]>>;
}

export type NodeData = Record<string, string | number | boolean>;
export type EdgeData = Record<string, string | number | boolean>;

export interface BaseGraphStorage extends StorageNameSpace {
  has_node(node_id: string): Promise<boolean>;
  has_edge(source_node_id: string, target_node_id: string): Promise<boolean>;
  node_degree(node_id: string): Promise<number>;
  edge_degree(src_id: string, tgt_id: string): Promise<number>;
  get_node(node_id: string): Promise<NodeData | null>;
  get_edge(source_node_id: string, target_node_id: string): Promise<EdgeData | null>;
  get_node_edges(source_node_id: string): Promise<[string, string][] | null>;
  get_nodes_batch(node_ids: string[]): Promise<Map<string, NodeData>>;
  node_degrees_batch(node_ids: string[]): Promise<Map<string, number>>;
  /** keyed by `edgePairKey(src, tgt)` (Python uses the tuple) */
  edge_degrees_batch(edge_pairs: [string, string][]): Promise<Map<string, number>>;
  get_edges_batch(pairs: { src: string; tgt: string }[]): Promise<Map<string, EdgeData>>;
  get_nodes_edges_batch(node_ids: string[]): Promise<Map<string, [string, string][]>>;
  upsert_node(node_id: string, node_data: NodeData): Promise<void>;
  upsert_nodes_batch(nodes: [string, NodeData][]): Promise<void>;
  has_nodes_batch(node_ids: string[]): Promise<Set<string>>;
  upsert_edges_batch(edges: [string, string, EdgeData][]): Promise<void>;
  upsert_edge(source_node_id: string, target_node_id: string, edge_data: EdgeData): Promise<void>;
  delete_node(node_id: string): Promise<void>;
  remove_nodes(nodes: string[]): Promise<void>;
  remove_edges(edges: [string, string][]): Promise<void>;
  get_all_labels(): Promise<string[]>;
  get_knowledge_graph(node_label: string, max_depth?: number, max_nodes?: number | null): Promise<KnowledgeGraph>;
  get_all_nodes(): Promise<NodeData[]>;
  get_all_edges(): Promise<EdgeData[]>;
  get_popular_labels(limit?: number): Promise<string[]>;
  search_labels(query: string, limit?: number): Promise<string[]>;
}

/** Python keys these batch maps by a (src, tgt) tuple; a NUL-joined string stands in. */
export function edgePairKey(src: string, tgt: string): string {
  return `${src}\u0000${tgt}`;
}

export const DocStatus = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  PREPROCESSED: 'preprocessed',
  PROCESSED: 'processed',
  FAILED: 'failed',
} as const;
export type DocStatusValue = (typeof DocStatus)[keyof typeof DocStatus];
export const DOC_STATUS_VALUES: readonly DocStatusValue[] = Object.values(DocStatus);

export type DocProcessingStatus = {
  content_summary: string;
  content_length: number;
  file_path: string;
  status: string;
  created_at: string;
  updated_at: string;
  track_id: string | null;
  chunks_count: number | null;
  chunks_list: string[] | null;
  error_msg: string | null;
  metadata: Record<string, unknown>;
  multimodal_processed: boolean | null;
};

/** What Python raises from `DocProcessingStatus(**data)` on a missing or unknown field. */
export class PyTypeError extends Error {
  override name = 'TypeError';
}

const DPS_REQUIRED = ['content_summary', 'content_length', 'file_path', 'status', 'created_at', 'updated_at'] as const;
const DPS_FIELDS = new Set<string>([
  ...DPS_REQUIRED,
  'track_id',
  'chunks_count',
  'chunks_list',
  'error_msg',
  'metadata',
  'multimodal_processed',
]);

/** `DocProcessingStatus(**data)` including its __post_init__ status rule. */
export function makeDocProcessingStatus(data: Record<string, unknown>): DocProcessingStatus {
  for (const k of Object.keys(data)) {
    if (!DPS_FIELDS.has(k)) throw new PyTypeError(`DocProcessingStatus.__init__() got an unexpected keyword argument '${k}'`);
  }
  const missing = DPS_REQUIRED.filter((k) => !(k in data));
  if (missing.length) throw new PyTypeError(`DocProcessingStatus.__init__() missing required argument(s): ${missing.join(', ')}`);
  const out: DocProcessingStatus = {
    content_summary: data.content_summary as string,
    content_length: data.content_length as number,
    file_path: data.file_path as string,
    status: data.status as string,
    created_at: data.created_at as string,
    updated_at: data.updated_at as string,
    track_id: (data.track_id ?? null) as string | null,
    chunks_count: (data.chunks_count ?? null) as number | null,
    chunks_list: 'chunks_list' in data ? (data.chunks_list as string[] | null) : [],
    error_msg: (data.error_msg ?? null) as string | null,
    metadata: 'metadata' in data ? (data.metadata as Record<string, unknown>) : {},
    multimodal_processed: (data.multimodal_processed ?? null) as boolean | null,
  };
  if (out.multimodal_processed === false && out.status === DocStatus.PROCESSED) out.status = DocStatus.PREPROCESSED;
  return out;
}

/** Working directory of a storage: `<working_dir>[/<workspace>]`. */
export function workspaceDir(opts: StorageOptions): string {
  const wd = opts.global_config.working_dir;
  return opts.workspace ? join(wd, opts.workspace) : wd;
}

/** Python `int(time.time())`; storages take a clock so tests are deterministic. */
export type Clock = () => number;
export const systemClock: Clock = () => Math.floor(Date.now() / 1000);

// ---------------------------------------------------------------- query (B3)

export type QueryMode = 'local' | 'global' | 'hybrid' | 'naive' | 'mix' | 'bypass';

export type QueryLlmOptions = {
  system_prompt?: string | null;
  history_messages?: ChatMessage[];
  enable_cot?: boolean;
  stream?: boolean;
  keyword_extraction?: boolean;
  _priority?: number;
};
/** An LLM binding as kg_query calls it: a string, or an async iterator when streaming. */
export type QueryLlmFunc = (prompt: string, opts?: QueryLlmOptions) => Promise<string | AsyncIterable<string>>;

/** base.QueryParam. `null` stands for Python's None where upstream allows it. */
export type QueryParam = {
  mode: QueryMode;
  only_need_context: boolean;
  only_need_prompt: boolean;
  response_type: string | null;
  stream: boolean;
  top_k: number;
  chunk_top_k: number | null;
  max_entity_tokens: number;
  max_relation_tokens: number;
  max_total_tokens: number;
  hl_keywords: string[];
  ll_keywords: string[];
  conversation_history: ChatMessage[];
  /** deprecated upstream, kept for the dataclass shape */
  history_turns: number;
  model_func: QueryLlmFunc | null;
  user_prompt: string | null;
  enable_rerank: boolean;
  include_references: boolean;
};

// base.py: `load_dotenv(dotenv_path=".env", override=False)` at import, before
// the QueryParam defaults read the environment. Node's loader keeps variables
// already set, like override=False; no .env in the cwd is not an error. This
// writes into process.env of the whole process, as Python's does - kept for
// key parity (TOP_K & co. change query cache keys); LIGHTRAG_TS_NO_DOTENV=1
// turns it off.
if (process.env.LIGHTRAG_TS_NO_DOTENV !== '1') {
  try {
    process.loadEnvFile('.env');
  } catch {
    // missing or unreadable .env: upstream's load_dotenv returns False and goes on
  }
}

/** Python's `int(os.getenv(name, str(default)))`, read once like the class attribute. */
function envInt(name: string, dflt: number): number {
  const v = process.env[name];
  if (v === undefined) return dflt;
  const t = v.trim();
  if (!/^[+-]?\d+(_\d+)*$/.test(t)) throw new Error(`invalid literal for int() with base 10: '${v}'`);
  return Number.parseInt(t.replaceAll('_', ''), 10);
}

/** Field defaults, evaluated at import like the dataclass's `os.getenv` defaults. */
const DEFAULTS = {
  top_k: envInt('TOP_K', DEFAULT_TOP_K),
  chunk_top_k: envInt('CHUNK_TOP_K', DEFAULT_CHUNK_TOP_K),
  max_entity_tokens: envInt('MAX_ENTITY_TOKENS', DEFAULT_MAX_ENTITY_TOKENS),
  max_relation_tokens: envInt('MAX_RELATION_TOKENS', DEFAULT_MAX_RELATION_TOKENS),
  max_total_tokens: envInt('MAX_TOTAL_TOKENS', DEFAULT_MAX_TOTAL_TOKENS),
  history_turns: envInt('HISTORY_TURNS', DEFAULT_HISTORY_TURNS),
  enable_rerank: (process.env.RERANK_BY_DEFAULT ?? 'true').toLowerCase() === 'true',
};

/** makeQueryParam's input: any subset of fields, undefined meaning "not given". */
export type QueryParamInit = { [K in keyof QueryParam]?: QueryParam[K] | undefined };

/**
 * `QueryParam(**fields)` with upstream's defaults (fresh lists per instance).
 * A field given as `undefined` keeps its default, as an omitted keyword does in
 * Python; `null` is Python's None and is kept.
 */
export function makeQueryParam(fields: QueryParamInit = {}): QueryParam {
  const given = Object.fromEntries(Object.entries(fields).filter(([, v]) => v !== undefined)) as Partial<QueryParam>;
  return {
    mode: 'mix',
    only_need_context: false,
    only_need_prompt: false,
    response_type: 'Multiple Paragraphs',
    stream: false,
    top_k: DEFAULTS.top_k,
    chunk_top_k: DEFAULTS.chunk_top_k,
    max_entity_tokens: DEFAULTS.max_entity_tokens,
    max_relation_tokens: DEFAULTS.max_relation_tokens,
    max_total_tokens: DEFAULTS.max_total_tokens,
    hl_keywords: [],
    ll_keywords: [],
    conversation_history: [],
    history_turns: DEFAULTS.history_turns,
    model_func: null,
    user_prompt: null,
    enable_rerank: DEFAULTS.enable_rerank,
    include_references: false,
    ...given,
  };
}

export type ReferenceEntry = { reference_id: string; file_path: string };

/** The dict convert_to_user_format builds, plus what callers add to it. */
export type QueryRawData = {
  status: string;
  message: string;
  data: {
    entities?: Record<string, unknown>[];
    relationships?: Record<string, unknown>[];
    chunks?: Record<string, unknown>[];
    references?: ReferenceEntry[];
  };
  metadata: Record<string, unknown>;
  [k: string]: unknown;
};

/** base.QueryResult */
export class QueryResult {
  content: string | null;
  response_iterator: AsyncIterable<string> | null;
  raw_data: QueryRawData | null;
  is_streaming: boolean;
  constructor(f: { content?: string | null; response_iterator?: AsyncIterable<string> | null; raw_data?: QueryRawData | null; is_streaming?: boolean } = {}) {
    this.content = f.content ?? null;
    this.response_iterator = f.response_iterator ?? null;
    this.raw_data = f.raw_data ?? null;
    this.is_streaming = f.is_streaming ?? false;
  }
  get reference_list(): ReferenceEntry[] {
    return this.raw_data ? (this.raw_data.data?.references ?? []) : [];
  }
  get metadata(): Record<string, unknown> {
    return this.raw_data ? (this.raw_data.metadata ?? {}) : {};
  }
}

/** base.QueryContextResult */
export class QueryContextResult {
  context: string;
  raw_data: QueryRawData;
  constructor(context: string, rawData: QueryRawData) {
    this.context = context;
    this.raw_data = rawData;
  }
  get reference_list(): ReferenceEntry[] {
    return this.raw_data.data?.references ?? [];
  }
}
