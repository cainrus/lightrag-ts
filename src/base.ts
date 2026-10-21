// Port of lightrag/base.py: storage contracts, DocStatus, DocProcessingStatus.
// Abstract dataclasses become interfaces; the default-method bodies upstream
// gives BaseGraphStorage (the *_batch helpers) live in `graphBatchDefaults`.
// QueryParam and the result dataclasses land with the query path (B3).
import { join } from 'node:path';
import type { KnowledgeGraph } from './types.ts';

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
  readonly cosine_better_than_threshold: number;
  readonly meta_fields: ReadonlySet<string>;
  query(query: string, top_k: number, query_embedding?: ArrayLike<number> | null): Promise<VectorQueryResult[]>;
  upsert(data: Record<string, KvRecord>): Promise<unknown>;
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
