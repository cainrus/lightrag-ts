// TS side of the query probes that tools/upstream/query_parity.py runs on
// upstream: same store, same stub LLM / rerank, same embeddings (replayed from
// the vectors upstream used), one record per probe. Shared by
// test/query.test.ts (synthetic store) and parity/query-context.ts (a copy of
// a real store).
import { createHash } from 'node:crypto';
import { closeSync, openSync, readFileSync, readSync } from 'node:fs';
import type { EmbeddingFunc, GlobalConfig } from '../src/base.ts';
import { JsonKVStorage } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDBStorage, vdbPath } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from '../src/kg/networkx_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import {
  type Dict,
  kgQuery,
  makeQueryParam,
  naiveQuery,
  pyJsonDumpsAny,
  type QueryGlobalConfig,
  type QueryLlmOptions,
  type QueryMode,
  type QueryParam,
  type QueryResult,
  type QueryTextChunksStorage,
  type RerankFunc,
} from '../src/query.ts';
import { type LlmResponseCache, pyStrip, TiktokenTokenizer } from '../src/utils.ts';

export const PINNED_TIME = 1760000000;

export type Probe = {
  id: string;
  query: string;
  mode: QueryMode;
  variant: 'context' | 'prompt' | 'full' | 'both';
  param: Partial<QueryParam>;
  gc: Dict & { rerank?: string };
  kw_response: string;
  answer: 'hash' | 'long' | 'stream';
  system_prompt: string | null;
};

export type ProbeRecord = {
  id: string;
  result: { content: string | null; is_streaming: boolean; streamed: string | null; raw_data: unknown } | null;
  error: string | null;
  llm_calls: Dict[];
  cache_writes: Dict[];
};

const META_FIELDS = {
  entities: ['entity_name', 'source_id', 'content', 'file_path'],
  relationships: ['src_id', 'tgt_id', 'source_id', 'content', 'file_path'],
  chunks: ['full_doc_id', 'content', 'file_path'],
};

const md5 = (s: string) => createHash('md5').update(s, 'utf8').digest('hex');

type State = { probe: Probe | null; calls: Dict[]; writes: Dict[] };

export type QueryStore = {
  gc: QueryGlobalConfig & GlobalConfig;
  base: Dict;
  graph: NetworkXStorage;
  entities: NanoVectorDBStorage;
  relationships: NanoVectorDBStorage;
  chunks: NanoVectorDBStorage;
  textChunks: QueryTextChunksStorage;
  cache: LlmResponseCache;
  state: State;
  /** texts the run asked to embed that the fixture lacks (a divergence from upstream) */
  missingVectors: Set<string>;
};

/** The stub upstream's harness installs as llm_model_func; same answers byte for byte. */
function stubLlm(state: State) {
  return async (prompt: string, o: QueryLlmOptions = {}): Promise<string | AsyncIterable<string>> => {
    const call = {
      prompt,
      system_prompt: o.system_prompt ?? null,
      history_messages: o.history_messages ?? null,
      keyword_extraction: o.keyword_extraction ?? false,
      enable_cot: o.enable_cot ?? false,
      stream: o.stream ?? false,
      _priority: o._priority ?? null,
    };
    state.calls.push(JSON.parse(JSON.stringify(call)) as Dict);
    const p = state.probe!;
    if (call.keyword_extraction) return p.kw_response;
    const digest = md5((o.system_prompt ?? '') + '\n' + prompt);
    if (p.answer === 'stream') {
      return (async function* () {
        for (const piece of ['stub ', 'stream ', digest]) yield piece;
      })();
    }
    if (p.answer === 'long') return `${o.system_prompt ?? ''} user said: model answer for ${prompt} <system>${digest}</system>`;
    return `STUB ${digest}`;
  };
}

/** upstream harness's stub_rerank: reversed order, scores 0.95, 0.85, ... */
const stubRerank: RerankFunc = async ({ documents, top_n }) => {
  const n = documents.length;
  const out = Array.from({ length: n }, (_, k) => ({ index: n - 1 - k, relevance_score: Number((0.95 - 0.1 * k).toFixed(2)) }));
  return top_n ? out.slice(0, top_n) : out;
};

export async function openQueryStore(dir: string, workspace: string, vectors: Record<string, number[]>): Promise<QueryStore> {
  const state: State = { probe: null, calls: [], writes: [] };
  const missingVectors = new Set<string>();
  const head = Buffer.alloc(256);
  const fd = openSync(vdbPath(dir, NameSpace.VECTOR_STORE_CHUNKS, workspace), 'r');
  readSync(fd, head, 0, 256, 0);
  closeSync(fd);
  const dim = Number(/"embedding_dim":\s*(\d+)/.exec(head.toString('latin1'))?.[1]);
  const embed = Object.assign(
    async (texts: string[]) =>
      texts.map((t) => {
        const v = vectors[t];
        if (!v) {
          missingVectors.add(t);
          throw new Error(`no fixture vector for ${JSON.stringify(t)}`);
        }
        return Float32Array.from(v);
      }),
    { embedding_dim: dim },
  ) as EmbeddingFunc;
  const gc: QueryGlobalConfig & GlobalConfig = {
    working_dir: dir,
    tokenizer: new TiktokenTokenizer('gpt-4o-mini'),
    kg_chunk_pick_method: 'VECTOR',
    related_chunk_number: 5,
    min_rerank_score: 0.0,
    rerank_model_func: null,
    enable_llm_cache: true,
    enable_llm_cache_for_entity_extract: true,
    addon_params: { language: 'English' },
    embedding_batch_num: 10,
    max_graph_nodes: 1000,
    vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 },
    llm_model_func: stubLlm(state),
  };
  const base: Dict = { ...gc };
  const opts = (namespace: string) => ({ namespace, workspace, global_config: gc });
  const graph = new NetworkXStorage(opts(NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION));
  const vdb = (ns: string, meta: string[]) => new NanoVectorDBStorage({ ...opts(ns), embedding_func: embed, meta_fields: meta });
  const entities = vdb(NameSpace.VECTOR_STORE_ENTITIES, META_FIELDS.entities);
  const relationships = vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS, META_FIELDS.relationships);
  const chunks = vdb(NameSpace.VECTOR_STORE_CHUNKS, META_FIELDS.chunks);
  const text = new JsonKVStorage({ ...opts(NameSpace.KV_STORE_TEXT_CHUNKS), clock: () => PINNED_TIME });
  const cacheKv = new JsonKVStorage({ ...opts(NameSpace.KV_STORE_LLM_RESPONSE_CACHE), clock: () => PINNED_TIME });
  for (const s of [graph, entities, relationships, chunks, text, cacheKv]) await s.initialize();
  const cache: LlmResponseCache = {
    global_config: gc,
    get_by_id: (k) => cacheKv.get_by_id(k),
    upsert: async (d) => {
      state.writes.push(JSON.parse(JSON.stringify(d)) as Dict);
      await cacheKv.upsert(d);
    },
  };
  const textChunks: QueryTextChunksStorage = { global_config: gc, embedding_func: embed, get_by_ids: (ids) => text.get_by_ids(ids) };
  return { gc, base, graph, entities, relationships, chunks, textChunks, cache, state, missingVectors };
}

function paramOf(p: Probe): QueryParam {
  const q: Partial<QueryParam> = { ...p.param, mode: p.mode };
  if (p.variant === 'context' || p.variant === 'both') q.only_need_context = true;
  if (p.variant === 'prompt' || p.variant === 'both') q.only_need_prompt = true;
  return makeQueryParam(q);
}

export async function runProbe(s: QueryStore, p: Probe): Promise<ProbeRecord> {
  for (const k of Object.keys(s.gc)) if (!Object.hasOwn(s.base, k)) delete (s.gc as Dict)[k];
  Object.assign(s.gc, s.base);
  const { rerank, ...extra } = p.gc;
  if (rerank === 'reverse') s.gc.rerank_model_func = stubRerank;
  Object.assign(s.gc, extra);
  s.state.probe = p;
  s.state.calls = [];
  s.state.writes = [];
  const rec: ProbeRecord = { id: p.id, result: null, error: null, llm_calls: [], cache_writes: [] };
  try {
    const qp = paramOf(p);
    const query = pyStrip(p.query);
    let r: QueryResult | null;
    if (p.mode === 'naive') r = await naiveQuery(query, s.chunks, qp, s.gc, s.cache, p.system_prompt);
    else r = await kgQuery(query, s.graph, s.entities, s.relationships, s.textChunks, qp, s.gc, s.cache, p.system_prompt, s.chunks);
    if (r !== null) {
      let streamed: string | null = null;
      if (r.is_streaming) {
        streamed = '';
        for await (const c of r.response_iterator!) streamed += c;
      }
      rec.result = { content: r.content, is_streaming: r.is_streaming, streamed, raw_data: r.raw_data };
    }
  } catch (e) {
    const err = e as Error;
    rec.error = `${err.name}: ${err.message}`;
  }
  rec.llm_calls = s.state.calls;
  rec.cache_writes = s.state.writes;
  return rec;
}

const canon = (v: unknown): string => JSON.stringify(v === undefined ? null : JSON.parse(JSON.stringify(v)));

function firstDiff(a: string, b: string): string {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return `at ${i}: ts=${JSON.stringify(a.slice(Math.max(0, i - 40), i + 60))} py=${JSON.stringify(b.slice(Math.max(0, i - 40), i + 60))}`;
}

/**
 * Differences between a TS record and upstream's: content and every LLM prompt
 * byte for byte; raw_data, LLM calls and cache writes as JSON values, key order
 * included (numbers compare by value: JS has no 1.0 vs 1).
 */
export function diffRecords(ts: ProbeRecord, py: ProbeRecord): string[] {
  const out: string[] = [];
  if (ts.error !== py.error) out.push(`error: ts=${ts.error} py=${py.error}`);
  if ((ts.result === null) !== (py.result === null)) out.push(`result: ts=${ts.result === null ? 'None' : 'value'} py=${py.result === null ? 'None' : 'value'}`);
  if (ts.result && py.result) {
    if (ts.result.content !== py.result.content) out.push(`content ${firstDiff(ts.result.content ?? '<None>', py.result.content ?? '<None>')}`);
    if (ts.result.is_streaming !== py.result.is_streaming) out.push(`is_streaming: ts=${ts.result.is_streaming} py=${py.result.is_streaming}`);
    if (ts.result.streamed !== py.result.streamed) out.push(`streamed: ts=${ts.result.streamed} py=${py.result.streamed}`);
    const a = canon(ts.result.raw_data);
    const b = canon(py.result.raw_data);
    if (a !== b) out.push(`raw_data ${firstDiff(a, b)}`);
  }
  if (ts.llm_calls.length !== py.llm_calls.length) out.push(`llm_calls: ts=${ts.llm_calls.length} py=${py.llm_calls.length}`);
  for (let i = 0; i < Math.min(ts.llm_calls.length, py.llm_calls.length); i++) {
    const a = canon(ts.llm_calls[i]);
    const b = canon(py.llm_calls[i]);
    if (a !== b) out.push(`llm_calls[${i}] ${firstDiff(a, b)}`);
  }
  const wa = canon(ts.cache_writes);
  const wb = canon(py.cache_writes);
  if (wa !== wb) out.push(`cache_writes ${firstDiff(wa, wb)}`);
  return out;
}

/** JSON Lines of upstream records; tolerant of a trailing newline. */
export function readRecords(file: string): ProbeRecord[] {
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as ProbeRecord);
}

/** For a record that diverged: the probe in a form a reader can rerun upstream. */
export const describeProbe = (p: Probe): string => pyJsonDumpsAny({ id: p.id, mode: p.mode, variant: p.variant, query: p.query, param: p.param, gc: p.gc });
