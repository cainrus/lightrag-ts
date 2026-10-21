// The Node half of the upstream-test adapter (d11): upstream's own pytest
// suite calls lightrag-ts through tools/upstream/lightrag_ts_adapter.py.
//
// Protocol: JSON lines on stdin/stdout, strictly request/response, blocking.
//   py -> ts   {"call": name, "params": {...}}          params bound by upstream's signature
//   ts -> py   {"ok": value, "mut": {...}} | {"err": {"type", "message", "fields"}}
//   ts -> py   {"cb": {"target", "method", "args", "kwargs"}}   while a call runs
//   py -> ts   {"cbok": value} | {"cberr": {"type", "message"}}
// Python objects arrive as handles ({"$h": id, "kind": "fn"|"obj", ...}); calling
// one is a callback, answered before the TS code continues (fs.readSync blocks),
// so synchronous TS code (a tokenizer inside chunking) can use them too.
import { readSync, writeSync } from 'node:fs';
import {
  chunkingByTokenSize,
  edgeKeyParts,
  extractEntities,
  handleEntityRelationSummary,
  handleSingleEntityExtraction,
  handleSingleRelationshipExtraction,
  mergeEdgesThenUpsert,
  mergeNodesThenUpsert,
  type MergeLog,
} from '../../src/operate.ts';
import { makeQueryParam } from '../../src/base.ts';
import { performKgSearch, type QueryGraphStorage, type QueryTextChunksStorage, type QueryVectorStorage } from '../../src/query.ts';
import { removeThinkTags } from '../../src/utils.ts';

// stdout carries the protocol; nothing else may write there
console.log = (...a: unknown[]) => console.error(...a);

let pending = Buffer.alloc(0);
function readLine(): string | null {
  const chunk = Buffer.alloc(1 << 16);
  for (;;) {
    const nl = pending.indexOf(10);
    if (nl >= 0) {
      const line = pending.subarray(0, nl).toString('utf8');
      pending = pending.subarray(nl + 1);
      return line;
    }
    let n: number;
    try {
      n = readSync(0, chunk, 0, chunk.length, null);
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'EAGAIN') {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2);
        continue;
      }
      throw e;
    }
    if (n === 0) return null;
    pending = Buffer.concat([pending, chunk.subarray(0, n)]);
  }
}
function writeLine(v: unknown): void {
  const s = Buffer.from(JSON.stringify(v) + '\n', 'utf8');
  let off = 0;
  while (off < s.length) off += writeSync(1, s, off);
}

class PyError extends Error {}

/** Typed arrays travel as {$nd, dtype} (numpy on the Python side), Maps as objects. */
function enc(v: unknown): unknown {
  if (v instanceof Float32Array || v instanceof Float64Array) return { $nd: Array.from(v), dtype: v instanceof Float32Array ? 'float32' : 'float64' };
  if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [String(k), enc(x)]));
  if (Array.isArray(v)) return v.map(enc);
  if (v !== null && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, enc(x)]));
  if (typeof v === 'function') return null;
  return v;
}

function callback(target: number, method: string | null, args: unknown[], kwargs: Record<string, unknown> = {}): unknown {
  writeLine({ cb: { target, method, args: enc(args), kwargs: enc(kwargs) } });
  const line = readLine();
  if (line === null) throw new Error('bridge: python went away during a callback');
  const msg = JSON.parse(line) as { cbok?: unknown; cberr?: { type: string; message: string } };
  if (msg.cberr) {
    const e = new PyError(msg.cberr.message);
    e.name = msg.cberr.type;
    throw e;
  }
  return decode(msg.cbok);
}

type Handle = { $h: number; kind: 'fn' | 'obj'; methods?: string[]; attrs?: Record<string, unknown> };
const HANDLE = Symbol('py-handle');

function decode(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(decode);
  if (v === null || typeof v !== 'object') return v;
  const nd = v as { $nd?: unknown[]; dtype?: string };
  if (Array.isArray(nd.$nd)) {
    const toTyped = (x: unknown): unknown =>
      Array.isArray(x) && x.length && Array.isArray(x[0]) ? x.map(toTyped) : nd.dtype === 'float32' ? Float32Array.from(x as number[]) : nd.dtype === 'float64' ? Float64Array.from(x as number[]) : x;
    return toTyped(nd.$nd);
  }
  const h = v as Partial<Handle>;
  if (typeof h.$h === 'number') {
    const id = h.$h;
    // a fn handle may carry methods and attributes too (a Python mock stands for both)
    const o: Record<string | symbol, unknown> =
      h.kind === 'fn' ? Object.assign((...args: unknown[]) => callback(id, null, args), { [HANDLE]: id }) : { [HANDLE]: id };
    const put = (k: string, value: unknown) => Object.defineProperty(o, k, { value, writable: true, enumerable: true, configurable: true });
    for (const [k, x] of Object.entries(h.attrs ?? {})) put(k, decode(x));
    for (const m of h.methods ?? []) put(m, (...args: unknown[]) => callback(id, m, args));
    return o;
  }
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v)) out[k] = decode(x);
  return out;
}

/** An upstream LLM function takes keyword arguments; TS passes them as one options object. */
function asLlm(f: unknown): unknown {
  const id = (f as { [HANDLE]?: number } | null)?.[HANDLE];
  if (id === undefined) return f;
  return (prompt: string, opts: Record<string, unknown> = {}) => callback(id, null, [prompt], opts);
}
function withLlm(gc: Record<string, unknown> | null | undefined): Record<string, unknown> | null | undefined {
  if (gc && 'llm_model_func' in gc) gc.llm_model_func = asLlm(gc.llm_model_func);
  return gc;
}

/** A Python method called the way upstream calls it: positional args, then keywords. */
function kwMethod(obj: unknown, name: string, nPositional: number, kwNames: string[]) {
  const id = (obj as { [HANDLE]: number })[HANDLE];
  return (...args: unknown[]) => {
    const kwargs: Record<string, unknown> = {};
    kwNames.forEach((k, i) => {
      if (args.length > nPositional + i) kwargs[k] = args[nPositional + i] ?? null;
    });
    const last = args[nPositional];
    // a trailing options object (the TS spelling of keyword arguments)
    if (kwNames.length === 0 && last && typeof last === 'object' && !ArrayBuffer.isView(last) && !Array.isArray(last)) Object.assign(kwargs, last);
    return callback(id, name, args.slice(0, nPositional), kwargs);
  };
}

/** A vector storage handle as the query path uses it: query(query, top_k=, query_embedding=) as upstream calls it. */
function queryVdb(h: any): QueryVectorStorage | null {
  if (!h) return null;
  return { cosine_better_than_threshold: h.cosine_better_than_threshold, query: kwMethod(h, 'query', 1, ['top_k', 'query_embedding']) as never, get_vectors_by_ids: kwMethod(h, 'get_vectors_by_ids', 1, []) as never };
}

type P = Record<string, any>;
const FUNCS: Record<string, (p: P, log: MergeLog) => unknown> = {
  chunking_by_token_size: (p) =>
    chunkingByTokenSize(p.tokenizer, p.content, p.split_by_character, p.split_by_character_only, p.chunk_overlap_token_size, p.chunk_token_size),
  remove_think_tags: (p) => removeThinkTags(p.text),
  _handle_single_entity_extraction: (p) => handleSingleEntityExtraction(p.record_attributes, p.chunk_key, p.timestamp, p.file_path),
  _handle_single_relationship_extraction: (p) => handleSingleRelationshipExtraction(p.record_attributes, p.chunk_key, p.timestamp, p.file_path),
  extract_entities: async (p, log) => {
    const r = await extractEntities(p.chunks, withLlm(p.global_config) as never, {
      pipelineStatus: p.pipeline_status ?? undefined,
      llmResponseCache: p.llm_response_cache,
      textChunksStorage: p.text_chunks_storage,
      log,
    });
    return r.map(({ nodes, edges }) => ({ nodes: [...nodes], edges: [...edges].map(([k, v]) => [edgeKeyParts(k), v]) }));
  },
  _handle_entity_relation_summary: (p, log) =>
    handleEntityRelationSummary(p.description_type, p.entity_or_relation_name, p.description_list, p.separator, withLlm(p.global_config) as never, p.llm_response_cache, log),
  _merge_nodes_then_upsert: (p, log) =>
    mergeNodesThenUpsert(p.entity_name, p.nodes_data, p.knowledge_graph_inst, p.entity_vdb, withLlm(p.global_config) as never, {
      pipelineStatus: p.pipeline_status ?? undefined,
      llmResponseCache: p.llm_response_cache,
      entityChunksStorage: p.entity_chunks_storage,
      log,
    }),
  _perform_kg_search: async (p, log) => {
    const tc = p.text_chunks_db;
    const textChunks: QueryTextChunksStorage = {
      global_config: { ...(tc.global_config ?? {}), log },
      embedding_func: tc.embedding_func ? (kwMethod(tc, 'embedding_func', 1, []) as never) : null,
      get_by_ids: kwMethod(tc, 'get_by_ids', 1, []) as never,
    };
    const g = p.knowledge_graph_inst;
    const graph = Object.fromEntries(
      ['get_nodes_batch', 'node_degrees_batch', 'get_nodes_edges_batch', 'get_edges_batch', 'edge_degrees_batch'].map((m) => [m, kwMethod(g, m, 1, [])]),
    ) as unknown as QueryGraphStorage;
    const { [HANDLE]: _h, ...qp } = p.query_param;
    const r = await performKgSearch(p.query, p.ll_keywords, p.hl_keywords, graph, queryVdb(p.entities_vdb)!, queryVdb(p.relationships_vdb)!, textChunks, makeQueryParam(qp), queryVdb(p.chunks_vdb));
    return enc(r);
  },
  _merge_edges_then_upsert: async (p, log) => {
    const added: Record<string, unknown>[] = [];
    const value = await mergeEdgesThenUpsert(p.src_id, p.tgt_id, p.edges_data, p.knowledge_graph_inst, p.relationships_vdb, p.entity_vdb, withLlm(p.global_config) as never, {
      pipelineStatus: p.pipeline_status ?? undefined,
      llmResponseCache: p.llm_response_cache,
      entityChunksStorage: p.entity_chunks_storage,
      relationChunksStorage: p.relation_chunks_storage,
      addedEntities: added,
      log,
    });
    return { value, added };
  },
};

function encodeError(e: unknown): { type: string; message: string; fields: Record<string, unknown> } {
  const err = e as Error & Record<string, unknown>;
  const fields: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(err ?? {})) if (k !== 'message' && k !== 'name' && k !== 'stack') fields[k] = v;
  return { type: err?.name ?? 'Error', message: err?.message ?? String(e), fields };
}

for (;;) {
  const line = readLine();
  if (line === null) break;
  const req = JSON.parse(line) as { call: string; params: P; log?: unknown };
  const fn = FUNCS[req.call];
  try {
    if (!fn) throw new Error(`bridge: ${req.call} is not wired`);
    const params = decode(req.params) as P;
    const logFn = decode(req.log) as ((level: string, msg: string) => unknown) | undefined;
    const log: MergeLog = (level, msg) => void logFn?.(level, msg);
    const value = await fn(params, log);
    writeLine({ ok: value ?? null, mut: { pipeline_status: params.pipeline_status ?? null } });
  } catch (e) {
    writeLine({ err: encodeError(e) });
  }
}
