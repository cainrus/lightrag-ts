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

function callback(target: number, method: string | null, args: unknown[], kwargs: Record<string, unknown> = {}): unknown {
  writeLine({ cb: { target, method, args, kwargs } });
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
  const h = v as Partial<Handle>;
  if (typeof h.$h === 'number') {
    const id = h.$h;
    if (h.kind === 'fn') return Object.assign((...args: unknown[]) => callback(id, null, args), { [HANDLE]: id });
    const o: Record<string | symbol, unknown> = { [HANDLE]: id };
    for (const [k, x] of Object.entries(h.attrs ?? {})) o[k] = decode(x);
    for (const m of h.methods ?? []) o[m] = (...args: unknown[]) => callback(id, m, args);
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
