// Merge behaviour the byte-parity replays cannot see: concurrent callers,
// failures, cancellation, limits. Each test pins a review finding (S1-S4, M1, M3).
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { EmbeddingFunc } from '../src/base.ts';
import { PipelineCancelledException } from '../src/exceptions.ts';
import { JsonKVStorage } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDBStorage } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from '../src/kg/networkx_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import {
  type ChunkExtraction,
  defaultMergeConfig,
  edgeKey,
  type ExtractedEntity,
  type ExtractedRelation,
  type MergeConfig,
  mergeNodesAndEdges,
  mergeNodesThenUpsert,
  summarizeDescriptions,
} from '../src/operate.ts';
import { type LlmFunc, priorityLimitAsyncFuncCall, pyJsonDumps, TiktokenTokenizer } from '../src/utils.ts';

const tok = new TiktokenTokenizer('gpt-4o-mini');
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const ent = (name: string, description: string, chunk: string, timestamp = 1): ExtractedEntity => ({
  entity_name: name,
  entity_type: 'concept',
  description,
  source_id: chunk,
  file_path: `${chunk}.md`,
  timestamp,
});
const rel = (src: string, tgt: string, description: string, chunk: string, timestamp = 1): ExtractedRelation => ({
  src_id: src,
  tgt_id: tgt,
  weight: 1.0,
  description,
  keywords: 'k',
  source_id: chunk,
  file_path: `${chunk}.md`,
  timestamp,
});
function chunk(ents: ExtractedEntity[], rels: ExtractedRelation[] = []): ChunkExtraction {
  const nodes = new Map<string, ExtractedEntity[]>();
  for (const e of ents) nodes.set(e.entity_name, [...(nodes.get(e.entity_name) ?? []), e]);
  const edges = new Map<string, ExtractedRelation[]>();
  for (const r of rels) edges.set(edgeKey(r.src_id, r.tgt_id), [...(edges.get(edgeKey(r.src_id, r.tgt_id)) ?? []), r]);
  return { nodes, edges };
}

/** Live counter of concurrent calls. */
function gauge() {
  const g = { now: 0, max: 0, calls: 0 };
  const enter = () => {
    g.calls++;
    g.max = Math.max(g.max, ++g.now);
  };
  const leave = () => void g.now--;
  return Object.assign(g, { enter, leave });
}

function embedder(dim = 4, opts: { delayMs?: number; fail?: boolean; g?: ReturnType<typeof gauge> } = {}): EmbeddingFunc {
  const f = async (texts: string[]) => {
    opts.g?.enter();
    try {
      if (opts.delayMs) await sleep(opts.delayMs);
      if (opts.fail) throw new Error('embedding service down');
      return texts.map((t) => Float32Array.from({ length: dim }, (_, j) => ((t.charCodeAt(j % t.length) % 13) + 1) / 13));
    } finally {
      opts.g?.leave();
    }
  };
  return Object.assign(f, { embedding_dim: dim });
}

async function stores(embedding = embedder()) {
  const dir = mkdtempSync(join(tmpdir(), 'lightrag-merge-conc-'));
  const gcStore = { working_dir: dir, vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 }, enable_llm_cache_for_entity_extract: true };
  const kv = async (namespace: string) => {
    const s = new JsonKVStorage({ namespace, global_config: gcStore, clock: () => 1 });
    await s.initialize();
    return s;
  };
  const vdb = (namespace: string, meta: string[]) =>
    new NanoVectorDBStorage({ namespace, global_config: gcStore, embedding_func: embedding, meta_fields: meta, clock: () => 1 });
  return {
    graph: new NetworkXStorage({ namespace: NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION, global_config: gcStore }),
    entVdb: vdb(NameSpace.VECTOR_STORE_ENTITIES, ['entity_name', 'source_id', 'content', 'file_path']),
    relVdb: vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS, ['src_id', 'tgt_id', 'source_id', 'content', 'file_path']),
    entityChunks: await kv(NameSpace.KV_STORE_ENTITY_CHUNKS),
    relationChunks: await kv(NameSpace.KV_STORE_RELATION_CHUNKS),
    fullEntities: await kv(NameSpace.KV_STORE_FULL_ENTITIES),
    fullRelations: await kv(NameSpace.KV_STORE_FULL_RELATIONS),
    cache: await kv(NameSpace.KV_STORE_LLM_RESPONSE_CACHE),
  };
}
type Stores = Awaited<ReturnType<typeof stores>>;

const quietLlm: LlmFunc = async (p) => `summary of ${p.length}`;

function merge(s: Stores, results: ChunkExtraction[], gc: MergeConfig, docId: string, status = { history_messages: [] as string[] }) {
  return mergeNodesAndEdges(results, s.graph, s.entVdb, s.relVdb, gc, {
    docId,
    pipelineStatus: status,
    llmResponseCache: s.cache,
    entityChunksStorage: s.entityChunks,
    relationChunksStorage: s.relationChunks,
    fullEntitiesStorage: s.fullEntities,
    fullRelationsStorage: s.fullRelations,
    clock: () => 1,
  });
}

test('S1: two documents merged at once keep both chunk lists and descriptions, as if merged one after another', async () => {
  const doc1 = [chunk([ent('X', 'X as doc one sees it', 'chunk-1')], [rel('X', 'A', 'X knows A (doc one)', 'chunk-1')])];
  const doc2 = [chunk([ent('X', 'X as doc two sees it', 'chunk-2')], [rel('X', 'A', 'X knows A (doc two)', 'chunk-2')])];
  const gc = defaultMergeConfig(tok, quietLlm);

  const seq = await stores();
  await merge(seq, doc1, gc, 'doc-1');
  await merge(seq, doc2, gc, 'doc-2');

  const par = await stores();
  await Promise.all([merge(par, doc1, gc, 'doc-1'), merge(par, doc2, gc, 'doc-2')]);

  const x = await par.graph.get_node('X');
  assert.deepEqual(String(x?.source_id).split('<SEP>').sort(), ['chunk-1', 'chunk-2']);
  assert.deepEqual((await par.entityChunks.get_by_id('X'))?.chunk_ids, ['chunk-1', 'chunk-2']);
  assert.deepEqual(x, await seq.graph.get_node('X'));
  assert.deepEqual(await par.graph.get_edge('A', 'X'), await seq.graph.get_edge('A', 'X'));
  assert.deepEqual((await par.relationChunks.get_by_id('A<SEP>X'))?.chunk_ids, ['chunk-1', 'chunk-2']);
});

test('S2: vector batches run at most embedding_func_max_async at once', async () => {
  const g = gauge();
  const s = await stores(embedder(4, { delayMs: 5, g }));
  const many = Array.from({ length: 200 }, (_, i) => ent(`E${i}`, `entity number ${i}`, 'chunk-1'));
  await merge(s, [chunk(many)], defaultMergeConfig(tok, quietLlm), 'doc-1');
  assert.equal(g.calls, 20, '200 records in batches of 10');
  assert.ok(g.max <= 8, `embedding calls in flight: ${g.max}`);
});

test('S2: when the entity vectors cannot be written, the relation phase does not run', async () => {
  const s = await stores(embedder(4, { fail: true }));
  const doc = [chunk([ent('X', 'x', 'chunk-1'), ent('A', 'a', 'chunk-1')], [rel('X', 'A', 'X knows A', 'chunk-1')])];
  await assert.rejects(merge(s, doc, defaultMergeConfig(tok, quietLlm), 'doc-1'), /embedding service down/);
  assert.equal(await s.graph.has_edge('X', 'A'), false);
  assert.equal(await s.relationChunks.get_by_id('A<SEP>X'), null);
});

test('S3: summaries call the LLM at most llm_model_max_async at once', async () => {
  const g = gauge();
  const llm: LlmFunc = async (p) => {
    g.enter();
    await sleep(10);
    g.leave();
    return `summary ${p.length}`;
  };
  const s = await stores();
  const ents = Array.from({ length: 12 }, (_, i) => [ent(`E${i}`, `first view of ${i}`, 'chunk-1', 1), ent(`E${i}`, `second view of ${i}`, 'chunk-1', 2)]).flat();
  const gc = defaultMergeConfig(tok, llm, { llm_model_max_async: 2, force_llm_summary_on_merge: 2 });
  await merge(s, [chunk(ents)], gc, 'doc-1');
  assert.equal(g.calls, 12);
  assert.ok(g.max <= 2, `LLM calls in flight: ${g.max}`);
});

test('S3: the limiter serves a lower _priority first', async () => {
  const order: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const f = priorityLimitAsyncFuncCall(1)(async (p: string, _opts: { _priority?: number }) => {
    if (p === 'first') await gate;
    order.push(p);
    return p;
  });
  const first = f('first', {});
  const low = f('normal', { _priority: 10 });
  const high = f('summary', { _priority: 8 });
  const query = f('query', { _priority: 5 });
  release();
  await Promise.all([first, low, high, query]);
  assert.deepEqual(order, ['first', 'query', 'summary', 'normal']);
});

test('S4: one failing summary is reported once, not once per merge still in flight', async () => {
  const llm: LlmFunc = async () => {
    await sleep(5);
    throw new Error('LLM is down');
  };
  const s = await stores();
  const ents = Array.from({ length: 6 }, (_, i) => [ent(`E${i}`, `first view of ${i}`, 'chunk-1', 1), ent(`E${i}`, `second view of ${i}`, 'chunk-1', 2)]).flat();
  const status = { history_messages: [] as string[] };
  await assert.rejects(merge(s, [chunk(ents)], defaultMergeConfig(tok, llm, { force_llm_summary_on_merge: 2 }), 'doc-1', status), /LLM is down/);
  const errors = status.history_messages.filter((m) => m.startsWith('Error processing'));
  assert.deepEqual(errors, ['Error processing entity `E0`: LLM is down']);
});

test('S4: a cancellation stops a map-reduce summary between its LLM calls', async () => {
  const status = { history_messages: [] as string[], cancellation_requested: false };
  let calls = 0;
  const llm: LlmFunc = async () => {
    calls++;
    status.cancellation_requested = true;
    return 'partial summary';
  };
  const s = await stores();
  // tiny context: the twelve descriptions map into several groups, each its own LLM call
  const ents = Array.from({ length: 12 }, (_, i) => ent('X', `description number ${i} of X, long enough to fill a group`, `chunk-${i}`, i));
  const gc = defaultMergeConfig(tok, llm, { summary_context_size: 40, summary_max_tokens: 20, force_llm_summary_on_merge: 3 });
  await assert.rejects(merge(s, [chunk(ents)], gc, 'doc-1', status), PipelineCancelledException);
  assert.equal(calls, 1);
});

test('M1: the KEEP skip log names the kept chunks like upstream (a list repr)', async () => {
  const s = await stores();
  await s.graph.upsert_node('X', { entity_id: 'X', entity_type: 'concept', description: 'x', source_id: 'chunk-1<SEP>chunk-2', file_path: 'a.md' });
  await s.entityChunks.upsert({ X: { chunk_ids: ['chunk-1', 'chunk-2'], count: 2 } });
  const logs: string[] = [];
  const gc = defaultMergeConfig(tok, quietLlm, { source_ids_limit_method: 'KEEP', max_source_ids_per_entity: 2 });
  await mergeNodesThenUpsert('X', [ent('X', 'new view', 'chunk-3')], s.graph, null, gc, {
    entityChunksStorage: s.entityChunks,
    log: (_l, m) => void logs.push(m),
  });
  assert.ok(logs.includes("Skipped `X`: KEEP old chunks ['chunk-1', 'chunk-2']/3"), logs.join('\n'));
});

test('M3: json.dumps leaves a lone surrogate raw, so the summary cache key matches upstream', async () => {
  assert.equal(pyJsonDumps({ Description: 'a\ud800b' }), '{"Description": "a\ud800b"}');
  assert.equal(pyJsonDumps('q"\\\n\x01\x7f'), '"q\\"\\\\\\n\\u0001\x7f"');
  const keys: string[] = [];
  const cache = {
    global_config: { enable_llm_cache_for_entity_extract: true },
    get_by_id: async () => null,
    upsert: async (d: Record<string, Record<string, unknown>>) => void keys.push(...Object.keys(d)),
  };
  await summarizeDescriptions('Entity', 'X', ['a\ud800b', 'plain c'], defaultMergeConfig(tok, async () => 'S'), cache);
  // upstream's _summarize_descriptions on the same input (lightrag-hku 1.4.16)
  assert.deepEqual(keys, ['default:summary:26d209df38ab37f15048464282274661']);
});
