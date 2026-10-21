// B1 storages on synthetic data. Expected values marked "networkx"/"numpy"/
// "upstream" were produced by running that code (see parity/storages.ts for the
// full cross-check against a real store).
import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { UndirectedGraph } from 'graphology';
import type { EmbeddingFunc, GlobalConfig } from '../src/base.ts';
import { makeDocProcessingStatus } from '../src/base.ts';
import { normalizeRowsF32 } from '../src/internal/numpy.ts';
import { atomicWrite, pyFloatRepr, writeJsonPy } from '../src/internal/py_json.ts';
import { JsonDocStatusStorage } from '../src/kg/json_doc_status_impl.ts';
import { JsonKVStorage, migrateLegacyCacheStructure, readJsonKv } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDB, NanoVectorDBStorage, decodeVectorField, encodeVectorField, floatToHalf } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage, NxGraph, loadNxGraph, writeNxGraphml } from '../src/kg/networkx_impl.ts';
import { makeRelationVdbIds } from '../src/utils.ts';

const fx = (p: string) => fileURLToPath(new URL(`./fixtures/${p}`, import.meta.url));
const store = fx('synthetic-store');
const tmp = () => mkdtempSync(join(tmpdir(), 'lightrag-ts-test-'));
const cfg = (dir: string, extra: Partial<GlobalConfig> = {}): GlobalConfig => ({
  working_dir: dir,
  embedding_batch_num: 2,
  vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 },
  ...extra,
});
function copyStore(): string {
  const d = tmp();
  for (const f of readdirSync(store)) copyFileSync(join(store, f), join(d, f));
  return d;
}

// ------------------------------------------------------------- python json

test('pyFloatRepr matches Python repr()', () => {
  // upstream: [[x, repr(x)] for x in xs] in CPython 3.12
  const cases: [number, string][] = [
    [1.0, '1.0'], [0.1, '0.1'], [-2.5, '-2.5'], [1e-5, '1e-05'], [0.0001, '0.0001'], [1e16, '1e+16'],
    [1e15, '1000000000000000.0'], [123456789012345.6, '123456789012345.6'], [1.5e-7, '1.5e-07'],
    [2 ** 53, '9007199254740992.0'], [5e-324, '5e-324'], [1.7976931348623157e308, '1.7976931348623157e+308'],
    [0.30000000000000004, '0.30000000000000004'], [-0, '-0.0'], [1e22, '1e+22'], [12345.678, '12345.678'],
  ];
  for (const [x, want] of cases) assert.equal(pyFloatRepr(x), want, String(x));
});

test('writeJsonPy reproduces upstream write_json byte for byte', () => {
  const d = tmp();
  for (const f of readdirSync(store).filter((f) => f.startsWith('kv_store_'))) {
    writeJsonPy(readJsonKv(join(store, f)), join(d, f));
    assert.ok(readFileSync(join(d, f)).equals(readFileSync(join(store, f))), f);
  }
});

test('writeJsonPy: a lone surrogate triggers the sanitizing rewrite, which also drops U+FFFE/U+FFFF', () => {
  const f = join(tmp(), 'kv.json');
  assert.equal(writeJsonPy({ a: 'ok￿' }, f).sanitized, false);
  assert.equal(readFileSync(f, 'utf8'), '{\n  "a": "ok￿"\n}');
  assert.equal(writeJsonPy({ a: 'x\uD800y', b: 'ok￿', c: ['😀'] }, f).sanitized, true);
  assert.equal(readFileSync(f, 'utf8'), '{\n  "a": "xy",\n  "b": "ok",\n  "c": [\n    "😀"\n  ]\n}');
});

test('atomicWrite: a failing writer leaves the old file and no temp file', () => {
  const d = tmp();
  const f = join(d, 'x.json');
  writeFileSync(f, 'old');
  assert.throws(() =>
    atomicWrite(f, (w) => {
      w.write('half');
      throw new Error('boom');
    }),
  );
  assert.equal(readFileSync(f, 'utf8'), 'old');
  assert.deepEqual(readdirSync(d), ['x.json']);
});

// ------------------------------------------------------------- kv

test('JsonKVStorage: upsert timestamps, _id, llm_cache_list, get_by_id defaults, filter_keys, delete', async () => {
  const d = copyStore();
  const s = new JsonKVStorage({ namespace: 'text_chunks', global_config: cfg(d), clock: () => 1700000000 });
  await s.initialize();
  const [first] = [...s.keys()] as [string];
  const before = (await s.get_by_id(first))!;
  const incoming = { [first]: { content: 'changed' }, 'chunk-new': { content: 'new', llm_cache_list: ['k'] } } as Record<string, Record<string, unknown>>;
  await s.upsert(incoming);
  // upstream mutates the caller's dicts
  assert.deepEqual(incoming[first], { content: 'changed', llm_cache_list: [], update_time: 1700000000, _id: first });
  assert.deepEqual(incoming['chunk-new'], { content: 'new', llm_cache_list: ['k'], create_time: 1700000000, update_time: 1700000000, _id: 'chunk-new' });
  // replacing an existing record drops its create_time; get_by_id fills 0 back in
  assert.equal((await s.get_by_id(first))!.create_time, 0);
  assert.notEqual(before.content, 'changed');
  assert.deepEqual(await s.filter_keys(new Set([first, 'nope'])), new Set(['nope']));
  assert.deepEqual(await s.get_by_ids(['nope', 'chunk-new']), [null, { ...incoming['chunk-new'] }]);
  await s.delete(['chunk-new', 'nope']);
  assert.equal(await s.get_by_id('chunk-new'), null);
  assert.equal(await s.index_done_callback(), true);
  assert.equal(await s.index_done_callback(), false, 'nothing to write twice');
  assert.equal(readJsonKv(join(d, 'kv_store_text_chunks.json')).get(first)!.content, 'changed');
});

test('JsonKVStorage: an empty record is falsy, as in Python', async () => {
  const d = tmp();
  writeFileSync(join(d, 'kv_store_full_docs.json'), '{"e": {}, "x": {"a": 1}}');
  const s = new JsonKVStorage({ namespace: 'full_docs', global_config: cfg(d) });
  await s.initialize();
  assert.deepEqual(await s.get_by_id('e'), {});
  assert.deepEqual(await s.get_by_ids(['e', 'x']), [null, { a: 1, create_time: 0, update_time: 0, _id: 'x' }]);
});

test('legacy llm cache layout is flattened to mode:type:hash', () => {
  const legacy = new Map<string, Record<string, unknown>>([
    ['default', { h1: { return: 'r', cache_type: 'extract' }, h2: { return: 'q' } }],
    ['global', {}],
    ['plain', { notcache: 1 }],
  ]);
  const { data, count } = migrateLegacyCacheStructure(legacy);
  assert.equal(count, 2);
  // Python all() over an empty dict is True: an empty mode disappears
  assert.deepEqual([...data.keys()], ['default:extract:h1', 'default:extract:h2', 'plain']);
  const flat = new Map([['default:extract:h1', { return: 'r' }]]);
  assert.equal(migrateLegacyCacheStructure(flat).count, 0);
});

// ------------------------------------------------------------- doc status

test('JsonDocStatusStorage: counts, malformed docs, pagination stability, multimodal rule', async () => {
  const d = tmp();
  const doc = (status: string, updated: string, extra = {}) => ({
    status, content_summary: 's', content_length: 1, file_path: '', created_at: updated, updated_at: updated, ...extra,
  });
  const s = new JsonDocStatusStorage({ namespace: 'doc_status', global_config: cfg(d) });
  await s.initialize();
  await s.upsert({
    a: doc('processed', '2026-01-02'),
    b: doc('processed', '2026-01-02', { multimodal_processed: false }),
    c: doc('failed', '2026-01-01', { track_id: 't1' }),
    bad: { status: 'pending' },
  });
  // upsert persists immediately and adds chunks_list
  assert.deepEqual(readJsonKv(join(d, 'kv_store_doc_status.json')).get('a')!.chunks_list, []);
  assert.deepEqual(await s.get_all_status_counts(), { pending: 1, processing: 0, preprocessed: 0, processed: 2, failed: 1, all: 4 });
  const byStatus = await s.get_docs_by_statuses(['processed', 'pending']);
  assert.deepEqual([...byStatus.keys()], ['a', 'b'], 'malformed "bad" skipped (TypeError caught upstream)');
  assert.equal(byStatus.get('b')!.status, 'preprocessed');
  assert.equal(byStatus.get('a')!.file_path, 'no-file-path');
  await assert.rejects(s.get_docs_paginated(), /missing required/);
  await s.delete(['bad']);
  const [page, total] = await s.get_docs_paginated(null, 1, 10, 'updated_at', 'desc');
  assert.equal(total, 3);
  assert.deepEqual(page.map(([k]) => k), ['a', 'b', 'c'], 'desc sort keeps insertion order for equal keys');
  assert.deepEqual([...(await s.get_docs_by_track_id('t1')).keys()], ['c']);
  await s.upsert({ z: doc('weird', 'x') });
  await assert.rejects(s.get_status_counts(), /KeyError: 'weird'/);
  assert.throws(() => makeDocProcessingStatus({ ...doc('pending', 'x'), extra_field: 1 }), /unexpected keyword/);
});

// ------------------------------------------------------------- graph

test('NxGraph keeps networkx neighbour order where graphology does not', () => {
  const g = new NxGraph();
  for (const n of ['b', '10', 'a', '2']) g.addNode(n);
  for (const [u, v] of [['a', 'b'], ['10', 'b'], ['2', 'b'], ['a', '10'], ['2', '2']] as const) g.addEdge(u, v, { weight: 1 });
  // networkx: {"b": ["a", "10", "2"], "10": ["b", "a"], "a": ["b", "10"], "2": ["b", "2"]}
  assert.deepEqual([...g.neighbors('b')], ['a', '10', '2']);
  assert.deepEqual([...g.edges()].map(([u, v]) => [u, v]), [['b', 'a'], ['b', '10'], ['b', '2'], ['10', 'a'], ['2', '2']]);
  assert.equal(g.degree('2'), 3, 'self-loop counts twice');
  const gg = g.toGraphology();
  assert.ok(gg instanceof UndirectedGraph);
  assert.deepEqual(gg.neighbors('b'), ['2', '10', 'a'], 'graphology puts integer-like keys first');
});

test('loadNxGraph replays read_graphml: MultiGraph -> Graph rebuild order, "" for empty data', () => {
  const g = loadNxGraph(fx('nx/reorder.graphml'));
  // networkx: {"a": ["b"], "b": ["a", "c"], "c": ["b"]} although the file lists b-c first
  assert.deepEqual([...g.neighbors('b')], ['a', 'c']);
  assert.deepEqual(Object.fromEntries(g.nodes), { a: { n: 1 }, b: {}, c: { n: '' } });
  const out = join(tmp(), 'g.graphml');
  writeNxGraphml(g, out);
  assert.equal(readFileSync(out, 'utf8'), readFileSync(fx('nx/reorder.networkx-rewrite.graphml'), 'utf8'));
});

test('writeNxGraphml reproduces networkx write_graphml on the synthetic store', () => {
  const out = join(tmp(), 'g.graphml');
  writeNxGraphml(loadNxGraph(join(store, 'graph_chunk_entity_relation.graphml')), out);
  assert.ok(readFileSync(out).equals(readFileSync(join(store, 'graph_chunk_entity_relation.graphml'))));
});

test('NetworkXStorage: BaseGraphStorage semantics', async () => {
  const d = tmp();
  const s = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(d, { max_graph_nodes: 3 }) });
  await s.upsert_nodes_batch([['Hub', { entity_type: 'x' }], ['A', {}], ['B', {}], ['C', {}], ['Lonely', {}]]);
  await s.upsert_edges_batch([['Hub', 'A', { weight: 1 }], ['Hub', 'B', { weight: 1 }], ['B', 'C', { weight: 2 }]]);
  await s.upsert_edge('A', 'Hub', { description: 'merged' });
  assert.deepEqual(await s.get_edge('Hub', 'A'), { weight: 1, description: 'merged' });
  assert.equal(await s.get_edge('A', 'Nope'), null);
  assert.deepEqual(await s.get_node_edges('Hub'), [['Hub', 'A'], ['Hub', 'B']]);
  assert.equal(await s.get_node_edges('Nope'), null);
  assert.equal(await s.edge_degree('Hub', 'Nope'), 2);
  assert.deepEqual(await s.get_popular_labels(2), ['Hub', 'B']);
  assert.deepEqual(await s.search_labels('b'), ['B', 'Hub'], 'exact match first, then substring');
  assert.deepEqual(await s.has_nodes_batch(['A', 'Nope']), new Set(['A']));
  const kg = await s.get_knowledge_graph('Hub', 3, 10);
  assert.deepEqual(kg.nodes.map((n) => n.id), ['Hub', 'A', 'B'], 'capped at max_graph_nodes = 3');
  assert.equal(kg.is_truncated, true);
  assert.deepEqual(kg.edges.map((e) => e.id), ['A-Hub', 'B-Hub']);
  assert.deepEqual((await s.get_knowledge_graph('*')).nodes.map((n) => n.id), ['Hub', 'A', 'B']);
  assert.deepEqual(await s.get_knowledge_graph('Nope'), { nodes: [], edges: [], is_truncated: false });
  await s.remove_edges([['B', 'C'], ['C', 'Nope']]);
  await s.delete_node('Hub');
  await s.remove_nodes(['Lonely', 'Nope']);
  assert.deepEqual(await s.get_all_labels(), ['A', 'B', 'C']);
  assert.equal(await s.index_done_callback(), true);
  const back = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(d) });
  assert.deepEqual(await back.get_all_nodes(), [{ id: 'A' }, { id: 'B' }, { id: 'C' }]);
});

// ------------------------------------------------------------- vectors

const DIM = 8;
const emb: EmbeddingFunc = Object.assign(
  async (texts: string[]) => texts.map((t) => Array.from({ length: DIM }, (_, i) => ((t.charCodeAt(i % t.length) * (i + 1)) % 17) - 8 + 0.5)),
  { embedding_dim: DIM },
);
const vdb = (d: string, ns = 'entities', meta = ['entity_name', 'content']) =>
  new NanoVectorDBStorage({ namespace: ns, global_config: cfg(d), embedding_func: emb, meta_fields: meta, clock: () => 1700000000 });

test('NanoVectorDB save reproduces upstream load + save byte for byte (matrix renormalized as numpy does)', () => {
  const d = copyStore();
  for (const ns of ['entities', 'relationships', 'chunks']) {
    new NanoVectorDB(DIM, join(d, `vdb_${ns}.json`)).save();
    assert.ok(readFileSync(join(d, `vdb_${ns}.json`)).equals(readFileSync(fx(`upstream-rewrite/vdb_${ns}.json`))), ns);
  }
});

test('normalizeRowsF32 is numpy normalize bit for bit (pairwise float32 sum)', () => {
  // numpy: (m / np.linalg.norm(m, axis=-1, keepdims=True)).tobytes() for this matrix
  const m = new Float32Array([3, 4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1e-3]);
  normalizeRowsF32(m, 8);
  assert.deepEqual(Array.from(m.slice(0, 2)), [0.6000000238418579, 0.800000011920929]);
  const norm = Math.hypot(...m.slice(8));
  assert.ok(Math.abs(norm - 1) < 1e-6);
});

test('floatToHalf rounds like a hardware float16 (Float16Array) on edge and random values', () => {
  const F16 = (globalThis as unknown as { Float16Array?: Float32ArrayConstructor }).Float16Array;
  if (!F16) return; // Node 22: nothing to compare against
  const probe = new F16(1);
  const bits = new Uint16Array(probe.buffer);
  const vals = [0, -0, 1, -1, 65504, 65519.99, 65520, 1e9, 2 ** -24, 2 ** -25, 1.5 * 2 ** -25, 2 ** -14, 2 ** -14 - 2 ** -25, 1 + 2 ** -11, 1 + 3 * 2 ** -11, NaN];
  let s = 42;
  for (let i = 0; i < 20000; i++) {
    s = (s * 1103515245 + 12345) % 2147483648;
    vals.push(((s / 2147483648 - 0.5) * 2) ** 3 * 10 ** ((i % 12) - 6));
  }
  for (const v of vals) {
    probe[0] = v;
    assert.equal(floatToHalf(v), bits[0], String(v));
  }
});

test('float16 vector field round-trips like numpy astype(float16)', () => {
  const v = [0.1, -2.5, 65504, 1e-8, 0.33333];
  assert.deepEqual(Array.from(decodeVectorField(encodeVectorField(v))), [0.0999755859375, -2.5, 65504, 0, 0.333251953125]);
});

test('NanoVectorDBStorage: upsert keeps meta fields, updates in place, queries with threshold', async () => {
  const d = tmp();
  const s = vdb(d);
  const r1 = await s.upsert({
    'ent-a': { content: 'alpha', entity_name: 'Alpha', ignored: 1 },
    'ent-b': { content: 'bravo', entity_name: 'Bravo' },
    'ent-c': { content: 'charlie', entity_name: 'Charlie' },
  });
  assert.deepEqual(r1, { update: [], insert: ['ent-a', 'ent-b', 'ent-c'] });
  assert.deepEqual(await s.upsert({ 'ent-b': { content: 'bravo two', entity_name: 'Bravo' } }), { update: ['ent-b'], insert: [] });
  assert.deepEqual(s.db.data.map((x) => x.__id__), ['ent-a', 'ent-b', 'ent-c']);
  const got = await s.get_by_id('ent-a');
  assert.deepEqual(got, { __id__: 'ent-a', __created_at__: 1700000000, entity_name: 'Alpha', content: 'alpha', id: 'ent-a', created_at: 1700000000 });
  const q = (await emb(['alpha']))[0]!;
  const res = await s.query('', 2, Array.from(q));
  assert.equal(res[0]!.id, 'ent-a');
  assert.ok(Math.abs(res[0]!.distance - 1) < 1e-6);
  assert.ok(res.length <= 2);
  // threshold: results stop at the first score below it
  const strict = new NanoVectorDBStorage({
    namespace: 'entities', global_config: { ...cfg(d), vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.999 } }, embedding_func: emb,
  });
  await s.index_done_callback();
  const reloaded = await new NanoVectorDBStorage({
    namespace: 'entities', global_config: { ...cfg(d), vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.999 } }, embedding_func: emb,
  }).query('alpha', 10);
  assert.deepEqual(reloaded.map((r) => r.id), ['ent-a']);
  assert.equal(strict.db.length, 0, 'constructed before the save: empty');
  // get() returns storage order; get_by_ids request order with nulls
  assert.deepEqual((await s.get_by_ids(['ent-c', 'nope', 'ent-a'])).map((r) => r?.id ?? null), ['ent-c', null, 'ent-a']);
  assert.deepEqual(Object.keys(await s.get_vectors_by_ids(['ent-c', 'ent-a'])), ['ent-a', 'ent-c']);
});

test('NanoVectorDBStorage: delete, delete_entity, delete_entity_relation', async () => {
  const d = tmp();
  const s = vdb(d, 'relationships', ['src_id', 'tgt_id', 'content']);
  await s.upsert({
    r1: { src_id: 'A', tgt_id: 'B', content: 'ab' },
    r2: { src_id: 'B', tgt_id: 'C', content: 'bc' },
    r3: { src_id: 'C', tgt_id: 'D', content: 'cd' },
  });
  await s.delete_entity_relation('B');
  assert.deepEqual(s.db.data.map((x) => x.__id__), ['r3']);
  assert.equal(s.db.rowIndex('r3'), 0);
  // a record without src_id makes upstream's comprehension raise: nothing is deleted
  const e = vdb(d, 'entities', ['content']);
  await e.upsert({ 'ent-x': { content: 'x' } });
  await e.delete_entity_relation('x');
  assert.equal(e.db.length, 1);
  await e.delete_entity('nobody');
  await e.delete(['ent-x', 'missing']);
  assert.equal(e.db.length, 0);
});

test('edges without a relationship vector (workflow-3361): lookups return null, nothing throws', async () => {
  // synthetic store: 2 graph edges, the single relationship row matches neither
  const d = copyStore();
  const g = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(d) });
  const rel = vdb(d, 'relationships', ['src_id', 'tgt_id', 'content']);
  const edges = [...g.graph.edges()];
  assert.equal(edges.length, 2);
  for (const [u, v] of edges) {
    const ids = makeRelationVdbIds(u, v);
    assert.deepEqual(await rel.get_by_ids(ids), ids.map(() => null));
    assert.deepEqual(await rel.get_vectors_by_ids(ids), {});
    await rel.delete(ids);
  }
  assert.equal(rel.db.length, 1, 'deleting absent ids is a no-op');
  assert.equal(await g.has_edge(edges[0]![0], edges[0]![1]), true);
});
