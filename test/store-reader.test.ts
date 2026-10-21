// Readers against a store written by upstream code (networkx, nano-vectordb,
// lightrag write_json). Small read buffers force every value across chunk edges.
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { iterateJsonObject } from '../src/internal/json_stream.ts';
import { readJsonKv } from '../src/kg/json_kv_impl.ts';
import { decodeVectorField, readNanoVectorDb } from '../src/kg/nano_vector_db_impl.ts';
import { edgeKey, readGraphml } from '../src/kg/networkx_impl.ts';
import { storeStats } from '../src/store.ts';

const store = fileURLToPath(new URL('./fixtures/synthetic-store/', import.meta.url));

test('graphml: typed attrs, xml escapes, unicode ids', () => {
  const g = readGraphml(join(store, 'graph_chunk_entity_relation.graphml'));
  assert.equal(g.directed, false);
  assert.equal(g.nodes.size, 3);
  assert.equal(g.edges.size, 2);
  assert.equal(g.nodes.get('Alice')?.description, 'Founder <of> Acme & co');
  assert.equal(g.nodes.get('Acme Robotics')?.description, 'Builds\nrobots');
  assert.equal(g.nodes.get('Тарту')?.created_at, 1779035542);
  const e = g.edges.get(edgeKey('Alice', 'Тарту'));
  assert.equal(e?.attrs.weight, 1.0);
  assert.equal(g.edges.get(edgeKey('Acme Robotics', 'Alice'))?.attrs.weight, 2.5);
});

test('kv: upstream write_json output, surrogate already stripped by upstream', () => {
  const chunks = readJsonKv<{ content: string }>(join(store, 'kv_store_text_chunks.json'));
  assert.equal(chunks.size, 3);
  assert.ok(!/[\uD800-\uDFFF](?![\uDC00-\uDFFF])/.test(chunks.get('chunk-synthetic-3')!.content));
  assert.equal(readJsonKv(join(store, 'kv_store_llm_response_cache.json')).size, 0);
});

for (const chunkSize of [7, 64, 1 << 22]) {
  test(`kv streaming with ${chunkSize}-byte buffer equals JSON.parse`, () => {
    const file = join(store, 'kv_store_text_chunks.json');
    const streamed = Object.fromEntries(iterateJsonObject(file, chunkSize));
    assert.deepEqual(streamed, JSON.parse(readFileSync(file, 'utf8')));
  });

  test(`vdb with ${chunkSize}-byte buffer: records, matrix rows, f16 vector field`, () => {
    const v = readNanoVectorDb(join(store, 'vdb_entities.json'), { matrix: 'load', keepVectorField: true, chunkSize });
    assert.equal(v.embeddingDim, 8);
    assert.deepEqual(v.data.map((d) => d.__id__), ['ent-a', 'ent-b', 'ent-c']);
    assert.equal(v.matrix!.length, 3 * 8);
    // matrix rows are the L2-normalized float32 vectors; the f16 field is the raw vector
    for (let i = 0; i < 3; i++) {
      const raw = decodeVectorField(v.data[i]!.vector!);
      const norm = Math.hypot(...raw);
      for (let j = 0; j < 8; j++) assert.ok(Math.abs(raw[j]! / norm - v.matrix![i * 8 + j]!) < 2e-3);
    }
  });
}

test('json stream: escapes, nesting, empty containers, BOM, trailing garbage', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lrts-'));
  const obj = { a: 'x\\"}{]["', 'b"': [{}, [], { c: [1, -2.5e3, true, null] }], '': '' };
  const f = join(dir, 'x.json');
  writeFileSync(f, '﻿' + JSON.stringify(obj, null, 2));
  for (const n of [1, 2, 3, 5, 4096]) assert.deepEqual(Object.fromEntries(iterateJsonObject(f, n)), obj);
  writeFileSync(f, '{}');
  assert.deepEqual([...iterateJsonObject(f)], []);
  writeFileSync(f, '{"a": 1} x');
  assert.throws(() => [...iterateJsonObject(f)], /trailing data/);
  writeFileSync(f, '{"a": [1, 2}');
  assert.throws(() => [...iterateJsonObject(f)]);
});

test('storeStats over the synthetic store', () => {
  const s = storeStats(store);
  assert.deepEqual(s.graph, { nodes: 3, edges: 2, edgeElements: 2 });
  assert.equal(s.kv.text_chunks, 3);
  assert.equal(s.kv.full_entities, null);
  assert.deepEqual(s.docStatus, { processed: 1 });
  assert.deepEqual(s.vdb.relationships, { records: 1, embeddingDim: 8, matrixRows: 1 });
});
