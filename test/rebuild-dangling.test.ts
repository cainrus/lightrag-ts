// B5f-6: a rebuilt relation whose two endpoints are both missing.
// Upstream (operate.py ~1503-1530) creates the missing endpoints in the order of
// the set display `{src, tgt}`; TS creates `src` first, and the pipeline harness
// pins upstream to that order. The branch cannot be reached on the default
// storages: _rebuild_single_relationship returns when get_edge finds nothing
// (operate.py:1366), and NetworkX keeps no edge without its nodes (removing a
// node drops its edges, adding an edge adds its nodes). It exists for graph
// backends that can hold a dangling edge, so it is checked here on such a graph:
// the default NetworkX storage behind a view that hides the two endpoints.
import { strict as assert } from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { LightRAG } from '../src/lightrag.ts';
import { edgeKey, rebuildKnowledgeFromChunks } from '../src/operate.ts';
import { FIXED_NOW, stubEmbedding } from '../parity/pipeline-driver.ts';
import type { BaseGraphStorage } from '../src/base.ts';

const EXTRACTION = [
  'entity<|#|>Zeta Point<|#|>location<|#|>Zeta Point is a harbour.',
  'entity<|#|>Alpha Works<|#|>organization<|#|>Alpha Works builds ships.',
  'relation<|#|>Zeta Point<|#|>Alpha Works<|#|>location<|#|>Alpha Works builds ships at Zeta Point.',
  '<|COMPLETE|>',
].join('\n');

const dirs: string[] = [];
after(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

async function freshRag(): Promise<LightRAG> {
  const dir = mkdtempSync(join(tmpdir(), 'lightrag-dangling-'));
  dirs.push(dir);
  const rag = new LightRAG({
    working_dir: dir,
    llm_model_name: 'stub',
    llm_model_func: async (prompt: string, o: Record<string, unknown> = {}) =>
      String(o.system_prompt ?? '').includes('<|#|>') ? EXTRACTION : `Stub summary of ${prompt.length} chars`,
    embedding_func: stubEmbedding(8),
    entity_extract_max_gleaning: 0,
    now: () => FIXED_NOW,
  } as ConstructorParameters<typeof LightRAG>[0]);
  await rag.initialize_storages();
  await rag.ainsert('Alpha Works builds ships at Zeta Point.', null, false, ['doc-1'], ['harbour.md']);
  return rag;
}

test('on NetworkX an edge never outlives its endpoints, so the branch is unreachable there', async () => {
  const rag = await freshRag();
  const g = rag.chunk_entity_relation_graph;
  assert.ok(await g.get_edge('Zeta Point', 'Alpha Works'));
  await g.remove_nodes(['Zeta Point', 'Alpha Works']);
  assert.equal(await g.get_edge('Zeta Point', 'Alpha Works'), null);
  await rag.finalize_storages();
});

test('a dangling edge: both endpoints are created, src first, as UNKNOWN, in the vdb and entity_chunks in that order', async () => {
  const rag = await freshRag();
  const real = rag.chunk_entity_relation_graph;
  const edge = (await real.get_edge('Zeta Point', 'Alpha Works'))!;
  const chunkIds = String(edge.source_id).split('<SEP>');
  // a backend holding the edge but neither node; the node writes still land in the real graph
  const hidden = new Set(['Zeta Point', 'Alpha Works']);
  const created: string[] = [];
  const kg = new Proxy(real, {
    get(t, p, r) {
      if (p === 'has_node') return async (id: string) => (hidden.has(id) ? false : t.has_node(id));
      if (p === 'upsert_node')
        return async (id: string, data: Record<string, unknown>) => {
          created.push(id);
          hidden.delete(id);
          return t.upsert_node(id, data as never);
        };
      const v = Reflect.get(t, p, r);
      return typeof v === 'function' ? v.bind(t) : v;
    },
  }) as BaseGraphStorage;

  const upserts: string[][] = [];
  const entityChunks = new Proxy(rag.entity_chunks!, {
    get(t, p, r) {
      if (p === 'upsert') return async (d: Record<string, unknown>) => (upserts.push(Object.keys(d)), t.upsert(d as never));
      const v = Reflect.get(t, p, r);
      return typeof v === 'function' ? v.bind(t) : v;
    },
  });
  const vdbUpserts: string[][] = [];
  const entitiesVdb = new Proxy(rag.entities_vdb, {
    get(t, p, r) {
      if (p === 'upsert')
        return async (d: Record<string, { entity_name: string }>) => (vdbUpserts.push(Object.values(d).map((x) => x.entity_name)), t.upsert(d as never));
      const v = Reflect.get(t, p, r);
      return typeof v === 'function' ? v.bind(t) : v;
    },
  });

  const cfg = (rag as unknown as { mergeConfig(): Parameters<typeof rebuildKnowledgeFromChunks>[7] }).mergeConfig();
  await rebuildKnowledgeFromChunks(new Map(), new Map([[edgeKey('Zeta Point', 'Alpha Works'), chunkIds]]), kg, entitiesVdb, rag.relationships_vdb, rag.text_chunks, rag.llm_response_cache, cfg, {
    entityChunksStorage: entityChunks,
    clock: () => FIXED_NOW,
  });

  assert.deepEqual(created, ['Zeta Point', 'Alpha Works'], 'src first, then tgt');
  assert.deepEqual(upserts, [['Zeta Point'], ['Alpha Works']]);
  assert.deepEqual(vdbUpserts, [['Zeta Point'], ['Alpha Works']]);
  for (const id of created) {
    const n = (await real.get_node(id))!;
    assert.equal(n.entity_type, 'UNKNOWN');
    assert.equal(n.source_id, chunkIds.join('<SEP>'));
    assert.equal(n.created_at, FIXED_NOW);
  }
  await rag.finalize_storages();
});
