// Query path (B3). Fixtures under test/fixtures/query/ come from upstream code
// on synthetic data (tools/upstream/query_parity.py synthetic + run): a store,
// 136 probes over the five modes, upstream's record of each run, the query
// vectors it used, and Python-semantics cases recorded from the interpreter.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { describeProbe, diffRecords, openQueryStore, type Probe, readRecords, runProbe } from '../parity/query-runner.ts';
import {
  computeArgsHashPy,
  jsonRepairLoads,
  makeQueryParam,
  performKgSearch,
  pickByWeightedPolling,
  pyFloatRepr,
  pyJsonDumpsAny,
  pyRound,
  type QueryGraphStorage,
  type QueryTextChunksStorage,
  type QueryVectorStorage,
  queryCacheKeys,
} from '../src/query.ts';

const dir = fileURLToPath(new URL('./fixtures/query/', import.meta.url));
const read = <T>(f: string): T => JSON.parse(readFileSync(dir + f, 'utf8')) as T;

// ------------------------------------------------- upstream parity, synthetic

test('upstream parity: every probe matches upstream byte for byte', async (t) => {
  const { probes } = read<{ probes: Probe[] }>('probes.json');
  const upstream = new Map(readRecords(dir + 'upstream.jsonl').map((r) => [r.id, r]));
  assert.equal(upstream.size, probes.length);
  const store = await openQueryStore(dir + 'store', '', read<Record<string, number[]>>('emb.json'));
  // sequential on purpose: probes share the LLM cache, as they did upstream
  for (const p of probes) {
    const rec = await runProbe(store, p);
    await t.test(p.id, () => {
      const d = diffRecords(rec, upstream.get(p.id)!);
      assert.deepEqual(d, [], `${describeProbe(p)}\n${d.join('\n')}`);
    });
  }
  assert.deepEqual([...store.missingVectors], []);
});

test('queryCacheKeys names the keys upstream wrote for a kg and a naive query', () => {
  const upstream = new Map(readRecords(dir + 'upstream.jsonl').map((r) => [r.id, r]));
  const { probes } = read<{ probes: Probe[] }>('probes.json');
  const byId = new Map(probes.map((p) => [p.id, p]));
  for (const id of ['q2-mix-full', 'q2-local-full', 'q2-naive-full']) {
    const p = byId.get(id)!;
    const kw = JSON.parse(p.kw_response) as { high_level_keywords: string[]; low_level_keywords: string[] };
    const keys = queryCacheKeys(p.query, makeQueryParam({ ...p.param, mode: p.mode }), { hlKeywords: kw.high_level_keywords, llKeywords: kw.low_level_keywords });
    const written = upstream.get(id)!.cache_writes.flatMap((w) => Object.keys(w));
    assert.deepEqual(written, p.mode === 'naive' ? [keys.query] : [keys.keywords, keys.query], id);
  }
});

// ---------------------------------------------------- Python semantics cases

type Cases = {
  float_repr: [number, string][];
  round: [number, number][];
  args_hash: [unknown[], string][];
  json_dumps: [unknown, string, string | null][];
  json_repair: [string, unknown][];
  weighted_polling: [string[][], number, number, string[]][];
};
const cases = read<Cases>('py-cases.json');

test('pyFloatRepr matches repr(float)', () => {
  for (const [x, r] of cases.float_repr) assert.equal(pyFloatRepr(x), r, String(x));
});

test('pyRound matches round() (half to even)', () => {
  for (const [x, r] of cases.round) assert.equal(pyRound(x), r, String(x));
});

test('computeArgsHashPy matches compute_args_hash (str of None, bool, float; lone surrogate)', () => {
  for (const [args, h] of cases.args_hash) assert.equal(computeArgsHashPy(...args), h, JSON.stringify(args));
});

test('pyJsonDumpsAny matches json.dumps with and without ensure_ascii', () => {
  for (const [v, ascii, raw] of cases.json_dumps) {
    assert.equal(pyJsonDumpsAny(v, true), ascii);
    if (raw !== null) assert.equal(pyJsonDumpsAny(v, false), raw);
  }
});

test('jsonRepairLoads agrees with json_repair.loads on the shapes LLMs return', () => {
  for (const [s, want] of cases.json_repair) assert.deepEqual(jsonRepairLoads(s), want, JSON.stringify(s));
});

test('pickByWeightedPolling matches pick_by_weighted_polling', () => {
  for (const [items, mx, mn, want] of cases.weighted_polling) {
    assert.deepEqual(pickByWeightedPolling(items.map((c) => ({ sorted_chunks: c })), mx, mn), want, JSON.stringify([items, mx, mn]));
  }
});

// ------------------------------- upstream tests/test_batch_embeddings.py, ported
// (d11: B2's pytest adapter is not in main yet; these five tests mock storages
// with MagicMock, so they are ported as-is rather than bridged)

type Call = { texts: string[]; opts: unknown };
function batchFixture(fail = false) {
  const dim = 1536;
  const calls: Call[] = [];
  const embed = async (texts: string[], opts?: unknown) => {
    calls.push({ texts, opts });
    if (fail) throw new Error('API error');
    return texts.map((_, i) => new Float32Array(dim).fill(i + 1));
  };
  const vdb = () => {
    const queries: { query: string; top_k: number; query_embedding: ArrayLike<number> | null | undefined }[] = [];
    const v: QueryVectorStorage & { queries: typeof queries } = {
      cosine_better_than_threshold: 0.2,
      queries,
      query: async (query, top_k, query_embedding) => {
        queries.push({ query, top_k, query_embedding });
        return [];
      },
      get_vectors_by_ids: async () => ({}),
    };
    return v;
  };
  const textChunks: QueryTextChunksStorage = { global_config: { kg_chunk_pick_method: 'VECTOR' }, embedding_func: embed, get_by_ids: async () => [] };
  const graph = {} as QueryGraphStorage;
  return { calls, textChunks, graph, entities: vdb(), relationships: vdb() };
}

test('test_hybrid_mode_batches_embeddings', async () => {
  const f = batchFixture();
  await performKgSearch('test query', 'entity1, entity2', 'theme1, theme2', f.graph, f.entities, f.relationships, f.textChunks, makeQueryParam({ mode: 'hybrid', top_k: 5 }));
  assert.equal(f.calls.length, 1);
  assert.deepEqual(f.calls[0]!.texts, ['test query', 'entity1, entity2', 'theme1, theme2']);
});

test('test_hybrid_mode_passes_embeddings_to_vdbs', async () => {
  const f = batchFixture();
  await performKgSearch('test query', 'entity keywords', 'theme keywords', f.graph, f.entities, f.relationships, f.textChunks, makeQueryParam({ mode: 'hybrid', top_k: 5 }));
  const ll = f.entities.queries[0]?.query_embedding;
  assert.ok(ll, 'll_embedding was not passed to entities_vdb.query');
  assert.ok(Array.from(ll).every((x) => x === 2));
  const hl = f.relationships.queries[0]?.query_embedding;
  assert.ok(hl, 'hl_embedding was not passed to relationships_vdb.query');
  assert.ok(Array.from(hl).every((x) => x === 3));
});

test('test_local_mode_skips_hl_keywords', async () => {
  const f = batchFixture();
  await performKgSearch('test query', 'entity keywords', 'theme keywords', f.graph, f.entities, f.relationships, f.textChunks, makeQueryParam({ mode: 'local', top_k: 5 }));
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0]!.texts.length, 2);
  assert.ok(!f.calls[0]!.texts.includes('theme keywords'));
});

test('test_global_mode_skips_ll_keywords', async () => {
  const f = batchFixture();
  await performKgSearch('test query', 'entity keywords', 'theme keywords', f.graph, f.entities, f.relationships, f.textChunks, makeQueryParam({ mode: 'global', top_k: 5 }));
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0]!.texts.length, 2);
  assert.ok(!f.calls[0]!.texts.includes('entity keywords'));
});

test('test_embedding_failure_falls_back_gracefully', async () => {
  const f = batchFixture(true);
  await performKgSearch('test query', 'entity keywords', 'theme keywords', f.graph, f.entities, f.relationships, f.textChunks, makeQueryParam({ mode: 'hybrid', top_k: 5 }));
  assert.equal(f.entities.queries.length, 1);
  assert.equal(f.entities.queries[0]!.query_embedding, null);
  assert.equal(f.relationships.queries.length, 1);
  assert.equal(f.relationships.queries[0]!.query_embedding, null);
});
