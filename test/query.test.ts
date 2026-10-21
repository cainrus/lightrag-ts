// Query path (B3). Fixtures under test/fixtures/query/ come from upstream code
// on synthetic data (tools/upstream/query_parity.py synthetic + run): a store,
// 136 probes over the five modes, upstream's record of each run, the query
// vectors it used, and Python-semantics cases recorded from the interpreter.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { computeArgsHash, pyJsonDumps, pyReprFloat, TiktokenTokenizer } from '../src/utils.ts';
import { describeProbe, diffRecords, openQueryStore, type Probe, readRecords, runProbe } from '../parity/query-runner.ts';
import {
  jsonRepairLoads,
  kgQueryArgsHash,
  lightragQueryConfig,
  makeQueryParam,
  pickByWeightedPolling,
  pyRound,
  queryCacheKeys,
  queryTextChunks,
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

// ------------------------------------------------------------ review fixes

test('makeQueryParam: a field passed as undefined keeps its default (no "None" in the cache key)', () => {
  const dflt = makeQueryParam({ mode: 'mix' });
  const withUndef = makeQueryParam({ mode: 'mix', top_k: undefined, chunk_top_k: undefined, response_type: undefined });
  assert.equal(withUndef.top_k, dflt.top_k);
  assert.equal(withUndef.chunk_top_k, dflt.chunk_top_k);
  assert.equal(withUndef.response_type, dflt.response_type);
  assert.equal(kgQueryArgsHash(withUndef, 'q', 'h', 'l'), kgQueryArgsHash(dflt, 'q', 'h', 'l'));
  // an explicit null is Python's None and stays
  assert.equal(makeQueryParam({ chunk_top_k: null }).chunk_top_k, null);
});

test('queryCacheKeys: empty keywords follow kg_query (query as ll under 50 chars, no answer key from 50)', () => {
  const upstream = new Map(readRecords(dir + 'upstream.jsonl').map((r) => [r.id, r]));
  const { probes } = read<{ probes: Probe[] }>('probes.json');
  const byId = new Map(probes.map((p) => [p.id, p]));
  const short = byId.get('kw-mix-empty')!;
  const keys = queryCacheKeys(short.query, makeQueryParam({ ...short.param, mode: short.mode }), { hlKeywords: [], llKeywords: [] });
  const written = upstream.get('kw-mix-empty')!.cache_writes.flatMap((w) => Object.keys(w));
  assert.deepEqual(written, [keys.query]); // upstream wrote no keywords entry: the extraction came back empty
  const long = byId.get('kw-mix-empty-long-query')!;
  assert.equal(queryCacheKeys(long.query, makeQueryParam({ ...long.param, mode: long.mode })).query, null);
  assert.deepEqual(upstream.get('kw-mix-empty-long-query')!.cache_writes, []);
});

test('jsonRepairLoads warns when non-blank text holds no JSON', () => {
  const seen: string[] = [];
  const log = (level: string, msg: string) => seen.push(`${level}: ${msg}`);
  assert.equal(jsonRepairLoads('no keywords here', log), '');
  assert.equal(jsonRepairLoads('   ', log), '');
  assert.equal(seen.length, 1);
  assert.match(seen[0]!, /^warning: /);
});

test('base.ts reads .env from the working directory without overriding the environment (load_dotenv)', () => {
  const tmp = mkdtempSync(join(tmpdir(), 'lightrag-ts-env-'));
  writeFileSync(join(tmp, '.env'), 'TOP_K=7\nCHUNK_TOP_K=3\n');
  const base = new URL('../src/base.ts', import.meta.url).href;
  const script = `import { makeQueryParam } from ${JSON.stringify(base)}; const q = makeQueryParam(); console.log(q.top_k, q.chunk_top_k);`;
  const run = (env: NodeJS.ProcessEnv) =>
    execFileSync(process.execPath, ['--input-type=module', '-e', script], { cwd: tmp, env, encoding: 'utf8' }).trim();
  const env = { ...process.env };
  delete env.TOP_K;
  delete env.CHUNK_TOP_K;
  assert.equal(run(env), '7 3');
  assert.equal(run({ ...env, TOP_K: '9' }), '9 3');
  // opt-out: the library leaves process.env alone
  assert.equal(run({ ...env, LIGHTRAG_TS_NO_DOTENV: '1' }), '40 20');
});

test('lightragQueryConfig carries LightRAG defaults the query path reads', () => {
  const gc = lightragQueryConfig();
  assert.equal(gc.kg_chunk_pick_method, 'VECTOR');
  assert.equal(gc.related_chunk_number, 5);
  assert.equal(gc.min_rerank_score, 0);
  assert.equal(gc.enable_llm_cache, true);
  assert.equal(gc.addon_params?.language, 'English');
  assert.equal(gc.rerank_model_func, null);
  assert.ok(gc.tokenizer instanceof TiktokenTokenizer && gc.tokenizer.modelName === 'gpt-4o-mini');
  assert.equal(lightragQueryConfig({ related_chunk_number: 2 }).related_chunk_number, 2);
});

test('queryTextChunks warns loudly when VECTOR chunk picking has no embedding function', async () => {
  const warnings: string[] = [];
  const onWarning = (w: Error) => warnings.push(w.message);
  process.on('warning', onWarning);
  try {
    queryTextChunks({ get_by_ids: async () => [] }, lightragQueryConfig(), null);
    queryTextChunks({ get_by_ids: async () => [] }, lightragQueryConfig({ kg_chunk_pick_method: 'WEIGHT' }), null);
    await new Promise((r) => setImmediate(r));
  } finally {
    process.off('warning', onWarning);
  }
  assert.equal(warnings.length, 1);
  assert.match(warnings[0]!, /VECTOR/);
});

test('a graph edge without weight gets weight 1.0 written into the graph itself, as upstream mutates it', async () => {
  const { probes } = read<{ probes: Probe[] }>('probes.json');
  const store = await openQueryStore(dir + 'store', '', read<Record<string, number[]>>('emb.json'));
  const before = await store.graph.get_edge('Alice', 'Rust');
  assert.ok(before && !('weight' in before));
  await runProbe(store, probes.find((p) => p.id === 'q1-local-context')!);
  const after = await store.graph.get_edge('Rust', 'Alice');
  assert.equal(after?.weight, 1.0);
  assert.equal(after, before); // the same object, not a copy
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

test('pyReprFloat matches repr(float)', () => {
  for (const [x, r] of cases.float_repr) assert.equal(pyReprFloat(x), r, String(x));
});

test('pyRound matches round() (half to even)', () => {
  for (const [x, r] of cases.round) assert.equal(pyRound(x), r, String(x));
});

test('computeArgsHash matches compute_args_hash (str of None, bool, float; lone surrogate)', () => {
  for (const [args, h] of cases.args_hash) assert.equal(computeArgsHash(...args), h, JSON.stringify(args));
});

test('pyJsonDumps matches json.dumps with and without ensure_ascii', () => {
  for (const [v, ascii, raw] of cases.json_dumps) {
    assert.equal(pyJsonDumps(v, { ensureAscii: true }), ascii);
    if (raw !== null) assert.equal(pyJsonDumps(v), raw);
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

// upstream tests/test_batch_embeddings.py runs unchanged against this code through
// the pytest bridge (npm run test:upstream, d11), so it is not ported here.
