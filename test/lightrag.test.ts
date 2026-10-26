// LightRAG (B4): the synthetic pipeline scenario against upstream's digests,
// plus the class's own edges (config, pipeline status, cancellation, helpers).
import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';
import { makeQueryParam } from '../src/base.ts';
import { chunkFieldsFromStatusDoc, getPipelineStatus, LightRAG, normalizeStringList, resolveDocFilePath } from '../src/lightrag.ts';
import { computeIncrementalChunkIds, getContentSummary, getEnvValue, pyFloatParse, pyIntParse, pyIsoformatUtc, subtractSourceIds } from '../src/utils.ts';
import {
  dirDiffs,
  FIXED_NOW,
  type FixtureDigest,
  fixtureDigest,
  makeLlm,
  NO_CACHE_VARIANTS,
  noCacheRebuildSpec,
  runSpecTs,
  stubEmbedding,
  syntheticDoc,
  syntheticSpec,
} from '../parity/pipeline-driver.ts';

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/pipeline-synthetic.json', import.meta.url), 'utf8')) as FixtureDigest;
const tmp = () => mkdtempSync(join(tmpdir(), 'lightrag-b4-'));

/** Pairs of calls that end in the other order than they started (a call key may repeat: matched first-in, first-out). */
function completionInversions(started: string[], ended: string[]): number {
  const pos = new Map<string, number[]>();
  started.forEach((k, i) => pos.set(k, [...(pos.get(k) ?? []), i]));
  const at = ended.map((k) => pos.get(k)!.shift()!);
  let n = 0;
  for (let i = 0; i < at.length; i++) for (let j = i + 1; j < at.length; j++) if (at[i]! > at[j]!) n++;
  return n;
}

function assertDigest(dir: string, records: Awaited<ReturnType<typeof runSpecTs>>): void {
  const got = fixtureDigest(dir, records);
  assert.equal(got.steps.length, FIXTURE.steps.length);
  got.steps.forEach((s, i) => assert.deepEqual(s, FIXTURE.steps[i], `step ${i} (${s.op}) differs from upstream`));
  assert.deepEqual(Object.keys(got.files), Object.keys(FIXTURE.files));
  for (const [f, d] of Object.entries(got.files)) assert.deepEqual(d, FIXTURE.files[f], `${f} differs from upstream`);
}

describe('LightRAG pipeline vs upstream (synthetic scenario)', () => {
  it('one task at a time: every record and every file equal upstream', async () => {
    const dir = tmp();
    try {
      const spec = syntheticSpec(dir);
      const records = await runSpecTs({ ...spec, config: { ...spec.config, llm_model_max_async: 1 } }, { maxParallelInsert: 1 });
      assertDigest(dir, records);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('default concurrency (2 documents, 4 LLM calls at once): the same files', async () => {
    const dir = tmp();
    try {
      const records = await runSpecTs(syntheticSpec(dir));
      const digest = fixtureDigest(dir, records);
      for (const [f, d] of Object.entries(digest.files)) {
        // the LLM cache gains its keys in completion order under concurrency: compared as a set of entries there
        if (f.endsWith('kv_store_llm_response_cache.json')) continue;
        assert.deepEqual(d, FIXTURE.files[f], `${f} differs from upstream`);
      }
      digest.steps.forEach((s, i) => assert.equal(s.result, FIXTURE.steps[i]!.result, `step ${i} (${s.op}) result`));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('default concurrency, answers finishing out of order (seeded delays): the same files', async () => {
    const dir = tmp();
    const seq = tmp();
    try {
      const started: string[] = [];
      const ended: string[] = [];
      const records = await runSpecTs(syntheticSpec(dir), {
        jitter: { seed: 'b4f', maxMs: 4, onCall: (kind, ev, key) => (ev === 'start' ? started : ended).push(`${kind}:${key}`) },
      });
      // pairs of calls that finish in the other order than they started: 114-136 such pairs
      // across the 526 calls of a run, over 8 runs (21.10.2026); a handful would leave this test checking almost nothing
      const inv = completionInversions(started, ended);
      assert.ok(inv >= 50, `the delays must reorder completions, or this test checks nothing (${inv} pairs out of order)`);
      const digest = fixtureDigest(dir, records);
      for (const [f, d] of Object.entries(digest.files)) {
        if (f.endsWith('kv_store_llm_response_cache.json')) continue;
        assert.deepEqual(d, FIXTURE.files[f], `${f} differs from upstream`);
      }
      digest.steps.forEach((s, i) => assert.equal(s.result, FIXTURE.steps[i]!.result, `step ${i} (${s.op}) result`));
      // the cache: the same entries as the one-at-a-time run (which equals upstream byte for byte)
      const spec = syntheticSpec(seq);
      await runSpecTs({ ...spec, config: { ...spec.config, llm_model_max_async: 1 } }, { maxParallelInsert: 1 });
      const cacheFile = Object.keys(digest.files).find((f) => f.endsWith('kv_store_llm_response_cache.json'))!;
      const cache = (d: string) => new Map(Object.entries(JSON.parse(readFileSync(join(d, cacheFile), 'utf8')) as object));
      assert.deepEqual(cache(dir), cache(seq));
    } finally {
      rmSync(dir, { recursive: true, force: true });
      rmSync(seq, { recursive: true, force: true });
    }
  });

  it('a second run on the same scenario is deterministic (incl. the cache bytes)', async () => {
    const a = tmp();
    const b = tmp();
    try {
      await runSpecTs(syntheticSpec(a));
      await runSpecTs(syntheticSpec(b));
      assert.deepEqual(
        dirDiffs(a, b).filter(([, d]) => d !== null),
        [],
      );
    } finally {
      rmSync(a, { recursive: true, force: true });
      rmSync(b, { recursive: true, force: true });
    }
  });
});

describe('rebuild without cached extractions vs upstream', () => {
  for (const variant of NO_CACHE_VARIANTS) {
    it(`${variant}: every record and file equal upstream, one at a time and at default concurrency`, async () => {
      const want = JSON.parse(readFileSync(new URL(`./fixtures/pipeline-nocache-${variant}.json`, import.meta.url), 'utf8')) as FixtureDigest;
      for (const sequential of [true, false]) {
        const dir = tmp();
        try {
          const spec = noCacheRebuildSpec(dir, variant);
          const records = sequential
            ? await runSpecTs({ ...spec, config: { ...spec.config, llm_model_max_async: 1 } }, { maxParallelInsert: 1 })
            : await runSpecTs(spec, { jitter: { seed: variant, maxMs: 3 } });
          assert.ok(
            records.some((r) => r.history.includes('No cached extraction results found, cannot rebuild')),
            'the scenario must reach the no-cache branch',
          );
          const got = fixtureDigest(dir, records, want.note);
          for (const [f, d] of Object.entries(got.files)) {
            if (!sequential && f.endsWith('kv_store_llm_response_cache.json')) continue;
            assert.deepEqual(d, want.files[f], `${f} differs from upstream (${sequential ? 'sequential' : 'default'})`);
          }
          assert.deepEqual(Object.keys(got.files), Object.keys(want.files));
          got.steps.forEach((s, i) => {
            if (sequential) assert.deepEqual(s, want.steps[i], `step ${i} (${s.op})`);
            else assert.equal(s.result, want.steps[i]!.result, `step ${i} (${s.op}) result`);
          });
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      }
    });
  }
});

describe('LightRAG class', () => {
  const make = (dir: string, extra: Record<string, unknown> = {}) =>
    new LightRAG({ working_dir: dir, workspace: `t-${Math.random()}`, embedding_func: stubEmbedding(8), llm_model_func: makeLlm('synthetic'), now: () => FIXED_NOW, ...extra });

  it('rejects storages that are not ported, a missing LLM, unknown fields', () => {
    const dir = tmp();
    try {
      assert.throws(() => make(dir, { kv_storage: 'RedisKVStorage' }), /not ported/);
      assert.throws(() => make(dir, { llm_model_func: null }), /callable/);
      assert.throws(() => make(dir, { no_such_field: 1 }), /unexpected keyword argument 'no_such_field'/);
      assert.throws(() => new LightRAG({ working_dir: dir, llm_model_func: makeLlm('synthetic') }), /embedding_func is required/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('fills cosine_better_than_threshold into vector_db_storage_cls_kwargs and keeps the caller embedding function unwrapped', () => {
    const dir = tmp();
    try {
      const emb = stubEmbedding(8);
      const rag = make(dir, { embedding_func: emb, vector_db_storage_cls_kwargs: { extra: 1 } });
      assert.deepEqual(rag.vector_db_storage_cls_kwargs, { cosine_better_than_threshold: 0.2, extra: 1 });
      assert.notEqual(rag.embedding_func, emb);
      assert.equal(rag.embedding_func!.embedding_dim, 8);
      assert.equal(rag._storages_status, 'created');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('a busy pipeline queues the request; the running one picks the documents up', async () => {
    const dir = tmp();
    try {
      let open!: () => void;
      const gate = new Promise<void>((r) => (open = r));
      const base = makeLlm('synthetic');
      const rag = make(dir, {
        llm_model_func: async (p: string, o?: Record<string, unknown>) => {
          await gate;
          return base(p, o);
        },
      });
      await rag.initialize_storages();
      const first = rag.ainsert([syntheticDoc(1)], null, false, null, null, 'a');
      while (!rag.pipeline_status.busy) await new Promise((r) => setImmediate(r));
      // the second call enqueues, sees the pipeline busy and returns at once
      await rag.ainsert([syntheticDoc(2)], null, false, null, null, 'b');
      assert.equal(rag.pipeline_status.request_pending, true);
      open();
      await first;
      const counts = await rag.get_processing_status();
      assert.equal(counts.processed, 2);
      assert.ok(rag.pipeline_status.history_messages.includes('Processing additional documents due to pending request'));
      assert.equal(rag.pipeline_status.busy, false);
      await rag.finalize_storages();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('chunks_only stores chunks and vectors without an LLM call; naive finds them, delete removes them', async () => {
    const dir = tmp();
    try {
      const rag = make(dir, {
        chunks_only: true,
        cosine_better_than_threshold: -1,
        llm_model_func: async () => {
          throw new Error('chunks_only must not call the LLM');
        },
      });
      await rag.initialize_storages();
      await rag.ainsert([syntheticDoc(1), syntheticDoc(2)], null, false, ['doc-a', 'doc-b'], ['a.md', 'b.md'], 'co');
      const processed = await rag.get_docs_by_status('processed');
      assert.deepEqual([...processed.keys()].sort(), ['doc-a', 'doc-b']);
      const chunkIds = processed.get('doc-a')!.chunks_list!;
      assert.ok(chunkIds.length > 0);
      assert.ok((await rag.chunks_vdb.get_by_ids(chunkIds)).every((c) => c !== null));
      assert.ok((await rag.text_chunks.get_by_ids(chunkIds)).every((c) => c !== null));
      assert.equal((await rag.chunk_entity_relation_graph.get_all_nodes()).length, 0);
      assert.equal(await rag.full_entities.get_by_id('doc-a'), null);

      const ctx = await rag.aquery(syntheticDoc(1).slice(0, 200), makeQueryParam({ mode: 'naive', only_need_context: true }));
      assert.match(String(ctx), /a\.md/);

      const r = await rag.adelete_by_doc_id('doc-a');
      assert.equal(r.status, 'success');
      assert.ok((await rag.chunks_vdb.get_by_ids(chunkIds)).every((c) => c === null));
      assert.ok((await rag.text_chunks.get_by_ids(chunkIds)).every((c) => c === null));
      assert.equal((await rag.get_docs_by_status('processed')).size, 1);
      await rag.finalize_storages();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('a deletion is refused while the pipeline runs another job', async () => {
    const dir = tmp();
    try {
      const rag = make(dir);
      await rag.initialize_storages();
      Object.assign(getPipelineStatus(rag.workspace), { busy: true, job_name: 'Default Job' });
      const r = await rag.adelete_by_doc_id('doc-x');
      assert.equal(r.status, 'not_allowed');
      assert.equal(r.status_code, 403);
      getPipelineStatus(rag.workspace).busy = false;
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('cancellation stops queued documents with FAILED "User cancelled"', async () => {
    const dir = tmp();
    try {
      let calls = 0;
      const base = makeLlm('synthetic');
      const rag = make(dir, {
        max_parallel_insert: 1,
        llm_model_func: async (p: string, o?: Record<string, unknown>) => {
          if (++calls === 1) getPipelineStatus(rag.workspace).cancellation_requested = true;
          return base(p, o);
        },
      });
      await rag.initialize_storages();
      await rag.ainsert([syntheticDoc(3), syntheticDoc(4)], null, false, null, null, 'c');
      const failed = await rag.get_docs_by_status('failed');
      assert.equal(failed.size, 2);
      assert.ok([...failed.values()].every((d) => /User cancelled/.test(d.error_msg ?? '')));
      assert.equal(rag.pipeline_status.cancellation_requested, false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('aquery returns the answer string; bypass streams when asked', async () => {
    const dir = tmp();
    try {
      const rag = make(dir, {
        cosine_better_than_threshold: -1,
        llm_model_func: async (p: string, o: Record<string, unknown> = {}) =>
          o.stream
            ? (async function* () {
                yield 'a';
                yield 'b';
              })()
            : makeLlm('synthetic')(p, o),
      });
      await rag.initialize_storages();
      await rag.ainsert([syntheticDoc(5)], null, false, null, null, 'q');
      assert.match(String(await rag.aquery('archive?', makeQueryParam({ mode: 'local' }))), /^STUB [0-9a-f]{32}$/);
      const it = (await rag.aquery('hi', makeQueryParam({ mode: 'bypass', stream: true }))) as AsyncIterable<string>;
      let s = '';
      for await (const c of it) s += c;
      assert.equal(s, 'ab');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('lightrag.py helpers', () => {
  it('_resolve_doc_file_path prefers a real path, then full_docs, then a placeholder', () => {
    assert.equal(resolveDocFilePath({ file_path: ' a.md ' }), 'a.md');
    assert.equal(resolveDocFilePath({ file_path: 'unknown_source' }, { file_path: 'b.md' }), 'b.md');
    assert.equal(resolveDocFilePath({ file_path: 'no-file-path' }), 'unknown_source');
    assert.equal(resolveDocFilePath({ file_path: '' }, { file_path: '  ' }), 'unknown_source');
    assert.equal(resolveDocFilePath(null, null), 'unknown_source');
  });

  it('_chunk_fields_from_status_doc and _normalize_string_list', () => {
    assert.deepEqual(chunkFieldsFromStatusDoc({ chunks_list: ['a', '', 'b'], chunks_count: null }), [['a', 'b'], 2]);
    assert.deepEqual(chunkFieldsFromStatusDoc({ chunks_list: ['a'], chunks_count: 5 }), [['a'], 5]);
    assert.deepEqual(chunkFieldsFromStatusDoc({ chunks_list: null, chunks_count: -1 }), [[], 0]);
    const warnings: string[] = [];
    assert.deepEqual(normalizeStringList(['x', 1, '', 'y'], 'ctx', (_l, m) => warnings.push(m)), ['x', 'y']);
    assert.deepEqual(warnings, ['Non-string element dropped from list (ctx) at index 1: 1', "Non-string element dropped from list (ctx) at index 2: ''"]);
    assert.deepEqual(normalizeStringList('nope'), []);
  });

  it('utils helpers: content summary, subtract / incremental chunk ids, env values, isoformat', () => {
    assert.equal(getContentSummary('  short  '), 'short');
    assert.equal(getContentSummary('é'.repeat(300)), 'é'.repeat(250) + '...');
    assert.deepEqual(subtractSourceIds(['a', '', 'b', 'c'], ['b']), ['a', 'c']);
    // upstream's docstring example says ['chunk-3', 'chunk-2', 'chunk-4']; its code (ported) keeps chunk-2 in place
    assert.deepEqual(computeIncrementalChunkIds(['chunk-1', 'chunk-2', 'chunk-3'], ['chunk-1', 'chunk-2'], ['chunk-2', 'chunk-4']), ['chunk-2', 'chunk-3', 'chunk-4']);
    assert.equal(pyIntParse(' 1_000 '), 1000);
    assert.equal(pyIntParse('1.0'), null);
    assert.equal(pyFloatParse('1e3'), 1000);
    assert.equal(pyFloatParse('-inf'), -Infinity);
    assert.equal(pyFloatParse('x'), null);
    process.env.LIGHTRAG_TS_TEST_ENV = '0.5';
    assert.equal(getEnvValue('LIGHTRAG_TS_TEST_ENV', 3, 'int'), 3);
    assert.equal(getEnvValue('LIGHTRAG_TS_TEST_ENV', 3, 'float'), 0.5);
    process.env.LIGHTRAG_TS_TEST_ENV = 'Yes';
    assert.equal(getEnvValue('LIGHTRAG_TS_TEST_ENV', false, 'bool'), true);
    process.env.LIGHTRAG_TS_TEST_ENV = '["a"]';
    assert.deepEqual(getEnvValue('LIGHTRAG_TS_TEST_ENV', [], 'list'), ['a']);
    delete process.env.LIGHTRAG_TS_TEST_ENV;
    assert.equal(pyIsoformatUtc(1760000000), '2025-10-09T08:53:20+00:00');
    assert.equal(pyIsoformatUtc(1760000000.25), '2025-10-09T08:53:20.250000+00:00');
    assert.equal(createHash('md5').update('').digest('hex').length, 32);
  });
});

describe('LightRAG.aquery_llm vs upstream kg_query / naive_query (B3 probes)', () => {
  it('every probe a LightRAG can express builds what upstream built', async (t) => {
    const { cpSync } = await import('node:fs');
    const { describeProbe, diffRecords, expectedViaLightRAG, openLightRAGStore, probeFitsLightRAG, readRecords, runProbeViaLightRAG } = await import('../parity/query-runner.ts');
    const q = new URL('./fixtures/query/', import.meta.url);
    const read = <T>(f: string): T => JSON.parse(readFileSync(new URL(f, q), 'utf8')) as T;
    const dir = tmp();
    try {
      cpSync(new URL('store', q), dir, { recursive: true });
      const { probes } = read<{ probes: import('../parity/query-runner.ts').Probe[] }>('probes.json');
      const upstream = new Map(readRecords(new URL('upstream.jsonl', q).pathname).map((r) => [r.id, r]));
      const store = await openLightRAGStore(dir, '', read<Record<string, number[]>>('emb.json'));
      const fit = probes.filter(probeFitsLightRAG);
      assert.ok(fit.length >= 130, `${fit.length} probes`);
      for (const p of fit) {
        const rec = await runProbeViaLightRAG(store, p);
        const want = upstream.get(p.id)!;
        await t.test(p.id, () => {
          const d = diffRecords(rec, expectedViaLightRAG(want));
          assert.deepEqual(d, [], `${describeProbe(p)}\n${d.join('\n')}`);
        });
      }
      assert.deepEqual([...store.missingVectors], []);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
