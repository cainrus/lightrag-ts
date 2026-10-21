// Parity harness, query context on a real store: the same query must build the
// same context, prompt and raw_data in TS as upstream, byte for byte.
//
// 1. probes (read-only on the source store): every `*:keywords:*` cache entry
//    becomes a query, run in each mode with the QueryParam the entry recorded;
//    the stub LLM answers keyword extraction with the entry's own JSON.
//      node parity/query-context.ts probes <working_dir> --workspace W --out probes.json
// 2. a COPY of the store (APFS clone: cp -c), then upstream on it:
//      <rag-venv>/bin/python -I tools/upstream/query_parity.py run <copy> --workspace W \
//        --probes probes.json --emb-from-store --emb-out emb.json --out py.jsonl
//    query vectors are stored chunk vectors picked by md5(text): no live embedding
// 3. TS on the same copy with the vectors upstream used, diffed record by record:
//      node parity/query-context.ts run <copy> --workspace W --probes probes.json \
//        --emb emb.json --py py.jsonl [--json report.json]
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { forEachJsonKv, kvStorePath } from '../src/kg/json_kv_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import type { Dict, QueryMode } from '../src/query.ts';
import { describeProbe, diffRecords, expectedViaLightRAG, openLightRAGStore, openQueryStore, type Probe, type ProbeRecord, readRecords, runProbe, runProbeViaLightRAG } from './query-runner.ts';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    workspace: { type: 'string', default: '' },
    out: { type: 'string' },
    modes: { type: 'string', default: 'mix,local,global,hybrid,naive' },
    variants: { type: 'string', default: 'context,full' },
    limit: { type: 'string' },
    probes: { type: 'string' },
    emb: { type: 'string' },
    py: { type: 'string' },
    json: { type: 'string' },
    'show-diffs': { type: 'string', default: '5' },
    // B4: run the probes through LightRAG.aquery_llm instead of kg_query / naive_query
    via: { type: 'string', default: 'functions' },
  },
});
const [cmd, dir] = positionals;

if (cmd === 'probes' && dir && values.out) {
  type Entry = { return: string; original_prompt?: string; queryparam?: Dict | null };
  const seen = new Set<string>();
  const queries: { query: string; qp: Dict; kw: string }[] = [];
  forEachJsonKv<Entry>(kvStorePath(dir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, values.workspace), (k, e) => {
    if (k.split(':')[1] !== 'keywords' || !e.original_prompt || seen.has(e.original_prompt)) return;
    seen.add(e.original_prompt);
    queries.push({ query: e.original_prompt, qp: e.queryparam ?? {}, kw: e.return });
  });
  const take = values.limit ? queries.slice(0, Number(values.limit)) : queries;
  const probes: Probe[] = [];
  take.forEach((q, i) => {
    const param: Dict = {};
    for (const f of ['response_type', 'top_k', 'chunk_top_k', 'max_entity_tokens', 'max_relation_tokens', 'max_total_tokens', 'enable_rerank']) {
      if (Object.hasOwn(q.qp, f)) param[f] = q.qp[f];
    }
    if (q.qp.user_prompt) param.user_prompt = q.qp.user_prompt;
    for (const mode of values.modes.split(',')) {
      for (const variant of values.variants.split(',')) {
        probes.push({
          id: `q${String(i).padStart(3, '0')}-${mode}-${variant}`,
          query: q.query,
          mode: mode as QueryMode,
          variant: variant as Probe['variant'],
          param,
          gc: {},
          kw_response: q.kw,
          answer: 'hash',
          system_prompt: null,
        });
      }
    }
  });
  // private data: the probes carry real queries, keep them out of the repo
  writeFileSync(values.out, JSON.stringify({ embeddings: {}, probes }, null, 1) + '\n');
  console.log(JSON.stringify({ queries: take.length, probes: probes.length, out: values.out }));
} else if (cmd === 'run' && dir && values.probes && values.emb && values.py) {
  const t0 = performance.now();
  const { probes } = JSON.parse(readFileSync(values.probes, 'utf8')) as { probes: Probe[] };
  const py = new Map(readRecords(values.py).map((r) => [r.id, r]));
  const vectors = JSON.parse(readFileSync(values.emb, 'utf8')) as Record<string, number[]>;
  const viaRag = values.via === 'lightrag';
  const store = viaRag ? await openLightRAGStore(dir, values.workspace, vectors) : await openQueryStore(dir, values.workspace, vectors);
  const run1 = (p: Probe): Promise<ProbeRecord> =>
    viaRag ? runProbeViaLightRAG(store as Awaited<ReturnType<typeof openLightRAGStore>>, p) : runProbe(store as Awaited<ReturnType<typeof openQueryStore>>, p);
  const loadSeconds = (performance.now() - t0) / 1000;
  const diffs: { id: string; probe: string; diff: string[] }[] = [];
  const tally = { probes: 0, match: 0, noUpstreamRecord: 0, contextNone: 0, errors: 0, llmCalls: 0, cacheHitAnswers: 0, byMode: {} as Record<string, { n: number; match: number }> };
  for (const p of probes) {
    const want = py.get(p.id);
    if (!want) {
      tally.noUpstreamRecord++;
      continue;
    }
    tally.probes++;
    const rec = await run1(p);
    // through LightRAG the priority limiter takes `_priority`; upstream's records come from kg_query directly
    const d = diffRecords(rec, viaRag ? expectedViaLightRAG(want) : want);
    const m = (tally.byMode[p.mode] ??= { n: 0, match: 0 });
    m.n++;
    if (!d.length) {
      tally.match++;
      m.match++;
    } else diffs.push({ id: p.id, probe: describeProbe(p), diff: d });
    if (rec.result === null) tally.contextNone++;
    if (rec.error) tally.errors++;
    tally.llmCalls += rec.llm_calls.length;
    if (p.variant === 'full' && rec.result && !rec.llm_calls.some((c) => !c.keyword_extraction)) tally.cacheHitAnswers++;
  }
  const report = {
    dir,
    loadSeconds: Number(loadSeconds.toFixed(1)),
    seconds: Number(((performance.now() - t0) / 1000).toFixed(1)),
    ...tally,
    missingVectors: [...store.missingVectors].length,
    diffSamples: diffs.slice(0, Number(values['show-diffs'])),
  };
  console.log(JSON.stringify(report, null, 2));
  if (values.json) writeFileSync(values.json, JSON.stringify({ ...report, diffs }, null, 2) + '\n');
  process.exitCode = diffs.length || tally.noUpstreamRecord ? 1 : 0;
} else {
  console.error('usage: query-context.ts probes <dir> --out FILE [--workspace W] | run <copy> --probes FILE --emb FILE --py FILE [--workspace W]');
  process.exit(2);
}
