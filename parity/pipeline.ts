// B4 pipeline parity: the same scenario through upstream's LightRAG and the TS
// LightRAG, every step's result and pipeline messages compared, then every
// file both stores end with, byte for byte.
//
//   node parity/pipeline.ts synthetic [--work DIR] [--write-fixture test/fixtures/pipeline]
//       the public scenario (parity/pipeline-driver.ts syntheticSpec): insert,
//       duplicates, queries in every mode, deletions with rebuild, the editing
//       API, a custom KG. --write-fixture stores upstream's records and files
//       for test/lightrag.test.ts.
//   node parity/pipeline.ts store <store-dir> --workspace W --work DIR [--docs N] [--batch B] [--deletes M]
//       a COPY-based replay of a real store (private data, nothing committed):
//       N processed documents, B per ainsert call, inserted into an empty store
//       seeded with the source's LLM cache entries the run uses (a TS pre-run
//       records them); then M of them deleted (rebuild from cache), then
//       query probes. Extraction must hit the cache; summaries and answers that
//       miss get deterministic stand-ins on both sides.
//
// Upstream side: tools/upstream/pipeline_parity.py (LIGHTRAG_PY, default the
// upstream test venv). It runs with max_parallel_insert=1 and its merge tasks
// one at a time; TS runs at the spec's parallelism (--ts-parallel overrides).
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { mkdtempSync } from 'node:fs';
import { forEachJsonKv, kvStorePath } from '../src/kg/json_kv_impl.ts';
import { writeJsonPy } from '../src/internal/py_json.ts';
import { NameSpace } from '../src/namespace.ts';
import { isDeepStrictEqual } from 'node:util';
import { dirDiffs, diffStepRecords, fixtureDigest, type PipelineSpec, readStepRecords, runSpecTs, type Step, syntheticSpec } from './pipeline-driver.ts';

const PY = process.env.LIGHTRAG_PY ?? join(homedir(), '.cache/lightrag-ts/upstream-venv/bin/python');
const PY_TOOL = resolve(import.meta.dirname, '../tools/upstream/pipeline_parity.py');

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    work: { type: 'string' },
    workspace: { type: 'string', default: '' },
    'write-fixture': { type: 'string' },
    docs: { type: 'string', default: '100' },
    offset: { type: 'string', default: '0' },
    batch: { type: 'string', default: '10' },
    deletes: { type: 'string', default: '10' },
    queries: { type: 'string', default: '4' },
    'ts-mode': { type: 'string', default: 'both' },
    'reuse-py': { type: 'boolean', default: false },
    json: { type: 'string' },
  },
});
const [cmd, src] = positionals;
const WORK = resolve(values.work ?? mkdtempSync(join(tmpdir(), 'lightrag-pipeline-')));
mkdirSync(WORK, { recursive: true });
const t0 = Date.now();
const log = (s: string) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${s}`);

function runPy(specFile: string, out: string): void {
  try {
    execFileSync(PY, ['-I', PY_TOOL, specFile, '--out', out], { maxBuffer: 1 << 30, stdio: ['ignore', 'inherit', 'pipe'] });
  } catch (e) {
    const err = (e as { stderr?: Buffer }).stderr?.toString('utf8') ?? '';
    throw new Error(`upstream run failed:\n${err.split('\n').slice(-15).join('\n')}`);
  }
}

type Variant = { name: string; recordDiffs: string[]; fileDiffs: [string, string][]; files: number };
type Report = { steps: number; variants: Variant[]; seconds: Record<string, number> };

/** The LLM cache as a sorted map: the comparison for runs whose LLM calls finish in another order. */
function cacheAsMapDiff(a: string, b: string): string | null {
  const m = (p: string) => Object.entries(JSON.parse(readFileSync(p, 'utf8')) as object).sort(([x], [y]) => (x < y ? -1 : 1));
  return isDeepStrictEqual(m(a), m(b)) ? null : 'cache entries differ (compared as a map)';
}

/**
 * Run `spec` on both sides from the same seed dir and compare. TS runs twice:
 * "sequential" (llm_model_max_async=1, max_parallel_insert=1) must match
 * upstream byte for byte; "default" (the spec's concurrency) too, except the
 * LLM cache, whose new keys follow completion order there (compared as a map).
 */
async function compare(spec: Omit<PipelineSpec, 'working_dir'>, seed: string | null): Promise<Report> {
  const pyDir = join(WORK, 'py');
  const seconds: Record<string, number> = {};
  const fresh = (d: string) => {
    rmSync(d, { recursive: true, force: true });
    if (seed) cpSync(seed, d, { recursive: true });
    else mkdirSync(d, { recursive: true });
  };
  const pyOut = join(WORK, 'py.jsonl');
  if (!(values['reuse-py'] && existsSync(pyOut))) {
    fresh(pyDir);
    const specFile = join(WORK, 'spec-py.json');
    writeFileSync(specFile, JSON.stringify({ ...spec, working_dir: pyDir }));
    const t = Date.now();
    runPy(specFile, pyOut);
    seconds.upstream = (Date.now() - t) / 1000;
    log(`upstream done in ${seconds.upstream}s`);
  }
  const pyRecords = readStepRecords(pyOut);
  const variants: Variant[] = [];
  const modes = values['ts-mode'] === 'both' ? ['sequential', 'default'] : [values['ts-mode']!];
  for (const name of modes) {
    const dir = join(WORK, `ts-${name}`);
    fresh(dir);
    const t = Date.now();
    const config = name === 'sequential' ? { ...(spec.config ?? {}), llm_model_max_async: 1 } : spec.config;
    const records = await runSpecTs({ ...spec, config, working_dir: dir } as PipelineSpec, name === 'sequential' ? { maxParallelInsert: 1 } : {});
    seconds[`ts-${name}`] = (Date.now() - t) / 1000;
    log(`ts ${name} done in ${seconds[`ts-${name}`]}s`);
    writeFileSync(join(WORK, `ts-${name}.jsonl`), records.map((r) => JSON.stringify(r)).join('\n') + '\n');
    // concurrent runs interleave pipeline messages: only the results compare there
    const recordDiffs = diffStepRecords(records, pyRecords, { history: name === 'sequential', asyncLimit: name === 'sequential' });
    const files = dirDiffs(pyDir, dir);
    const fileDiffs: [string, string][] = [];
    for (const [f, d] of files) {
      if (d === null) continue;
      if (name !== 'sequential' && f.endsWith('kv_store_llm_response_cache.json')) {
        const m = cacheAsMapDiff(join(pyDir, f), join(dir, f));
        if (m) fileDiffs.push([f, m]);
      } else fileDiffs.push([f, d]);
    }
    variants.push({ name, recordDiffs, fileDiffs, files: files.length });
  }
  return { steps: pyRecords.length, variants, seconds };
}

function report(r: Report): void {
  let bad = 0;
  for (const v of r.variants) {
    for (const d of v.recordDiffs.slice(0, 20)) log(`[${v.name}] DIFF ${d}`);
    for (const [f, d] of v.fileDiffs) log(`[${v.name}] FILE ${f}: ${d}`);
    log(`[${v.name}] ${r.steps} steps, ${v.recordDiffs.length} record differences; ${v.files} files, ${v.fileDiffs.length} differ`);
    bad += v.recordDiffs.length + v.fileDiffs.length;
  }
  if (values.json) writeFileSync(values.json, JSON.stringify(r, null, 2) + '\n');
  process.exitCode = bad ? 1 : 0;
}

if (cmd === 'synthetic') {
  const spec = syntheticSpec('');
  const r = await compare(spec, null);
  report(r);
  const fixture = values['write-fixture'];
  if (fixture) {
    if (r.variants.some((v) => v.recordDiffs.length || v.fileDiffs.length)) throw new Error('not writing a fixture from a run that differs');
    writeFileSync(fixture, JSON.stringify(fixtureDigest(join(WORK, 'py'), readStepRecords(join(WORK, 'py.jsonl'))), null, 1) + '\n');
    log(`fixture written to ${fixture}`);
  }
} else if (cmd === 'store' && src) {
  await storeReplay(src);
} else {
  console.error('usage: pipeline.ts synthetic [--work DIR] [--write-fixture DIR] | store <store-dir> --workspace W --work DIR [--docs N]');
  process.exit(2);
}

// ---------------------------------------------------------------- store replay

async function storeReplay(storeDir: string): Promise<void> {
  const ws = values.workspace!;
  type Status = { status: string; file_path?: string; chunks_list?: string[] };
  const status = new Map<string, Status>();
  forEachJsonKv<Status>(kvStorePath(storeDir, NameSpace.DOC_STATUS, ws), (k, v) => void status.set(k, v));
  const fullDocs = new Map<string, { content: string; file_path?: string }>();
  forEachJsonKv<{ content: string; file_path?: string }>(kvStorePath(storeDir, NameSpace.KV_STORE_FULL_DOCS, ws), (k, v) => void fullDocs.set(k, v));
  const ids = [...status].filter(([k, v]) => v.status === 'processed' && fullDocs.has(k)).map(([k]) => k);
  const n = Number(values.docs);
  const picked = ids.slice(Number(values.offset), Number(values.offset) + n);
  const batch = Number(values.batch);
  const steps: Step[] = [];
  for (let i = 0; i < picked.length; i += batch) {
    const part = picked.slice(i, i + batch);
    steps.push({
      op: 'insert',
      docs: part.map((d) => fullDocs.get(d)!.content),
      file_paths: part.map((d) => status.get(d)!.file_path ?? fullDocs.get(d)!.file_path ?? 'unknown_source'),
      track_id: `replay-${i / batch}`,
    });
  }
  // a re-insert of the first document: the duplicate path
  if (picked.length) steps.push({ op: 'insert', docs: [fullDocs.get(picked[0]!)!.content], file_paths: ['again.md'], track_id: 'replay-dup' });
  // queries from the store's own keyword cache entries
  const kw: string[] = [];
  forEachJsonKv<{ original_prompt?: string }>(kvStorePath(storeDir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ws), (k, e) => {
    if (k.split(':')[1] === 'keywords' && e.original_prompt && kw.length < Number(values.queries) && !kw.includes(e.original_prompt)) kw.push(e.original_prompt);
  });
  const queryStep = (q: string, mode: string, tag: string): Step => ({ op: 'query', query: q, param: { mode, ...(tag === 'ctx' ? { only_need_context: true } : {}) } });
  for (const q of kw) for (const mode of ['mix', 'local', 'global', 'hybrid', 'naive']) steps.push(queryStep(q, mode, 'ctx'), queryStep(q, mode, 'full'));
  // deletions: every k-th inserted document, half of them with their cache
  const deletes = Number(values.deletes);
  const stride = Math.max(1, Math.floor(picked.length / Math.max(1, deletes)));
  let dn = 0;
  for (let i = 0; i < picked.length && dn < deletes; i += stride, dn++) steps.push({ op: 'delete', doc_id: picked[i]!, delete_llm_cache: dn % 2 === 1 });
  for (const q of kw) for (const mode of ['mix', 'local']) steps.push(queryStep(q, mode, 'full'));
  steps.push({ op: 'counts' });

  const spec: Omit<PipelineSpec, 'working_dir'> = { workspace: ws, dim: 8, llm: 'cache', config: {}, steps };
  writeFileSync(join(WORK, 'spec.json'), JSON.stringify(spec));
  log(`${picked.length} documents in ${Math.ceil(picked.length / batch)} inserts, ${dn} deletes, ${kw.length} queries x 5 modes`);

  // seed: the source's cache entries this run reads, in file order (a TS pre-run finds them)
  const seed = join(WORK, 'seed');
  const seedCache = kvStorePath(seed, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ws);
  if (!existsSync(seedCache)) {
    const hits = new Set<string>();
    const pre = join(WORK, 'pre');
    rmSync(pre, { recursive: true, force: true });
    mkdirSync(join(pre, ws), { recursive: true });
    cpSync(kvStorePath(storeDir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ws), kvStorePath(pre, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ws));
    await runSpecTs({ ...spec, working_dir: pre } as PipelineSpec, {
      onRag: (rag) => {
        const c = rag.llm_response_cache;
        const get = c.get_by_id.bind(c);
        const gets = c.get_by_ids.bind(c);
        c.get_by_id = async (id) => {
          const r = await get(id);
          if (r) hits.add(id);
          return r;
        };
        c.get_by_ids = async (ids2) => {
          const r = await gets(ids2);
          ids2.forEach((id, i) => r[i] && hits.add(id));
          return r;
        };
        // the pre-run never writes the 300 MB cache back
        c.index_done_callback = async () => false;
        c.finalize = async () => {};
      },
    });
    const slice = new Map<string, unknown>();
    forEachJsonKv(kvStorePath(storeDir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ws), (k, v) => void (hits.has(k) && slice.set(k, v)));
    mkdirSync(join(seed, ws), { recursive: true });
    writeJsonPy(slice, seedCache);
    rmSync(pre, { recursive: true, force: true });
    log(`seed cache: ${slice.size} entries the run reads`);
  }
  const r = await compare(spec, seed);
  report(r);
  void readFileSync;
}
