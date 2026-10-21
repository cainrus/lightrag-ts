// B2 operate parity: the TS chunking / extraction / merge against upstream's, on a copy of a store.
//
//   node parity/operate.ts <store-dir> [--work <dir>] [--phase chunk,...]
//
// The store is only read; upstream writes its JSON Lines under <work>.
//   merge  merge_nodes_and_edges over whole documents replayed from the cache
//          (--select processed|unmerged, --docs N, --offset K, --base empty|graph,
//          --vdb, --dim D): both sides start from the same files, extract each
//          chunk from the cache, merge doc by doc with a fixed clock, a summary
//          stub for cache misses and a hash embedding; every file they write
//          must be byte-identical. Upstream runs its merge tasks one at a time
//          (see SCOPE.md); --concurrency-check reruns TS with the default
//          concurrency, --py-concurrent measures how far upstream's own
//          concurrent run drifts from its sequential one.
//   chunk  chunking_by_token_size for every doc in full_docs: token ids (md5),
//          chunk ids, tokens, content equal upstream's for the same doc; with the
//          default settings also equal to what kv_store_text_chunks holds.
//          A few sampled variants (small windows, split_by_character, -only)
//          exercise the code paths the store's settings never reach.
// Exit code 1 on any mismatch. Upstream side: tools/upstream/operate_parity.py in the rag-venv.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { isDeepStrictEqual } from 'node:util';
import { kvStorePath, readJsonKv } from '../src/kg/json_kv_impl.ts';
import { type GraphmlGraph, readGraphml } from '../src/kg/networkx_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import { DEFAULT_ENTITY_TYPES, DEFAULT_MAX_EXTRACT_INPUT_TOKENS } from '../src/constants.ts';
import {
  chunkingByTokenSize,
  edgeKeyParts,
  extractEntities,
  type ExtractChunk,
  type MergeConfig,
  processExtractionResult,
} from '../src/operate.ts';
import { PROMPTS } from '../src/prompt.ts';
import { computeMdhashId, pyFormat, TiktokenTokenizer } from '../src/utils.ts';
import { CacheMiss, fileDiff, type MergeDoc, mergeDirDiffs, MERGE_FILES, type MergeRec, tsMergeDocs } from './merge-driver.ts';

const PY = process.env.LIGHTRAG_PY ?? join(process.env.HOME!, '.claude-home/rag-venv/bin/python');
const PY_TOOL = resolve(import.meta.dirname, '../tools/upstream/operate_parity.py');

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    work: { type: 'string' },
    phase: { type: 'string', default: 'chunk' },
    model: { type: 'string', default: 'gpt-4o-mini' },
    // merge phase
    select: { type: 'string', default: 'processed' },
    base: { type: 'string', default: 'empty' },
    docs: { type: 'string', default: '200' },
    offset: { type: 'string', default: '0' },
    vdb: { type: 'boolean', default: false },
    dim: { type: 'string', default: '8' },
    'concurrency-check': { type: 'boolean', default: false },
    'py-concurrent': { type: 'boolean', default: false },
    gc: { type: 'string', default: '{}' },
    'ts-only': { type: 'boolean', default: false },
    'reuse-py': { type: 'boolean', default: false },
  },
});
const SRC = positionals[0];
if (!SRC) throw new Error('usage: operate.ts <store-dir> [--work dir] [--phase chunk]');
const WORK = values.work ?? mkdtempSync(join(tmpdir(), 'lightrag-operate-'));
mkdirSync(WORK, { recursive: true });
const PHASES = new Set(values.phase!.split(','));

let failures = 0;
const t0 = Date.now();
const log = (s: string) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${s}`);
function check(name: string, ok: boolean, detail = ''): void {
  if (!ok) failures++;
  log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` - ${detail}` : ''}`);
}

function py(args: string[]): void {
  try {
    execFileSync(PY, ['-I', PY_TOOL, ...args], { maxBuffer: 1 << 30, stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (e) {
    const err = (e as { stderr?: Buffer }).stderr?.toString('utf8') ?? '';
    throw new Error(`upstream ${args[0]} failed:\n${err.split('\n').slice(-8).join('\n')}`);
  }
}
const readJsonl = <T>(file: string): T[] =>
  readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as T);

// ---------------------------------------------------------------- chunk

type ChunkRec = {
  doc: string;
  n_tokens?: number;
  tokens_md5?: string;
  encode_error?: string;
  chunks?: { id: string; tokens: number; index: number; content: string }[];
  error?: string;
};
type Doc = { content: string };
type StoredChunk = { content: string; tokens: number; chunk_order_index: number; full_doc_id?: string };

const tokensMd5 = (t: number[]) => createHash('md5').update(t.join(',')).digest('hex');

function tsChunkRec(tok: TiktokenTokenizer, doc: string, content: string, v: Variant): ChunkRec {
  const rec: ChunkRec = { doc };
  try {
    const t = tok.encode(content);
    rec.n_tokens = t.length;
    rec.tokens_md5 = tokensMd5(t);
  } catch {
    rec.encode_error = 'ValueError';
  }
  try {
    rec.chunks = chunkingByTokenSize(tok, content, v.split ?? null, v.splitOnly ?? false, v.overlap, v.size).map((c) => ({
      id: computeMdhashId(c.content, 'chunk-'),
      tokens: c.tokens,
      index: c.chunk_order_index,
      content: c.content,
    }));
  } catch (e) {
    rec.error = String(e);
  }
  return rec;
}

type Variant = { name: string; size: number; overlap: number; split?: string; splitOnly?: boolean; sample?: number };
const VARIANTS: Variant[] = [
  { name: 'default 1200/100', size: 1200, overlap: 100 },
  { name: 'window 50/7', size: 50, overlap: 7, sample: 400 },
  { name: 'window 3/1 (cuts inside UTF-8 sequences)', size: 3, overlap: 1, sample: 60 },
  { name: 'split \\n\\n 200/20', size: 200, overlap: 20, split: '\n\n', sample: 400 },
  { name: 'split-only \\n 300', size: 300, overlap: 20, split: '\n', splitOnly: true, sample: 400 },
  { name: 'split-only \\n 40 (ChunkTokenLimitExceededError)', size: 40, overlap: 5, split: '\n', splitOnly: true, sample: 200 },
];

function phaseChunk(): void {
  const tok = new TiktokenTokenizer(values.model);
  const docs = readJsonKv<Doc>(kvStorePath(SRC!, NameSpace.KV_STORE_FULL_DOCS, ''));
  const stored = readJsonKv<StoredChunk>(kvStorePath(SRC!, NameSpace.KV_STORE_TEXT_CHUNKS, ''));
  log(`chunk: ${docs.size} docs, ${stored.size} stored chunks, tokenizer ${values.model}`);
  for (const v of VARIANTS) {
    const out = join(WORK, `py-chunk-${v.size}-${v.overlap}-${v.split ? 's' : ''}${v.splitOnly ? 'o' : ''}.jsonl`);
    const args = ['chunk', SRC!, '--out', out, '--model', values.model!, '--size', String(v.size), '--overlap', String(v.overlap)];
    if (v.split) args.push('--split', JSON.stringify(v.split).slice(1, -1));
    if (v.splitOnly) args.push('--split-only');
    if (v.sample) args.push('--sample', String(v.sample));
    py(args);
    const pyRecs = readJsonl<ChunkRec>(out);
    let same = 0;
    let chunkCount = 0;
    let errors = 0;
    const bad: string[] = [];
    for (const p of pyRecs) {
      const t = tsChunkRec(tok, p.doc, docs.get(p.doc)!.content, v);
      if (p.error) errors++;
      chunkCount += p.chunks?.length ?? 0;
      if (isDeepStrictEqual(t, p)) same++;
      else if (bad.length < 3) bad.push(firstDifference(p, t));
    }
    check(`chunk ${v.name}: TS == upstream for every doc`, same === pyRecs.length, `${same}/${pyRecs.length} docs, ${chunkCount} chunks, ${errors} upstream errors${bad.length ? `; ${bad.join(' | ')}` : ''}`);

    if (v.name.startsWith('default')) {
      // against the store itself: every stored chunk is reproduced
      const byDoc = new Map<string, Map<number, [string, StoredChunk]>>();
      for (const [id, c] of stored) {
        if (!c.full_doc_id) continue;
        const m = byDoc.get(c.full_doc_id) ?? new Map();
        m.set(c.chunk_order_index, [id, c]);
        byDoc.set(c.full_doc_id, m);
      }
      let docsWith = 0;
      let match = 0;
      for (const p of pyRecs) {
        const have = byDoc.get(p.doc);
        if (!have) continue;
        docsWith++;
        const ok = p.chunks?.length === have.size && p.chunks.every((c) => {
          const s = have.get(c.index);
          return s && s[0] === c.id && s[1].tokens === c.tokens && s[1].content === c.content;
        });
        if (ok) match++;
      }
      check('chunk default: TS reproduces kv_store_text_chunks (id, content, tokens)', match === docsWith, `${match}/${docsWith} docs with stored chunks, ${docs.size - docsWith} docs without`);
    }
  }
}

function firstDifference(p: ChunkRec, t: ChunkRec): string {
  if (p.tokens_md5 !== t.tokens_md5) return `${p.doc}: token ids differ (${p.n_tokens} vs ${t.n_tokens})`;
  if (p.error !== t.error) return `${p.doc}: error ${JSON.stringify(p.error)} vs ${JSON.stringify(t.error)}`;
  const n = Math.max(p.chunks?.length ?? 0, t.chunks?.length ?? 0);
  for (let i = 0; i < n; i++) {
    const a = p.chunks?.[i];
    const b = t.chunks?.[i];
    if (!isDeepStrictEqual(a, b)) return `${p.doc} chunk ${i}: py=${JSON.stringify(a)?.slice(0, 160)} ts=${JSON.stringify(b)?.slice(0, 160)}`;
  }
  return `${p.doc}: differs`;
}

// ---------------------------------------------------------------- extract

type ExtractRec = {
  chunk: string;
  nodes?: [string, Record<string, unknown>[]][];
  edges?: [[string, string], Record<string, unknown>[]][];
  llm_cache_list?: string[] | null;
  error?: string;
};

/** A read-only view of a loaded KV map, as LlmResponseCache / ChunkKv. */
function memKv(data: Map<string, Record<string, unknown>>, globalConfig = {}) {
  return {
    global_config: globalConfig,
    get_by_id: async (id: string) => data.get(id) ?? null,
    upsert: async (d: Record<string, Record<string, unknown>>) => {
      for (const [k, v] of Object.entries(d)) data.set(k, v);
    },
  };
}

export async function tsExtractRecs(
  chunkIds: string[],
  chunks: Map<string, Record<string, unknown>>,
  cache: Map<string, Record<string, unknown>>,
  language: string,
): Promise<ExtractRec[]> {
  const tok = new TiktokenTokenizer(values.model);
  const cacheKv = memKv(cache, { enable_llm_cache_for_entity_extract: true });
  const gc = {
    llm_model_func: async () => {
      throw new CacheMiss('cache miss: the parity run never calls an LLM');
    },
    entity_extract_max_gleaning: 1,
    addon_params: { language, entity_types: DEFAULT_ENTITY_TYPES },
    tokenizer: tok,
    max_extract_input_tokens: DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    llm_model_max_async: 4,
  };
  const out: ExtractRec[] = [];
  for (const cid of chunkIds) {
    const chunk = chunks.get(cid)!;
    const store = new Map([[cid, structuredClone(chunk)]]);
    const rec: ExtractRec = { chunk: cid };
    try {
      const [r] = await extractEntities(new Map([[cid, chunk as ExtractChunk]]), gc, { llmResponseCache: cacheKv, textChunksStorage: memKv(store) });
      rec.nodes = [...r!.nodes].map(([k, v]) => [k, v]);
      rec.edges = [...r!.edges].map(([k, v]) => [edgeKeyParts(k), v]);
      rec.llm_cache_list = (store.get(cid)!.llm_cache_list as string[] | undefined) ?? null;
    } catch (e) {
      rec.error = `${(e as Error).name}: ${(e as Error).message}`;
    }
    out.push(rec);
  }
  return out;
}

async function phaseExtract(): Promise<void> {
  const cache = readJsonKv<Record<string, unknown>>(kvStorePath(SRC!, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ''));
  const chunks = readJsonKv<Record<string, unknown>>(kvStorePath(SRC!, NameSpace.KV_STORE_TEXT_CHUNKS, ''));
  const withEntries = new Set<string>();
  for (const e of cache.values()) if (e.cache_type === 'extract' && typeof e.chunk_id === 'string') withEntries.add(e.chunk_id);
  log(`extract: ${chunks.size} chunks, ${withEntries.size} with extract cache entries, ${cache.size} cache entries`);
  const out = join(WORK, 'py-extract.jsonl');
  py(['extract', SRC!, '--out', out, '--model', values.model!]);
  const pyRecs = readJsonl<ExtractRec>(out);
  const tsRecs = await tsExtractRecs(pyRecs.map((r) => r.chunk), chunks, cache, 'English');

  let same = 0;
  let tsMissOnCached = 0;
  let pyMissOnCached = 0;
  let missOther = 0;
  let entities = 0;
  let relations = 0;
  const bad: string[] = [];
  pyRecs.forEach((p, i) => {
    const t = tsRecs[i]!;
    const cached = withEntries.has(p.chunk);
    if (cached && t.error) tsMissOnCached++;
    if (cached && p.error) pyMissOnCached++;
    if (!cached && t.error) missOther++;
    entities += p.nodes?.reduce((n, [, v]) => n + v.length, 0) ?? 0;
    relations += p.edges?.reduce((n, [, v]) => n + v.length, 0) ?? 0;
    if (isDeepStrictEqual(t, p)) same++;
    else if (bad.length < 3) bad.push(`${p.chunk}: py=${JSON.stringify(p).slice(0, 300)} ts=${JSON.stringify(t).slice(0, 300)}`);
  });
  check(
    `extract: zero cache misses on the ${withEntries.size} chunks with extract entries`,
    tsMissOnCached === 0 && pyMissOnCached === 0,
    `ts ${tsMissOnCached}, upstream ${pyMissOnCached}; chunks without entries miss on both sides: ${missOther}`,
  );
  check(
    'extract: parsed entities, relations, timestamps, llm_cache_list == upstream for every chunk',
    same === pyRecs.length,
    `${same}/${pyRecs.length} chunks, ${entities} entity records, ${relations} relation records${bad.length ? `; ${bad.join(' | ')}` : ''}`,
  );
}

// ---------------------------------------------------------------- parse (fuzz)

/** mulberry32: a seeded PRNG so the corpus is the same on every run. */
function prng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Damage an LLM answer the ways the upstream parser has a branch for, plus
// characters where Python and JS regex/str semantics part ways.
const DELIM_DAMAGE = ['<|##|>', '<|#||#|>', '<|\\#|>', '<|>', '<||>', '<X|#|>', '<|#|Y>', '<#>', '<#|>', '<|#>', '<a#|>', '<|#b>',
  '<|#|', '<|#||', '<|#:', '<||#>', '<||', '|#|>', '<|#|>|', '||#||', '<😀|#|>', '<\r|#|>', '< |#|>', '<|#|\n>'];
const INSERTS = ['\r', ' ', '\u0085', '\x1c', '﻿', ' ', ' ', '1 ', '　', '（', '）', '—', '－', 'Ｈｅｌｌｏ', '１２', '中 文',
  '中 A', 'A 中', '"中"', "'中'", '“q”', '‘q’', '《书》', '<p>', '</P >', '<br/>', '<BR >', '&amp;', '&#55357;', '\ud800', '١٢', '²', '1.2', '.5',
  '٣.٤', ',', '，', 'Σ', 'ΑΣ', 'İ', 'ß', '<think>x</think>', '<|COMPLETE|>', '<|complete|>', '<|#|>entity<|#|>', '<|#|>relation<|#|>',
  '<|#|>relationship<|#|>', '\n', '\n\n', '"', "'", '(', '|', '/', '\\', ' ', '\t', 'x'.repeat(300), '😀'.repeat(130), '0', '12', '123', '+3.5', '-0.25'];

function fuzzCorpus(base: readonly string[], n: number): string[] {
  const rnd = prng(20261021);
  const pick = <T>(a: readonly T[]) => a[Math.floor(rnd() * a.length)]!;
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    let s = pick(base);
    const lines = s.split('\n').slice(0, 1 + Math.floor(rnd() * 12));
    s = lines.join(pick(['\n', '\n', '<|#|>', '\r\n', '']));
    const edits = 1 + Math.floor(rnd() * 8);
    for (let k = 0; k < edits; k++) {
      const r = rnd();
      if (r < 0.35) s = s.replace('<|#|>', pick(DELIM_DAMAGE));
      else {
        const at = Math.floor(rnd() * (s.length + 1));
        s = s.slice(0, at) + pick(INSERTS) + s.slice(at);
      }
    }
    out.push(s);
  }
  return out;
}

async function phaseParse(): Promise<void> {
  const cache = readJsonKv<Record<string, unknown>>(kvStorePath(SRC!, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, ''));
  const corpus = fuzzCorpus([...cache.values()].filter((e) => e.cache_type === 'extract').map((e) => e.return as string), 20000);
  const inp = join(WORK, 'parse-corpus.jsonl');
  const out = join(WORK, 'py-parse.jsonl');
  writeFileSync(inp, corpus.map((result) => JSON.stringify({ result })).join('\n') + '\n');
  py(['parse', SRC!, '--in', inp, '--out', out]);
  const pyRecs = readJsonl<{ nodes: unknown; edges: unknown }>(out);
  let same = 0;
  let nonEmpty = 0;
  const bad: string[] = [];
  corpus.forEach((result, i) => {
    const r = processExtractionResult(result, 'chunk-x', 7, 'f.md');
    const t = { nodes: [...r.nodes].map(([k, v]) => [k, v]), edges: [...r.edges].map(([k, v]) => [edgeKeyParts(k), v]) };
    // JSON round trip: upstream's lines went through json.dumps/JSON.parse too (lone surrogates -> escapes)
    const tj = JSON.parse(JSON.stringify(t));
    if (r.nodes.size || r.edges.size) nonEmpty++;
    if (isDeepStrictEqual(tj, pyRecs[i])) same++;
    else if (bad.length < 3) bad.push(`#${i} input=${JSON.stringify(result).slice(0, 200)} py=${JSON.stringify(pyRecs[i]).slice(0, 400)} ts=${JSON.stringify(tj).slice(0, 400)}`);
  });
  check('parse: fuzzed extraction answers parse the same as upstream', same === corpus.length, `${same}/${corpus.length} (${nonEmpty} non-empty)${bad.length ? `; ${bad.join(' | ')}` : ''}`);
}

/**
 * test/fixtures/parse-cases.json: the same fuzzing over upstream's own prompt
 * examples (public text, unlike a store's cache) with upstream's parse recorded.
 */
function phaseFixture(): void {
  const examples = PROMPTS.entity_extraction_examples.map((e) =>
    pyFormat(e.slice(e.indexOf('<Output>') + '<Output>'.length), {
      tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
      completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    }).trim(),
  );
  const corpus = fuzzCorpus(examples, 600);
  const inp = join(WORK, 'fixture-corpus.jsonl');
  const out = join(WORK, 'py-fixture.jsonl');
  writeFileSync(inp, corpus.map((result) => JSON.stringify({ result })).join('\n') + '\n');
  py(['parse', SRC!, '--in', inp, '--out', out]);
  const pyRecs = readJsonl<{ nodes: unknown; edges: unknown }>(out);
  const file = resolve(import.meta.dirname, '../test/fixtures/parse-cases.json');
  writeFileSync(file, `[\n${corpus.map((result, i) => JSON.stringify({ result, ...pyRecs[i] })).join(',\n')}\n]\n`);
  log(`fixture: ${corpus.length} cases -> ${file}`);
}

// ---------------------------------------------------------------- merge

function selectMergeDocs(): MergeDoc[] {
  const status = readJsonKv<{ status: string; updated_at: string; file_path?: string; chunks_list?: string[] }>(
    kvStorePath(SRC!, NameSpace.DOC_STATUS, ''),
  );
  const chunks = readJsonKv<{ chunk_order_index: number }>(kvStorePath(SRC!, NameSpace.KV_STORE_TEXT_CHUNKS, ''));
  const want = values.select === 'processed' ? (s: string) => s === 'processed' : (s: string) => s !== 'processed';
  const docs: (MergeDoc & { at: string })[] = [];
  for (const [id, s] of status) {
    const list = s.chunks_list ?? [];
    if (!want(s.status) || !list.length || !list.every((c) => chunks.has(c))) continue;
    const ordered = [...list].sort((a, b) => chunks.get(a)!.chunk_order_index - chunks.get(b)!.chunk_order_index);
    docs.push({ id, file_path: s.file_path ?? 'unknown_source', chunks: ordered, at: s.updated_at });
  }
  docs.sort((a, b) => (a.at < b.at ? -1 : a.at > b.at ? 1 : a.id < b.id ? -1 : 1));
  const offset = Number(values.offset);
  return docs.slice(offset, offset + Number(values.docs)).map(({ at: _, ...d }) => d);
}

/** A fresh work dir holding the starting files (APFS clones of the store copy). */
function seedWorkDir(dir: string): void {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const seed = values.base === 'graph' ? MERGE_FILES.slice(0, 6) : MERGE_FILES.slice(0, 1);
  for (const f of seed) if (existsSync(join(SRC!, f))) execFileSync('cp', ['-c', join(SRC!, f), join(dir, f)]);
}

function compareMergeDirs(label: string, py: string, ts: string, opts: { cacheAsMap?: boolean } = {}): void {
  for (const [f, diff] of mergeDirDiffs(py, ts, { vdb: values.vdb!, ...opts })) {
    const size = existsSync(join(ts, f)) ? statSync(join(ts, f)).size : 0;
    check(`${label}: ${f}${opts.cacheAsMap && f.includes('cache') ? ' (as a map)' : ''}`, diff === null, diff ?? `${size} bytes identical`);
  }
}

const mergeOpts = () => ({ vdb: values.vdb!, dim: Number(values.dim), gc: JSON.parse(values.gc!) as Partial<MergeConfig>, model: values.model! });

/** Summary calls answered by the store's own cache (TS rebuilt upstream's historical prompt) vs sent to the stub. */
function mergeStats(tsDir: string, summaryHits: Set<string>, recs: MergeRec[]): void {
  const cacheAfter = readJsonKv<{ cache_type?: string }>(join(tsDir, 'kv_store_llm_response_cache.json'));
  const cacheBefore = readJsonKv<{ cache_type?: string }>(join(SRC!, 'kv_store_llm_response_cache.json'));
  let stub = 0;
  let stored = 0;
  for (const [k, e] of cacheAfter) if (e.cache_type === 'summary' && !cacheBefore.has(k)) stub++;
  for (const e of cacheBefore.values()) if (e.cache_type === 'summary') stored++;
  const storeHits = [...summaryHits].filter((k) => cacheBefore.has(k)).length;
  const llmMerges = recs.reduce((n, r) => n + (r.messages?.filter((m) => m.startsWith('LLMmrg')).length ?? 0), 0);
  log(`merge: ${llmMerges} LLM merges; summary calls: ${storeHits} distinct store entries hit (of ${stored} summary entries in the store), ${stub} went to the stub`);
}

async function phaseMerge(): Promise<void> {
  const docs = selectMergeDocs();
  const gcTag = values.gc === '{}' ? '' : `-gc${createHash('md5').update(values.gc!).digest('hex').slice(0, 6)}`;
  const base = join(WORK, `merge-${values.select}-${values.base}-${values.offset}-${docs.length}${values.vdb ? '-vdb' : ''}${gcTag}`);
  mkdirSync(base, { recursive: true });
  const docsFile = join(base, 'docs.json');
  // --reuse-py: upstream's half of an earlier run in this work dir, if it merged exactly these docs
  const reuse = values['reuse-py'] && existsSync(join(base, 'py-merge.jsonl')) && existsSync(docsFile) && readFileSync(docsFile, 'utf8') === JSON.stringify(docs);
  if (values['reuse-py'] && !reuse) throw new Error(`--reuse-py: no earlier upstream run of the same docs under ${base}`);
  writeFileSync(docsFile, JSON.stringify(docs));
  const nChunks = docs.reduce((n, d) => n + (d.chunks?.length ?? 0), 0);
  log(
    `merge: ${docs.length} ${values.select} docs (${nChunks} chunks) onto ${values.base === 'graph' ? 'the store graph' : 'an empty store'}${values.vdb ? `, vdb dim ${values.dim}` : ', no vdb'}${values.gc === '{}' ? '' : `, config ${values.gc}`}`,
  );

  const pyDir = join(base, 'py');
  const pyArgs = ['merge', SRC!, '--work', pyDir, '--docs', docsFile, '--out', join(base, 'py-merge.jsonl'), '--dim', values.dim!, '--gc', values.gc!, '--sequential'];
  if (values.vdb) pyArgs.push('--vdb');
  if (reuse) log('merge: upstream side reused from the earlier run (same docs.json)');
  else if (!values['ts-only']) {
    seedWorkDir(pyDir);
    py(pyArgs);
    log('merge: upstream done');
  }

  const chunks = readJsonKv<Record<string, unknown>>(kvStorePath(SRC!, NameSpace.KV_STORE_TEXT_CHUNKS, ''));
  const tsDir = join(base, 'ts');
  seedWorkDir(tsDir);
  const summaryHits = new Set<string>();
  const tsRecs = await tsMergeDocs(tsDir, docs, {
    ...mergeOpts(),
    chunks,
    maxConcurrency: 1,
    onCacheHit: (k) => k.startsWith('default:summary:') && summaryHits.add(k),
  });
  log('merge: TS done');
  if (values['ts-only']) return mergeStats(tsDir, summaryHits, tsRecs);
  const pyRecs = readJsonl<MergeRec>(join(base, 'py-merge.jsonl'));
  const merged = pyRecs.filter((r) => !r.skipped && !r.error).length;
  const skipped = pyRecs.filter((r) => r.skipped).length;
  const errors = pyRecs.filter((r) => r.error).length;
  const msgs = pyRecs.reduce((n, r) => n + (r.messages?.length ?? 0), 0);
  const llmMerges = pyRecs.reduce((n, r) => n + (r.messages?.filter((m) => m.startsWith('LLMmrg')).length ?? 0), 0);
  let same = 0;
  const bad: string[] = [];
  pyRecs.forEach((p, i) => {
    if (isDeepStrictEqual(p, tsRecs[i])) same++;
    else if (bad.length < 2) bad.push(`py=${JSON.stringify(p).slice(0, 600)} ts=${JSON.stringify(tsRecs[i]).slice(0, 600)}`);
  });
  check(
    'merge: per-doc outcome and pipeline messages == upstream',
    same === pyRecs.length,
    `${same}/${pyRecs.length} docs (${merged} merged, ${skipped} skipped on a cache miss, ${errors} failed; ${msgs} messages, ${llmMerges} LLM merges)${bad.length ? `; ${bad.join(' | ')}` : ''}`,
  );
  compareMergeDirs('merge sequential', pyDir, tsDir);

  mergeStats(tsDir, summaryHits, tsRecs);

  if (values['concurrency-check']) {
    const ts8 = join(base, 'ts-concurrent');
    seedWorkDir(ts8);
    const recs8 = await tsMergeDocs(ts8, docs, { ...mergeOpts(), chunks });
    check('merge TS concurrent: per-doc outcome and messages == sequential', isDeepStrictEqual(recs8, tsRecs));
    compareMergeDirs('merge TS concurrent vs upstream sequential', pyDir, ts8, { cacheAsMap: true });
  }
  if (values['py-concurrent']) {
    const pyc = join(base, 'py-concurrent');
    seedWorkDir(pyc);
    const args = pyArgs
      .filter((a) => a !== '--sequential')
      .map((a) => (a === pyDir ? pyc : a === join(base, 'py-merge.jsonl') ? join(base, 'pyc-merge.jsonl') : a));
    py(args);
    const g1 = readGraphml(join(pyDir, 'graph_chunk_entity_relation.graphml'));
    const g2 = readGraphml(join(pyc, 'graph_chunk_entity_relation.graphml'));
    let nodeDiff = 0;
    for (const [id, a] of g1.nodes) if (!isDeepStrictEqual(a, g2.nodes.get(id))) nodeDiff++;
    const edgeMap = (g: GraphmlGraph) => new Map([...g.edges.values()].map((e) => [[e.source, e.target].sort().join('\u0000'), e.attrs]));
    const e1 = edgeMap(g1);
    const e2 = edgeMap(g2);
    let edgeDiff = 0;
    for (const [k, a] of e1) if (!isDeepStrictEqual(a, e2.get(k))) edgeDiff++;
    const sameBytes = fileDiff(join(pyDir, 'graph_chunk_entity_relation.graphml'), join(pyc, 'graph_chunk_entity_relation.graphml')) === null;
    log(
      `merge: upstream concurrent vs sequential: graph bytes ${sameBytes ? 'identical' : 'differ'}; ${nodeDiff}/${g1.nodes.size} nodes and ${edgeDiff}/${e1.size} edges differ in attributes (${g2.nodes.size} nodes, ${e2.size} edges)`,
    );
  }
}

/**
 * test/fixtures/merge-cases: upstream's merge of public input - the parsed
 * extraction answers in parse-cases.json (fuzzed prompt examples), grouped
 * into documents - under a few configs, recorded for test/merge.test.ts.
 */
function phaseMergeFixture(): void {
  type Case = { nodes: [string, Record<string, unknown>[]][]; edges: [[string, string], Record<string, unknown>[]][] };
  const cases = (JSON.parse(readFileSync(resolve(import.meta.dirname, '../test/fixtures/parse-cases.json'), 'utf8')) as Case[]).filter(
    (c) => c.nodes.length || c.edges.length,
  );
  const docs: MergeDoc[] = [];
  const perDoc = 4;
  for (let d = 0; d * perDoc < Math.min(cases.length, 160); d++) {
    const id = `doc-${String(d).padStart(3, '0')}`;
    const results = cases.slice(d * perDoc, (d + 1) * perDoc).map((c, j) => {
      const i = d * perDoc + j;
      const chunk = `chunk-${String(i).padStart(4, '0')}`;
      // pairs of chunks share a timestamp, so the (timestamp, -length) order meets ties
      const stamp = (r: Record<string, unknown>) => ({ ...r, source_id: chunk, file_path: `${id}.md`, timestamp: 1000 + Math.floor(i / 2) });
      return { nodes: c.nodes.map(([k, v]) => [k, v.map(stamp)]), edges: c.edges.map(([k, v]) => [k, v.map(stamp)]) };
    });
    docs.push({ id, file_path: `${id}.md`, results: results as NonNullable<MergeDoc['results']> });
  }
  const root = resolve(import.meta.dirname, '../test/fixtures/merge-cases');
  rmSync(root, { recursive: true, force: true });
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, 'docs.json'), JSON.stringify(docs) + '\n');
  const variants: Record<string, Record<string, unknown>> = {
    default: {},
    'keep-small': {
      source_ids_limit_method: 'KEEP',
      max_source_ids_per_entity: 3,
      max_source_ids_per_relation: 2,
      max_file_paths: 2,
      force_llm_summary_on_merge: 3,
      summary_max_tokens: 100,
      summary_context_size: 150,
    },
    'fifo-small': {
      source_ids_limit_method: 'FIFO',
      max_source_ids_per_entity: 3,
      max_source_ids_per_relation: 2,
      max_file_paths: 2,
      force_llm_summary_on_merge: 4,
      summary_max_tokens: 120,
      summary_context_size: 200,
    },
  };
  for (const [name, gc] of Object.entries(variants)) {
    const dir = join(root, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'gc.json'), JSON.stringify(gc) + '\n');
    py(['merge', SRC!, '--work', dir, '--docs', join(root, 'docs.json'), '--out', join(dir, 'messages.jsonl'), '--dim', '4', '--gc', JSON.stringify(gc), '--sequential', '--vdb']);
    const recs = readJsonl<MergeRec>(join(dir, 'messages.jsonl'));
    const llm = recs.reduce((n, r) => n + (r.messages?.filter((m) => m.startsWith('LLMmrg')).length ?? 0), 0);
    log(`merge-fixture ${name}: ${recs.length} docs, ${llm} LLM merges, errors ${recs.filter((r) => r.error).length} -> ${dir}`);
  }
}

if (PHASES.has('merge')) await phaseMerge();
if (PHASES.has('merge-fixture')) phaseMergeFixture();
if (PHASES.has('fixture')) phaseFixture();
if (PHASES.has('chunk')) phaseChunk();
if (PHASES.has('extract')) await phaseExtract();
if (PHASES.has('parse')) await phaseParse();
log(failures ? `${failures} FAILED` : 'all operate parity checks passed');
process.exit(failures ? 1 : 0);
