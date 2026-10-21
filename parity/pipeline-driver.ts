// The TS side of the pipeline comparisons (B4), shared by parity/pipeline.ts
// (a store copy) and test/lightrag.test.ts (a synthetic scenario). Mirrors
// tools/upstream/pipeline_parity.py: the same spec, the same stand-ins for the
// LLM and the embedding model, the same fixed clock.
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import type { EmbeddingFunc } from '../src/base.ts';
import { makeQueryParam } from '../src/base.ts';
import { getPipelineStatus, LightRAG, type LightRAGInit, type LightRAGLlmFunc } from '../src/lightrag.ts';
import { pyJsonDumps } from '../src/utils.ts';

export const FIXED_NOW = 1760000000;

export type Step = { op: string; [k: string]: unknown };
export type PipelineSpec = {
  working_dir: string;
  workspace?: string;
  dim: number;
  /** "synthetic": the stub answers every prompt; "cache": a store replay, extraction and keywords must hit the cache */
  llm: 'synthetic' | 'cache';
  config?: Record<string, unknown>;
  steps: Step[];
};
export type StepRecord = { i: number; op: string; result: unknown; error: string | null; history: string[] };

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;
const md5 = (s: string) => createHash('md5').update(Buffer.from(s.replace(LONE_SURROGATE, '?'), 'utf8'));
const md5Hex = (s: string) => md5(s).digest('hex');
const md5Bytes = (s: string) => md5(s).digest();

/** stub_vector(): md5 bytes scaled to [-1, 1), md5 of `${i}:${text}` past 16 dims. */
export function stubVector(text: string, dim: number): Float32Array {
  const blocks: Buffer[] = [];
  for (let i = 0; i < Math.ceil(dim / 16); i++) blocks.push(md5Bytes(i === 0 ? text : `${i}:${text}`));
  const raw = Buffer.concat(blocks);
  return Float32Array.from({ length: dim }, (_, j) => (raw[j]! - 128) / 128);
}

export function stubEmbedding(dim: number): EmbeddingFunc {
  return Object.assign(async (texts: string[]) => texts.map((t) => stubVector(t, dim)), { embedding_dim: dim });
}

const SUMMARY_HEAD = '---Role---\nYou are a Knowledge Graph Specialist, proficient in data curation and synthesis.';
const VOCAB = ['Alice Moreau', 'Bob Tanaka', 'Carol Diaz', 'Acme Corp', 'Globex', 'Paris', 'Berlin', 'Quantum Engine', 'Project Atlas', 'Red River', 'Old Library', 'Night Train'];
const TYPES = ['person', 'organization', 'location', 'artifact', 'concept'];
const KWS = ['collaboration', 'conflict', 'travel', 'ownership', 'research', 'funding'];

export class CacheMiss extends Error {
  override name = 'CacheMiss';
}

/** synth_extraction(): 2-4 entities (1 when gleaning) from a fixed vocabulary, chained by relations. */
export function synthExtraction(prompt: string, glean: boolean): string {
  const d = md5Bytes(prompt);
  const h = md5Hex(prompt);
  const n = glean ? 1 : 2 + (d[0]! % 3);
  const names: string[] = [];
  for (let i = 0; i < 6; i++) {
    const nm = VOCAB[d[1 + i]! % VOCAB.length]!;
    if (!names.includes(nm)) names.push(nm);
    if (names.length === n) break;
  }
  const lines: string[] = [];
  names.forEach((nm, i) => {
    const t = TYPES[d[7 + i]! % TYPES.length]!;
    lines.push(`entity<|#|>${nm}<|#|>${t}<|#|>${nm} is mentioned in passage ${h.slice(i * 4, i * 4 + 8)}${glean ? ' with more detail' : ''}.`);
  });
  for (let i = 0; i + 1 < names.length; i++) {
    const [a, b] = [names[i]!, names[i + 1]!];
    lines.push(`relation<|#|>${a}<|#|>${b}<|#|>${KWS[d[12]! % KWS.length]}, ${KWS[d[13]! % KWS.length]}<|#|>${a} and ${b} are linked in passage ${h.slice(0, 8)}.`);
  }
  lines.push('<|COMPLETE|>');
  return lines.join('\n');
}

export function synthKeywords(prompt: string): string {
  const d = md5Bytes(prompt);
  return pyJsonDumps({ high_level_keywords: [KWS[d[0]! % 6], KWS[d[1]! % 6]], low_level_keywords: [VOCAB[d[2]! % 12], VOCAB[d[3]! % 12]] });
}

/** make_llm() */
export function makeLlm(kind: 'synthetic' | 'cache'): LightRAGLlmFunc {
  return async (prompt, o = {}) => {
    if (prompt.startsWith(SUMMARY_HEAD)) return `Stub summary ${md5Hex(prompt)}`;
    if (o.keyword_extraction) {
      if (kind === 'synthetic') return synthKeywords(prompt);
      throw new CacheMiss('cache miss: keywords');
    }
    const system = (o.system_prompt as string | null | undefined) ?? null;
    if (system && system.includes('<|#|>')) {
      if (kind === 'synthetic') return synthExtraction(prompt, Array.isArray(o.history_messages) && o.history_messages.length > 0);
      throw new CacheMiss('cache miss: the parity run never calls an LLM');
    }
    return `STUB ${md5Hex((system ?? '') + '\n' + prompt)}`;
  };
}

/** jsonable(): what a step returned, as the Python side writes it (response iterators dropped, Maps as objects). */
export function jsonable(v: unknown): unknown {
  if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [String(k), jsonable(x)]));
  if (Array.isArray(v)) return v.map(jsonable);
  if (v instanceof Float32Array || v instanceof Float64Array) return Array.from(v);
  if (v && typeof v === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, x] of Object.entries(v)) if (k !== 'response_iterator') out[k] = jsonable(x);
    return out;
  }
  if (typeof v === 'function' || v === undefined) return null;
  return v;
}

const pyErr = (e: unknown) => {
  const err = e as Error;
  return `${err?.name === 'Error' ? 'Exception' : (err?.name ?? 'Exception')}: ${err?.message ?? String(e)}`;
};

export type RunOptions = {
  /** override of max_parallel_insert (upstream's side always runs with 1) */
  maxParallelInsert?: number;
  /** wrap the rag right after construction (cache-hit recording, persistence stubs) */
  onRag?: (rag: LightRAG) => void | Promise<void>;
};

/** Run a spec through the TS LightRAG; one record per step. */
export async function runSpecTs(spec: PipelineSpec, o: RunOptions = {}): Promise<StepRecord[]> {
  const init: LightRAGInit = {
    ...(spec.config ?? {}),
    working_dir: spec.working_dir,
    workspace: spec.workspace ?? '',
    embedding_func: stubEmbedding(spec.dim),
    llm_model_func: makeLlm(spec.llm),
    now: () => FIXED_NOW,
  };
  if (o.maxParallelInsert !== undefined) init.max_parallel_insert = o.maxParallelInsert;
  const rag = new LightRAG(init);
  await o.onRag?.(rag);
  await rag.initialize_storages();
  const status = getPipelineStatus(rag.workspace);
  const records: StepRecord[] = [];
  for (const [i, s] of spec.steps.entries()) {
    const rec: StepRecord = { i, op: s.op, result: null, error: null, history: [] };
    status.history_messages.length = 0;
    const a = s as Record<string, any>;
    try {
      let r: unknown;
      switch (s.op) {
        case 'insert':
          r = await rag.ainsert(a.docs, a.split_by_character ?? null, a.split_by_character_only ?? false, a.ids ?? null, a.file_paths ?? null, a.track_id);
          break;
        case 'enqueue':
          r = await rag.apipeline_enqueue_documents(a.docs, a.ids ?? null, a.file_paths ?? null, a.track_id);
          break;
        case 'process':
          r = await rag.apipeline_process_enqueue_documents(a.split_by_character ?? null, a.split_by_character_only ?? false);
          break;
        case 'error_docs':
          r = await rag.apipeline_enqueue_error_documents(a.files, a.track_id);
          break;
        case 'custom_kg':
          r = await rag.ainsert_custom_kg(a.kg, a.full_doc_id ?? null);
          break;
        case 'delete':
          r = await rag.adelete_by_doc_id(a.doc_id, a.delete_llm_cache ?? false);
          break;
        case 'query':
          r = await rag.aquery_llm(a.query, makeQueryParam(a.param ?? {}), a.system_prompt ?? null);
          break;
        case 'aquery':
          r = await rag.aquery(a.query, makeQueryParam(a.param ?? {}), a.system_prompt ?? null);
          break;
        case 'query_data':
          r = await rag.aquery_data(a.query, makeQueryParam(a.param ?? {}));
          break;
        case 'delete_entity':
          r = await rag.adelete_by_entity(a.name);
          break;
        case 'delete_relation':
          r = await rag.adelete_by_relation(a.src, a.tgt);
          break;
        case 'edit_entity':
          r = await rag.aedit_entity(a.name, a.data, a.allow_rename ?? true, a.allow_merge ?? false);
          break;
        case 'edit_relation':
          r = await rag.aedit_relation(a.src, a.tgt, a.data);
          break;
        case 'create_entity':
          r = await rag.acreate_entity(a.name, a.data);
          break;
        case 'create_relation':
          r = await rag.acreate_relation(a.src, a.tgt, a.data);
          break;
        case 'merge_entities':
          r = await rag.amerge_entities(a.sources, a.target, a.strategy ?? null, a.target_data ?? null);
          break;
        case 'entity_info':
          r = await rag.get_entity_info(a.name, a.include_vector_data ?? false);
          break;
        case 'relation_info':
          r = await rag.get_relation_info(a.src, a.tgt, a.include_vector_data ?? false);
          break;
        case 'counts':
          r = await rag.get_processing_status();
          break;
        default:
          throw new Error(`unknown op ${s.op}`);
      }
      rec.result = jsonable(r);
    } catch (e) {
      rec.error = pyErr(e);
    }
    rec.history = [...status.history_messages];
    records.push(rec);
  }
  await rag.finalize_storages();
  return records;
}

/** Tracebacks (Python's format_exc, TS's stack) cannot match: both drop out of the comparison. */
const isTraceback = (m: string) => m.startsWith('Traceback (most recent call last)') || /\n {4}at /.test(m);

/** Differences between TS and upstream records: results as JSON values (key order included), history line by line. */
export function diffStepRecords(ts: StepRecord[], py: StepRecord[], o: { history?: boolean; asyncLimit?: boolean } = {}): string[] {
  const out: string[] = [];
  if (ts.length !== py.length) out.push(`records: ts=${ts.length} py=${py.length}`);
  const canon = (v: unknown) => JSON.stringify(JSON.parse(JSON.stringify(v ?? null)));
  for (let i = 0; i < Math.min(ts.length, py.length); i++) {
    const a = ts[i]!;
    const b = py[i]!;
    const tag = `step ${i} (${a.op})`;
    if (a.error !== b.error) out.push(`${tag} error: ts=${a.error} py=${b.error}`);
    const ra = canon(a.result);
    const rb = canon(b.result);
    if (ra !== rb) out.push(`${tag} result: ${firstDiff(ra, rb)}`);
    // a run made sequential through llm_model_max_async prints its own limit
    const norm = (h: string[]) => h.filter((m) => !isTraceback(m)).map((m) => (o.asyncLimit ? m.replace(/\(async: \d+\)/, '(async: N)') : m));
    const ha = norm(a.history);
    const hb = norm(b.history);
    if ((o.history ?? true) && !isDeepStrictEqual(ha, hb)) {
      let k = 0;
      while (k < ha.length && k < hb.length && ha[k] === hb[k]) k++;
      out.push(`${tag} history (${ha.length} vs ${hb.length} lines) first difference at ${k}: ts=${JSON.stringify(ha[k])} py=${JSON.stringify(hb[k])}`);
    }
  }
  return out;
}

export function firstDiff(a: string, b: string): string {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return `at ${i}: ts=${JSON.stringify(a.slice(Math.max(0, i - 60), i + 80))} py=${JSON.stringify(b.slice(Math.max(0, i - 60), i + 80))}`;
}

export function readStepRecords(file: string): StepRecord[] {
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as StepRecord);
}

/** Every file under two store dirs, compared byte for byte: [relative path, difference | null]. */
export function dirDiffs(expected: string, actual: string): [string, string | null][] {
  const files = new Set<string>();
  const walk = (root: string, rel = '') => {
    if (!existsSync(join(root, rel))) return;
    for (const f of readdirSync(join(root, rel))) {
      const r = rel ? join(rel, f) : f;
      if (statSync(join(root, r)).isDirectory()) walk(root, r);
      else files.add(r);
    }
  };
  walk(expected);
  walk(actual);
  const out: [string, string | null][] = [];
  for (const f of [...files].sort()) {
    const a = join(expected, f);
    const b = join(actual, f);
    if (!existsSync(a) || !existsSync(b)) {
      out.push([f, `${existsSync(a) ? 'ts' : 'upstream'} side missing`]);
      continue;
    }
    const x = readFileSync(a);
    const y = readFileSync(b);
    if (x.equals(y)) {
      out.push([f, null]);
      continue;
    }
    let i = 0;
    while (i < x.length && i < y.length && x[i] === y[i]) i++;
    const ctx = (buf: Buffer) => JSON.stringify(buf.subarray(Math.max(0, i - 100), i + 150).toString('utf8'));
    out.push([f, `${x.length} vs ${y.length} bytes, first difference at ${i}: upstream=${ctx(x)} ts=${ctx(y)}`]);
  }
  return out;
}

// ---------------------------------------------------------------- the synthetic scenario

const WORDS = ['river', 'archive', 'engine', 'harbor', 'signal', 'meadow', 'lantern', 'circuit', 'quarry', 'orchard', 'beacon', 'canyon', 'ledger', 'mosaic', 'tundra', 'violet'];

/** Document k of the synthetic corpus: a few sentences, deterministic, long enough to chunk at 80 tokens. */
export function syntheticDoc(k: number, sentences = 6 + (k % 5) * 3): string {
  const out: string[] = [];
  for (let s = 0; s < sentences; s++) {
    const w = (j: number) => WORDS[(k * 7 + s * 3 + j) % WORDS.length]!;
    out.push(`Record ${k}.${s}: the ${w(0)} near the ${w(1)} kept the ${w(2)} and the ${w(3)} for ${k + s} seasons, as noted ${s % 2 ? 'twice' : 'once'} in the ${w(4)} log.`);
  }
  return out.join(' ');
}

/**
 * Insert (fresh, duplicate, custom ids, an empty document), error records,
 * queries in every mode (bypass, an unknown mode, aquery_data, aquery),
 * deletions that rebuild shared entities and relations, the editing API,
 * a custom KG, and deletions after the edits. Small chunks and low caps so the
 * summaries, source-id and file-path limits all come into play.
 */
export function syntheticSpec(workingDir: string, extra: Partial<PipelineSpec> = {}): PipelineSpec {
  const docs = Array.from({ length: 14 }, (_, k) => syntheticDoc(k));
  const paths = docs.map((_, k) => `notes/doc-${k}.md`);
  const queries = (tag: string): Step[] => [
    ...['mix', 'local', 'global', 'hybrid', 'naive'].map((mode) => ({ op: 'query', query: `What did the archive keep? ${tag}`, param: { mode } })),
    { op: 'query', query: `Who works with Globex? ${tag}`, param: { mode: 'mix', only_need_context: true } },
    { op: 'query', query: `Who works with Globex? ${tag}`, param: { mode: 'local', only_need_prompt: true } },
    { op: 'query', query: `Plain question ${tag}`, param: { mode: 'bypass' }, system_prompt: 'Answer briefly.' },
    { op: 'query', query: `Unknown mode ${tag}`, param: { mode: 'weird' } },
    { op: 'query_data', query: `What did the archive keep? ${tag}`, param: { mode: 'hybrid' } },
    { op: 'query_data', query: `naive data ${tag}`, param: { mode: 'naive', chunk_top_k: 3 } },
    { op: 'query_data', query: `bypass data ${tag}`, param: { mode: 'bypass' } },
    { op: 'aquery', query: `Short answer about Paris ${tag}`, param: { mode: 'global', top_k: 5 } },
    { op: 'query', query: `Tell me about Berlin ${tag}`, param: { mode: 'mix', hl_keywords: ['travel'], ll_keywords: ['Berlin'] } },
  ];
  return {
    working_dir: workingDir,
    workspace: 'ws',
    dim: 8,
    llm: 'synthetic',
    config: {
      chunk_token_size: 80,
      chunk_overlap_token_size: 10,
      force_llm_summary_on_merge: 3,
      max_source_ids_per_entity: 6,
      max_source_ids_per_relation: 4,
      max_file_paths: 3,
      summary_max_tokens: 200,
      addon_params: { language: 'English' },
    },
    steps: [
      { op: 'insert', docs: docs.slice(0, 6), file_paths: paths.slice(0, 6), track_id: 'track-1' },
      { op: 'counts' },
      { op: 'insert', docs: [docs[3]!, ...docs.slice(6, 9)], file_paths: [paths[3]!, ...paths.slice(6, 9)], track_id: 'track-2' },
      { op: 'insert', docs: docs.slice(9, 11), ids: ['custom-a', 'custom-b'], file_paths: ['  ', paths[10]!], track_id: 'track-3' },
      { op: 'enqueue', docs: [docs[11]!], file_paths: [paths[11]!], track_id: 'track-4' },
      { op: 'insert', docs: ['   '], track_id: 'track-5' },
      { op: 'error_docs', files: [{ file_path: 'broken.pdf', error_description: 'PDF parse failed', original_error: 'EOF marker not found', file_size: 1234 }], track_id: 'track-6' },
      ...queries('before'),
      { op: 'delete', doc_id: computeDocId(docs[1]!) },
      { op: 'delete', doc_id: computeDocId(docs[4]!), delete_llm_cache: true },
      { op: 'delete', doc_id: 'doc-does-not-exist' },
      { op: 'delete', doc_id: 'custom-a' },
      ...queries('after-delete'),
      { op: 'entity_info', name: 'Globex', include_vector_data: true },
      { op: 'relation_info', src: 'Bob Tanaka', tgt: 'Alice Moreau', include_vector_data: true },
      { op: 'edit_entity', name: 'Paris', data: { description: 'Capital of France, edited.', entity_type: 'location' } },
      { op: 'edit_entity', name: 'Berlin', data: { entity_name: 'Berlin City', description: 'Renamed Berlin.' } },
      { op: 'edit_entity', name: 'Night Train', data: { description: '' } },
      { op: 'create_entity', name: 'Zed Lab', data: { description: 'A new lab.', entity_type: 'organization', source_id: 'chunk-manual' } },
      { op: 'create_entity', name: 'Zed Lab', data: { description: 'Again.' } },
      { op: 'create_relation', src: 'Zed Lab', tgt: 'Globex', data: { description: 'Zed Lab partners with Globex.', keywords: 'partnership', weight: 2 } },
      { op: 'edit_relation', src: 'Globex', tgt: 'Zed Lab', data: { description: 'Edited partnership.', keywords: 'partnership, funding' } },
      { op: 'merge_entities', sources: ['Red River', 'Old Library'], target: 'Riverside Library' },
      { op: 'edit_entity', name: 'Carol Diaz', data: { entity_name: 'Acme Corp' }, allow_merge: true },
      { op: 'delete_relation', src: 'Zed Lab', tgt: 'Globex' },
      { op: 'delete_relation', src: 'Nobody', tgt: 'Globex' },
      { op: 'delete_entity', name: 'Quantum Engine' },
      { op: 'delete_entity', name: 'Nobody' },
      {
        op: 'custom_kg',
        kg: {
          chunks: [{ content: 'Custom chunk about Globex and Zed Lab.', source_id: 'src-1', file_path: 'custom.md' }],
          entities: [{ entity_name: 'Zed Lab', entity_type: 'organization', description: 'Custom description.', source_id: 'src-1' }],
          relationships: [{ src_id: 'Zed Lab', tgt_id: 'New Partner', description: 'Custom relation.', keywords: 'custom', source_id: 'src-1', weight: 1.5 }],
        },
        full_doc_id: 'custom-doc',
      },
      { op: 'delete', doc_id: computeDocId(docs[0]!), delete_llm_cache: true },
      { op: 'insert', docs: [docs[1]!, docs[12]!, docs[13]!], file_paths: [paths[1]!, paths[12]!, paths[13]!], track_id: 'track-7' },
      ...queries('end'),
      { op: 'counts' },
    ],
    ...extra,
  };
}

const computeDocId = (content: string) => `doc-${md5Hex(content)}`;

/** What test/lightrag.test.ts checks a run against: sha256 of every file and of every step's record. */
export type FixtureDigest = {
  note: string;
  files: Record<string, { bytes: number; sha256: string }>;
  steps: { op: string; error: string | null; result: string; history: string }[];
};

const sha256 = (b: Buffer | string) => createHash('sha256').update(b).digest('hex');
const canonJson = (v: unknown) => JSON.stringify(JSON.parse(JSON.stringify(v ?? null)));

export function fixtureDigest(storeDir: string, records: StepRecord[]): FixtureDigest {
  const files: FixtureDigest['files'] = {};
  for (const [f] of dirDiffs(storeDir, storeDir)) {
    const b = readFileSync(join(storeDir, f));
    files[f] = { bytes: b.length, sha256: sha256(b) };
  }
  return {
    note: 'Upstream LightRAG 1.4.16 on parity/pipeline-driver.ts syntheticSpec (node parity/pipeline.ts synthetic --write-fixture <this file>): file and record digests.',
    files,
    steps: records.map((r) => ({
      op: r.op,
      error: r.error,
      result: sha256(canonJson(r.result)),
      history: sha256(canonJson(r.history.filter((m) => !isTraceback(m)))),
    })),
  };
}
