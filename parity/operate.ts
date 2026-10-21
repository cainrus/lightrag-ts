// B2 operate parity: the TS chunking / extraction / merge against upstream's, on a copy of a store.
//
//   node parity/operate.ts <store-dir> [--work <dir>] [--phase chunk,...]
//
// The store is only read; upstream writes its JSON Lines under <work>.
//   chunk  chunking_by_token_size for every doc in full_docs: token ids (md5),
//          chunk ids, tokens, content equal upstream's for the same doc; with the
//          default settings also equal to what kv_store_text_chunks holds.
//          A few sampled variants (small windows, split_by_character, -only)
//          exercise the code paths the store's settings never reach.
// Exit code 1 on any mismatch. Upstream side: tools/upstream/operate_parity.py in the rag-venv.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { isDeepStrictEqual } from 'node:util';
import { kvStorePath, readJsonKv } from '../src/kg/json_kv_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import { chunkingByTokenSize } from '../src/operate.ts';
import { computeMdhashId, TiktokenTokenizer } from '../src/utils.ts';

const PY = process.env.LIGHTRAG_PY ?? join(process.env.HOME!, '.claude-home/rag-venv/bin/python');
const PY_TOOL = resolve(import.meta.dirname, '../tools/upstream/operate_parity.py');

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    work: { type: 'string' },
    phase: { type: 'string', default: 'chunk' },
    model: { type: 'string', default: 'gpt-4o-mini' },
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

if (PHASES.has('chunk')) phaseChunk();
log(failures ? `${failures} FAILED` : 'all operate parity checks passed');
process.exit(failures ? 1 : 0);
