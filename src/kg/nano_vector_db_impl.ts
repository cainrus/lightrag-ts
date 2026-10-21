// Port of lightrag/kg/nano_vector_db_impl.py + nano-vectordb 0.0.4.3, whose
// file format it persists:
//   {"embedding_dim": N, "data": [{"__id__", "__created_at__", ...meta, "vector"}], "matrix": "<b64>"}
// - data[i].vector: base64(zlib(float16[N])) of the raw embedding, written by
//   LightRAG (nano_vector_db_impl.py upsert); kept here as a Buffer of the
//   compressed bytes, so a 900 MB store does not hold 300 MB of base64 strings
// - matrix: base64(float32[rows*N]) little-endian, rows L2-normalized; row i is data[i]
//
// Single process: upstream's cross-process reload flags are not ported (SCOPE.md).
import { existsSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync, inflateSync } from 'node:zlib';
import type { BaseVectorStorage, DropResult, EmbeddingFunc, GlobalConfig, KvRecord, StorageOptions, VectorQueryResult } from '../base.ts';
import { workspaceDir } from '../base.ts';
import { ByteReader } from '../internal/json_stream.ts';
import { normalizeF32, normalizeF64, normalizeRowsF32 } from '../internal/numpy.ts';
import { PyJsonWriter, atomicWrite } from '../internal/py_json.ts';
import { computeMdhashId } from '../utils.ts';

export type VdbRecord = { __id__: string; __created_at__?: number; vector?: string | Buffer; [k: string]: unknown };

export type NanoVdbFile = {
  embeddingDim: number;
  data: VdbRecord[];
  /** rows x embeddingDim, present when opts.matrix === 'load' */
  matrix?: Float32Array;
  /** decoded byte length of the matrix field, always set */
  matrixBytes: number;
  /** top-level keys other than embedding_dim/data/matrix (nano-vectordb's additional_data), in file order */
  extra: [string, unknown][];
  /** top-level key order as read */
  keyOrder: string[];
};

export type ReadVdbOptions = {
  /** 'load' decodes the float32 matrix; 'skip' only measures it (default). */
  matrix?: 'load' | 'skip';
  /** per-record `vector` field: dropped (default), kept as the base64 string, or decoded to the compressed bytes */
  vectorField?: 'drop' | 'string' | 'buffer';
  /** @deprecated B0 name for vectorField: 'string' */
  keepVectorField?: boolean;
  /** read buffer size; tests shrink it to exercise chunk boundaries */
  chunkSize?: number;
};

/** `<working_dir>[/<workspace>]/vdb_<namespace>.json` */
export function vdbPath(workingDir: string, namespace: string, workspace = ''): string {
  return join(workspace ? join(workingDir, workspace) : workingDir, `vdb_${namespace}.json`);
}

export function readNanoVectorDb(file: string, opts: ReadVdbOptions = {}): NanoVdbFile {
  const vf = opts.vectorField ?? (opts.keepVectorField ? 'string' : 'drop');
  const r = new ByteReader(file, opts.chunkSize);
  let embeddingDim = -1;
  const data: VdbRecord[] = [];
  let matrix: Float32Array | undefined;
  let matrixBytes = 0;
  const extra: [string, unknown][] = [];
  const keyOrder: string[] = [];
  try {
    for (const key of r.objectKeys()) {
      keyOrder.push(key);
      if (key === 'embedding_dim') embeddingDim = r.readValue() as number;
      else if (key === 'data') {
        for (const v of r.arrayValues()) {
          const rec = v as VdbRecord;
          if (typeof rec.vector === 'string') {
            if (vf === 'drop') delete rec.vector;
            else if (vf === 'buffer') rec.vector = Buffer.from(rec.vector, 'base64');
          }
          data.push(rec);
        }
      } else if (key === 'matrix') {
        const dec = new Base64Stream(opts.matrix === 'load' ? data.length * embeddingDim * 4 : 0);
        r.streamString((b) => dec.push(b));
        matrixBytes = dec.finish();
        if (opts.matrix === 'load') matrix = new Float32Array(dec.out.buffer, dec.out.byteOffset, matrixBytes / 4);
      } else extra.push([key, r.readValue()]);
    }
  } finally {
    r.close();
  }
  if (embeddingDim < 0) throw new Error(`${file}: no embedding_dim`);
  const out: NanoVdbFile = { embeddingDim, data, matrixBytes, extra, keyOrder };
  if (matrix) out.matrix = matrix;
  return out;
}

/** Decode a record's `vector` field: base64 -> zlib -> float16 -> float32. */
export function decodeVectorField(field: string | Buffer): Float32Array {
  const raw = inflateSync(typeof field === 'string' ? Buffer.from(field, 'base64') : field);
  const out = new Float32Array(raw.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(raw.readUInt16LE(i * 2));
  return out;
}

/** LightRAG's encoding: float16 (round to nearest even, as numpy astype) -> zlib level 6 -> bytes. */
export function encodeVectorField(embedding: ArrayLike<number>): Buffer {
  const raw = Buffer.allocUnsafe(embedding.length * 2);
  for (let i = 0; i < embedding.length; i++) raw.writeUInt16LE(floatToHalf(embedding[i]!), i * 2);
  return deflateSync(raw);
}

export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

/**
 * IEEE half bits of a double, round to nearest even (numpy astype(float16)).
 * Written out because Float16Array / Math.f16round need Node 24; the scalings
 * by powers of two below are exact, so the only rounding is roundHalfEven.
 */
export function floatToHalf(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a >= 65520) return sign | 0x7c00; // 65520 is the tie between 65504 and 2^16: rounds to even = inf
  if (a < 2 ** -14) return sign | roundHalfEven(a * 2 ** 24); // subnormal; 1024 lands on the smallest normal
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  let m = roundHalfEven((a / 2 ** e - 1) * 1024);
  if (m === 1024) {
    m = 0;
    e++;
  }
  if (e > 15) return sign | 0x7c00;
  return sign | ((e + 15) << 10) | m;
}

function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Base64 decoder over a byte stream; keeps a <4 char tail between pushes. */
class Base64Stream {
  out: Buffer;
  private len = 0;
  private tail = '';
  private measureOnly: boolean;
  constructor(capacity: number) {
    this.measureOnly = capacity === 0;
    this.out = Buffer.allocUnsafe(capacity);
  }
  push(chunk: Buffer): void {
    const s = this.tail + chunk.toString('latin1');
    const whole = s.length - (s.length % 4);
    this.tail = s.slice(whole);
    this.write(s.slice(0, whole));
  }
  private write(b64: string): void {
    if (b64.length === 0) return;
    if (this.measureOnly) {
      let pad = 0;
      if (b64.endsWith('==')) pad = 2;
      else if (b64.endsWith('=')) pad = 1;
      this.len += (b64.length / 4) * 3 - pad;
      return;
    }
    const bytes = Buffer.from(b64, 'base64');
    if (this.len + bytes.length > this.out.length) throw new Error('matrix larger than data.length * embedding_dim');
    bytes.copy(this.out, this.len);
    this.len += bytes.length;
  }
  finish(): number {
    if (this.tail.length) throw new Error('base64 matrix length is not a multiple of 4');
    return this.len;
  }
}

/** One embedding as numpy would hold it: float32 rows stay float32, anything else is float64. */
type Embedding = ArrayLike<number>;

export type UpsertReport = { update: string[]; insert: string[] };

/**
 * NanoVectorDB (nano-vectordb 0.0.4.3), in memory: rows in a growable float32
 * matrix, an id -> row index, data dicts without `__vector__`.
 */
export class NanoVectorDB {
  readonly embedding_dim: number;
  readonly storage_file: string;
  data: VdbRecord[] = [];
  private index = new Map<string, number>();
  private mat: Float32Array;
  private extra: [string, unknown][] = [];
  private keyOrder: string[] = ['embedding_dim', 'data', 'matrix'];

  constructor(embedding_dim: number, storage_file: string) {
    this.embedding_dim = embedding_dim;
    this.storage_file = storage_file;
    this.mat = new Float32Array(0);
    if (existsSync(storage_file)) {
      const f = readNanoVectorDb(storage_file, { matrix: 'load', vectorField: 'buffer' });
      if (f.embeddingDim !== embedding_dim) {
        throw new Error(`Embedding dim mismatch, expected: ${embedding_dim}, but loaded: ${f.embeddingDim}`);
      }
      if (f.matrixBytes !== f.data.length * embedding_dim * 4) {
        // numpy reshape(-1, dim) would succeed with a different row count; upstream then misaligns rows
        throw new Error(`${storage_file}: matrix has ${f.matrixBytes / 4 / embedding_dim} rows, data has ${f.data.length}`);
      }
      this.data = f.data;
      this.mat = f.matrix ?? new Float32Array(0);
      this.extra = f.extra;
      this.keyOrder = f.keyOrder.includes('matrix') ? f.keyOrder : [...f.keyOrder, 'matrix'];
      this.data.forEach((d, i) => this.index.set(d.__id__, i));
    }
    // pre_process: cosine metric normalizes every row on load
    normalizeRowsF32(this.mat, embedding_dim, this.data.length);
  }

  get length(): number {
    return this.data.length;
  }

  /** Row view of the matrix (no copy). */
  row(i: number): Float32Array {
    return this.mat.subarray(i * this.embedding_dim, (i + 1) * this.embedding_dim);
  }

  rowIndex(id: string): number | undefined {
    return this.index.get(id);
  }

  get_additional_data(): unknown {
    return this.extra.find(([k]) => k === 'additional_data')?.[1] ?? {};
  }

  /** `datas` carry `__vector__`; it is normalized, stored as a float32 row and removed from the dict. */
  upsert(datas: (VdbRecord & { __vector__: Embedding })[]): UpsertReport {
    const dim = this.embedding_dim;
    const pending = new Map<string, VdbRecord & { __vector__: Embedding }>();
    for (const d of datas) {
      if (typeof d.__id__ !== 'string') throw new Error('upsert without __id__ is not supported (nano-vectordb would hash the vector)');
      pending.set(d.__id__, d);
    }
    const toRow = (v: Embedding): Float32Array => {
      if (v.length !== dim) throw new Error(`vector of length ${v.length}, embedding_dim ${dim}`);
      return v instanceof Float32Array ? normalizeF32(v) : Float32Array.from(normalizeF64(v));
    };
    const report: UpsertReport = { update: [], insert: [] };
    // nano-vectordb walks the stored rows in order and replaces matches in place
    for (let i = 0; i < this.data.length && pending.size > 0; i++) {
      const id = this.data[i]!.__id__;
      const u = pending.get(id);
      if (!u) continue;
      pending.delete(id);
      this.mat.set(toRow(u.__vector__), i * dim);
      delete (u as Partial<typeof u>).__vector__;
      this.data[i] = u;
      report.update.push(id);
    }
    if (pending.size === 0) return report;
    const start = this.data.length;
    this.ensureRows(start + pending.size);
    let i = start;
    for (const [id, d] of pending) {
      this.mat.set(toRow(d.__vector__), i * dim);
      delete (d as Partial<typeof d>).__vector__;
      this.data.push(d);
      this.index.set(id, i);
      report.insert.push(id);
      i++;
    }
    return report;
  }

  private ensureRows(rows: number): void {
    const need = rows * this.embedding_dim;
    if (this.mat.length >= need) return;
    const grown = new Float32Array(Math.max(need, Math.ceil(this.mat.length * 1.5)));
    grown.set(this.mat.subarray(0, this.data.length * this.embedding_dim));
    this.mat = grown;
  }

  /** Records whose id is in `ids`, in storage order (not request order), as upstream. */
  get(ids: string[]): VdbRecord[] {
    const want = new Set(ids);
    const rows = [...want].map((id) => this.index.get(id)).filter((i): i is number => i !== undefined);
    rows.sort((a, b) => a - b);
    return rows.map((i) => this.data[i]!);
  }

  delete(ids: string[]): void {
    const del = new Set(ids);
    const dim = this.embedding_dim;
    let w = 0;
    for (let r = 0; r < this.data.length; r++) {
      const d = this.data[r]!;
      if (del.has(d.__id__)) continue;
      if (w !== r) {
        this.data[w] = d;
        this.mat.copyWithin(w * dim, r * dim, (r + 1) * dim);
      }
      w++;
    }
    if (w === this.data.length) return;
    this.data.length = w;
    this.index.clear();
    this.data.forEach((d, i) => this.index.set(d.__id__, i));
  }

  /**
   * _cosine_query: scores = matrix . normalize(query) in float64 (a Python list
   * query is float64), top_k by score, stop at the first score below threshold.
   * numpy's BLAS dot sums in its own order, so scores agree to ~1e-7, not bitwise.
   */
  query(query: Embedding, top_k: number, better_than_threshold: number | null): (VdbRecord & { __metrics__: number })[] {
    const dim = this.embedding_dim;
    if (query.length !== dim) throw new Error(`query of length ${query.length}, embedding_dim ${dim}`);
    const q = query instanceof Float32Array ? normalizeF32(query) : normalizeF64(query);
    const n = this.data.length;
    const scores = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      let s = 0;
      const off = r * dim;
      for (let i = 0; i < dim; i++) s += this.mat[off + i]! * q[i]!;
      scores[r] = query instanceof Float32Array ? Math.fround(s) : s;
    }
    const k = Math.max(0, Math.min(top_k, n));
    // partial selection: keep the k best, highest first
    const best: number[] = [];
    for (let r = 0; r < n; r++) {
      const s = scores[r]!;
      if (best.length === k && (k === 0 || s <= scores[best[k - 1]!]!)) continue;
      let pos = best.length;
      while (pos > 0 && scores[best[pos - 1]!]! < s) pos--;
      best.splice(pos, 0, r);
      if (best.length > k) best.pop();
    }
    const out: (VdbRecord & { __metrics__: number })[] = [];
    for (const r of best) {
      if (better_than_threshold !== null && scores[r]! < better_than_threshold) break;
      out.push({ ...this.data[r]!, __metrics__: scores[r]! });
    }
    return out;
  }

  /** nano-vectordb save(): json.dump(storage, f, ensure_ascii=False), atomic here. Throws on lone surrogates, as Python would. */
  save(): void {
    const dim = this.embedding_dim;
    const rows = this.data.length;
    const extra = new Map(this.extra);
    atomicWrite(this.storage_file, (sink) => {
      const w = new PyJsonWriter(sink, null);
      sink.write('{');
      this.keyOrder.forEach((key, i) => {
        if (i) sink.write(', ');
        sink.write(`${JSON.stringify(key)}: `);
        if (key === 'embedding_dim') sink.write(String(dim));
        else if (key === 'data') w.value(this.data);
        else if (key === 'matrix') {
          sink.write('"');
          const bytes = new Uint8Array(this.mat.buffer, this.mat.byteOffset, rows * dim * 4);
          const step = 3 << 20;
          for (let off = 0; off < bytes.length; off += step) {
            sink.write(Buffer.from(bytes.buffer, bytes.byteOffset + off, Math.min(step, bytes.length - off)).toString('base64'));
          }
          sink.write('"');
        } else w.value(extra.get(key));
      });
      sink.write('}');
      if (w.sanitized) throw new Error(`UnicodeEncodeError: lone surrogate in ${this.storage_file} data (Python's save fails the same way)`);
    });
  }
}

const META_STRIP = (dp: VdbRecord): KvRecord => {
  const out: KvRecord = {};
  for (const [k, v] of Object.entries(dp)) if (k !== 'vector') out[k] = v;
  return out;
};

export type NanoVectorDBStorageOptions = StorageOptions & {
  embedding_func: EmbeddingFunc;
  meta_fields?: Iterable<string>;
  clock?: () => number;
  log?: (level: 'error' | 'warn' | 'info' | 'debug', msg: string) => void;
};

export class NanoVectorDBStorage implements BaseVectorStorage {
  readonly namespace: string;
  readonly workspace: string;
  readonly global_config: GlobalConfig;
  readonly embedding_func: EmbeddingFunc;
  readonly meta_fields: ReadonlySet<string>;
  readonly cosine_better_than_threshold: number;
  readonly client_file_name: string;
  private readonly maxBatch: number;
  private readonly clock: () => number;
  private readonly log: NonNullable<NanoVectorDBStorageOptions['log']>;
  private client: NanoVectorDB;

  constructor(opts: NanoVectorDBStorageOptions) {
    if (!opts.embedding_func) throw new Error('embedding_func is required for vector storage.');
    this.namespace = opts.namespace;
    this.workspace = opts.workspace ?? '';
    this.global_config = opts.global_config;
    this.embedding_func = opts.embedding_func;
    this.meta_fields = new Set(opts.meta_fields ?? []);
    const t = opts.global_config.vector_db_storage_cls_kwargs?.cosine_better_than_threshold;
    if (t === undefined || t === null) throw new Error('cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs');
    this.cosine_better_than_threshold = t;
    mkdirSync(workspaceDir(opts), { recursive: true });
    this.client_file_name = vdbPath(this.global_config.working_dir, this.namespace, this.workspace);
    this.maxBatch = opts.global_config.embedding_batch_num ?? 32;
    this.clock = opts.clock ?? (() => Math.floor(Date.now() / 1000));
    this.log = opts.log ?? (() => {});
    this.client = new NanoVectorDB(this.embedding_func.embedding_dim, this.client_file_name);
  }

  async initialize(): Promise<void> {}
  async finalize(): Promise<void> {}

  /** Direct access for tools and tests (upstream: `client_storage`). */
  get db(): NanoVectorDB {
    return this.client;
  }

  async upsert(data: Record<string, KvRecord>): Promise<UpsertReport | undefined> {
    const entries = Object.entries(data);
    if (entries.length === 0) return undefined;
    const now = this.clock();
    const list = entries.map(([k, v]) => {
      const d: VdbRecord = { __id__: k, __created_at__: now };
      for (const [k1, v1] of Object.entries(v)) if (this.meta_fields.has(k1)) d[k1] = v1;
      return d;
    });
    const contents = entries.map(([, v]) => v.content as string);
    const batches: string[][] = [];
    for (let i = 0; i < contents.length; i += this.maxBatch) batches.push(contents.slice(i, i + this.maxBatch));
    const results = await Promise.all(batches.map((b) => this.embedding_func(b, { context: 'document' })));
    const embeddings: ArrayLike<number>[] = [];
    for (const r of results) for (let i = 0; i < r.length; i++) embeddings.push(r[i]!);
    if (embeddings.length !== list.length) {
      this.log('error', `[${this.workspace}] embedding is not 1-1 with data, ${embeddings.length} != ${list.length}`);
      return undefined;
    }
    const withVec = list.map((d, i) => {
      d.vector = encodeVectorField(embeddings[i]!);
      return Object.assign(d, { __vector__: embeddings[i]! });
    });
    return this.client.upsert(withVec);
  }

  async query(query: string, top_k: number, query_embedding: ArrayLike<number> | null = null): Promise<VectorQueryResult[]> {
    let embedding = query_embedding;
    if (embedding === null || embedding === undefined) {
      embedding = (await this.embedding_func([query], { context: 'query', _priority: 5 }))[0]!;
    }
    return this.client.query(embedding, top_k, this.cosine_better_than_threshold).map((dp) => ({
      ...META_STRIP(dp),
      id: dp.__id__,
      distance: dp.__metrics__,
      created_at: dp.__created_at__ ?? null,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    try {
      this.client.delete(ids);
    } catch (e) {
      this.log('error', `[${this.workspace}] Error while deleting vectors from ${this.namespace}: ${e}`);
    }
  }

  async delete_entity(entity_name: string): Promise<void> {
    const id = computeMdhashId(entity_name, 'ent-');
    if (this.client.get([id]).length) this.client.delete([id]);
  }

  /**
   * Upstream's list comprehension reads dp["src_id"] / dp["tgt_id"]; a record
   * without them raises KeyError, which is logged and nothing is deleted.
   */
  async delete_entity_relation(entity_name: string): Promise<void> {
    const ids: string[] = [];
    for (const dp of this.client.data) {
      if (!('src_id' in dp)) return this.log('error', `[${this.workspace}] Error deleting relations for ${entity_name}: 'src_id'`);
      if (dp.src_id === entity_name) {
        ids.push(dp.__id__);
        continue;
      }
      if (!('tgt_id' in dp)) return this.log('error', `[${this.workspace}] Error deleting relations for ${entity_name}: 'tgt_id'`);
      if (dp.tgt_id === entity_name) ids.push(dp.__id__);
    }
    if (ids.length) this.client.delete(ids);
  }

  async index_done_callback(): Promise<boolean> {
    try {
      this.client.save();
      return true;
    } catch (e) {
      this.log('error', `[${this.workspace}] Error saving data for ${this.namespace}: ${e}`);
      return false;
    }
  }

  async get_by_id(id: string): Promise<KvRecord | null> {
    const r = this.client.get([id]);
    if (!r.length) return null;
    const dp = r[0]!;
    return { ...META_STRIP(dp), id: dp.__id__ ?? null, created_at: dp.__created_at__ ?? null };
  }

  async get_by_ids(ids: string[]): Promise<(KvRecord | null)[]> {
    if (ids.length === 0) return [];
    const m = new Map<string, KvRecord>();
    for (const dp of this.client.get(ids)) m.set(String(dp.__id__), { ...META_STRIP(dp), id: dp.__id__, created_at: dp.__created_at__ ?? null });
    return ids.map((id) => m.get(String(id)) ?? null);
  }

  /** Decoded float16 `vector` fields (not the normalized matrix rows), storage order. */
  async get_vectors_by_ids(ids: string[]): Promise<Record<string, number[]>> {
    const out: Record<string, number[]> = {};
    if (ids.length === 0) return out;
    for (const dp of this.client.get(ids)) {
      if (dp.vector !== undefined && dp.vector !== null && '__id__' in dp) out[dp.__id__] = Array.from(decodeVectorField(dp.vector));
    }
    return out;
  }

  async drop(): Promise<DropResult> {
    try {
      rmSync(this.client_file_name, { force: true });
      this.client = new NanoVectorDB(this.embedding_func.embedding_dim, this.client_file_name);
      return { status: 'success', message: 'data dropped' };
    } catch (e) {
      return { status: 'error', message: String((e as Error).message ?? e) };
    }
  }
}
