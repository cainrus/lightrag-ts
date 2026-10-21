// Port of lightrag/kg/nano_vector_db_impl.py + the nano-vectordb 0.0.4.3 file
// format it persists. B0: read-only loader. File shape:
//   {"embedding_dim": N, "data": [{"__id__", "__created_at__", ...meta, "vector"}], "matrix": "<b64>"}
// - data[i].vector: base64(zlib(float16[N]))  - written by LightRAG (nano_vector_db_impl.py:131-135)
// - matrix: base64(float32[rows*N]), little-endian, rows L2-normalized by nano-vectordb
//   for cosine metric; row i belongs to data[i].
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { ByteReader } from '../internal/json_stream.ts';

export type VdbRecord = { __id__: string; __created_at__?: number; vector?: string; [k: string]: unknown };

export type NanoVdbFile = {
  embeddingDim: number;
  data: VdbRecord[];
  /** rows x embeddingDim, present when opts.matrix === 'load' */
  matrix?: Float32Array;
  /** decoded byte length of the matrix field, always set */
  matrixBytes: number;
};

export type ReadVdbOptions = {
  /** 'load' decodes the float32 matrix; 'skip' only measures it (default). */
  matrix?: 'load' | 'skip';
  /** keep the per-record compressed `vector` field (default false, it is ~3 KB a record) */
  keepVectorField?: boolean;
  /** read buffer size; tests shrink it to exercise chunk boundaries */
  chunkSize?: number;
};

/** `<working_dir>[/<workspace>]/vdb_<namespace>.json` */
export function vdbPath(workingDir: string, namespace: string, workspace = ''): string {
  return join(workspace ? join(workingDir, workspace) : workingDir, `vdb_${namespace}.json`);
}

export function readNanoVectorDb(file: string, opts: ReadVdbOptions = {}): NanoVdbFile {
  const r = new ByteReader(file, opts.chunkSize);
  let embeddingDim = -1;
  const data: VdbRecord[] = [];
  let matrix: Float32Array | undefined;
  let matrixBytes = 0;
  try {
    for (const key of r.objectKeys()) {
      if (key === 'embedding_dim') embeddingDim = r.readValue() as number;
      else if (key === 'data') {
        for (const v of r.arrayValues()) {
          const rec = v as VdbRecord;
          if (!opts.keepVectorField) delete rec.vector;
          data.push(rec);
        }
      } else if (key === 'matrix') {
        const dec = new Base64Stream(opts.matrix === 'load' ? data.length * embeddingDim * 4 : 0);
        r.streamString((b) => dec.push(b));
        matrixBytes = dec.finish();
        if (opts.matrix === 'load') matrix = new Float32Array(dec.out.buffer, dec.out.byteOffset, matrixBytes / 4);
      } else r.readValue();
    }
  } finally {
    r.close();
  }
  if (embeddingDim < 0) throw new Error(`${file}: no embedding_dim`);
  const out: NanoVdbFile = { embeddingDim, data, matrixBytes };
  if (matrix) out.matrix = matrix;
  return out;
}

/** Decode a record's `vector` field: base64 -> zlib -> float16 -> float32. */
export function decodeVectorField(b64: string): Float32Array {
  const raw = inflateSync(Buffer.from(b64, 'base64'));
  const out = new Float32Array(raw.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(raw.readUInt16LE(i * 2));
  return out;
}

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
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
