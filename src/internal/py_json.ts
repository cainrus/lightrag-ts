// Port-only: byte-compatible writers for the two `json.dump` calls upstream
// storages make, streamed to disk because vdb files outgrow V8's max string length.
//   JsonKV / JsonDocStatus: json.dump(obj, f, indent=2, ensure_ascii=False)   (utils.write_json)
//   NanoVectorDB.save:      json.dump(obj, f, ensure_ascii=False)              (", " / ": ")
//
// Numbers: a JS number carries no int/float distinction. Integers are written as
// Python ints, everything else with Python's float repr. A Python float with an
// integral value (1.0) would come back as 1; no kv store has one (measured on
// the live store, B1), and the vdb/graph writers keep their own type info.
import { closeSync, fsyncSync, openSync, renameSync, rmSync, writeSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

/** Python `repr(float)`: shortest round-trip digits, fixed for 1e-4 <= |x| < 1e16, else exponent. */
export function pyFloatRepr(x: number): string {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return 'Infinity'; // json.dumps spelling; repr() would say 'inf'
  if (x === -Infinity) return '-Infinity';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const [mant, expS] = Math.abs(x).toExponential().split('e') as [string, string];
  const digits = mant.replace('.', '');
  const exp = Number(expS);
  const sign = x < 0 ? '-' : '';
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const e = Math.abs(exp) < 10 ? `0${Math.abs(exp)}` : String(Math.abs(exp));
    return `${sign}${m}e${exp < 0 ? '-' : '+'}${e}`;
  }
  if (exp < 0) return `${sign}0.${'0'.repeat(-exp - 1)}${digits}`;
  const intPart = digits.length > exp + 1 ? digits.slice(0, exp + 1) : digits + '0'.repeat(exp + 1 - digits.length);
  const frac = digits.length > exp + 1 ? digits.slice(exp + 1) : '0';
  return `${sign}${intPart}.${frac}`;
}

/** Python `str(int)` / json int, or float repr. */
export function pyNumber(x: number): string {
  return Number.isInteger(x) ? String(x) : pyFloatRepr(x);
}

// A JS string can hold lone surrogates; Python's write_json fails on them in the
// fast path and rewrites the file through SanitizingJSONEncoder, which drops them.
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

export function hasLoneSurrogate(s: string): boolean {
  LONE_SURROGATE.lastIndex = 0;
  return LONE_SURROGATE.test(s);
}

// upstream _SURROGATE_PATTERN = [\uD800-\uDFFF\uFFFE\uFFFF]; it is applied to every
// string, but only once the fast path has failed on a surrogate somewhere in the file.
// `u` flag: a surrogate PAIR is one code point (one Python char) and must survive
const SANITIZE = /[\uD800-\uDFFF\uFFFE\uFFFF]/gu;

/** upstream utils._sanitize_string_for_json */
export function sanitizeStringForJson(s: string): string {
  return s.replace(SANITIZE, '');
}

/** Sink that batches small writes into a buffer and flushes to an fd. */
export class FdSink {
  private buf = Buffer.allocUnsafe(1 << 20);
  private len = 0;
  bytes = 0;
  private fd: number;
  constructor(fd: number) {
    this.fd = fd;
  }
  write(s: string): void {
    const need = Buffer.byteLength(s);
    if (this.len + need > this.buf.length) this.flush();
    if (need > this.buf.length) {
      writeSync(this.fd, s);
    } else {
      this.len += this.buf.write(s, this.len);
    }
    this.bytes += need;
  }
  writeBytes(b: Uint8Array): void {
    if (this.len + b.length > this.buf.length) this.flush();
    if (b.length > this.buf.length) writeSync(this.fd, b);
    else {
      this.buf.set(b, this.len);
      this.len += b.length;
    }
    this.bytes += b.length;
  }
  flush(): void {
    let off = 0;
    while (off < this.len) off += writeSync(this.fd, this.buf, off, this.len - off);
    this.len = 0;
  }
}

export type PyJsonWriteResult = { sanitized: boolean; bytes: number };

/**
 * Stream `json.dump(value, f, indent=indent, ensure_ascii=False)`.
 * `indent: null` gives the compact default separators (", ", ": ").
 * Map values are written like dicts, in insertion order.
 * Callers can hook `raw` to emit a precomputed fragment (e.g. a base64 stream) for a key.
 */
export class PyJsonWriter {
  /** set when a lone surrogate was seen: upstream's fast path would have failed */
  sanitized = false;
  private sink: FdSink;
  private indent: number | null;
  /** second pass, like upstream's SanitizingJSONEncoder */
  private sanitize: boolean;
  constructor(sink: FdSink, indent: number | null, sanitize = false) {
    this.sink = sink;
    this.indent = indent;
    this.sanitize = sanitize;
  }

  str(s: string): string {
    if (this.sanitize) return JSON.stringify(sanitizeStringForJson(s));
    if (hasLoneSurrogate(s)) this.sanitized = true;
    return JSON.stringify(s);
  }

  value(v: unknown, level = 0): void {
    const w = this.sink;
    if (v === null || v === undefined) return w.write('null');
    switch (typeof v) {
      case 'string':
        return w.write(this.str(v));
      case 'boolean':
        return w.write(v ? 'true' : 'false');
      case 'number':
        return w.write(pyNumber(v));
      case 'bigint':
        return w.write(v.toString());
    }
    if (v instanceof RawJson) return w.write(v.text);
    // bytes kept decoded in memory (vdb `vector` fields) go back out as the base64 string they were
    if (v instanceof Uint8Array) return w.write(`"${Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64')}"`);
    if (Array.isArray(v)) {
      if (v.length === 0) return w.write('[]');
      this.open('[', level);
      v.forEach((x, i) => {
        if (i) this.sep(level);
        this.value(x, level + 1);
      });
      return this.close(']', level);
    }
    const entries = v instanceof Map ? v.entries() : Object.entries(v as object)[Symbol.iterator]();
    let first = true;
    for (const [k, x] of entries) {
      if (first) this.open('{', level);
      else this.sep(level);
      first = false;
      w.write(this.str(String(k)) + ': ');
      this.value(x, level + 1);
    }
    if (first) return w.write('{}');
    this.close('}', level);
  }

  private open(ch: string, level: number): void {
    this.sink.write(this.indent === null ? ch : `${ch}\n${' '.repeat(this.indent * (level + 1))}`);
  }
  private sep(level: number): void {
    this.sink.write(this.indent === null ? ', ' : `,\n${' '.repeat(this.indent * (level + 1))}`);
  }
  private close(ch: string, level: number): void {
    this.sink.write(this.indent === null ? ch : `\n${' '.repeat(this.indent * level)}${ch}`);
  }
}

/** A pre-serialized JSON fragment, written verbatim. */
export class RawJson {
  readonly text: string;
  constructor(text: string) {
    this.text = text;
  }
}

/**
 * Write `file` atomically: stream into a temp file in the same directory, fsync,
 * rename over the target. Upstream writes in place (a crash leaves a truncated
 * store; the live store has a `.corrupt-*` file from exactly that).
 */
export function atomicWrite<T>(file: string, body: (sink: FdSink) => T): T {
  const tmp = join(dirname(file), `.${basename(file)}.tmp-${process.pid}-${Date.now().toString(36)}`);
  const fd = openSync(tmp, 'w');
  let ok = false;
  try {
    const sink = new FdSink(fd);
    const out = body(sink);
    sink.flush();
    fsyncSync(fd);
    ok = true;
    return out;
  } finally {
    closeSync(fd);
    if (ok) renameSync(tmp, file);
    else rmSync(tmp, { force: true });
  }
}

/** upstream utils.write_json, atomic. Returns whether lone surrogates were dropped (caller reloads, as upstream). */
export function writeJsonPy(value: unknown, file: string, indent: number | null = 2): PyJsonWriteResult {
  const fast = atomicWrite(file, (sink) => {
    const w = new PyJsonWriter(sink, indent);
    w.value(value);
    return { sanitized: w.sanitized, bytes: sink.bytes };
  });
  if (!fast.sanitized) return fast;
  return atomicWrite(file, (sink) => {
    new PyJsonWriter(sink, indent, true).value(value);
    return { sanitized: true, bytes: sink.bytes };
  });
}
