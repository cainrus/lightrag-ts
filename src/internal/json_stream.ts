// Synchronous streaming reader for the two JSON shapes upstream writes:
//   kv_store_*.json  - one big object, values are small
//   vdb_*.json       - {"embedding_dim", "data": [...], "matrix": "<base64>"}
// The vdb files outgrow V8's max string length (~512 MiB): a vault with ~100k
// relations writes a ~900 MB file, so JSON.parse(readFileSync()) is not an option.
// Each value is parsed on its own; a huge string value can be streamed to a sink.
import { closeSync, openSync, readSync } from 'node:fs';

const CHUNK = 1 << 22; // 4 MiB

const B = {
  QUOTE: 0x22,
  BACKSLASH: 0x5c,
  LBRACE: 0x7b,
  RBRACE: 0x7d,
  LBRACKET: 0x5b,
  RBRACKET: 0x5d,
  COMMA: 0x2c,
  COLON: 0x3a,
} as const;

function isWs(c: number): boolean {
  return c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09;
}

export class ByteReader {
  readonly path: string;
  private fd: number;
  private buf: Buffer;
  private pos = 0;
  private end = 0;
  private eof = false;
  /** bytes consumed before the current buffer; for error positions */
  private base = 0;

  constructor(path: string, chunkSize = CHUNK) {
    this.buf = Buffer.allocUnsafe(chunkSize);
    this.path = path;
    this.fd = openSync(path, 'r');
    // utf-8-sig, like upstream load_json
    const bom = Buffer.alloc(3);
    if (readSync(this.fd, bom, 0, 3, 0) === 3 && bom[0] === 0xef && bom[1] === 0xbb && bom[2] === 0xbf) this.base = 3;
    this.next = this.base;
    this.fill();
  }

  /** file offset of the next read */
  private next = 0;

  close(): void {
    closeSync(this.fd);
  }

  private fill(): boolean {
    if (this.eof) return false;
    this.base = this.next;
    this.end = readSync(this.fd, this.buf, 0, this.buf.length, this.next);
    this.next += this.end;
    this.pos = 0;
    if (this.end === 0) this.eof = true;
    return !this.eof;
  }

  private fail(msg: string): never {
    throw new Error(`${this.path}: ${msg} at byte ${this.base + this.pos}`);
  }

  skipWs(): void {
    for (;;) {
      if (this.pos >= this.end && !this.fill()) return;
      if (!isWs(this.buf[this.pos]!)) return;
      this.pos++;
    }
  }

  peek(): number {
    this.skipWs();
    return this.pos < this.end ? this.buf[this.pos]! : -1;
  }

  expect(c: number): void {
    if (this.peek() !== c) this.fail(`expected '${String.fromCharCode(c)}'`);
    this.pos++;
  }

  atEof(): boolean {
    return this.peek() === -1;
  }

  /** Read one complete JSON value and return its raw bytes (may alias the read buffer: valid until the next read). */
  readRaw(): Buffer {
    const first = this.peek();
    if (first === -1) this.fail('unexpected end of input');
    const parts: Buffer[] = [];
    let start = this.pos;
    let depth = 0;
    let inStr = false;
    let esc = false;
    const scalar = first !== B.LBRACE && first !== B.LBRACKET && first !== B.QUOTE;
    for (;;) {
      if (this.pos >= this.end) {
        parts.push(Buffer.from(this.buf.subarray(start, this.end)));
        if (!this.fill()) {
          if (scalar) return Buffer.concat(parts);
          this.fail('unterminated value');
        }
        start = 0;
      }
      const c = this.buf[this.pos]!;
      if (inStr) {
        if (esc) esc = false;
        else if (c === B.BACKSLASH) esc = true;
        else if (c === B.QUOTE) {
          inStr = false;
          if (depth === 0) {
            this.pos++;
            break;
          }
        }
        this.pos++;
        continue;
      }
      if (scalar) {
        if (c === B.COMMA || c === B.RBRACE || c === B.RBRACKET || isWs(c)) break;
        this.pos++;
        continue;
      }
      if (c === B.QUOTE) inStr = true;
      else if (c === B.LBRACE || c === B.LBRACKET) depth++;
      else if (c === B.RBRACE || c === B.RBRACKET) {
        depth--;
        if (depth === 0) {
          this.pos++;
          break;
        }
      }
      this.pos++;
    }
    parts.push(this.buf.subarray(start, this.pos));
    return parts.length === 1 ? parts[0]! : Buffer.concat(parts);
  }

  readValue(): unknown {
    return JSON.parse(this.readRaw().toString('utf8'));
  }

  /**
   * Stream the body of a JSON string without materializing it. Only for
   * strings without escapes (base64), which is what upstream stores there.
   */
  streamString(sink: (bytes: Buffer) => void): void {
    this.expect(B.QUOTE);
    for (;;) {
      if (this.pos >= this.end && !this.fill()) this.fail('unterminated string');
      const idx = this.buf.indexOf(B.QUOTE, this.pos);
      const stop = idx === -1 || idx >= this.end ? this.end : idx;
      const slice = this.buf.subarray(this.pos, stop);
      if (slice.includes(B.BACKSLASH)) this.fail('escape inside a streamed string');
      if (slice.length > 0) sink(slice);
      this.pos = stop;
      if (stop < this.end) {
        this.pos++;
        return;
      }
    }
  }

  /** Iterate `"key": value` pairs of the object at the cursor; the callback consumes the value. */
  *objectKeys(): Generator<string> {
    this.expect(B.LBRACE);
    if (this.peek() === B.RBRACE) {
      this.pos++;
      return;
    }
    for (;;) {
      const key = this.readValue();
      if (typeof key !== 'string') this.fail('object key is not a string');
      this.expect(B.COLON);
      const before = this.base + this.pos;
      yield key;
      if (this.base + this.pos === before) this.fail(`value of "${key}" was not consumed`);
      const c = this.peek();
      this.pos++;
      if (c === B.RBRACE) return;
      if (c !== B.COMMA) this.fail("expected ',' or '}'");
    }
  }

  /** Iterate elements of the array at the cursor, each as a parsed value. */
  *arrayValues(): Generator<unknown> {
    this.expect(B.LBRACKET);
    if (this.peek() === B.RBRACKET) {
      this.pos++;
      return;
    }
    for (;;) {
      yield this.readValue();
      const c = this.peek();
      this.pos++;
      if (c === B.RBRACKET) return;
      if (c !== B.COMMA) this.fail("expected ',' or ']'");
    }
  }
}

/** Entries of a top-level JSON object, each value parsed separately. */
export function* iterateJsonObject(path: string, chunkSize?: number): Generator<[string, unknown]> {
  const r = new ByteReader(path, chunkSize);
  try {
    for (const key of r.objectKeys()) yield [key, r.readValue()];
    if (!r.atEof()) throw new Error(`${path}: trailing data after the top-level object`);
  } finally {
    r.close();
  }
}
