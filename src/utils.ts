// Port of lightrag/utils.py (subset needed so far). Function names keep the
// upstream names in camelCase so a reviewer can diff them side by side.
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { getEncodingNameForModel, Tiktoken, type TiktokenBPE, type TiktokenModel } from 'js-tiktoken/lite';
import { HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS } from './internal/html5_entities.ts';

/** Lone UTF-16 surrogates; Python's `encode('utf-8', errors='replace')` turns each into '?'. */
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

/** utils.compute_args_hash: md5 hex of the concatenated str() of every argument. */
export function computeArgsHash(...args: unknown[]): string {
  const s = args.map((a) => String(a)).join('');
  return createHash('md5').update(Buffer.from(s.replace(LONE_SURROGATE, '?'), 'utf8')).digest('hex');
}

/** utils.compute_mdhash_id */
export function computeMdhashId(content: string, prefix = ''): string {
  return prefix + computeArgsHash(content);
}

/** utils.make_relation_vdb_ids: normalized id first, reverse-order id as a fallback. */
export function makeRelationVdbIds(srcEntity: string, tgtEntity: string): string[] {
  const [a, b] = [srcEntity, tgtEntity].sort(pyCompare) as [string, string];
  const ids = [computeMdhashId(a + b, 'rel-')];
  const reverse = computeMdhashId(b + a, 'rel-');
  if (!ids.includes(reverse)) ids.push(reverse);
  return ids;
}

/** utils.TokenizerInterface */
export interface TokenizerInterface {
  encode(content: string): number[];
  decode(tokens: number[]): string;
}

/** utils.Tokenizer */
export class Tokenizer {
  readonly modelName: string;
  readonly tokenizer: TokenizerInterface;
  constructor(modelName: string, tokenizer: TokenizerInterface) {
    this.modelName = modelName;
    this.tokenizer = tokenizer;
  }
  encode(content: string): number[] {
    return this.tokenizer.encode(content);
  }
  decode(tokens: number[]): string {
    return this.tokenizer.decode(tokens);
  }
}

const require = createRequire(import.meta.url);
const encodings = new Map<string, Tiktoken>();

/**
 * utils.TiktokenTokenizer over js-tiktoken. Same BPE ranks and split regex as
 * tiktoken; encode() rejects special-token text like tiktoken's default
 * (`disallowed_special="all"`), decode() replaces broken UTF-8 with U+FFFD
 * like `errors="replace"`. Ranks load lazily, once per encoding.
 */
export class TiktokenTokenizer extends Tokenizer {
  constructor(modelName = 'gpt-4o-mini') {
    let name: string;
    try {
      name = getEncodingNameForModel(modelName as TiktokenModel);
    } catch {
      throw new Error(`Invalid model_name: ${modelName}.`);
    }
    let enc = encodings.get(name);
    if (!enc) {
      enc = new Tiktoken(require(`js-tiktoken/ranks/${name}`) as TiktokenBPE);
      encodings.set(name, enc);
    }
    const e = enc;
    super(modelName, { encode: (s) => e.encode(s), decode: (t) => e.decode(t) });
  }
}

/** utils.generate_cache_key: `{mode}:{cache_type}:{hash}` */
export function generateCacheKey(mode: string, cacheType: string, hashValue: string): string {
  return `${mode}:${cacheType}:${hashValue}`;
}

/** utils.pack_user_ass_to_openai_messages */
export function packUserAssToOpenaiMessages(...args: string[]): { role: 'user' | 'assistant'; content: string }[] {
  return args.map((content, i) => ({ role: i % 2 === 0 ? 'user' : 'assistant', content }));
}

export type ChatMessage = { role: string; content: string };

/**
 * The cache-key half of utils.use_llm_func_with_cache: sanitize the inputs the
 * same way, join them, hash. Returns everything the caller needs to look the
 * call up in `kv_store_llm_response_cache.json` without calling the LLM.
 */
export function llmCacheLookup(
  userPrompt: string,
  opts: { systemPrompt?: string | null; historyMessages?: ChatMessage[] | null; cacheType: string; mode?: string },
): { prompt: string; argsHash: string; cacheKey: string } {
  const safeUser = sanitizeTextForEncoding(userPrompt);
  const safeSystem = opts.systemPrompt ? sanitizeTextForEncoding(opts.systemPrompt) : null;
  let history: string | null = null;
  if (opts.historyMessages && opts.historyMessages.length > 0) {
    history = pyJsonDumps(opts.historyMessages.map((m) => ({ ...m, content: sanitizeTextForEncoding(m.content) })));
  }
  const parts: string[] = [];
  if (safeUser) parts.push(safeUser);
  if (safeSystem) parts.push(safeSystem);
  if (history) parts.push(history);
  const prompt = parts.join('\n');
  const argsHash = computeArgsHash(prompt);
  return { prompt, argsHash, cacheKey: generateCacheKey(opts.mode ?? 'default', opts.cacheType, argsHash) };
}

// `u` flag: Python strings are code points, so only LONE surrogates match;
// without it the pattern eats both halves of every emoji.
const SURROGATE_PATTERN = /[\uD800-\uDFFF\u{FFFE}\u{FFFF}]/gu;
const CONTROL_CHAR_PATTERN_ALL = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g;

/** utils.sanitize_text_for_encoding */
export function sanitizeTextForEncoding(text: string, replacementChar = ''): string {
  if (!text) return text;
  text = pyStrip(text);
  if (!text) return text;
  text = htmlUnescape(text);
  text = text.replace(SURROGATE_PATTERN, replacementChar);
  text = text.replace(CONTROL_CHAR_PATTERN_ALL, replacementChar);
  return pyStrip(text);
}

// ---------------------------------------------------------------------------
// Python semantics the port depends on. Kept here (not in a generic helper
// package) because each one exists to reproduce a specific upstream byte.
// ---------------------------------------------------------------------------

/** Characters for which Python's `str.isspace()` is true (differs from JS `trim`: no U+FEFF, adds \x1c-\x1f, \x85). */
const PY_WS = '\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0                　';
const PY_WS_SET = new Set(PY_WS);

/** Python `str.strip()` with no arguments. */
export function pyStrip(s: string): string {
  let start = 0;
  let end = s.length;
  while (start < end && PY_WS_SET.has(s[start]!)) start++;
  while (end > start && PY_WS_SET.has(s[end - 1]!)) end--;
  return start === 0 && end === s.length ? s : s.slice(start, end);
}

/** Python default str ordering: by code point, not by UTF-16 unit. */
export function pyCompare(a: string, b: string): number {
  const ia = a[Symbol.iterator]();
  const ib = b[Symbol.iterator]();
  for (;;) {
    const x = ia.next();
    const y = ib.next();
    if (x.done) return y.done ? 0 : -1;
    if (y.done) return 1;
    const d = x.value.codePointAt(0)! - y.value.codePointAt(0)!;
    if (d !== 0) return d;
  }
}

const CHARREF = /&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)/g;

/** CPython `html.unescape` (HTML5 rules, longest-prefix named refs, invalid charref table). */
export function htmlUnescape(s: string): string {
  if (!s.includes('&')) return s;
  return s.replace(CHARREF, (_m, ref: string) => {
    if (ref[0] === '#') {
      const hex = ref[1] === 'x' || ref[1] === 'X';
      const digits = (hex ? ref.slice(2) : ref.slice(1)).replace(/;+$/, '');
      // BigInt: Python ints are unbounded, "&#99999999999999999999;" must still map to U+FFFD
      const num = BigInt(hex ? `0x${digits}` : digits);
      const key = num.toString();
      if (Object.hasOwn(INVALID_CHARREFS, key)) return INVALID_CHARREFS[key]!;
      if ((num >= 0xd800n && num <= 0xdfffn) || num > 0x10ffffn) return '�';
      if (INVALID_CODEPOINTS.has(Number(num))) return '';
      return String.fromCodePoint(Number(num));
    }
    if (Object.hasOwn(HTML5, ref)) return HTML5[ref]!;
    // longest matching prefix, as the standard defines (Python iterates over code points)
    const cps = Array.from(ref);
    for (let x = cps.length - 1; x > 1; x--) {
      const head = cps.slice(0, x).join('');
      if (Object.hasOwn(HTML5, head)) return HTML5[head]! + cps.slice(x).join('');
    }
    return '&' + ref;
  });
}

/**
 * Python `str.format(**kwargs)` for the subset upstream prompts use: `{name}`,
 * `{{`, `}}`. Anything else (positional fields, format specs, conversions)
 * throws, so a new upstream template cannot silently format differently.
 */
export function pyFormat(template: string, values: Readonly<Record<string, unknown>>): string {
  let out = '';
  let i = 0;
  while (i < template.length) {
    const ch = template[i]!;
    if (ch === '{') {
      if (template[i + 1] === '{') {
        out += '{';
        i += 2;
        continue;
      }
      const close = template.indexOf('}', i);
      if (close < 0) throw new Error(`pyFormat: single '{' at ${i}`);
      const name = template.slice(i + 1, close);
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(`pyFormat: unsupported field {${name}}`);
      if (!Object.hasOwn(values, name)) throw new Error(`pyFormat: missing key '${name}'`);
      out += pyStr(values[name]);
      i = close + 1;
      continue;
    }
    if (ch === '}') {
      if (template[i + 1] !== '}') throw new Error(`pyFormat: single '}' at ${i}`);
      out += '}';
      i += 2;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

function pyStr(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number' && Number.isInteger(v)) return String(v);
  throw new Error(`pyFormat: value of type ${typeof v} has no verified Python str() equivalent`);
}

/**
 * Python `json.dumps(obj, ensure_ascii=False)` for JSON-compatible values:
 * `", "` / `": "` separators, same string escaping as JSON.stringify.
 * Known gap: a lone surrogate stays raw in Python but JSON.stringify escapes it;
 * irrelevant on the cache path because sanitizeTextForEncoding strips them first.
 */
export function pyJsonDumps(v: unknown): string {
  if (v === null) return 'null';
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') {
    if (!Number.isInteger(v)) throw new Error('pyJsonDumps: float repr not ported yet');
    return String(v);
  }
  if (Array.isArray(v)) return '[' + v.map(pyJsonDumps).join(', ') + ']';
  if (typeof v === 'object') {
    return '{' + Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${pyJsonDumps(x)}`).join(', ') + '}';
  }
  throw new Error(`pyJsonDumps: unsupported ${typeof v}`);
}
