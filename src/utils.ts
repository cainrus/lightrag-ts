// Port of lightrag/utils.py (subset needed so far). Function names keep the
// upstream names in camelCase so a reviewer can diff them side by side.
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { getEncodingNameForModel, Tiktoken, type TiktokenBPE, type TiktokenModel } from 'js-tiktoken/lite';
import { HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS } from './internal/html5_entities.ts';
import { PY_ISDIGIT, PY_RE_DIGIT, PY_RE_SPACE } from './internal/py_unicode.ts';

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

// ---------------------------------------------------------------------------
// Extraction-output helpers. Python regex semantics differ from JS in `.`
// (Python matches \r and U+2028), `\s` and `\d` (Unicode tables of CPython's
// Unicode version) and `$` (also before a final \n); each pattern below spells
// those out instead of relying on the JS meaning.
// ---------------------------------------------------------------------------

const PY_S = `[${PY_RE_SPACE}]`;
const isPyDigit = (c: string) => PY_ISDIGIT_RE.test(c);
const PY_ISDIGIT_RE = new RegExp(`^[${PY_ISDIGIT}]$`, 'u');

/** re.escape for the characters a `u`-flag RegExp accepts escaped. */
function reEscape(s: string): string {
  return s.replace(/[\\^$.*+?()[\]{}|/]/g, '\\$&');
}

/** utils.split_string_by_multi_markers */
export function splitStringByMultiMarkers(content: string | null | undefined, markers: readonly string[]): string[] {
  if (!markers.length) return [content as string];
  const results = (content ?? '').split(new RegExp(markers.map(reEscape).join('|'), 'u'));
  return results.map(pyStrip).filter((r) => r);
}

/** utils.is_float_regex: `re.match(r"^[-+]?[0-9]*\.?[0-9]+$")` */
export function isFloatRegex(value: string): boolean {
  return /^[-+]?[0-9]*\.?[0-9]+\n?$/.test(value);
}

/** Python `len()`: code points, not UTF-16 units. */
export const pyLen = (s: string): number => {
  let n = 0;
  for (const _ of s) n++;
  return n;
};

/** Python `s[:n]` by code point. */
export const pySliceTo = (s: string, n: number): string => (s.length <= n ? s : Array.from(s).slice(0, n).join(''));

const FULLWIDTH: Record<string, string> = {};
for (let i = 0; i < 26; i++) {
  FULLWIDTH[String.fromCharCode(0xff21 + i)] = String.fromCharCode(0x41 + i);
  FULLWIDTH[String.fromCharCode(0xff41 + i)] = String.fromCharCode(0x61 + i);
}
for (let i = 0; i < 10; i++) FULLWIDTH[String.fromCharCode(0xff10 + i)] = String.fromCharCode(0x30 + i);
const FULLWIDTH_RE = /[Ａ-Ｚａ-ｚ０-９]/g;
const HAN = '[\\u4e00-\\u9fa5]';
const ASCIIISH = '[a-zA-Z0-9\\(\\)\\[\\]@#$%!&\\*\\-=+_]';
const RE_P_TAGS = new RegExp(`</p${PY_S}*>|<p${PY_S}*>|<p/>`, 'giu');
const RE_BR_TAGS = new RegExp(`</br${PY_S}*>|<br${PY_S}*>|<br/>`, 'giu');
const RE_HAN_HAN = new RegExp(`(?<=${HAN})${PY_S}+(?=${HAN})`, 'gu');
const RE_HAN_ASCII = new RegExp(`(?<=${HAN})${PY_S}+(?=${ASCIIISH})`, 'gu');
const RE_ASCII_HAN = new RegExp(`(?<=${ASCIIISH})${PY_S}+(?=${HAN})`, 'gu');
const RE_QUOTES_BEFORE_HAN = new RegExp(`['"]+(?=${HAN})`, 'gu');
const RE_QUOTES_AFTER_HAN = new RegExp(`(?<=${HAN})['"]+`, 'gu');
const RE_NNBSP_AFTER_NON_DIGIT = new RegExp(`(?<=[^${PY_RE_DIGIT}])\\u202F`, 'gu');

function stripOuter(name: string, open: string, close: string, forbidden: readonly string[]): string {
  if (name.startsWith(open) && name.endsWith(close)) {
    const inner = name.slice(open.length, -close.length);
    if (!forbidden.some((q) => inner.includes(q))) return inner;
  }
  return name;
}

/** utils.normalize_extracted_info */
export function normalizeExtractedInfo(name: string, removeInnerQuotes = false): string {
  name = name.replace(RE_P_TAGS, '').replace(RE_BR_TAGS, '');
  name = name.replace(FULLWIDTH_RE, (c) => FULLWIDTH[c]!);
  name = name.replaceAll('－', '-').replaceAll('＋', '+').replaceAll('／', '/').replaceAll('＊', '*');
  name = name.replaceAll('（', '(').replaceAll('）', ')');
  name = name.replaceAll('—', '-').replaceAll('－', '-');
  name = name.replaceAll('　', ' ');
  name = name.replace(RE_HAN_HAN, '').replace(RE_HAN_ASCII, '').replace(RE_ASCII_HAN, '');
  if (pyLen(name) >= 2) {
    name = stripOuter(name, '"', '"', ['"']);
    name = stripOuter(name, "'", "'", ["'"]);
    name = stripOuter(name, '“', '”', ['“', '”']);
    name = stripOuter(name, '‘', '’', ['‘', '’']);
    name = stripOuter(name, '《', '》', ['《', '》']);
  }
  if (removeInnerQuotes) {
    name = name.replaceAll('“', '').replaceAll('”', '').replaceAll('‘', '').replaceAll('’', '');
    name = name.replace(RE_QUOTES_BEFORE_HAN, '').replace(RE_QUOTES_AFTER_HAN, '');
    name = name.replaceAll(' ', ' ');
    name = name.replace(RE_NNBSP_AFTER_NON_DIGIT, ' ');
  }
  name = pyStrip(name);
  const len = pyLen(name);
  if (len < 3 && /^[0-9]+\n?$/.test(name)) return '';
  if (len < 6 && name.includes('.') && Array.from(name).every((c) => c === '.' || isPyDigit(c))) return '';
  return name;
}

/** utils.sanitize_and_normalize_extracted_text */
export function sanitizeAndNormalizeExtractedText(inputText: string, removeInnerQuotes = false): string {
  const safe = sanitizeTextForEncoding(inputText);
  return safe ? normalizeExtractedInfo(safe, removeInnerQuotes) : '';
}

/** utils.fix_tuple_delimiter_corruption */
export function fixTupleDelimiterCorruption(record: string, delimiterCore: string, tupleDelimiter: string): string {
  if (!record || !delimiterCore || !tupleDelimiter) return record;
  const d = reEscape(delimiterCore);
  const subs: string[] = [
    `<\\|${d}\\|*?${d}\\|>`,
    `<\\|\\\\${d}\\|>`,
    `<\\|+>`,
    `<[^\\n]?\\|${d}\\|[^\\n]?>`,
    `<\\|?${d}\\|?>`,
    `<[^|]${d}\\|>|<\\|${d}[^|]>`,
    `<\\|${d}\\|+(?!>)`,
    `<\\|${d}:(?!>)`,
    `<\\|+${d}>`,
    `<\\|\\|(?!>)`,
    `(?<!<)\\|${d}\\|>`,
    `<\\|${d}\\|>\\|`,
    `\\|\\|${d}\\|\\|`,
  ];
  // replace() with a function: a `$` in the delimiter must stay literal
  for (const p of subs) record = record.replace(new RegExp(p, 'gu'), () => tupleDelimiter);
  return record;
}

/** utils.remove_think_tags */
export function removeThinkTags(text: string): string {
  // Python's `.` with DOTALL is any code point; [\s\S] is the JS spelling
  text = text.replace(/^(?:(?!<think>)[\s\S])*?<\/think>/u, '');
  text = text.replace(/<think>[\s\S]*?<\/think>/gu, '');
  return pyStrip(text);
}

/** The slice of BaseKVStorage the LLM cache path uses. */
export type LlmResponseCache = {
  global_config: { enable_llm_cache?: unknown; enable_llm_cache_for_entity_extract?: unknown; [k: string]: unknown };
  get_by_id(id: string): Promise<Record<string, unknown> | null>;
  upsert(data: Record<string, Record<string, unknown>>): Promise<void>;
};

export type LlmFunc = (
  prompt: string,
  opts: { system_prompt?: string | null; history_messages?: ChatMessage[]; max_tokens?: number; _priority?: number },
) => Promise<string>;

/** utils.handle_cache */
export async function handleCache(
  hashingKv: LlmResponseCache | null | undefined,
  argsHash: string,
  _prompt: string,
  mode = 'default',
  cacheType = 'unknown',
): Promise<[string, number] | null> {
  if (!hashingKv) return null;
  if (mode !== 'default' ? !hashingKv.global_config.enable_llm_cache : !hashingKv.global_config.enable_llm_cache_for_entity_extract) return null;
  const entry = await hashingKv.get_by_id(generateCacheKey(mode, cacheType, argsHash));
  if (entry) return [entry.return as string, (entry.create_time as number | undefined) ?? 0];
  return null;
}

/** utils.save_to_cache (CacheData flattened into arguments) */
export async function saveToCache(
  hashingKv: LlmResponseCache | null | undefined,
  d: { argsHash: string; content: string; prompt: string; mode?: string; cacheType?: string; chunkId?: string | null; queryparam?: Record<string, unknown> | null },
): Promise<void> {
  if (!hashingKv || !d.content) return;
  const key = generateCacheKey(d.mode ?? 'default', d.cacheType ?? 'query', d.argsHash);
  const existing = await hashingKv.get_by_id(key);
  if (existing && existing.return === d.content) return;
  await hashingKv.upsert({
    [key]: {
      return: d.content,
      cache_type: d.cacheType ?? 'query',
      chunk_id: d.chunkId ?? null,
      original_prompt: d.prompt,
      queryparam: d.queryparam ?? null,
    },
  });
}

/**
 * utils.use_llm_func_with_cache. Returns [response, timestamp]: the entry's
 * create_time on a hit, now (seconds) after a call.
 */
export async function useLlmFuncWithCache(
  userPrompt: string,
  useLlmFunc: LlmFunc,
  opts: {
    llmResponseCache?: LlmResponseCache | null;
    systemPrompt?: string | null;
    maxTokens?: number;
    historyMessages?: ChatMessage[] | null;
    cacheType?: string;
    chunkId?: string | null;
    cacheKeysCollector?: string[];
    now?: () => number;
  } = {},
): Promise<[string, number]> {
  const cacheType = opts.cacheType ?? 'extract';
  const now = opts.now ?? (() => Math.floor(Date.now() / 1000));
  const safeUser = sanitizeTextForEncoding(userPrompt);
  const safeSystem = opts.systemPrompt ? sanitizeTextForEncoding(opts.systemPrompt) : null;
  const safeHistory = opts.historyMessages?.length
    ? opts.historyMessages.map((m) => ({ ...m, content: sanitizeTextForEncoding(m.content) }))
    : null;
  const kwargs: { history_messages?: ChatMessage[]; max_tokens?: number } = {};
  if (safeHistory) kwargs.history_messages = safeHistory;
  if (opts.maxTokens !== undefined) kwargs.max_tokens = opts.maxTokens;

  const cache = opts.llmResponseCache;
  if (cache) {
    const { prompt, argsHash, cacheKey } = llmCacheLookup(userPrompt, {
      systemPrompt: opts.systemPrompt ?? null,
      historyMessages: opts.historyMessages ?? null,
      cacheType,
    });
    const hit = await handleCache(cache, argsHash, prompt, 'default', cacheType);
    if (hit) {
      opts.cacheKeysCollector?.push(cacheKey);
      return hit;
    }
    const res = removeThinkTags(await useLlmFunc(safeUser, { system_prompt: safeSystem, ...kwargs }));
    const ts = now();
    if (cache.global_config.enable_llm_cache_for_entity_extract) {
      await saveToCache(cache, { argsHash, content: res, prompt, cacheType, chunkId: opts.chunkId ?? null });
      opts.cacheKeysCollector?.push(cacheKey);
    }
    return [res, ts];
  }
  let res: string;
  try {
    res = await useLlmFunc(safeUser, { system_prompt: safeSystem, ...kwargs });
  } catch (e) {
    if (e instanceof Error) e.message = `[LLM func] ${e.message}`;
    throw e;
  }
  return [removeThinkTags(res), now()];
}

// ---------------------------------------------------------------------------
// Merge helpers (operate.merge_nodes_and_edges and friends).
// ---------------------------------------------------------------------------

/** utils.truncate_list_by_token_size: the longest prefix whose summed token counts stay within the budget. */
export function truncateListByTokenSize<T>(listData: readonly T[], key: (x: T) => string, maxTokenSize: number, tokenizer: Tokenizer): T[] {
  if (maxTokenSize <= 0) return [];
  let tokens = 0;
  for (let i = 0; i < listData.length; i++) {
    tokens += tokenizer.encode(key(listData[i]!)).length;
    if (tokens > maxTokenSize) return listData.slice(0, i);
  }
  return [...listData];
}

/** utils.normalize_source_ids_limit_method */
export function normalizeSourceIdsLimitMethod(method: string | null | undefined): 'FIFO' | 'KEEP' {
  if (!method) return 'FIFO';
  const m = method.toUpperCase();
  return m === 'KEEP' || m === 'FIFO' ? m : 'FIFO';
}

/** utils.merge_source_ids: order-preserving union, empty ids dropped. */
export function mergeSourceIds(existingIds: Iterable<string> | null | undefined, newIds: Iterable<string> | null | undefined): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const seq of [existingIds, newIds]) {
    if (!seq) continue;
    for (const id of seq) {
      if (!id || seen.has(id)) continue;
      seen.add(id);
      merged.push(id);
    }
  }
  return merged;
}

/** utils.apply_source_ids_limit: FIFO keeps the newest (tail), KEEP the oldest (head). */
export function applySourceIdsLimit(sourceIds: readonly string[], limit: number, method: string | null | undefined): string[] {
  if (limit <= 0) return [];
  if (sourceIds.length <= limit) return [...sourceIds];
  return normalizeSourceIdsLimitMethod(method) === 'FIFO' ? sourceIds.slice(-limit) : sourceIds.slice(0, limit);
}

/** utils.make_relation_chunk_key: the pair sorted (code point order) and joined with GRAPH_FIELD_SEP. */
export function makeRelationChunkKey(src: string, tgt: string): string {
  return [src, tgt].sort(pyCompare).join('<SEP>');
}

/** utils.parse_relation_chunk_key */
export function parseRelationChunkKey(key: string): [string, string] {
  const parts = key.split('<SEP>');
  if (parts.length !== 2) throw new Error(`Invalid relation chunk key: ${key}`);
  return [parts[0]!, parts[1]!];
}

/**
 * Builtin `sum()` of Python 3.12 over floats, start 0: the first item is added
 * to int 0 exactly, the rest with Neumaier compensation (CPython bltinmodule.c).
 * A plain left-to-right `+` can differ in the last bit.
 */
export function pySumFloats(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let f = 0 + xs[0]!;
  let c = 0;
  for (let i = 1; i < xs.length; i++) {
    const x = xs[i]!;
    const t = f + x;
    if (Math.abs(f) >= Math.abs(x)) c += f - t + x;
    else c += x - t + f;
    f = t;
  }
  if (c !== 0 && Number.isFinite(c)) f += c;
  return f;
}

/** Python `repr()` of a str, close enough for messages: quote choice and the common escapes. */
export function pyReprStr(s: string): string {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (ch === '\\' || ch === q) out += '\\' + ch;
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else if (cp < 0x20 || cp === 0x7f) out += '\\x' + cp.toString(16).padStart(2, '0');
    else out += ch;
  }
  return out + q;
}

/**
 * utils.safe_vdb_operation_with_exception: retry, then raise
 * `VDB {operation} failed for {name} after {n} attempts: {e}`.
 */
export async function safeVdbOperationWithException(
  operation: () => Promise<unknown>,
  operationName: string,
  entityName = '',
  maxRetries = 3,
  retryDelay = 0.2,
  log: (msg: string) => void = () => {},
): Promise<void> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      await operation();
      return;
    } catch (e) {
      if (attempt >= maxRetries - 1) {
        const msg = `VDB ${operationName} failed for ${entityName} after ${maxRetries} attempts: ${(e as Error)?.message ?? e}`;
        log(msg);
        throw new Error(msg, { cause: e });
      }
      log(`VDB ${operationName} attempt ${attempt + 1} failed for ${entityName}: ${(e as Error)?.message ?? e}, retrying...`);
      if (retryDelay > 0) await new Promise((r) => setTimeout(r, retryDelay * 1000));
    }
  }
}

/** The slice of BaseKVStorage update_chunk_cache_list uses. */
export type ChunkKv = {
  get_by_id(id: string): Promise<Record<string, unknown> | null>;
  upsert(data: Record<string, Record<string, unknown>>): Promise<void>;
};

/** utils.update_chunk_cache_list (failures are logged upstream, swallowed here too) */
export async function updateChunkCacheList(chunkId: string, textChunksStorage: ChunkKv, cacheKeys: string[]): Promise<void> {
  if (!cacheKeys.length) return;
  try {
    const chunk = await textChunksStorage.get_by_id(chunkId);
    if (!chunk) return;
    const list = (chunk.llm_cache_list ??= []) as string[];
    const existing = new Set(list);
    const fresh = cacheKeys.filter((k) => !existing.has(k));
    if (fresh.length) {
      list.push(...fresh);
      await textChunksStorage.upsert({ [chunkId]: chunk });
    }
  } catch {
    // upstream: logger.warning and continue
  }
}
