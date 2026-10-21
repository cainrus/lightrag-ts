// Port of the lightrag/utils.py helpers the query path uses. Kept out of
// src/utils.ts while B2 edits that file; same upstream names in camelCase.
import { createHash } from 'node:crypto';
import { DEFAULT_MAX_TOTAL_TOKENS } from '../constants.ts';
import { computeArgsHash, pyLen, pyStrip, type Tokenizer } from '../utils.ts';
import type { QueryParam, QueryRawData, ReferenceEntry } from './base.ts';

export type Dict = Record<string, unknown>;

/** Python `d.get(key, default)`: a present key wins even when its value is None. */
export function pyGet<T = unknown>(d: Dict | null | undefined, key: string, dflt: T): T {
  return d && Object.hasOwn(d, key) ? (d[key] as T) : dflt;
}

/** Python truthiness for JSON-shaped values. */
export function pyTruthy(v: unknown): boolean {
  if (v === null || v === undefined || v === false || v === 0 || v === '') return false;
  if (Number.isNaN(v)) return true;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof Map || v instanceof Set) return v.size > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return true;
}

/** Python `repr(float)`: shortest round-trip digits, Python's switch to exponent form. */
export function pyFloatRepr(x: number): string {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const [mant, expStr] = x.toExponential().split('e') as [string, string];
  const exp = Number(expStr);
  const neg = mant.startsWith('-');
  const digits = mant.replace('-', '').replace('.', '');
  const sign = neg ? '-' : '';
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  if (exp < 0) return `${sign}0.${'0'.repeat(-exp - 1)}${digits}`;
  const intLen = exp + 1;
  if (digits.length <= intLen) return `${sign}${digits}${'0'.repeat(intLen - digits.length)}.0`;
  return `${sign}${digits.slice(0, intLen)}.${digits.slice(intLen)}`;
}

/**
 * Python `str(x)` for the argument types compute_args_hash receives. JS has
 * no int/float split: integral numbers print as Python ints (QueryParam's
 * numeric fields are ints), others as `repr(float)`.
 */
export function pyStr(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'bigint') return v.toString();
  if (typeof v === 'number') return Number.isInteger(v) ? BigInt(v).toString() : pyFloatRepr(v);
  throw new Error(`pyStr: no verified Python str() for ${typeof v}`);
}

/** utils.compute_args_hash with Python `str()` of each argument (True, None, 0.5, ...). */
export function computeArgsHashPy(...args: unknown[]): string {
  return computeArgsHash(...args.map(pyStr));
}

/** Python `round(x)` for a float: round half to even. */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const ESC: Record<string, string> = { '"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f' };
const hex4 = (c: number) => '\\u' + c.toString(16).padStart(4, '0');

function pyJsonString(s: string, ensureAscii: boolean): string {
  const re = ensureAscii ? /[\\"\u0000-\u001f\u007f-￿]/g : /[\\"\u0000-\u001f]/g;
  return '"' + s.replace(re, (c) => ESC[c] ?? hex4(c.charCodeAt(0))) + '"';
}

/**
 * Python `json.dumps(v, ensure_ascii=...)` with default separators for
 * JSON-shaped values, floats in `repr` form. Lone surrogates behave as in
 * Python: raw without ensure_ascii, `\udXXX` with it. JS cannot tell 1.0 from
 * 1, so an integral float prints as an int ("1", Python "1.0").
 */
export function pyJsonDumpsAny(v: unknown, ensureAscii = false): string {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'string') return pyJsonString(v, ensureAscii);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'NaN';
    if (!Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
    return Number.isInteger(v) ? BigInt(v).toString() : pyFloatRepr(v);
  }
  if (Array.isArray(v)) return '[' + v.map((x) => pyJsonDumpsAny(x, ensureAscii)).join(', ') + ']';
  if (typeof v === 'object') {
    return '{' + Object.entries(v).map(([k, x]) => `${pyJsonString(k, ensureAscii)}: ${pyJsonDumpsAny(x, ensureAscii)}`).join(', ') + '}';
  }
  throw new Error(`pyJsonDumpsAny: unsupported ${typeof v}`);
}

const isEmptyContainer = (v: unknown): boolean =>
  v === '' || (Array.isArray(v) && v.length === 0) || (v !== null && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0);

/** json_repair's ObjectComparer.is_same_object: same type and shape, atoms compare by type only. */
function isSameObject(a: unknown, b: unknown): boolean {
  const kind = (v: unknown) => (v === null ? 'null' : Array.isArray(v) ? 'list' : typeof v === 'number' ? (Number.isInteger(v) ? 'int' : 'float') : typeof v);
  if (kind(a) !== kind(b)) return false;
  if (Array.isArray(a)) return a.length === (b as unknown[]).length && a.every((x, i) => isSameObject(x, (b as unknown[])[i]));
  if (a !== null && typeof a === 'object') {
    const ka = Object.keys(a);
    const bo = b as Dict;
    return ka.length === Object.keys(bo).length && ka.every((k) => Object.hasOwn(bo, k) && isSameObject((a as Dict)[k], bo[k]));
  }
  return true;
}

const QUOTES = new Set(['"', "'", '“', '”']);
const CLOSE_QUOTE: Record<string, string> = { '"': '"', "'": "'", '“': '”', '”': '”' };
const JSON_ESC: Record<string, string> = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };

/**
 * A lenient JSON reader for what LLMs answer, modelled on json_repair 0.61
 * (`json_repair.loads`, which upstream uses for keyword JSON). Valid JSON
 * parses as JSON. Otherwise: text before the first `{` / `[` is skipped,
 * trailing and doubled commas dropped, missing commas tolerated, strings,
 * arrays and objects cut off by the end of the text closed, a key without a
 * value dropped, single-quoted strings and unquoted keys / words read,
 * `true`/`false`/`null` and Python's True/False recognised (None stays a word), `#` and
 * `//` comments skipped. Several top-level values become a list, except that a
 * value shaped like the previous one replaces it (json_repair's "update"
 * rule). No JSON at all gives `""`, and `log` gets a warning when the text was
 * not blank. Not ported: json_repair's heuristics for unescaped quotes inside
 * strings, parenthesised tuples and schema-guided repair.
 */
export function jsonRepairLoads(text: string, log?: QueryLog): unknown {
  try {
    return JSON.parse(text);
  } catch {
    // fall through to the lenient reader
  }
  const s = text;
  let i = 0;
  const ws = () => {
    while (i < s.length && /\s/.test(s[i]!)) i++;
  };
  const skipComment = (): boolean => {
    if (s[i] === '#' || (s[i] === '/' && s[i + 1] === '/')) {
      while (i < s.length && s[i] !== '\n') i++;
      return true;
    }
    if (s[i] === '/' && s[i + 1] === '*') {
      const end = s.indexOf('*/', i + 2);
      i = end < 0 ? s.length : end + 2;
      return true;
    }
    return false;
  };

  function parseString(): string {
    const open = s[i]!;
    const close = CLOSE_QUOTE[open]!;
    i++;
    let out = '';
    while (i < s.length && s[i] !== close) {
      if (s[i] === '\\' && i + 1 < s.length) {
        const e = s[i + 1]!;
        if (e === 'u' && /^[0-9a-fA-F]{4}$/.test(s.slice(i + 2, i + 6))) {
          out += String.fromCharCode(Number.parseInt(s.slice(i + 2, i + 6), 16));
          i += 6;
          continue;
        }
        out += JSON_ESC[e] ?? e;
        i += 2;
        continue;
      }
      out += s[i]!;
      i++;
    }
    if (i < s.length) i++; // closing quote; at the end of text the string is closed for it
    return out;
  }

  /** A bare word: a literal, or an unquoted string ended by a delimiter. */
  function parseBare(stops: string): unknown {
    const start = i;
    while (i < s.length && !stops.includes(s[i]!) && s[i] !== '\n') i++;
    const word = s.slice(start, i).trim();
    if (word === 'true' || word === 'True') return true;
    if (word === 'false' || word === 'False') return false;
    if (word === 'null') return null; // json_repair reads True/False but leaves None a string
    if (/^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(word)) return Number(word);
    return word;
  }

  function parseArray(): unknown[] {
    i++;
    const arr: unknown[] = [];
    for (;;) {
      ws();
      while (s[i] === ',') {
        i++;
        ws();
      }
      if (i >= s.length || s[i] === ']' || s[i] === '}') break;
      if (skipComment()) continue;
      const before = i;
      const v = parseValue(',]}');
      if (isEmptyContainer(v) && s[i] !== ']' && s[i] !== ',') {
        if (i === before) i++;
      } else arr.push(v);
    }
    if (i < s.length) i++; // `]`, or a stray `}` json_repair also eats
    return arr;
  }

  function parseObject(): Dict {
    i++;
    const obj: Dict = {};
    for (;;) {
      ws();
      while (s[i] === ',') {
        i++;
        ws();
      }
      if (i >= s.length || s[i] === '}') break;
      if (s[i] === ']') {
        i++;
        continue;
      }
      if (skipComment()) continue;
      let key: string;
      if (QUOTES.has(s[i]!)) key = parseString();
      else if (/[\p{L}\p{N}_$-]/u.test(s[i]!)) key = String(parseBare(':,}'));
      else {
        i++;
        continue;
      }
      ws();
      if (s[i] !== ':') {
        // a key with no value: dropped (a cut-off answer ends like this)
        continue;
      }
      i++;
      ws();
      obj[key] = i >= s.length || s[i] === ',' || s[i] === '}' ? '' : parseValue(',}');
    }
    if (i < s.length) i++;
    return obj;
  }

  function parseValue(stops: string): unknown {
    ws();
    if (i >= s.length) return '';
    const c = s[i]!;
    if (c === '{') return parseObject();
    if (c === '[') return parseArray();
    if (QUOTES.has(c)) return parseString();
    return parseBare(stops);
  }

  /** json_repair's parse_json at top level: skip prose until a container starts. */
  function parseTop(): unknown {
    while (i < s.length) {
      const c = s[i]!;
      if (c === '{') return parseObject();
      if (c === '[') return parseArray();
      if (!skipComment()) i++;
    }
    return '';
  }

  let result: unknown = parseTop();
  if (i < s.length) {
    const list: unknown[] = [result];
    while (i < s.length) {
      let j = i;
      while (j < s.length && /\s/.test(s[j]!)) j++;
      let k = i - 1;
      while (k >= 0 && /\s/.test(s[k]!)) k--;
      const commaSeparated = s[j] === ',' || (k >= 0 && s[k] === ',');
      const v = parseTop();
      if (pyTruthy(v)) {
        if (!commaSeparated && isSameObject(list.at(-1), v)) list.pop();
        else if (!pyTruthy(list.at(-1))) list.pop();
        list.push(v);
      } else i++;
    }
    result = list.length === 1 ? list[0] : list;
  }
  if (result === '' && pyStrip(text)) log?.('warning', `json_repair: no JSON found in ${pyLen(text)} chars of LLM output`);
  return result;
}

/** utils.truncate_list_by_token_size */
export function truncateListByTokenSize<T>(listData: T[], key: (x: T) => string, maxTokenSize: number, tokenizer: Tokenizer): T[] {
  if (maxTokenSize <= 0) return [];
  let tokens = 0;
  for (let i = 0; i < listData.length; i++) {
    tokens += tokenizer.encode(key(listData[i]!)).length;
    if (tokens > maxTokenSize) return listData.slice(0, i);
  }
  return listData;
}

/** The truncation key upstream uses everywhere: `"\n".join(json.dumps(item, ensure_ascii=False) for item in [x])`. */
export const jsonLineKey = (x: unknown): string => pyJsonDumpsAny(x, false);

/**
 * utils.cosine_similarity. Upstream mixes a float32 query norm with float64
 * math; that norm is a common factor of every score in one call, so the
 * ranking is the same in double precision.
 */
export function cosineSimilarity(v1: ArrayLike<number>, v2: ArrayLike<number>): number {
  let dot = 0;
  let n1 = 0;
  let n2 = 0;
  for (let i = 0; i < v1.length; i++) {
    const a = v1[i]!;
    const b = v2[i]!;
    dot += a * b;
    n1 += a * a;
    n2 += b * b;
  }
  if (v1.length !== v2.length) throw new Error(`shapes (${v1.length},) and (${v2.length},) not aligned`);
  return dot / (Math.sqrt(n1) * Math.sqrt(n2));
}

export type ChunkCarrier = { sorted_chunks?: string[]; [k: string]: unknown };

/** utils.pick_by_weighted_polling */
export function pickByWeightedPolling(entitiesOrRelations: ChunkCarrier[], maxRelatedChunks: number, minRelatedChunks = 1): string[] {
  if (!entitiesOrRelations.length) return [];
  const n = entitiesOrRelations.length;
  // JS slice(0, negative) drops from the end exactly like Python's [:negative]
  if (n === 1) return (entitiesOrRelations[0]!.sorted_chunks ?? []).slice(0, maxRelatedChunks);
  const expectedCounts: number[] = [];
  for (let i = 0; i < n; i++) {
    const ratio = i / (n - 1);
    expectedCounts.push(pyRound(maxRelatedChunks - ratio * (maxRelatedChunks - minRelatedChunks)));
  }
  const selected: string[] = [];
  const used: number[] = [];
  let totalRemaining = 0;
  entitiesOrRelations.forEach((er, i) => {
    const chunks = er.sorted_chunks ?? [];
    const expected = expectedCounts[i]!;
    const actual = Math.min(expected, chunks.length);
    selected.push(...chunks.slice(0, actual));
    used.push(actual);
    const remaining = expected - actual;
    if (remaining > 0) totalRemaining += remaining;
  });
  for (let r = 0; r < totalRemaining; r++) {
    let allocated = false;
    for (let i = 0; i < n; i++) {
      const chunks = entitiesOrRelations[i]!.sorted_chunks ?? [];
      if (used[i]! < chunks.length) {
        selected.push(chunks.at(used[i]!)!);
        used[i]!++;
        allocated = true;
        break;
      }
    }
    if (!allocated) break;
  }
  return selected;
}

export type QueryEmbeddingFunc = (texts: string[], opts?: { context?: 'document' | 'query'; _priority?: number }) => Promise<ArrayLike<ArrayLike<number>>>;

/** utils.pick_by_vector_similarity. Candidate order is first-seen order; upstream iterates a Python set. */
export async function pickByVectorSimilarity(a: {
  query: string;
  chunksVdb: { get_vectors_by_ids(ids: string[]): Promise<Record<string, ArrayLike<number>>> };
  numOfChunks: number;
  entityInfo: ChunkCarrier[];
  embeddingFunc: QueryEmbeddingFunc;
  queryEmbedding?: ArrayLike<number> | null;
  log?: QueryLog;
}): Promise<string[]> {
  const log = a.log ?? noLog;
  if (!a.entityInfo.length || a.numOfChunks <= 0) return [];
  const seen = new Set<string>();
  for (const e of a.entityInfo) for (const id of e.sorted_chunks ?? []) seen.add(id);
  if (!seen.size) {
    log('warning', 'Vector similarity chunk selection:  no chunk IDs found in entity_info');
    return [];
  }
  const allChunkIds = [...seen];
  try {
    let qe = a.queryEmbedding ?? null;
    if (qe === null) qe = (await a.embeddingFunc([a.query], { context: 'query' }))[0]!;
    const vectors = await a.chunksVdb.get_vectors_by_ids(allChunkIds);
    const got = Object.keys(vectors).length;
    if (!got || got !== allChunkIds.length) {
      log('warning', got ? `Vector similarity chunk selection: found ${got} but expecting ${allChunkIds.length}` : 'Vector similarity chunk selection: no vectors retrieved from chunks_vdb');
      return [];
    }
    const sims: [string, number][] = [];
    for (const id of allChunkIds) {
      if (!Object.hasOwn(vectors, id)) {
        log('warning', `Vector similarity chunk selection:  no vector found for chunk ${id}`);
        continue;
      }
      try {
        sims.push([id, cosineSimilarity(qe, vectors[id]!)]);
      } catch (e) {
        log('warning', `Vector similarity chunk selection: failed to calculate similarity for chunk ${id}: ${e}`);
      }
    }
    sims.sort((x, y) => y[1] - x[1]);
    return sims.slice(0, a.numOfChunks).map(([id]) => id);
  } catch (e) {
    log('error', `[VECTOR_SIMILARITY] Error in vector similarity sorting: ${e}`);
    return allChunkIds.slice(0, a.numOfChunks);
  }
}

export type RerankResult = { index: number; relevance_score: number } | Dict;
export type RerankFunc = (a: { query: string; documents: string[]; top_n: number | null }) => Promise<RerankResult[]>;

/** utils.apply_rerank_if_enabled */
export async function applyRerankIfEnabled(
  query: string,
  retrievedDocs: Dict[],
  globalConfig: { rerank_model_func?: RerankFunc | null; log?: QueryLog },
  enableRerank = true,
  topN: number | null = null,
): Promise<Dict[]> {
  const log = globalConfig.log ?? noLog;
  if (!enableRerank || !retrievedDocs.length) return retrievedDocs;
  const rerank = globalConfig.rerank_model_func;
  if (!rerank) {
    log('warning', 'Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.');
    return retrievedDocs;
  }
  try {
    const texts = retrievedDocs.map((d) => {
      for (const k of ['content', 'text', 'chunk_content', 'document']) if (pyTruthy(d[k])) return d[k] as string;
      return pyJsonDumpsAny(d); // upstream: str(doc), a Python repr; no ported caller reaches it
    });
    const results = await rerank({ query, documents: texts, top_n: topN });
    if (results && results.length > 0) {
      const first = results[0];
      if (first && typeof first === 'object' && 'index' in first) {
        const out: Dict[] = [];
        for (const r of results as { index: number; relevance_score: number }[]) {
          if (r.index >= 0 && r.index < retrievedDocs.length) out.push({ ...retrievedDocs[r.index]!, rerank_score: r.relevance_score });
        }
        log('info', `Successfully reranked: ${out.length} chunks from ${retrievedDocs.length} original chunks`);
        return out;
      }
      log('info', `Using legacy rerank format: ${results.length} chunks`);
      return (topN ? results.slice(0, topN) : results) as Dict[];
    }
    log('warning', 'Rerank returned empty results, using original chunks');
    return retrievedDocs;
  } catch (e) {
    log('error', `Error during reranking: ${e}, using original chunks`);
    return retrievedDocs;
  }
}

export type ChunkGlobalConfig = {
  tokenizer?: Tokenizer | null;
  rerank_model_func?: RerankFunc | null;
  min_rerank_score?: number;
  log?: QueryLog;
  [k: string]: unknown;
};

/** utils.process_chunks_unified */
export async function processChunksUnified(a: {
  query: string;
  uniqueChunks: Dict[];
  queryParam: QueryParam;
  globalConfig: ChunkGlobalConfig;
  sourceType?: string;
  chunkTokenLimit?: number | null;
}): Promise<Dict[]> {
  let chunks = a.uniqueChunks;
  if (!chunks.length) return [];
  const qp = a.queryParam;
  if (qp.enable_rerank && a.query && chunks.length) {
    const rerankTopK = qp.chunk_top_k || chunks.length;
    chunks = await applyRerankIfEnabled(a.query, chunks, a.globalConfig, qp.enable_rerank, rerankTopK);
  }
  if (qp.enable_rerank && chunks.length) {
    const minScore = pyGet(a.globalConfig, 'min_rerank_score', 0.5);
    if (minScore > 0.0) {
      chunks = chunks.filter((c) => pyGet(c, 'rerank_score', 1.0) >= minScore);
      if (!chunks.length) return [];
    }
  }
  if (qp.chunk_top_k !== null && qp.chunk_top_k > 0 && chunks.length > qp.chunk_top_k) chunks = chunks.slice(0, qp.chunk_top_k);
  const tokenizer = a.globalConfig.tokenizer;
  if (tokenizer && chunks.length) {
    const limit = a.chunkTokenLimit ?? qp.max_total_tokens ?? DEFAULT_MAX_TOTAL_TOKENS;
    chunks = truncateListByTokenSize(chunks, jsonLineKey, limit, tokenizer);
  }
  return chunks.map((c, i) => ({ ...c, id: `DC${i + 1}` }));
}

/** Map key for Python's (a, b) tuple keys. */
export const pairKey = (a: unknown, b: unknown): string => JSON.stringify([a ?? null, b ?? null]);

/** utils.convert_to_user_format */
export function convertToUserFormat(
  entitiesContext: Dict[],
  relationsContext: Dict[],
  chunks: Dict[],
  references: ReferenceEntry[],
  queryMode: string,
  entityIdToOriginal?: Map<string, Dict> | null,
  relationIdToOriginal?: Map<string, Dict> | null,
): QueryRawData {
  const entities = entitiesContext.map((e) => {
    const name = pyGet(e, 'entity', '');
    const o = entityIdToOriginal && entityIdToOriginal.has(name as string) ? entityIdToOriginal.get(name as string)! : null;
    if (pyTruthy(o)) {
      return {
        entity_name: pyGet(o, 'entity_name', name),
        entity_type: pyGet(o, 'entity_type', 'UNKNOWN'),
        description: pyGet(o, 'description', ''),
        source_id: pyGet(o, 'source_id', ''),
        file_path: pyGet(o, 'file_path', 'unknown_source'),
        created_at: pyGet(o, 'created_at', ''),
      };
    }
    return {
      entity_name: name,
      entity_type: pyGet(e, 'type', 'UNKNOWN'),
      description: pyGet(e, 'description', ''),
      source_id: pyGet(e, 'source_id', ''),
      file_path: pyGet(e, 'file_path', 'unknown_source'),
      created_at: pyGet(e, 'created_at', ''),
    };
  });
  const relationships = relationsContext.map((r) => {
    const e1 = pyGet(r, 'entity1', '');
    const e2 = pyGet(r, 'entity2', '');
    const k = pairKey(e1, e2);
    const o = relationIdToOriginal && relationIdToOriginal.has(k) ? relationIdToOriginal.get(k)! : null;
    const src = pyTruthy(o) ? o! : null;
    return {
      src_id: src ? pyGet(src, 'src_id', e1) : e1,
      tgt_id: src ? pyGet(src, 'tgt_id', e2) : e2,
      description: pyGet(src ?? r, 'description', ''),
      keywords: pyGet(src ?? r, 'keywords', ''),
      weight: pyGet(src ?? r, 'weight', 1.0),
      source_id: pyGet(src ?? r, 'source_id', ''),
      file_path: pyGet(src ?? r, 'file_path', 'unknown_source'),
      created_at: pyGet(src ?? r, 'created_at', ''),
    };
  });
  const formattedChunks = chunks.map((c) => ({
    reference_id: pyGet(c, 'reference_id', ''),
    content: pyGet(c, 'content', ''),
    file_path: pyGet(c, 'file_path', 'unknown_source'),
    chunk_id: pyGet(c, 'chunk_id', ''),
  }));
  return {
    status: 'success',
    message: 'Query processed successfully',
    data: { entities, relationships, chunks: formattedChunks, references },
    metadata: { query_mode: queryMode, keywords: { high_level: [], low_level: [] } },
  };
}

/** utils.generate_reference_list_from_chunks */
export function generateReferenceListFromChunks(chunks: Dict[]): [ReferenceEntry[], Dict[]] {
  if (!chunks.length) return [[], []];
  const valid = (p: unknown): p is string => pyTruthy(p) && p !== 'unknown_source';
  const counts = new Map<string, number>();
  for (const c of chunks) {
    const p = pyGet(c, 'file_path', '');
    if (valid(p)) counts.set(p, (counts.get(p) ?? 0) + 1);
  }
  const withIdx: [string, number, number][] = [];
  const seen = new Set<string>();
  chunks.forEach((c, i) => {
    const p = pyGet(c, 'file_path', '');
    if (valid(p) && !seen.has(p)) {
      withIdx.push([p, counts.get(p)!, i]);
      seen.add(p);
    }
  });
  withIdx.sort((x, y) => y[1] - x[1] || x[2] - y[2]);
  const ids = new Map<string, string>();
  withIdx.forEach(([p], i) => ids.set(p, String(i + 1)));
  const updated = chunks.map((c) => {
    const p = pyGet(c, 'file_path', '');
    return { ...c, reference_id: valid(p) ? ids.get(p)! : '' };
  });
  return [withIdx.map(([p], i) => ({ reference_id: String(i + 1), file_path: p })), updated];
}

export type QueryLog = (level: 'error' | 'warning' | 'info' | 'debug', msg: string) => void;
export const noLog: QueryLog = () => {};

/** md5 hex of a string, for diagnostics. */
export const md5 = (s: string): string => createHash('md5').update(s).digest('hex');
