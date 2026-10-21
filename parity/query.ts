// Parity harness, query stage (plan fact f3, step B3).
//
//   node parity/query.ts keys <working_dir> [--workspace W] [--language English] [--json out.json]
//
// Read-only. Unlike extraction, query cache entries are keyed by a hash of the
// query PARAMETERS, not of the prompt (compute_args_hash over mode, query,
// QueryParam fields and the joined keywords). For every `*:keywords:*` and
// `*:query:*` entry of kv_store_llm_response_cache.json this recomputes the key
// with the TS port from the entry's own original_prompt and queryparam:
//   keywords  - keywordsArgsHash(mode, query, language)
//   query     - stored: kgQueryArgsHash with the hl/ll strings the entry recorded
//               chained: the keywords come from replaying extract_keywords_only
//               against the cache (TS parses the cached JSON and joins it), so the
//               whole kg_query key path runs, not just the hash function
//   naive     - naiveQueryArgsHash
// It also checks that the TS port would write the same `queryparam` dict (key
// order included) and, for keywords, the same JSON `return` string.
//
//   node parity/query.ts context ...   (see parity/query-context.ts)
import { writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { forEachJsonKv, kvStorePath } from '../src/kg/json_kv_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import {
  type Dict,
  jsonRepairLoads,
  keywordsArgsHash,
  kgQueryArgsHash,
  kgQueryParamDict,
  makeQueryParam,
  naiveQueryArgsHash,
  pyJsonDumpsAny,
  type QueryGlobalConfig,
  type QueryMode,
  type QueryParam,
  queryParamDict,
  resolveKgKeywords,
} from '../src/query.ts';
import { computeArgsHash, generateCacheKey, type LlmResponseCache, TiktokenTokenizer } from '../src/utils.ts';

type Entry = { return: string; cache_type?: string; original_prompt?: string; queryparam?: Dict | null; create_time?: number };

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    workspace: { type: 'string', default: '' },
    language: { type: 'string', default: 'English' },
    json: { type: 'string' },
    'show-misses': { type: 'string', default: '5' },
  },
});
const [cmd, dir] = positionals;
if (cmd !== 'keys' || !dir) {
  console.error('usage: query.ts keys <working_dir> [--workspace W] [--language English] [--json out.json]');
  process.exit(2);
}

const t0 = performance.now();
const file = kvStorePath(dir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, values.workspace);
const entries = new Map<string, Entry>();
const byType: Record<string, number> = {};
forEachJsonKv<Entry>(file, (k, v) => {
  const [mode, type] = k.split(':');
  byType[`${mode}:${type}`] = (byType[`${mode}:${type}`] ?? 0) + 1;
  if (type === 'keywords' || type === 'query') entries.set(k, v);
});

const view: LlmResponseCache = {
  global_config: { enable_llm_cache: true },
  get_by_id: async (k) => (entries.get(k) as Dict | undefined) ?? null,
  upsert: async () => {
    throw new Error('read-only cache view: the replay asked to write');
  },
};
// Upstream saves a keywords entry only when the LLM returned some keywords. A
// query entry without one therefore had empty keywords (kg_query then forces
// the query itself as ll keyword): the stub replays exactly that answer, and
// the run counts how often it was needed.
let emptyKeywordReplays = 0;
const gc: QueryGlobalConfig = {
  addon_params: { language: values.language },
  tokenizer: new TiktokenTokenizer('gpt-4o-mini'),
  llm_model_func: async (_prompt, opts) => {
    if (!opts?.keyword_extraction) throw new Error('the replay asked the LLM for an answer');
    emptyKeywordReplays++;
    return '{"high_level_keywords": [], "low_level_keywords": []}';
  },
};

/** The QueryParam a cache entry was produced with, as far as its `queryparam` records it. */
function paramOf(mode: string, qp: Dict | null | undefined): QueryParam {
  const q = qp ?? {};
  return makeQueryParam({
    mode: mode as QueryMode,
    ...(Object.hasOwn(q, 'response_type') ? { response_type: q.response_type as string | null } : {}),
    ...(Object.hasOwn(q, 'top_k') ? { top_k: q.top_k as number } : {}),
    ...(Object.hasOwn(q, 'chunk_top_k') ? { chunk_top_k: q.chunk_top_k as number | null } : {}),
    ...(Object.hasOwn(q, 'max_entity_tokens') ? { max_entity_tokens: q.max_entity_tokens as number } : {}),
    ...(Object.hasOwn(q, 'max_relation_tokens') ? { max_relation_tokens: q.max_relation_tokens as number } : {}),
    ...(Object.hasOwn(q, 'max_total_tokens') ? { max_total_tokens: q.max_total_tokens as number } : {}),
    user_prompt: (q.user_prompt as string | undefined) || null,
    ...(Object.hasOwn(q, 'enable_rerank') ? { enable_rerank: q.enable_rerank as boolean } : {}),
  });
}

const misses: { key: string; check: string; detail: string }[] = [];
const miss = (key: string, check: string, detail: string) => misses.push({ key, check, detail });
const r = {
  keywords: { total: 0, keyHit: 0, queryparamSame: 0, returnSame: 0, promptHashWouldMatch: 0 },
  query: { total: 0, kg: 0, naive: 0, keyHitStored: 0, keyHitChained: 0, chainedViaEmptyKeywords: 0, keywordsAgree: 0, queryparamSame: 0, promptHashWouldMatch: 0 },
};
const keywordsClaimed = new Set<string>();

for (const [key, e] of entries) {
  const [mode, type, hash] = key.split(':') as [string, string, string];
  const query = e.original_prompt ?? '';
  const qp = paramOf(mode, e.queryparam);
  if (computeArgsHash(query) === hash) (type === 'keywords' ? r.keywords : r.query).promptHashWouldMatch++;

  if (type === 'keywords') {
    r.keywords.total++;
    const h = keywordsArgsHash(mode, query, values.language);
    if (h === hash) r.keywords.keyHit++;
    else miss(key, 'keywords key', `ts=${h} (mode=${mode}, language=${values.language})`);
    const want = JSON.stringify(e.queryparam);
    const got = JSON.stringify(queryParamDict(qp));
    if (got === want) r.keywords.queryparamSame++;
    else miss(key, 'keywords queryparam', `ts=${got} py=${want}`);
    const back = pyJsonDumpsAny(jsonRepairLoads(e.return), true);
    if (back === e.return) r.keywords.returnSame++;
    else miss(key, 'keywords return', `ts=${back.slice(0, 120)} py=${e.return.slice(0, 120)}`);
    continue;
  }

  r.query.total++;
  if (mode === 'naive') {
    r.query.naive++;
    const h = naiveQueryArgsHash(qp, query);
    if (h === hash) {
      r.query.keyHitStored++;
      r.query.keyHitChained++;
    } else miss(key, 'naive query key', `ts=${h}`);
    const got = JSON.stringify(queryParamDict(qp));
    if (got === JSON.stringify(e.queryparam)) r.query.queryparamSame++;
    else miss(key, 'naive queryparam', `ts=${got} py=${JSON.stringify(e.queryparam)}`);
    continue;
  }
  r.query.kg++;
  const storedHl = (e.queryparam?.hl_keywords as string | undefined) ?? '';
  const storedLl = (e.queryparam?.ll_keywords as string | undefined) ?? '';
  const stored = kgQueryArgsHash(qp, query, storedHl, storedLl);
  if (stored === hash) r.query.keyHitStored++;
  else miss(key, 'kg query key (stored keywords)', `ts=${stored}`);
  try {
    const before = emptyKeywordReplays;
    const kw = await resolveKgKeywords(query, qp, gc, view);
    if (emptyKeywordReplays > before) r.query.chainedViaEmptyKeywords++;
    if (kw === null) {
      miss(key, 'kg query key (chained)', 'keywords empty and query >= 50 chars: kg_query answers fail_response');
    } else {
      if (emptyKeywordReplays === before) keywordsClaimed.add(generateCacheKey(mode, 'keywords', keywordsArgsHash(mode, query, values.language)));
      if (kw.hl === storedHl && kw.ll === storedLl) r.query.keywordsAgree++;
      else miss(key, 'chained keywords', `ts hl=${JSON.stringify(kw.hl)} ll=${JSON.stringify(kw.ll)} py hl=${JSON.stringify(storedHl)} ll=${JSON.stringify(storedLl)}`);
      const chained = kgQueryArgsHash(qp, query, kw.hl, kw.ll);
      if (chained === hash) r.query.keyHitChained++;
      else miss(key, 'kg query key (chained)', `ts=${chained}`);
      const got = JSON.stringify(kgQueryParamDict(qp, kw.hl, kw.ll));
      if (got === JSON.stringify(e.queryparam)) r.query.queryparamSame++;
      else miss(key, 'kg queryparam', `ts=${got} py=${JSON.stringify(e.queryparam)}`);
    }
  } catch (err) {
    miss(key, 'kg query key (chained)', String((err as Error).message ?? err));
  }
}

const keywordKeys = [...entries.keys()].filter((k) => k.split(':')[1] === 'keywords');
const result = {
  file,
  language: values.language,
  seconds: Number(((performance.now() - t0) / 1000).toFixed(1)),
  cacheByType: byType,
  entries: entries.size,
  ...r,
  keywordsWithoutQueryEntry: keywordKeys.filter((k) => !keywordsClaimed.has(k)).length,
  missesByCheck: misses.reduce<Record<string, number>>((acc, m) => ((acc[m.check] = (acc[m.check] ?? 0) + 1), acc), {}),
  missSamples: misses.slice(0, Number(values['show-misses'])),
};
console.log(JSON.stringify(result, null, 2));
if (values.json) writeFileSync(values.json, JSON.stringify({ ...result, misses }, null, 2) + '\n');
process.exitCode = misses.length ? 1 : 0;
