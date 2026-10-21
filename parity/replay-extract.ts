// Parity harness, extraction stage (plan fact f3):
// can the TS port recompute the LLM-cache keys of upstream's extract_entities
// for every chunk of a real store, so B2 can replay extraction with zero API calls?
//
//   node parity/replay-extract.ts <working_dir> [--language English] [--json out.json]
//
// Checks, all read-only:
//   1. integrity  - md5(original_prompt) equals the hash in every cache key
//   2. initial    - TS-built prompt for each chunk hits `default:extract:<md5>`
//   3. gleaning   - with the cached initial answer as history, the continue
//                   prompt hits a second extract entry
//   4. coverage   - which cache entries are claimed by no chunk (orphans)
// For misses it prints the first divergence between the TS prompt and the
// stored original_prompt of an entry with the same chunk_id.
import { writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { kvStorePath, readJsonKv } from '../src/kg/json_kv_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import { buildExtractionPrompts } from '../src/operate.ts';
import { computeArgsHash, llmCacheLookup, packUserAssToOpenaiMessages } from '../src/utils.ts';

type CacheEntry = {
  return: string;
  cache_type?: string;
  chunk_id?: string | null;
  original_prompt?: string;
  create_time?: number;
};
type Chunk = { content: string; full_doc_id?: string; file_path?: string; llm_cache_list?: string[] };

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    language: { type: 'string', default: 'English' },
    'entity-types': { type: 'string' },
    workspace: { type: 'string', default: '' },
    json: { type: 'string' },
    'show-diffs': { type: 'string', default: '3' },
  },
});
const dir = positionals[0];
if (!dir) {
  console.error('usage: replay-extract.ts <working_dir> [--language L] [--entity-types a,b] [--json out.json]');
  process.exit(2);
}
const cfg = {
  language: values.language,
  ...(values['entity-types'] ? { entityTypes: values['entity-types'].split(',') } : {}),
};

const t0 = performance.now();
const cache = readJsonKv<CacheEntry>(kvStorePath(dir, NameSpace.KV_STORE_LLM_RESPONSE_CACHE, values.workspace));
const chunks = readJsonKv<Chunk>(kvStorePath(dir, NameSpace.KV_STORE_TEXT_CHUNKS, values.workspace));

// 1. integrity + index by chunk
const byType: Record<string, number> = {};
const hashBadByType: Record<string, number> = {};
let hashOk = 0;
let hashBad = 0;
let noPrompt = 0;
const entriesByChunk = new Map<string, string[]>();
for (const [key, e] of cache) {
  const [, type = '?', hash] = key.split(':');
  byType[type] = (byType[type] ?? 0) + 1;
  if (typeof e.original_prompt !== 'string') noPrompt++;
  else if (computeArgsHash(e.original_prompt) === hash) hashOk++;
  else {
    hashBad++;
    hashBadByType[type] = (hashBadByType[type] ?? 0) + 1;
  }
  if (e.chunk_id) {
    const list = entriesByChunk.get(e.chunk_id) ?? [];
    list.push(key);
    entriesByChunk.set(e.chunk_id, list);
  }
}

// 2 + 3. recompute keys from chunk content
const claimed = new Set<string>();
let initialHit = 0;
let gleanHit = 0;
let gleanMissAfterInitialHit = 0;
let chunksWithEntries = 0;
let cacheListAgrees = 0;
let cacheListPresent = 0;
const misses: { chunk: string; reason: string; detail?: string }[] = [];

function firstDiff(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

for (const [chunkId, chunk] of chunks) {
  const own = entriesByChunk.get(chunkId) ?? [];
  if (own.length) chunksWithEntries++;
  const p = buildExtractionPrompts(chunk.content, cfg);
  const init = llmCacheLookup(p.user, { systemPrompt: p.system, cacheType: 'extract' });
  const hit = cache.get(init.cacheKey);
  const mine: string[] = [];
  if (!hit) {
    let detail: string | undefined;
    const other = own.map((k) => cache.get(k)!).find((e) => e.original_prompt && !e.original_prompt.includes('"role": "assistant"'));
    if (other?.original_prompt) {
      const at = firstDiff(init.prompt, other.original_prompt);
      detail = `diverges at ${at}/${init.prompt.length}: ts=${JSON.stringify(init.prompt.slice(Math.max(0, at - 40), at + 60))} py=${JSON.stringify(other.original_prompt.slice(Math.max(0, at - 40), at + 60))}`;
    }
    misses.push({ chunk: chunkId, reason: own.length ? 'initial key differs' : 'no cache entry for chunk', ...(detail ? { detail } : {}) });
    continue;
  }
  initialHit++;
  mine.push(init.cacheKey);
  const glean = llmCacheLookup(p.continueUser, {
    systemPrompt: p.system,
    historyMessages: packUserAssToOpenaiMessages(p.user, hit.return),
    cacheType: 'extract',
  });
  if (cache.has(glean.cacheKey)) {
    gleanHit++;
    mine.push(glean.cacheKey);
  } else if (own.length > 1) {
    gleanMissAfterInitialHit++;
    let detail: string | undefined;
    const other = own.filter((k) => k !== init.cacheKey).map((k) => cache.get(k)!).find((e) => e.original_prompt?.includes('"role": "assistant"'));
    if (other?.original_prompt) {
      const at = firstDiff(glean.prompt, other.original_prompt);
      detail = `diverges at ${at}/${glean.prompt.length}: ts=${JSON.stringify(glean.prompt.slice(Math.max(0, at - 60), at + 60))} py=${JSON.stringify(other.original_prompt.slice(Math.max(0, at - 60), at + 60))}`;
    }
    misses.push({ chunk: chunkId, reason: 'gleaning key differs (chunk has >1 entries)', ...(detail ? { detail } : {}) });
  }
  for (const k of mine) claimed.add(k);
  if (Array.isArray(chunk.llm_cache_list)) {
    cacheListPresent++;
    if (mine.every((k) => chunk.llm_cache_list!.includes(k))) cacheListAgrees++;
  }
}

// 4. extract entries claimed by no chunk: older prompt versions, deleted chunks
const extractKeys = [...cache.keys()].filter((k) => k.startsWith('default:extract:'));
const orphanExtract = extractKeys.filter((k) => !claimed.has(k));
const orphanWithLiveChunk = orphanExtract.filter((k) => {
  const c = cache.get(k)!.chunk_id;
  return c ? chunks.has(c) : false;
});

const result = {
  dir,
  config: cfg,
  seconds: Number(((performance.now() - t0) / 1000).toFixed(1)),
  cache: { entries: cache.size, byType, hashOk, hashBad, hashBadByType, noPrompt },
  chunks: {
    total: chunks.size,
    withCacheEntries: chunksWithEntries,
    initialHit,
    gleanHit,
    gleanMissAfterInitialHit,
    llmCacheListPresent: cacheListPresent,
    llmCacheListAgrees: cacheListAgrees,
  },
  extractEntries: {
    total: extractKeys.length,
    claimedByTsKeys: claimed.size,
    orphan: orphanExtract.length,
    orphanButChunkStillExists: orphanWithLiveChunk.length,
  },
  missReasons: misses.reduce<Record<string, number>>((acc, m) => ((acc[m.reason] = (acc[m.reason] ?? 0) + 1), acc), {}),
  missSamples: misses.filter((m) => m.detail).slice(0, Number(values['show-diffs'])),
};
console.log(JSON.stringify(result, null, 2));
if (values.json) writeFileSync(values.json, JSON.stringify(result, null, 2) + '\n');
