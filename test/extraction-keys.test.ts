// The TS port must produce the same LLM-cache keys (and stored prompts) as
// upstream extract_entities: that is what lets the parity harness replay a
// real store's extraction with zero API calls. Fixture: entries upstream wrote
// for synthetic chunks with a fake LLM (tools/upstream/make_fixtures.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { DEFAULT_ENTITY_TYPES } from '../src/constants.ts';
import { buildExtractionPrompts, extractionCacheKeys } from '../src/operate.ts';
import { llmCacheLookup, packUserAssToOpenaiMessages } from '../src/utils.ts';

type Entry = { return: string; cache_type: string; chunk_id: string; original_prompt: string };
const fx = JSON.parse(readFileSync(new URL('./fixtures/upstream-cases.json', import.meta.url), 'utf8')) as {
  chunks: Record<string, { content: string }>;
  extraction: { language: string; entity_types: string[] | null; cache: Record<string, Entry> }[];
};

for (const run of fx.extraction) {
  const cfg = { language: run.language, entityTypes: run.entity_types ?? DEFAULT_ENTITY_TYPES };
  for (const [chunkId, chunk] of Object.entries(fx.chunks)) {
    test(`${run.language}: ${chunkId} initial + gleaning keys match upstream`, () => {
      const own = Object.entries(run.cache).filter(([, e]) => e.chunk_id === chunkId);
      assert.equal(own.length, 2, 'upstream wrote initial + gleaning');

      const p = buildExtractionPrompts(chunk.content, cfg);
      const init = llmCacheLookup(p.user, { systemPrompt: p.system, cacheType: 'extract' });
      const initEntry = run.cache[init.cacheKey];
      assert.ok(initEntry, `initial key ${init.cacheKey} not in upstream cache`);
      assert.equal(init.prompt, initEntry.original_prompt);

      const glean = llmCacheLookup(p.continueUser, {
        systemPrompt: p.system,
        historyMessages: packUserAssToOpenaiMessages(p.user, initEntry.return),
        cacheType: 'extract',
      });
      assert.ok(run.cache[glean.cacheKey], 'gleaning key not in upstream cache');
      assert.equal(glean.prompt, run.cache[glean.cacheKey]!.original_prompt);

      assert.deepEqual(extractionCacheKeys(chunk.content, cfg, initEntry.return), {
        initial: init.cacheKey,
        gleaning: glean.cacheKey,
      });
    });
  }
}
