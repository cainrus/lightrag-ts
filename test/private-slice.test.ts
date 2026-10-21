// Runs only where a slice of a real store exists (fixtures/private/, gitignored;
// made by tools/upstream/slice_store.py). Proves the readers and the
// extraction-key replay on real upstream output, not only on synthetic data.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { storeStats } from '../src/store.ts';

const slice = fileURLToPath(new URL('../fixtures/private/store-slice/', import.meta.url));
const skip = existsSync(slice) ? false : 'no private store slice (tools/upstream/slice_store.py)';

test('private slice: every store file loads', { skip }, () => {
  const s = storeStats(slice);
  assert.ok(s.graph && s.graph.nodes > 0 && s.graph.edges > 0);
  assert.ok((s.kv.text_chunks ?? 0) > 0);
  assert.equal(s.vdb.chunks?.records, s.kv.text_chunks);
  assert.equal(s.vdb.entities?.records, s.graph.nodes);
});

test('private slice: every chunk replays initial + gleaning from the LLM cache', { skip }, () => {
  const script = fileURLToPath(new URL('../parity/replay-extract.ts', import.meta.url));
  const out = JSON.parse(execFileSync(process.execPath, [script, slice], { encoding: 'utf8' })) as {
    chunks: { total: number; initialHit: number; gleanHit: number };
    extractEntries: { total: number; claimedByTsKeys: number };
  };
  assert.equal(out.chunks.initialHit, out.chunks.total);
  assert.equal(out.chunks.gleanHit, out.chunks.total);
  assert.equal(out.extractEntries.claimedByTsKeys, out.extractEntries.total);
});
