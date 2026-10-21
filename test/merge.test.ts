// merge_nodes_and_edges against upstream's own output on public input
// (test/fixtures/merge-cases, recorded by `parity/operate.ts --phase merge-fixture`):
// documents built from the parsed prompt examples, merged one after another
// with a summary stub and a hash embedding, under three configs.
import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { type MergeDoc, mergeDirDiffs, type MergeRec, tsMergeDocs } from '../parity/merge-driver.ts';

const root = fileURLToPath(new URL('./fixtures/merge-cases/', import.meta.url));
const docs = JSON.parse(readFileSync(join(root, 'docs.json'), 'utf8')) as MergeDoc[];
const variants = readdirSync(root).filter((d) => statSync(join(root, d)).isDirectory());
const readJsonl = <T>(f: string): T[] => readFileSync(f, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as T);

for (const variant of variants) {
  const dir = join(root, variant);
  const gc = JSON.parse(readFileSync(join(dir, 'gc.json'), 'utf8'));
  const expected = readJsonl<MergeRec>(join(dir, 'messages.jsonl'));

  test(`merge == upstream (${variant}): every file byte for byte, pipeline messages`, async () => {
    const out = mkdtempSync(join(tmpdir(), `lightrag-merge-${variant}-`));
    const recs = await tsMergeDocs(out, docs, { vdb: true, dim: 4, gc, maxConcurrency: 1 });
    assert.deepEqual(recs, expected);
    for (const [file, diff] of mergeDirDiffs(dir, out, { vdb: true })) assert.equal(diff, null, `${variant}/${file}: ${diff}`);
  });

  test(`merge with concurrent summaries == upstream sequential (${variant})`, async () => {
    const out = mkdtempSync(join(tmpdir(), `lightrag-merge-${variant}-c-`));
    const recs = await tsMergeDocs(out, docs, { vdb: true, dim: 4, gc });
    assert.deepEqual(recs, expected);
    // only the cache's key order follows LLM completion order
    for (const [file, diff] of mergeDirDiffs(dir, out, { vdb: true, cacheAsMap: true })) assert.equal(diff, null, `${variant}/${file}: ${diff}`);
  });
}
