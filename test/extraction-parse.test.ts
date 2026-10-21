// _process_extraction_result against upstream's parse of fuzzed answers
// (test/fixtures/parse-cases.json, built by `parity/operate.ts --phase fixture`
// from upstream's public prompt examples). The store-wide check is the
// `extract` and `parse` phases of parity/operate.ts.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { edgeKeyParts, processExtractionResult } from '../src/operate.ts';

type Case = { result: string; nodes: unknown; edges: unknown };
const cases = JSON.parse(readFileSync(new URL('./fixtures/parse-cases.json', import.meta.url), 'utf8')) as Case[];

test(`${cases.length} fuzzed extraction answers parse like upstream`, () => {
  let nonEmpty = 0;
  for (const [i, c] of cases.entries()) {
    const r = processExtractionResult(c.result, 'chunk-x', 7, 'f.md');
    if (r.nodes.size || r.edges.size) nonEmpty++;
    const got = JSON.parse(JSON.stringify({ nodes: [...r.nodes], edges: [...r.edges].map(([k, v]) => [edgeKeyParts(k), v]) }));
    assert.deepEqual(got, { nodes: c.nodes, edges: c.edges }, `case ${i}: ${JSON.stringify(c.result).slice(0, 120)}`);
  }
  assert.ok(nonEmpty > cases.length / 2, 'most cases still parse to something');
});
