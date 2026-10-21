// Byte parity of the hashing / sanitizing helpers against outputs recorded
// from upstream lightrag (tools/upstream/make_fixtures.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { computeArgsHash, makeRelationVdbIds, pyFormat, pyStrip, sanitizeTextForEncoding } from '../src/utils.ts';

type Case = { input: string; sanitized: string; args_hash: string; strip: string };
const cases = JSON.parse(readFileSync(new URL('./fixtures/upstream-cases.json', import.meta.url), 'utf8')) as {
  sanitize: Case[];
};

for (const c of cases.sanitize) {
  test(`upstream parity: ${JSON.stringify(c.input).slice(0, 50)}`, () => {
    assert.equal(pyStrip(c.input), c.strip, 'str.strip');
    assert.equal(sanitizeTextForEncoding(c.input), c.sanitized, 'sanitize_text_for_encoding');
    assert.equal(computeArgsHash(c.input), c.args_hash, 'compute_args_hash');
  });
}

test('emoji survive sanitize (regression: regex without u flag split surrogate pairs)', () => {
  assert.equal(sanitizeTextForEncoding('a 💰 b'), 'a 💰 b');
});

test('pyFormat: fields and escaped braces, refuses unknown syntax', () => {
  assert.equal(pyFormat('{a}-{{b}}-}}', { a: 'x' }), 'x-{b}-}');
  assert.throws(() => pyFormat('{0}', {}), /unsupported/);
  assert.throws(() => pyFormat('{a:>3}', { a: 'x' }), /unsupported/);
  assert.throws(() => pyFormat('{missing}', {}), /missing/);
});

test('makeRelationVdbIds: normalized order first', () => {
  const [first, second] = makeRelationVdbIds('B', 'A');
  assert.equal(first, 'rel-' + computeArgsHash('AB'));
  assert.equal(second, 'rel-' + computeArgsHash('BA'));
});
