// tokenizer + chunking_by_token_size against values recorded from upstream
// (tiktoken 0.12.0, lightrag 1.4.16). The full-store check is parity/operate.ts.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ChunkTokenLimitExceededError } from '../src/exceptions.ts';
import { chunkingByTokenSize } from '../src/operate.ts';
import { TiktokenTokenizer } from '../src/utils.ts';

const tok = new TiktokenTokenizer();
const TEXT = 'Привет, мир! Hello world.\n\nВторой абзац 😀 с эмодзи.';

test('gpt-4o-mini encodes with o200k_base like tiktoken', () => {
  assert.deepEqual(
    tok.encode(TEXT),
    [23881, 131903, 11, 37934, 0, 32949, 2375, 364, 3540, 9885, 1742, 12306, 12819, 924, 88038, 669, 39098, 1267, 22849, 13],
  );
});

test('special-token text is refused, as tiktoken disallowed_special="all"', () => {
  assert.throws(() => tok.encode('a<|endoftext|>b'));
});

test('unknown model name', () => {
  assert.throws(() => new TiktokenTokenizer('no-such-model'), /Invalid model_name: no-such-model\./);
});

test('sliding window, overlap 2, size 5', () => {
  assert.deepEqual(chunkingByTokenSize(tok, TEXT, null, false, 2, 5), [
    { tokens: 5, content: 'Привет, мир!', chunk_order_index: 0 },
    { tokens: 5, content: 'мир! Hello world.', chunk_order_index: 1 },
    { tokens: 5, content: 'world.\n\nВторой', chunk_order_index: 2 },
    { tokens: 5, content: 'торой абзац', chunk_order_index: 3 },
    { tokens: 5, content: 'зац 😀 с эм', chunk_order_index: 4 },
    { tokens: 5, content: 'с эмодзи.', chunk_order_index: 5 },
    { tokens: 2, content: 'зи.', chunk_order_index: 6 },
  ]);
});

test('overlap larger than the window: range() with a negative step is empty', () => {
  assert.deepEqual(chunkingByTokenSize(tok, TEXT, null, false, 9, 5), []);
  assert.throws(() => chunkingByTokenSize(tok, TEXT, null, false, 5, 5), /must not be zero/);
});

test('split_by_character_only over the limit raises with a code-point preview', () => {
  assert.throws(
    () => chunkingByTokenSize(tok, TEXT, '\n\n', true, 1, 3),
    (e: unknown) =>
      e instanceof ChunkTokenLimitExceededError &&
      e.message === "Chunk token length 8 exceeds chunk_token_size 3. Preview: 'Привет, мир! Hello world.'",
  );
});
