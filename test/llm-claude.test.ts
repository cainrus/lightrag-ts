// claude-complete adapter (B5). test/fixtures/claude-prompts.json is the
// vault's own LLM function (vault_rag_core._anthropic_complete_str, through
// _claude_call) recorded against test/fixtures/fake-claude-complete.ts
// (tools/upstream/claude_prompts.py): the bridge's argv and stdin per call.
// The TS function must hand the same fake bridge the same bytes.
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  ClaudeCompleteError,
  classifyProviderError,
  claude_complete,
  describeClaudeModel,
  makeClaudeCompleteLlm,
  resolveClaudeCompleteBin,
} from '../src/llm/claude.ts';
import { RateLimitError, RetryError } from '../src/llm/openai.ts';

const FAKE = fileURLToPath(new URL('./fixtures/fake-claude-complete.ts', import.meta.url));
const fixture = JSON.parse(readFileSync(new URL('./fixtures/claude-prompts.json', import.meta.url), 'utf8')) as {
  calls: {
    name: string;
    prompt: string;
    kwargs: Record<string, unknown>;
    mode: string;
    result?: string;
    error?: { class: string; message: string };
    bridge: { argv: string[]; stdin: string }[];
  }[];
};

const tmp = mkdtempSync(join(tmpdir(), 'lightrag-ts-claude-'));
after(() => rmSync(tmp, { recursive: true, force: true }));
const readLog = (p: string) => readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as { argv: string[]; stdin: string });

for (const c of fixture.calls) {
  test(`vault parity: ${c.name}`, async () => {
    const log = join(tmp, `${c.name}.jsonl`);
    const llm = makeClaudeCompleteLlm({ bin: FAKE, env: { ...process.env, FAKE_CLAUDE_LOG: log, FAKE_CLAUDE_MODE: c.mode } });
    let result: string | undefined;
    let error: Error | undefined;
    try {
      result = await llm(c.prompt, c.kwargs);
    } catch (e) {
      error = e as Error;
    }
    assert.deepEqual(readLog(log), c.bridge);
    if (c.error) {
      assert.ok(error, 'expected an error');
      assert.equal(error.name, c.error.class);
      assert.equal(error.message, c.error.message);
    } else {
      if (error) throw error;
      assert.equal(result, c.result);
    }
  });
}

test('the bridge is found on PATH, as claude-complete.ts', async () => {
  const bin = join(tmp, 'bin');
  const { mkdirSync } = await import('node:fs');
  mkdirSync(bin);
  symlinkSync(FAKE, join(bin, 'claude-complete.ts'));
  const env = { ...process.env, PATH: `${bin}:${process.env.PATH}`, CLAUDE_COMPLETE_BIN: undefined };
  assert.equal(resolveClaudeCompleteBin({ env }), join(bin, 'claude-complete.ts'));
  assert.equal(await claude_complete('hello', { env: { ...env, FAKE_CLAUDE_REPLY: '\n  ok \n' } }), 'ok');
  assert.throws(() => resolveClaudeCompleteBin({ env: { PATH: tmp } }), /claude-complete CLI missing/);
});

test('a hung bridge is killed after timeout + grace', async () => {
  const t0 = Date.now();
  await assert.rejects(claude_complete('p', { bin: FAKE, timeout: 0, grace_sec: 0.3, env: { ...process.env, FAKE_CLAUDE_MODE: 'hang' } }), { name: 'TimeoutError' });
  assert.ok(Date.now() - t0 < 5000);
});

test('model null leaves --model out; --describe-model asks the bridge', async () => {
  const log = join(tmp, 'nomodel.jsonl');
  await claude_complete('p', { bin: FAKE, model: null, profile: 'work', timeout: 7, env: { ...process.env, FAKE_CLAUDE_LOG: log } });
  assert.deepEqual(readLog(log)[0]!.argv, ['--profile', 'work', '--timeout', '7']);
  assert.deepEqual(await describeClaudeModel('sonnet', { bin: FAKE }), { provider: 'anthropic', model: 'sonnet' });
});

test('stops: session limit, billing, network are told apart', async () => {
  const limit = await claude_complete('p', { bin: FAKE, env: { ...process.env, FAKE_CLAUDE_MODE: 'limit' } }).catch((e: unknown) => e);
  assert.ok(limit instanceof ClaudeCompleteError);
  assert.deepEqual(classifyProviderError(limit), { kind: 'session_limit', code: null });
  const fail = await claude_complete('p', { bin: FAKE, env: { ...process.env, FAKE_CLAUDE_MODE: 'fail' } }).catch((e: unknown) => e);
  assert.equal(classifyProviderError(fail), null);
  const quota = new RateLimitError('Error code: 429 - ...', 'u', { type: 'insufficient_quota', code: 'project_spend_limit_exceeded' }, 429, new Headers());
  assert.deepEqual(classifyProviderError(new RetryError(quota)), { kind: 'billing', code: 'project_spend_limit_exceeded' });
  assert.deepEqual(classifyProviderError(new Error('credit_balance_exhausted')), { kind: 'billing', code: null });
});
