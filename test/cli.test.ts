// CLI (B5): insert / query / delete / status against a fake OpenAI-compatible
// server (node:http: chat answers, keyword JSON, md5-seeded embeddings) and the
// fake claude-complete bridge; nothing leaves the machine.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { EXIT, main } from '../src/cli.ts';
import { openaiRuntime } from '../src/llm/openai.ts';

const FAKE_CLAUDE = fileURLToPath(new URL('./fixtures/fake-claude-complete.ts', import.meta.url));
const CLI = fileURLToPath(new URL('../src/cli.ts', import.meta.url));

const EXTRACTION = [
  'entity<|#|>Alice<|#|>person<|#|>Alice is an engineer who builds bridges.',
  'entity<|#|>Bob<|#|>person<|#|>Bob is a designer.',
  'relation<|#|>Alice<|#|>Bob<|#|>collaboration<|#|>Alice works with Bob on bridges.',
  '<|COMPLETE|>',
].join('\n');

const srv = { url: '', chat: 0, embed: 0, mode: 'ok' as 'ok' | 'quota' };
const httpd = createServer((req, res) => {
  const chunks: Buffer[] = [];
  req.on('data', (c: Buffer) => chunks.push(c));
  req.on('end', () => {
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
    const send = (status: number, obj: unknown) => {
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(obj));
    };
    if (srv.mode === 'quota') {
      send(429, { error: { message: 'You exceeded your current quota', type: 'insufficient_quota', param: null, code: 'credit_balance_exhausted' } });
      return;
    }
    if (req.url === '/v1/embeddings') {
      srv.embed++;
      const dim = (body.dimensions as number | undefined) ?? 8;
      // bag of words hashed into dim buckets: texts sharing words point the same way
      const data = (body.input as string[]).map((t, index) => {
        const v = Array.from({ length: dim }, () => 0.01);
        for (const w of t.toLowerCase().match(/[a-z]+/g) ?? []) v[createHash('md5').update(w).digest()[0]! % dim]! += 1;
        return { index, embedding: v };
      });
      send(200, { data, usage: { prompt_tokens: 1, total_tokens: 1 } });
      return;
    }
    srv.chat++;
    const content = body.response_format ? JSON.stringify({ high_level_keywords: ['engineering'], low_level_keywords: ['Alice', 'Bob'] }) : EXTRACTION;
    send(200, { choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }] });
  });
});

const tmp = mkdtempSync(join(tmpdir(), 'lightrag-ts-cli-'));
const savedRuntime = { ...openaiRuntime };
before(async () => {
  await new Promise<void>((r) => httpd.listen(0, '127.0.0.1', r));
  srv.url = `http://127.0.0.1:${(httpd.address() as AddressInfo).port}/v1`;
  openaiRuntime.sleep = async () => {};
});
after(() => {
  Object.assign(openaiRuntime, savedRuntime);
  httpd.close();
  rmSync(tmp, { recursive: true, force: true });
});

function envFor(dir: string, extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  return {
    PATH: process.env.PATH,
    HOME: process.env.HOME,
    WORKING_DIR: dir,
    LLM_BINDING: 'openai',
    LLM_MODEL: 'gpt-test',
    LLM_BINDING_HOST: srv.url,
    LLM_BINDING_API_KEY: 'sk-fake',
    EMBEDDING_BINDING_HOST: srv.url,
    EMBEDDING_BINDING_API_KEY: 'sk-fake',
    EMBEDDING_MODEL: 'text-embedding-3-small',
    EMBEDDING_DIM: '8',
    EMBEDDING_SEND_DIM: 'true',
    ...extra,
  };
}

async function run(argv: string[], env: NodeJS.ProcessEnv, stdin = '') {
  let out = '';
  let err = '';
  const code = await main(argv, { stdout: (s) => (out += s), stderr: (s) => (err += s), stdin: () => stdin, env });
  return { code, out, err };
}

test('openai binding: insert, status, query (then from cache), delete', async () => {
  const dir = join(tmp, 'openai');
  const env = envFor(dir);
  const doc = join(tmp, 'notes.md');
  writeFileSync(doc, 'Alice is an engineer. She works with Bob, a designer, on bridges.\n');

  const ins = await run(['insert', doc, '--id', 'doc-notes'], env);
  assert.equal(ins.code, EXIT.ok, ins.err);
  const inserted = JSON.parse(ins.out) as { documents: Record<string, { status: string; file_path: string }> };
  assert.deepEqual(Object.keys(inserted.documents), ['doc-notes']);
  assert.equal(inserted.documents['doc-notes']!.status, 'processed');
  assert.equal(inserted.documents['doc-notes']!.file_path, 'notes.md');

  const st = await run(['status'], env);
  const status = JSON.parse(st.out) as { documents: Record<string, number>; graph: { nodes: number; edges: number } };
  assert.deepEqual(status.documents, { processed: 1 });
  assert.equal(status.graph.nodes, 2);
  assert.equal(status.graph.edges, 1);

  const q1 = await run(['query', 'Who', 'works', 'with', 'Bob?', '--mode', 'mix'], env);
  assert.equal(q1.code, EXIT.ok, q1.err);
  assert.equal(q1.out, EXTRACTION + '\n');
  const chatAfterFirst = srv.chat;
  const q2 = await run(['query', 'Who works with Bob?', '--mode', 'mix'], env);
  assert.equal(q2.out, q1.out);
  assert.equal(srv.chat, chatAfterFirst, 'the second query is answered from the LLM cache');

  const ctx = await run(['query', 'Who works with Bob?', '--mode', 'local', '--only-context'], env);
  assert.match(ctx.out, /Alice/);

  const del = await run(['delete', '--doc-id', 'doc-notes', '--delete-llm-cache'], env);
  assert.equal(del.code, EXIT.ok, del.out + del.err);
  assert.equal((JSON.parse(del.out) as { status: string }).status, 'success');
  const st2 = JSON.parse((await run(['status'], env)).out) as { documents: Record<string, number>; graph: { nodes: number } };
  assert.deepEqual(st2.documents, {});
  assert.equal(st2.graph.nodes, 0);

  const missing = await run(['delete', '--doc-id', 'nope'], env);
  assert.equal(missing.code, EXIT.error);
});

test('claude-complete binding from PATH: stdin insert, the bridge gets the folded prompt', async () => {
  const dir = join(tmp, 'claude');
  const bin = join(tmp, 'bin');
  mkdirSync(bin, { recursive: true });
  symlinkSync(FAKE_CLAUDE, join(bin, 'claude-complete.ts'));
  const log = join(tmp, 'claude.jsonl');
  const env = envFor(dir, {
    PATH: `${bin}:${process.env.PATH}`,
    LLM_BINDING: 'claude-complete',
    FAKE_CLAUDE_LOG: log,
    FAKE_CLAUDE_REPLY: EXTRACTION,
  });
  delete env.LLM_MODEL; // the claude-complete binding defaults to haiku, as the vault asks
  const r = await run(['insert', '--file-path', 'stdin.md'], env, 'Alice met Bob at the bridge.');
  assert.equal(r.code, EXIT.ok, r.err);
  const calls = readFileSync(log, 'utf8').trim().split('\n').map((l) => JSON.parse(l) as { argv: string[]; stdin: string });
  assert.ok(calls.length >= 1);
  assert.deepEqual(calls[0]!.argv, ['--profile', 'home', '--timeout', '180', '--model', 'haiku']);
  assert.match(calls[0]!.stdin, /---Role---[\s\S]*\n\n[\s\S]*Alice met Bob at the bridge\./);
  const doc = Object.values((JSON.parse(r.out) as { documents: Record<string, { status: string; file_path: string }> }).documents)[0]!;
  assert.deepEqual(doc, { status: 'processed', file_path: 'stdin.md', chunks_count: 1 });
});

test('stops: empty OpenAI balance exits 3, spent Claude window exits 4', async () => {
  srv.mode = 'quota';
  try {
    const r = await run(['insert'], envFor(join(tmp, 'quota')), 'Some text about Alice.');
    assert.equal(r.code, EXIT.billing, r.out + r.err);
    assert.match(r.err, /stopped: billing/);
  } finally {
    srv.mode = 'ok';
  }
  const r2 = await run(
    ['insert'],
    envFor(join(tmp, 'limit'), { LLM_BINDING: 'claude-complete', CLAUDE_COMPLETE_BIN: FAKE_CLAUDE, FAKE_CLAUDE_MODE: 'limit' }),
    'Some text about Bob.',
  );
  assert.equal(r2.code, EXIT.session_limit, r2.out + r2.err);
  assert.match(JSON.stringify(JSON.parse(r2.out)), /hit your session limit/);
});

test('usage errors and an unported binding', async () => {
  assert.equal((await run([], envFor(tmp))).code, EXIT.usage);
  assert.equal((await run(['query'], envFor(tmp))).code, EXIT.usage);
  assert.equal((await run(['query', 'q', '--mode', 'weird'], envFor(tmp))).code, EXIT.usage);
  assert.equal((await run(['frobnicate'], envFor(tmp))).code, EXIT.usage);
  const r = await run(['query', 'q'], envFor(join(tmp, 'x'), { LLM_BINDING: 'ollama' }));
  assert.equal(r.code, EXIT.error);
  assert.match(r.err, /LLM_BINDING=ollama is not ported/);
});

test('the bin entry runs as a script', () => {
  const dir = join(tmp, 'empty');
  const out = execFileSync(process.execPath, [CLI, 'status', '--working-dir', dir], { encoding: 'utf8', env: { PATH: process.env.PATH, HOME: process.env.HOME, LIGHTRAG_TS_NO_DOTENV: '1' } });
  assert.deepEqual(JSON.parse(out).documents, {});
});
