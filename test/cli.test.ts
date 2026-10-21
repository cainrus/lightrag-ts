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
import { EXIT, lightragInitFromEnv, main, StopGuard } from '../src/cli.ts';
import { APIConnectionError, type EmbeddingFunc, openaiRuntime } from '../src/llm/openai.ts';

const FAKE_CLAUDE = fileURLToPath(new URL('./fixtures/fake-claude-complete.ts', import.meta.url));
const CLI = fileURLToPath(new URL('../src/cli.ts', import.meta.url));

const EXTRACTION = [
  'entity<|#|>Alice<|#|>person<|#|>Alice is an engineer who builds bridges.',
  'entity<|#|>Bob<|#|>person<|#|>Bob is a designer.',
  'relation<|#|>Alice<|#|>Bob<|#|>collaboration<|#|>Alice works with Bob on bridges.',
  '<|COMPLETE|>',
].join('\n');

const srv = { url: '', chat: 0, embed: 0, requests: 0, mode: 'ok' as 'ok' | 'quota', auth: [] as string[], embedInputs: [] as string[][] };
const httpd = createServer((req, res) => {
  const chunks: Buffer[] = [];
  req.on('data', (c: Buffer) => chunks.push(c));
  req.on('end', () => {
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
    srv.requests++;
    srv.auth.push(String(req.headers.authorization ?? ''));
    if (req.url === '/v1/embeddings') srv.embedInputs.push(body.input as string[]);
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

async function run(argv: string[], env: NodeJS.ProcessEnv, stdin = '', cwd = tmp) {
  let out = '';
  let err = '';
  const code = await main(argv, { stdout: (s) => (out += s), stderr: (s) => (err += s), stdin: () => stdin, env, cwd: () => cwd });
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

type Inserted = { documents: Record<string, { status: string; file_path: string }> };

test('B5f-1: a file is stored under its path relative to the current directory, so same-named files stay apart', async () => {
  const root = join(tmp, 'vault1');
  mkdirSync(join(root, 'notes'), { recursive: true });
  mkdirSync(join(root, 'daily'), { recursive: true });
  writeFileSync(join(root, 'notes', 'a.md'), 'Alice is an engineer.\n');
  writeFileSync(join(root, 'daily', 'a.md'), 'Bob is a designer.\n');
  const env = envFor(join(tmp, 'rel'));
  const r = await run(['insert', 'notes/a.md', join(root, 'daily', 'a.md')], env, '', root);
  assert.equal(r.code, EXIT.ok, r.err);
  const paths = Object.values((JSON.parse(r.out) as Inserted).documents).map((d) => d.file_path).sort();
  assert.deepEqual(paths, ['daily/a.md', 'notes/a.md']);
});

test('B5f-1: --file-path per input, in order, as --id; a count mismatch is a usage error', async () => {
  const root = join(tmp, 'vault2');
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, 'x.md'), 'Alice builds bridges.\n');
  writeFileSync(join(root, 'y.md'), 'Bob designs bridges.\n');
  const env = envFor(join(tmp, 'given'));
  const r = await run(['insert', 'x.md', 'y.md', '--id', 'dx', '--file-path', 'p/x.md', '--id', 'dy', '--file-path', 'q/y.md'], env, '', root);
  assert.equal(r.code, EXIT.ok, r.err);
  const docs = (JSON.parse(r.out) as Inserted).documents;
  assert.equal(docs.dx!.file_path, 'p/x.md');
  assert.equal(docs.dy!.file_path, 'q/y.md');
  const bad = await run(['insert', 'x.md', 'y.md', '--file-path', 'only-one.md'], env, '', root);
  assert.equal(bad.code, EXIT.usage);
  assert.match(bad.err, /--file-path given 1 times for 2 input/);
});

test('B5f-2: the EmbeddingFunc carries no max_token_size unless EMBEDDING_TOKEN_LIMIT is set; calls still cut at 8192', async () => {
  const emb = lightragInitFromEnv(envFor(join(tmp, 'tok'))).embedding_func!;
  assert.equal(emb.max_token_size, null);
  assert.equal(lightragInitFromEnv(envFor(join(tmp, 'tok'), { EMBEDDING_TOKEN_LIMIT: '512' })).embedding_func!.max_token_size, 512);
  srv.embedInputs.length = 0;
  const long = 'alpha '.repeat(9000);
  await emb([long]);
  const sent = srv.embedInputs.at(-1)![0]!;
  assert.ok(sent.length < long.length, 'the text is cut');
  assert.ok(sent.length > 'alpha '.length * 8000, 'at 8192 tokens, not at a smaller limit');
});

test('EMBEDDING_SEND_DIM comes from the passed environment, not process.env', () => {
  const saved = process.env.EMBEDDING_SEND_DIM;
  try {
    process.env.EMBEDDING_SEND_DIM = 'false';
    assert.equal((lightragInitFromEnv(envFor(join(tmp, 'sd'), { EMBEDDING_SEND_DIM: 'true' })).embedding_func as unknown as EmbeddingFunc).send_dimensions, true);
    process.env.EMBEDDING_SEND_DIM = 'true';
    assert.equal((lightragInitFromEnv(envFor(join(tmp, 'sd'), { EMBEDDING_SEND_DIM: 'false' })).embedding_func as unknown as EmbeddingFunc).send_dimensions, false);
    const unset = envFor(join(tmp, 'sd'));
    delete unset.EMBEDDING_SEND_DIM;
    assert.equal((lightragInitFromEnv(unset).embedding_func as unknown as EmbeddingFunc).send_dimensions, false, 'default false, whatever process.env says');
  } finally {
    if (saved === undefined) delete process.env.EMBEDDING_SEND_DIM;
    else process.env.EMBEDDING_SEND_DIM = saved;
  }
});

test('B5f-3: the OpenAI key comes from the passed environment, never from process.env', async () => {
  const saved = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = 'sk-from-process';
  try {
    const env = envFor(join(tmp, 'key'), { OPENAI_API_KEY: 'sk-from-env' });
    delete env.LLM_BINDING_API_KEY;
    delete env.EMBEDDING_BINDING_API_KEY;
    srv.auth.length = 0;
    const r = await run(['insert', '--file-path', 'k.md'], env, 'Alice met Bob.');
    assert.equal(r.code, EXIT.ok, r.err);
    assert.ok(srv.auth.length > 0);
    assert.deepEqual([...new Set(srv.auth)], ['Bearer sk-from-env']);

    const noKey = envFor(join(tmp, 'nokey'));
    delete noKey.LLM_BINDING_API_KEY;
    delete noKey.EMBEDDING_BINDING_API_KEY;
    srv.auth.length = 0;
    const r2 = await run(['insert', '--file-path', 'k.md'], noKey, 'Alice met Bob.');
    assert.notEqual(r2.code, EXIT.ok);
    assert.ok(!srv.auth.includes('Bearer sk-from-process'), 'process.env is not read');
  } finally {
    if (saved === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = saved;
  }
});

test('B5f-5: the first billing refusal stops the insert', async () => {
  srv.mode = 'quota';
  try {
    srv.requests = 0;
    const one = await run(['insert'], envFor(join(tmp, 'stop1')), 'Some text about Alice.');
    assert.equal(one.code, EXIT.billing);
    const oneChunk = srv.requests;
    srv.requests = 0;
    // four documents: without the stop each spends its own 9 requests
    const root = join(tmp, 'stopdocs');
    mkdirSync(root, { recursive: true });
    const names = ['a', 'b', 'c', 'd'].map((n) => {
      writeFileSync(join(root, `${n}.md`), `Text ${n} about Alice.`);
      return `${n}.md`;
    });
    const many = await run(['insert', ...names], envFor(join(tmp, 'stop2')), '', root);
    assert.equal(many.code, EXIT.billing, many.out + many.err);
    assert.match(many.err, /stopped: billing/);
    assert.ok(srv.requests <= oneChunk, `no more requests than a one-chunk document (got ${srv.requests}, one chunk ${oneChunk})`);
  } finally {
    srv.mode = 'ok';
  }
});

test('B5f-5: StopGuard keeps the highest-ranked stop and fails later calls with it', async () => {
  const g = new StopGuard();
  const net = new APIConnectionError('Connection error.');
  const bill = new Error("Error code: 429 - {'error': {'type': 'insufficient_quota'}}");
  const f = g.wrap(async (e: unknown) => {
    throw e;
  });
  await assert.rejects(f(net));
  assert.equal(g.stop!.kind, 'network');
  await assert.rejects(f(bill));
  assert.equal(g.stop!.kind, 'network', 'a stopped guard does not call through');
  g.record(bill);
  assert.equal(g.stop!.kind, 'billing');
  g.record(net);
  assert.equal(g.stop!.kind, 'billing');
  let called = 0;
  await assert.rejects(g.wrap(async () => ++called)(), (e) => e === bill);
  assert.equal(called, 0);
});

test('B5f-5: two calls in flight, network fails first, billing second: the run reports billing', async () => {
  const g = new StopGuard();
  const net = new APIConnectionError('Connection error.');
  const bill = new Error("Error code: 429 - {'error': {'type': 'insufficient_quota'}}");
  const later = (e: unknown, ms: number) => g.wrap(() => new Promise<never>((_, rej) => setTimeout(() => rej(e), ms)))();
  await Promise.allSettled([later(net, 1), later(bill, 5)]);
  assert.equal(g.stop!.kind, 'billing');
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
