// OpenAI-compatible adapter (B5). test/fixtures/openai-requests.json is
// upstream's llm/openai.py recorded against a local server
// (tools/upstream/openai_requests.py): every request it sent, the waits of
// both retry layers, the result or the error. Here the TS functions meet a
// node:http server that plays the same responses and must send the same bytes.
import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { readFileSync } from 'node:fs';
import { after, before, test } from 'node:test';
import {
  APIConnectionError,
  makeEmbeddingFunc,
  openai_complete,
  openai_complete_if_cache,
  openai_embed,
  openai_embed_func,
  openaiRuntime,
  RateLimitError,
  RetryError,
  unwrapRetryError,
} from '../src/llm/openai.ts';

type Resp = { status?: number; body?: string; headers?: Record<string, string>; sse?: string[]; close?: boolean };
type Req = { method: string; path: string; headers: Record<string, string>; body: string };
type Scenario = {
  name: string;
  fn: 'complete' | 'embed';
  args: Record<string, unknown>;
  responses: Resp[];
  use_base64?: boolean;
  result?: { text?: string; stream?: string[]; vectors?: number[][] };
  error?: { class: string; message: string; last?: { class: string; message: string }; status_code?: number; code?: string | null; type?: string | null };
  requests: Req[] | number;
  sleeps: number[];
};

const fixture = JSON.parse(readFileSync(new URL('./fixtures/openai-requests.json', import.meta.url), 'utf8')) as { scenarios: Scenario[] };

const server = { url: '', requests: [] as Req[], responses: [] as Resp[] };
const httpd = createServer((req: IncomingMessage, res: ServerResponse) => {
  const chunks: Buffer[] = [];
  req.on('data', (c: Buffer) => chunks.push(c));
  req.on('end', () => {
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(req.headers)) if (typeof v === 'string') headers[k] = v;
    server.requests.push({ method: req.method!, path: req.url!, headers, body: Buffer.concat(chunks).toString('utf8') });
    const r = server.responses.length > 1 ? server.responses.shift()! : server.responses[0]!;
    if (r.close) {
      req.socket.destroy();
      return;
    }
    const payload = Buffer.from(r.sse ? r.sse.map((e) => `data: ${e}\n\n`).join('') : r.body!, 'utf8');
    res.writeHead(r.status ?? 200, { 'content-type': r.sse ? 'text/event-stream' : 'application/json', 'content-length': payload.length, ...r.headers });
    res.end(payload);
  });
});

const sleeps: number[] = [];
const runtime = { ...openaiRuntime };
before(async () => {
  await new Promise<void>((r) => httpd.listen(0, '127.0.0.1', r));
  server.url = `http://127.0.0.1:${(httpd.address() as AddressInfo).port}/v1`;
  openaiRuntime.sleep = async (s) => {
    if (s > 0) sleeps.push(Math.round(s * 1e6) / 1e6);
  };
  openaiRuntime.random = () => 0.5;
  process.env.OPENAI_API_KEY = 'sk-test-env-key';
});
after(() => {
  Object.assign(openaiRuntime, runtime);
  httpd.close();
});

function reset(responses: Resp[]) {
  server.requests = [];
  server.responses = [...responses];
  sleeps.length = 0;
}

async function runTs(sc: Scenario): Promise<{ result?: Scenario['result']; error?: unknown }> {
  const a = { ...sc.args };
  try {
    if (sc.fn === 'complete') {
      const { model, prompt, kwargs, ...rest } = a as { model: string; prompt: string; kwargs: Record<string, unknown> };
      const r = await openai_complete_if_cache(model, prompt, { ...rest, ...kwargs, base_url: server.url });
      if (typeof r === 'string') return { result: { text: r } };
      const stream: string[] = [];
      for await (const x of r) stream.push(x);
      return { result: { stream } };
    }
    const { texts, ...rest } = a as { texts: string[] };
    const v = await openai_embed_func(texts, { ...rest, base_url: server.url, ...(sc.use_base64 === false ? { use_base64: false } : {}) });
    return { result: { vectors: v.map((row) => Array.from(row)) } };
  } catch (e) {
    return { error: e };
  }
}

// the headers a server can act on; the SDK's X-Stainless platform telemetry is not sent
const COMPARED_HEADERS = ['authorization', 'accept', 'content-type', 'user-agent', 'x-stainless-retry-count', 'x-stainless-read-timeout'];

for (const sc of fixture.scenarios) {
  test(`upstream openai.py parity: ${sc.name}`, async () => {
    reset(sc.responses);
    const got = await runTs(sc);
    if (typeof sc.requests === 'number') {
      assert.equal(server.requests.length, sc.requests);
    } else {
      assert.equal(server.requests.length, sc.requests.length, 'request count');
      sc.requests.forEach((want, i) => {
        const have = server.requests[i]!;
        assert.equal(have.method, want.method);
        assert.equal(have.path, want.path);
        assert.equal(have.body, want.body, `body of request ${i}`);
        for (const h of COMPARED_HEADERS) assert.equal(have.headers[h], want.headers[h], `header ${h}`);
      });
    }
    assert.deepEqual(sleeps, sc.sleeps, 'retry waits');
    if (sc.error) {
      const e = got.error as Error & Record<string, unknown>;
      assert.ok(e, `expected ${sc.error.class}, got ${JSON.stringify(got.result)}`);
      assert.equal(e.name, sc.error.class);
      const inner = (sc.error.last ? (e as unknown as RetryError).last_attempt : e) as Error & Record<string, unknown>;
      if (sc.error.last) {
        assert.equal(inner.name, sc.error.last.class);
        assert.equal(inner.message, sc.error.last.message);
        assert.equal(e.message, `RetryError[${sc.error.last.class}: ${sc.error.last.message}]`);
      } else {
        assert.equal(e.message, sc.error.message);
      }
      for (const k of ['status_code', 'code', 'type'] as const) if (k in sc.error) assert.equal(inner[k] ?? null, sc.error[k] ?? null, k);
    } else {
      if (got.error) throw got.error;
      assert.deepEqual(got.result, sc.result);
    }
  });
}

test('billing: insufficient_quota surfaces its code through the RetryError', async () => {
  const sc = fixture.scenarios.find((s) => s.name === 'embed-quota')!;
  reset(sc.responses);
  const { error } = await runTs(sc);
  assert.ok(error instanceof RetryError);
  const inner = unwrapRetryError(error);
  assert.ok(inner instanceof RateLimitError);
  assert.equal(inner.code, 'insufficient_quota');
  assert.equal(inner.type, 'insufficient_quota');
  assert.match(String(error), /insufficient_quota/);
});

test('connection refused: APIConnectionError after both retry layers', async () => {
  sleeps.length = 0;
  const err = await openai_embed_func(['x'], { base_url: 'http://127.0.0.1:9/v1' }).catch((e: unknown) => e);
  assert.ok(err instanceof RetryError);
  assert.ok(err.last_attempt instanceof APIConnectionError);
  assert.deepEqual(sleeps, [0.4375, 0.875, 4, 0.4375, 0.875, 4, 0.4375, 0.875]);
});

test('openai_embed (decorated): context passes through, no dimensions sent, shape checked', async () => {
  reset([{ body: JSON.stringify({ data: [{ embedding: [1, 2] }] }) }]);
  // the decorated function declares 1536 dims; a 2-dim answer is a shape error
  await assert.rejects(openai_embed(['q'], { context: 'query', base_url: server.url } as never), /Embedding dimension mismatch|Vector count mismatch/);
  assert.equal(JSON.parse(server.requests[0]!.body).dimensions, undefined);
  const ef = makeEmbeddingFunc({ embedding_dim: 2, func: openai_embed, send_dimensions: true, max_token_size: 3 });
  assert.equal(ef.func, openai_embed.func, 'a wrapped EmbeddingFunc is unwrapped');
  reset([{ body: JSON.stringify({ data: [{ embedding: [1, 2] }] }) }]);
  const v = await ef(['one two three four'], { context: 'query', base_url: server.url, query_prefix: 'q ' } as never);
  assert.deepEqual(Array.from(v[0]!), [1, 2]);
  const body = JSON.parse(server.requests[0]!.body) as { input: string[]; dimensions: number };
  assert.equal(body.dimensions, 2);
  // context dropped (supports_asymmetric false): no query prefix; max_token_size 3 injected
  assert.deepEqual(body.input, ['one two three']);
});

test('openai_complete takes the model from hashing_kv.global_config', async () => {
  reset([{ body: JSON.stringify({ choices: [{ message: { content: 'hi' } }] }) }]);
  const r = await openai_complete('p', { hashing_kv: { global_config: { llm_model_name: 'my-model' } }, base_url: server.url });
  assert.equal(r, 'hi');
  assert.equal(JSON.parse(server.requests[0]!.body).model, 'my-model');
});

test('an unknown keyword is a TypeError, as the SDK signature makes it', async () => {
  await assert.rejects(openai_complete_if_cache('m', 'p', { base_url: server.url, bogus: 1 }), { name: 'TypeError', message: "AsyncCompletions.create() got an unexpected keyword argument 'bogus'" });
});
