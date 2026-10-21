// Port of lightrag/llm/openai.py (the OpenAI-compatible subset): chat
// completions and embeddings over fetch, without the `openai` package. The
// two retry layers upstream stacks are both here: the openai SDK's own
// (max_retries=2 on 408/409/429/5xx and connection errors, 0.5 s doubling
// with jitter or the server's retry-after) and openai.py's tenacity @retry
// (3 attempts, exponential wait from 4 s). Request bodies are serialized the
// way httpx does (`json.dumps(ensure_ascii=False, separators=(",", ":"))`),
// keys in the SDK's own order, so a server sees the same bytes.
import { Buffer } from 'node:buffer';
import type { ChatMessage } from '../utils.ts';
import { pyFloatParse, pyJsonDumps, pyReprStr, pyStr, pyStrip, TiktokenTokenizer, type Tokenizer } from '../utils.ts';

/** lightrag.api.__api_version__ of 1.4.16; part of the User-Agent upstream sends. */
export const LIGHTRAG_API_VERSION = '0291';
export const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';
/** openai._constants */
export const DEFAULT_MAX_RETRIES = 2;
export const DEFAULT_TIMEOUT_SECONDS = 600;
const INITIAL_RETRY_DELAY = 0.5;
const MAX_RETRY_DELAY = 8.0;

/**
 * openai.py's module constant: read once at import, as upstream does
 * (`EMBEDDING_USE_BASE64`, default true).
 */
export const EMBEDDING_USE_BASE64 = ['true', '1', 'yes'].includes((process.env.EMBEDDING_USE_BASE64 ?? 'true').toLowerCase());

/** The process hooks the retry layers use; tests replace them (upstream's tests patch anyio/asyncio sleep). */
export const openaiRuntime = {
  sleep: (seconds: number): Promise<void> => new Promise((r) => setTimeout(r, seconds * 1000)),
  random: (): number => Math.random(),
  fetch: (input: string, init: RequestInit): Promise<Response> => fetch(input, init),
};

// ---------------------------------------------------------------- errors

/** Python `repr()` of a JSON value, as the SDK's `f"Error code: {status} - {body}"` prints the parsed body. */
export function pyReprJson(v: unknown): string {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'string') return pyReprStr(v);
  if (typeof v === 'boolean' || typeof v === 'number') return pyStr(v);
  if (Array.isArray(v)) return '[' + v.map(pyReprJson).join(', ') + ']';
  if (typeof v === 'object') return '{' + Object.entries(v).map(([k, x]) => `${pyReprStr(k)}: ${pyReprJson(x)}`).join(', ') + '}';
  return String(v);
}

const named = <T extends Error>(e: T, name: string): T => {
  e.name = name;
  return e;
};

/** openai.OpenAIError */
export class OpenAIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OpenAIError';
  }
}

/** openai.APIError: `code` / `param` / `type` come from the error body when it is an object. */
export class APIError extends OpenAIError {
  readonly url: string;
  readonly body: unknown;
  readonly code: string | null;
  readonly param: string | null;
  readonly type: string | null;
  constructor(message: string, url: string, body: unknown) {
    super(message);
    this.name = 'APIError';
    this.url = url;
    this.body = body;
    const o = body && typeof body === 'object' && !Array.isArray(body) ? (body as Record<string, unknown>) : null;
    this.code = o && typeof o.code === 'string' ? o.code : null;
    this.param = o && typeof o.param === 'string' ? o.param : null;
    this.type = o && typeof o.type === 'string' ? o.type : null;
  }
}

/** openai.APIStatusError and its per-status subclasses (RateLimitError, ...), told apart by `name`. */
export class APIStatusError extends APIError {
  readonly status_code: number;
  readonly request_id: string | null;
  readonly headers: Headers;
  constructor(message: string, url: string, body: unknown, status: number, headers: Headers) {
    super(message, url, body);
    this.name = 'APIStatusError';
    this.status_code = status;
    this.headers = headers;
    this.request_id = headers.get('x-request-id');
  }
}

export class RateLimitError extends APIStatusError {
  constructor(message: string, url: string, body: unknown, status: number, headers: Headers) {
    super(message, url, body, status, headers);
    this.name = 'RateLimitError';
  }
}

/** openai.APIConnectionError ("Connection error.") */
export class APIConnectionError extends APIError {
  constructor(url: string, message = 'Connection error.', cause?: unknown) {
    super(message, url, null);
    this.name = 'APIConnectionError';
    if (cause !== undefined) this.cause = cause;
  }
}

/** openai.APITimeoutError ("Request timed out."), a connection error like upstream's. */
export class APITimeoutError extends APIConnectionError {
  constructor(url: string) {
    super(url, 'Request timed out.');
    this.name = 'APITimeoutError';
  }
}

/** openai.py's InvalidResponseError: an empty or malformed completion, retried by @retry. */
export class InvalidResponseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidResponseError';
  }
}

/**
 * tenacity.RetryError: every attempt failed with a retryable error. Upstream's
 * str() names only the class of the last error (`RetryError[<Future at 0x...
 * raised RateLimitError>]`); here the message carries the last error's own
 * message too, and `last_attempt` is the error itself (what
 * `RetryError.last_attempt.exception()` returns).
 */
export class RetryError extends Error {
  readonly last_attempt: Error;
  constructor(last: Error) {
    super(`RetryError[${last.name}: ${last.message}]`);
    this.name = 'RetryError';
    this.last_attempt = last;
    this.cause = last;
  }
}

/** The error a RetryError wraps, or the error itself (tenacity's `reraise()`). */
export function unwrapRetryError(e: unknown): unknown {
  return e instanceof RetryError ? e.last_attempt : e;
}

const STATUS_CLASS: Record<number, string> = {
  400: 'BadRequestError',
  401: 'AuthenticationError',
  403: 'PermissionDeniedError',
  404: 'NotFoundError',
  409: 'ConflictError',
  422: 'UnprocessableEntityError',
  429: 'RateLimitError',
};

/** _make_status_error_from_response + _make_status_error */
function makeStatusError(status: number, text: string, url: string, headers: Headers): APIStatusError {
  const errText = pyStrip(text);
  let body: unknown = errText;
  let msg: string;
  try {
    body = JSON.parse(errText);
    msg = `Error code: ${status} - ${pyReprJson(body)}`;
  } catch {
    msg = errText || `Error code: ${status}`;
  }
  const data = body && typeof body === 'object' && !Array.isArray(body) && 'error' in body ? (body as { error: unknown }).error : body;
  if (status === 429) return new RateLimitError(msg, url, data, status, headers);
  const e = new APIStatusError(msg, url, data, status, headers);
  return named(e, STATUS_CLASS[status] ?? (status >= 500 ? 'InternalServerError' : 'APIStatusError'));
}

// ---------------------------------------------------------------- client

export type OpenAIClientConfigs = {
  /** SDK retries per request (default 2); 0 turns the SDK layer off. */
  max_retries?: number;
  /** seconds; the per-request `timeout` wins */
  timeout?: number;
  organization?: string | null;
  project?: string | null;
  /** Upstream overwrites a `default_headers` given here with its own (User-Agent, Content-Type); kept for parity. */
  default_headers?: Record<string, string>;
};

export type OpenAIClient = {
  base_url: string;
  /** POST `path` (relative to base_url) with the SDK's retries; resolves to the successful Response. */
  post(
    path: string,
    body: Record<string, unknown>,
    opts?: { timeout?: number | null | undefined; stream?: boolean; extra_headers?: Record<string, string> | undefined; extra_query?: Record<string, string> | undefined },
  ): Promise<Response>;
};

/** openai.py create_openai_async_client (not Azure). */
export function create_openai_async_client(
  opts: { api_key?: string | null | undefined; base_url?: string | null | undefined; timeout?: number | null | undefined; client_configs?: OpenAIClientConfigs | null | undefined } = {},
): OpenAIClient {
  let apiKey = opts.api_key;
  if (!apiKey) {
    apiKey = process.env.OPENAI_API_KEY;
    // os.environ["OPENAI_API_KEY"] raises KeyError when it is unset
    if (apiKey === undefined) throw named(new Error("'OPENAI_API_KEY'"), 'KeyError');
  }
  const configs = opts.client_configs ?? {};
  const baseUrl = opts.base_url ?? process.env.OPENAI_API_BASE ?? DEFAULT_OPENAI_BASE_URL;
  const clientTimeout = opts.timeout ?? configs.timeout ?? DEFAULT_TIMEOUT_SECONDS;
  const maxRetries = configs.max_retries ?? DEFAULT_MAX_RETRIES;
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'Content-Type': 'application/json',
    'User-Agent': `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/${LIGHTRAG_API_VERSION}`,
  };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  if (configs.organization) headers['OpenAI-Organization'] = configs.organization;
  if (configs.project) headers['OpenAI-Project'] = configs.project;
  // the SDK's _prepare_url: base_url always ends with "/" and paths join onto it
  const root = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';

  const retryDelay = (retriesTaken: number, h: Headers | null): number => {
    const ra = parseRetryAfter(h);
    if (ra !== null && ra > 0 && ra <= 60) return ra;
    const sleep = Math.min(INITIAL_RETRY_DELAY * 2 ** Math.min(retriesTaken, 1000), MAX_RETRY_DELAY);
    return Math.max(0, sleep * (1 - 0.25 * openaiRuntime.random()));
  };

  return {
    base_url: baseUrl,
    async post(path, body, o = {}) {
      let url = new URL(path.replace(/^\//, ''), root).toString();
      if (o.extra_query && Object.keys(o.extra_query).length) url += (url.includes('?') ? '&' : '?') + new URLSearchParams(o.extra_query).toString();
      const timeout = o.timeout ?? clientTimeout;
      const payload = Buffer.from(pyJsonDumps(body, { compact: true }), 'utf8');
      for (let retriesTaken = 0; ; retriesTaken++) {
        const remaining = maxRetries - retriesTaken;
        const ctl = new AbortController();
        let timedOut = false;
        const timer = setTimeout(() => {
          timedOut = true;
          ctl.abort();
        }, timeout * 1000);
        let resp: Response;
        try {
          resp = await openaiRuntime.fetch(url, {
            method: 'POST',
            headers: { ...headers, 'x-stainless-retry-count': String(retriesTaken), 'x-stainless-read-timeout': pyStr(timeout), ...o.extra_headers },
            body: payload,
            signal: ctl.signal,
          });
        } catch (e) {
          clearTimeout(timer);
          if (remaining > 0) {
            await openaiRuntime.sleep(retryDelay(retriesTaken, null));
            continue;
          }
          throw timedOut ? new APITimeoutError(url) : new APIConnectionError(url, undefined, e);
        }
        if (resp.status >= 400) {
          let text: string;
          try {
            text = await resp.text();
          } finally {
            clearTimeout(timer);
          }
          if (remaining > 0 && shouldRetry(resp)) {
            await openaiRuntime.sleep(retryDelay(retriesTaken, resp.headers));
            continue;
          }
          throw makeStatusError(resp.status, text, url, resp.headers);
        }
        if (o.stream) {
          // httpx's read timeout is per chunk; a stream is not cut off as a whole
          clearTimeout(timer);
          return resp;
        }
        try {
          const buf = await resp.arrayBuffer();
          return new Response(buf, { status: resp.status, headers: resp.headers });
        } catch (e) {
          if (remaining > 0) {
            await openaiRuntime.sleep(retryDelay(retriesTaken, null));
            continue;
          }
          throw timedOut ? new APITimeoutError(url) : new APIConnectionError(url, undefined, e);
        } finally {
          clearTimeout(timer);
        }
      }
    },
  };
}

/** _parse_retry_after_header: seconds, or null */
function parseRetryAfter(h: Headers | null): number | null {
  if (!h) return null;
  const ms = h.get('retry-after-ms');
  if (ms !== null) {
    const v = pyFloatParse(ms);
    if (v !== null) return v / 1000;
  }
  const ra = h.get('retry-after');
  if (ra === null) return null;
  const v = pyFloatParse(ra);
  if (v !== null) return v;
  const date = Date.parse(ra);
  return Number.isNaN(date) ? null : (date - Date.now()) / 1000;
}

/** _should_retry */
function shouldRetry(resp: Response): boolean {
  const h = resp.headers.get('x-should-retry');
  if (h === 'true') return true;
  if (h === 'false') return false;
  return resp.status === 408 || resp.status === 409 || resp.status === 429 || resp.status >= 500;
}

/** tenacity `@retry(stop=stop_after_attempt(n), wait=wait_exponential(multiplier=1, min, max), retry=...)` */
async function tenacityRetry<T>(fn: () => Promise<T>, o: { attempts: number; min: number; max: number; retryable: (e: unknown) => boolean }): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (!o.retryable(e)) throw e;
      if (attempt >= o.attempts) throw new RetryError(e as Error);
      await openaiRuntime.sleep(Math.max(o.min, Math.min(2 ** (attempt - 1), o.max)));
    }
  }
}

const isRateOrConnection = (e: unknown) => e instanceof RateLimitError || e instanceof APIConnectionError;

// ---------------------------------------------------------------- completion

/** chat.completions.create's parameters in the order its body dict lists them (openai 2.36). */
const CREATE_KEYS = [
  'messages', 'model', 'audio', 'frequency_penalty', 'function_call', 'functions', 'logit_bias', 'logprobs',
  'max_completion_tokens', 'max_tokens', 'metadata', 'modalities', 'n', 'parallel_tool_calls', 'prediction',
  'presence_penalty', 'prompt_cache_key', 'prompt_cache_retention', 'reasoning_effort', 'response_format',
  'safety_identifier', 'seed', 'service_tier', 'stop', 'store', 'stream', 'stream_options', 'temperature',
  'tool_choice', 'tools', 'top_logprobs', 'top_p', 'user', 'verbosity', 'web_search_options',
] as const;
/** chat.completions.parse's body order: `store` before `stop`, `stream` always false. */
const PARSE_KEYS = [
  'messages', 'model', 'audio', 'frequency_penalty', 'function_call', 'functions', 'logit_bias', 'logprobs',
  'max_completion_tokens', 'max_tokens', 'metadata', 'modalities', 'n', 'parallel_tool_calls', 'prediction',
  'presence_penalty', 'prompt_cache_key', 'prompt_cache_retention', 'reasoning_effort', 'response_format',
  'safety_identifier', 'seed', 'service_tier', 'store', 'stop', 'stream', 'stream_options', 'temperature',
  'tool_choice', 'tools', 'top_logprobs', 'top_p', 'user', 'verbosity', 'web_search_options',
] as const;
const REQUEST_OPTION_KEYS = new Set(['extra_headers', 'extra_query', 'extra_body', 'timeout']);

/**
 * types.GPTKeywordExtractionFormat as the SDK sends a pydantic `response_format`
 * (`type_to_response_format_param`, strict JSON schema).
 */
export const GPT_KEYWORD_EXTRACTION_FORMAT = {
  type: 'json_schema',
  json_schema: {
    schema: {
      properties: {
        high_level_keywords: { items: { type: 'string' }, title: 'High Level Keywords', type: 'array' },
        low_level_keywords: { items: { type: 'string' }, title: 'Low Level Keywords', type: 'array' },
      },
      required: ['high_level_keywords', 'low_level_keywords'],
      title: 'GPTKeywordExtractionFormat',
      type: 'object',
      additionalProperties: false,
    },
    name: 'GPTKeywordExtractionFormat',
    strict: true,
  },
} as const;

/** utils.safe_unicode_decode on the text upstream re-encodes: literal `\uXXXX` escapes become characters. */
export function safeUnicodeDecode(content: string): string {
  return content.replace(/\\u([0-9a-fA-F]{4})/g, (_, h: string) => String.fromCharCode(Number.parseInt(h, 16)));
}

const decodeIfEscaped = (s: string) => (s.includes('\\u') ? safeUnicodeDecode(s) : s);

/** pydantic's `GPTKeywordExtractionFormat.model_validate_json(...).model_dump_json()` */
function keywordFormatRoundTrip(content: string): string {
  const fail = (why: string) => named(new Error(`1 validation error for GPTKeywordExtractionFormat\n${why}`), 'ValidationError');
  let v: unknown;
  try {
    v = JSON.parse(content);
  } catch {
    throw fail('Invalid JSON');
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw fail('Input should be an object');
  const o = v as Record<string, unknown>;
  for (const k of ['high_level_keywords', 'low_level_keywords']) {
    const xs = o[k];
    if (!Array.isArray(xs)) throw fail(`${k}\n  ${xs === undefined ? 'Field required' : 'Input should be a valid list'}`);
    if (!xs.every((x) => typeof x === 'string')) throw fail(`${k}\n  Input should be a valid string`);
  }
  return JSON.stringify({ high_level_keywords: o.high_level_keywords, low_level_keywords: o.low_level_keywords });
}

export type TokenTracker = { add_usage(counts: Record<string, number>): void };

export type OpenAICompleteOptions = {
  system_prompt?: string | null;
  history_messages?: ChatMessage[] | Record<string, unknown>[] | null;
  enable_cot?: boolean;
  base_url?: string | null;
  api_key?: string | null;
  token_tracker?: TokenTracker | null;
  stream?: boolean | null;
  /** seconds */
  timeout?: number | null;
  keyword_extraction?: boolean;
  /** dropped, as upstream pops it */
  hashing_kv?: unknown;
  openai_client_configs?: OpenAIClientConfigs;
  /** replaces the built message list, as upstream's `kwargs.pop("messages", messages)` */
  messages?: Record<string, unknown>[];
  /** any chat.completions.create parameter (max_tokens, temperature, ...) or extra_headers / extra_query / extra_body */
  [k: string]: unknown;
};

/**
 * openai.py openai_complete_if_cache. Returns the completion text, or an async
 * iterator of chunks when `stream` is set (keyword extraction never streams:
 * it goes through `parse()`). Not ported: Azure, Langfuse, the debug logging.
 */
export async function openai_complete_if_cache(model: string, prompt: string, opts: OpenAICompleteOptions = {}): Promise<string | AsyncIterable<string>> {
  return tenacityRetry(() => completeOnce(model, prompt, opts), {
    attempts: 3,
    min: 4,
    max: 10,
    retryable: (e) => isRateOrConnection(e) || e instanceof InvalidResponseError,
  });
}

async function completeOnce(model: string, prompt: string, opts: OpenAICompleteOptions): Promise<string | AsyncIterable<string>> {
  const {
    system_prompt: systemPrompt = null,
    history_messages: history = null,
    enable_cot: enableCot = false,
    base_url: baseUrl = null,
    api_key: apiKey = null,
    token_tracker: tokenTracker = null,
    stream = null,
    timeout = null,
    keyword_extraction: keywordExtraction = false,
    hashing_kv: _hashingKv,
    openai_client_configs: clientConfigs = {},
    messages: messagesOverride,
    ...rest
  } = opts;
  const kwargs: Record<string, unknown> = { ...rest };
  if (keywordExtraction) kwargs.response_format = GPT_KEYWORD_EXTRACTION_FORMAT;
  const client = create_openai_async_client({ api_key: apiKey, base_url: baseUrl, timeout, client_configs: clientConfigs });

  let messages: Record<string, unknown>[] = [];
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  messages.push(...((history ?? []) as Record<string, unknown>[]));
  messages.push({ role: 'user', content: prompt });
  if (messagesOverride !== undefined) messages = messagesOverride;
  if (stream !== null && stream !== undefined) kwargs.stream = stream;

  const parse = 'response_format' in kwargs;
  const keys: readonly string[] = parse ? PARSE_KEYS : CREATE_KEYS;
  const fnName = parse ? 'parse' : 'create';
  const params: Record<string, unknown> = { messages, model, ...kwargs };
  if (parse) params.stream = false;
  for (const k of Object.keys(params)) {
    if (!keys.includes(k) && !REQUEST_OPTION_KEYS.has(k)) throw named(new TypeError(`AsyncCompletions.${fnName}() got an unexpected keyword argument '${k}'`), 'TypeError');
  }
  const body: Record<string, unknown> = {};
  for (const k of keys) if (params[k] !== undefined) body[k] = params[k];
  if (params.extra_body && typeof params.extra_body === 'object') Object.assign(body, params.extra_body);
  const isStream = !parse && body.stream === true;

  const resp = await client.post('chat/completions', body, {
    timeout,
    stream: isStream,
    extra_headers: params.extra_headers as Record<string, string> | undefined,
    extra_query: params.extra_query as Record<string, string> | undefined,
  });
  if (isStream) return streamChunks(resp, client.base_url, enableCot, tokenTracker);

  const response = (await resp.json().catch(() => null)) as {
    choices?: { message?: Record<string, unknown>; finish_reason?: string }[];
    usage?: Record<string, number>;
  } | null;
  if (!response || !response.choices?.length || !response.choices[0]!.message) {
    throw new InvalidResponseError('Invalid response from OpenAI API');
  }
  const choice = response.choices[0]!;
  const message = choice.message!;
  let finalContent: string;
  let parsed: string | null = null;
  if (parse) {
    // openai/lib/_parsing: finish reasons first, then the pydantic model
    for (const c of response.choices) {
      if (c.finish_reason === 'length') throw named(new OpenAIError(`Could not parse response content as the length limit was reached - ${JSON.stringify(response.usage ?? null)}`), 'LengthFinishReasonError');
      if (c.finish_reason === 'content_filter') throw named(new OpenAIError('Could not parse response content as the request was rejected by the content filter'), 'ContentFilterFinishReasonError');
    }
    const content = message.content;
    if (typeof content === 'string' && content && !message.refusal) parsed = keywordFormatRoundTrip(content);
  }
  if (parsed !== null) {
    finalContent = parsed;
  } else {
    const content = (message.content ?? null) as string | null;
    const reasoning = ('reasoning_content' in message ? message.reasoning_content : '') as string | null;
    finalContent = '';
    if (enableCot) {
      let includeReasoning = false;
      if (reasoning && pyStrip(reasoning)) {
        if (!content || pyStrip(content) === '') {
          includeReasoning = true;
          finalContent = content || '';
        } else {
          finalContent = content;
        }
      } else {
        finalContent = content || '';
      }
      if (includeReasoning) finalContent = `<think>${decodeIfEscaped(reasoning!)}</think>${finalContent}`;
    } else {
      finalContent = content || '';
    }
    if (!finalContent || pyStrip(finalContent) === '') throw new InvalidResponseError('Received empty content from OpenAI API');
  }
  finalContent = decodeIfEscaped(finalContent);
  if (tokenTracker && response.usage) {
    tokenTracker.add_usage({
      prompt_tokens: response.usage.prompt_tokens ?? 0,
      completion_tokens: response.usage.completion_tokens ?? 0,
      total_tokens: response.usage.total_tokens ?? 0,
    });
  }
  return finalContent;
}

/** The SDK's SSE decoder: one parsed JSON object per event until `[DONE]`. */
async function* sseEvents(resp: Response, url: string): AsyncGenerator<Record<string, unknown>> {
  const reader = resp.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  let data: string[] = [];
  let event: string | null = null;
  const flush = function* (): Generator<Record<string, unknown> | 'done'> {
    if (!data.length && event === null) return;
    const text = data.join('\n');
    data = [];
    event = null;
    if (text.startsWith('[DONE]')) {
      yield 'done';
      return;
    }
    const obj = JSON.parse(text) as Record<string, unknown>;
    if (obj && typeof obj === 'object' && !Array.isArray(obj) && obj.error) {
      const err = obj.error as Record<string, unknown>;
      throw new APIError(typeof err.message === 'string' ? err.message : 'An error occurred during streaming', url, err);
    }
    yield obj;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (value) buf += value;
      let nl: number;
      while ((nl = buf.search(/\r\n|\r|\n/)) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + (buf.startsWith('\r\n', nl) ? 2 : 1));
        if (line === '') {
          for (const x of flush()) {
            if (x === 'done') return;
            yield x;
          }
        } else if (!line.startsWith(':')) {
          const i = line.indexOf(':');
          const field = i < 0 ? line : line.slice(0, i);
          let v = i < 0 ? '' : line.slice(i + 1);
          if (v.startsWith(' ')) v = v.slice(1);
          if (field === 'data') data.push(v);
          else if (field === 'event') event = v;
        }
      }
      if (done) break;
    }
    for (const x of flush()) {
      if (x === 'done') return;
      yield x;
    }
  } finally {
    reader.releaseLock();
    await resp.body?.cancel().catch(() => {});
  }
}

/** openai_complete_if_cache's `inner()`: content chunks, `<think>` around leading reasoning when enable_cot. */
async function* streamChunks(resp: Response, url: string, enableCot: boolean, tokenTracker: TokenTracker | null): AsyncGenerator<string> {
  let finalUsage: Record<string, number> | null = null;
  let cotActive = false;
  let cotStarted = false;
  let initialContentSeen = false;
  try {
    for await (const chunk of sseEvents(resp, url)) {
      if (chunk.usage) finalUsage = chunk.usage as Record<string, number>;
      const choices = chunk.choices as { delta?: Record<string, unknown> }[] | undefined;
      if (!choices?.length) continue;
      const delta = choices[0]!.delta;
      if (!delta) continue;
      let content = (delta.content ?? null) as string | null;
      let reasoning = ('reasoning_content' in delta ? delta.reasoning_content : '') as string | null;
      if (enableCot) {
        if (content) {
          if (!initialContentSeen) {
            initialContentSeen = true;
            if (reasoning) {
              cotActive = false;
              cotStarted = false;
            }
          }
          if (cotActive) {
            yield '</think>';
            cotActive = false;
          }
          yield decodeIfEscaped(content);
        } else if (reasoning) {
          if (!initialContentSeen && !cotStarted && !cotActive) {
            yield '<think>';
            cotActive = true;
            cotStarted = true;
          }
          if (cotActive) yield decodeIfEscaped(reasoning);
        }
      } else if (content) {
        yield decodeIfEscaped(content);
      }
      content = null;
      reasoning = null;
    }
    if (enableCot && cotActive) {
      yield '</think>';
      cotActive = false;
    }
    if (tokenTracker && finalUsage) {
      tokenTracker.add_usage({
        prompt_tokens: finalUsage.prompt_tokens ?? 0,
        completion_tokens: finalUsage.completion_tokens ?? 0,
        total_tokens: finalUsage.total_tokens ?? 0,
      });
    }
  } finally {
    if (enableCot && cotActive) {
      cotActive = false;
      yield '</think>';
    }
  }
}

type HashingKvLike = { global_config: Record<string, unknown> };

/** openai.py openai_complete: the model is `llm_model_name` of the hashing_kv's global config. */
export async function openai_complete(prompt: string, opts: OpenAICompleteOptions = {}): Promise<string | AsyncIterable<string>> {
  const kv = opts.hashing_kv as HashingKvLike | undefined;
  if (!kv) throw named(new Error("'hashing_kv'"), 'KeyError');
  return openai_complete_if_cache(String(kv.global_config.llm_model_name), prompt, { history_messages: [], ...opts });
}

export const gpt_4o_complete = (prompt: string, opts: OpenAICompleteOptions = {}) => openai_complete_if_cache('gpt-4o', prompt, { history_messages: [], ...opts });
export const gpt_4o_mini_complete = (prompt: string, opts: OpenAICompleteOptions = {}) => openai_complete_if_cache('gpt-4o-mini', prompt, { history_messages: [], ...opts });

// ---------------------------------------------------------------- embeddings

export type EmbedContext = 'document' | 'query';

export type OpenAIEmbedOptions = {
  model?: string;
  base_url?: string | null;
  api_key?: string | null;
  /** sent as `dimensions` */
  embedding_dim?: number | null;
  /** tokens per text; longer texts are cut (tiktoken of the model, cl100k_base when unknown); 0 or null: no cut */
  max_token_size?: number | null;
  client_configs?: OpenAIClientConfigs | null;
  token_tracker?: TokenTracker | null;
  context?: EmbedContext;
  query_prefix?: string | null;
  document_prefix?: string | null;
  /** port-only: overrides EMBEDDING_USE_BASE64 for this call */
  use_base64?: boolean;
};

const embedTokenizers = new Map<string, Tokenizer>();
function embedTokenizer(model: string): Tokenizer {
  let t = embedTokenizers.get(model);
  if (!t) {
    try {
      t = new TiktokenTokenizer(model);
    } catch {
      t = new TiktokenTokenizer('gpt-4'); // cl100k_base
    }
    embedTokenizers.set(model, t);
  }
  return t;
}

/** The undecorated openai_embed (`openai_embed.func` upstream): vectors as float32 rows. */
export async function openai_embed_func(texts: string[], opts: OpenAIEmbedOptions = {}): Promise<Float32Array[]> {
  return tenacityRetry(() => embedOnce(texts, opts), { attempts: 3, min: 4, max: 60, retryable: isRateOrConnection });
}

async function embedOnce(input: string[], opts: OpenAIEmbedOptions): Promise<Float32Array[]> {
  const model = opts.model ?? 'text-embedding-3-small';
  const context = opts.context ?? 'document';
  let texts = input;
  if (context === 'query' && opts.query_prefix) texts = texts.map((t) => opts.query_prefix + t);
  else if (context === 'document' && opts.document_prefix) texts = texts.map((t) => opts.document_prefix + t);
  const maxTokens = opts.max_token_size;
  if (maxTokens !== null && maxTokens !== undefined && maxTokens > 0) {
    const enc = embedTokenizer(model);
    texts = texts.map((t) => {
      if (!t) return t;
      const tokens = enc.encode(t);
      return tokens.length > maxTokens ? enc.decode(tokens.slice(0, maxTokens)) : t;
    });
  }
  const client = create_openai_async_client({ api_key: opts.api_key, base_url: opts.base_url, client_configs: opts.client_configs });
  const b64 = opts.use_base64 ?? EMBEDDING_USE_BASE64;
  const body: Record<string, unknown> = { input: texts, model };
  if (opts.embedding_dim !== null && opts.embedding_dim !== undefined) body.dimensions = opts.embedding_dim;
  body.encoding_format = b64 ? 'base64' : 'float';
  const resp = await client.post('embeddings', body);
  const response = (await resp.json()) as { data: { embedding: string | number[] }[]; usage?: Record<string, number> };
  if (opts.token_tracker && response.usage) {
    opts.token_tracker.add_usage({ prompt_tokens: response.usage.prompt_tokens ?? 0, total_tokens: response.usage.total_tokens ?? 0 });
  }
  return response.data.map((dp) => {
    if (Array.isArray(dp.embedding)) return Float32Array.from(dp.embedding);
    const raw = Buffer.from(dp.embedding, 'base64');
    const out = new Float32Array(raw.byteLength / 4);
    for (let i = 0; i < out.length; i++) out[i] = raw.readFloatLE(i * 4);
    return out;
  });
}

/** What an embedding function receives besides the texts. */
export type EmbedCallOptions = { context?: EmbedContext; embedding_dim?: number | null; max_token_size?: number | null; [k: string]: unknown };
export type RawEmbedFunc = (texts: string[], opts: EmbedCallOptions) => Promise<ArrayLike<ArrayLike<number>>>;

/** An EmbeddingFunc as LightRAG and the vector storages take it, plus upstream's attributes. */
export type EmbeddingFunc = ((texts: string[], opts?: EmbedCallOptions) => Promise<ArrayLike<ArrayLike<number>>>) & {
  embedding_dim: number;
  max_token_size: number | null;
  send_dimensions: boolean;
  model_name: string | undefined;
  supports_asymmetric: boolean;
  func: RawEmbedFunc;
};

/**
 * utils.EmbeddingFunc: injects `embedding_dim` (send_dimensions) and
 * `max_token_size`, drops `context` unless the function supports_asymmetric,
 * and checks the result's shape. Upstream injects max_token_size only when the
 * function's signature names it; a JS function cannot be asked, so it is
 * always passed when set (functions ignore options they do not read). An
 * EmbeddingFunc given as `func` is unwrapped, as upstream does.
 */
export function makeEmbeddingFunc(cfg: {
  embedding_dim: number;
  func: RawEmbedFunc | EmbeddingFunc;
  max_token_size?: number | null;
  send_dimensions?: boolean;
  model_name?: string;
  supports_asymmetric?: boolean;
}): EmbeddingFunc {
  let func = cfg.func as RawEmbedFunc;
  for (let depth = 0; isEmbeddingFunc(func); depth++) {
    if (depth >= 3) throw named(new Error('EmbeddingFunc unwrap depth exceeded 3. Possible circular reference detected.'), 'ValueError');
    func = func.func;
  }
  const f = async (texts: string[], opts: EmbedCallOptions = {}): Promise<ArrayLike<ArrayLike<number>>> => {
    const kwargs: EmbedCallOptions = { ...opts };
    if (self.send_dimensions) kwargs.embedding_dim = self.embedding_dim;
    if ('context' in kwargs && !self.supports_asymmetric) delete kwargs.context;
    if (self.max_token_size !== null && !('max_token_size' in kwargs)) kwargs.max_token_size = self.max_token_size;
    const result = await func(texts, kwargs);
    let total = 0;
    for (let i = 0; i < result.length; i++) total += result[i]!.length;
    if (total % self.embedding_dim !== 0) {
      throw named(
        new Error(`Embedding dimension mismatch detected: total elements (${total}) cannot be evenly divided by expected dimension (${self.embedding_dim}). `),
        'ValueError',
      );
    }
    const actual = total / self.embedding_dim;
    if (actual !== texts.length) {
      throw named(new Error(`Vector count mismatch: expected ${texts.length} vectors but got ${actual} vectors (from embedding result).`), 'ValueError');
    }
    return result;
  };
  const self: EmbeddingFunc = Object.assign(f, {
    embedding_dim: cfg.embedding_dim,
    max_token_size: cfg.max_token_size ?? null,
    send_dimensions: cfg.send_dimensions ?? false,
    model_name: cfg.model_name,
    supports_asymmetric: cfg.supports_asymmetric ?? false,
    func,
  });
  return self;
}

function isEmbeddingFunc(f: unknown): f is EmbeddingFunc {
  return typeof f === 'function' && 'func' in f && 'embedding_dim' in f;
}

/**
 * openai.py openai_embed as decorated upstream
 * (`@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192,
 * model_name="text-embedding-3-small", supports_asymmetric=True)`).
 * `openai_embed.func` is the undecorated function.
 */
export const openai_embed: EmbeddingFunc = makeEmbeddingFunc({
  embedding_dim: 1536,
  max_token_size: 8192,
  model_name: 'text-embedding-3-small',
  supports_asymmetric: true,
  func: (texts: string[], o: EmbedCallOptions) => openai_embed_func(texts, o as OpenAIEmbedOptions),
});
