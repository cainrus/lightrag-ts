// LLM function over the `claude-complete` CLI bridge, the way the personal
// vault wires LightRAG today (port-only: upstream has no such binding). Two
// Python pieces are mirrored: `_claude_call.claude_complete_async` (spawn the
// bridge with `--profile --timeout --model`, prompt on stdin from a file,
// stripped stdout, `claude-complete failed (rc=N): ...` on a non-zero exit)
// and `vault_rag_core._anthropic_complete_str` (system prompt and history
// folded into one prompt string, LightRAG's extra keyword arguments dropped).
// The cache keys LightRAG writes do not depend on this function: they hash the
// prompt pieces before the call (operate / utils), so a store the Python vault
// filled is hit the same way.
import { spawn } from 'node:child_process';
import { accessSync, constants, mkdtempSync, openSync, closeSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { pyStr, pyStrip } from '../utils.ts';
import { APIConnectionError, APIStatusError, unwrapRetryError } from './openai.ts';

export const CLAUDE_COMPLETE_DEFAULT_MODEL = 'haiku';
export const CLAUDE_COMPLETE_DEFAULT_PROFILE = 'home';
export const CLAUDE_COMPLETE_DEFAULT_TIMEOUT = 180;
/** _claude_call._GRACE_SEC: the bridge enforces --timeout itself; the subprocess gets this much more. */
export const CLAUDE_COMPLETE_GRACE_SEC = 15;

export type ClaudeCompleteConfig = {
  /** the bridge executable; default: $CLAUDE_COMPLETE_BIN, else `claude-complete.ts` / `claude-complete` on PATH */
  bin?: string;
  /** `--model`; null leaves it to the bridge (default 'haiku', as the vault asks) */
  model?: string | null;
  profile?: string;
  /** seconds, `--timeout`; a `timeout` keyword of the call wins (default 180) */
  timeout?: number;
  env?: NodeJS.ProcessEnv;
  /** seconds past `timeout` before the subprocess is killed (default 15, _GRACE_SEC) */
  grace_sec?: number;
};

/** The bridge's failure, `RuntimeError` upstream; `kind` classifies it like the vault ingest does. */
export class ClaudeCompleteError extends Error {
  readonly returncode: number | null;
  readonly kind: ProviderStop | null;
  constructor(message: string, returncode: number | null) {
    super(message);
    this.name = 'RuntimeError';
    this.returncode = returncode;
    this.kind = classifyText(message);
  }
}

function findOnPath(names: string[], env: NodeJS.ProcessEnv): string | null {
  for (const dir of (env.PATH ?? '').split(delimiter)) {
    if (!dir) continue;
    for (const n of names) {
      const p = join(dir, n);
      try {
        accessSync(p, constants.X_OK);
        return p;
      } catch {
        // next
      }
    }
  }
  return null;
}

/** The bridge path the call would use; throws `claude-complete CLI missing` as _cli_args does. */
export function resolveClaudeCompleteBin(cfg: ClaudeCompleteConfig = {}): string {
  const env = cfg.env ?? process.env;
  const bin = cfg.bin ?? env.CLAUDE_COMPLETE_BIN ?? findOnPath(['claude-complete.ts', 'claude-complete'], env);
  if (!bin) throw new ClaudeCompleteError('claude-complete CLI missing: claude-complete.ts not on PATH (set CLAUDE_COMPLETE_BIN)', null);
  try {
    accessSync(bin, constants.X_OK);
  } catch {
    throw new ClaudeCompleteError(`claude-complete CLI missing: ${bin}`, null);
  }
  return bin;
}

/** _claude_call._fail: the first 500 code points of stderr (else stdout), stripped. */
function failMessage(rc: number | null, stdout: string, stderr: string): string {
  const text = Array.from(pyStrip(stderr) || pyStrip(stdout)).slice(0, 500).join('') || '<no error output>';
  return `claude-complete failed (rc=${rc === null ? 'None' : rc}): ${text}`;
}

/**
 * _claude_call.claude_complete_async: one call through the bridge. The prompt
 * goes in as a file on stdin (not argv: a prompt starting with `--` would be
 * eaten; not a pipe: the bridge reads a file); a run that outlives
 * timeout + 15 s is killed with a TimeoutError.
 */
export async function claude_complete(prompt: string, cfg: ClaudeCompleteConfig & { timeout?: number } = {}): Promise<string> {
  const env = cfg.env ?? process.env;
  const bin = resolveClaudeCompleteBin(cfg);
  const timeout = cfg.timeout ?? CLAUDE_COMPLETE_DEFAULT_TIMEOUT;
  const model = cfg.model === undefined ? CLAUDE_COMPLETE_DEFAULT_MODEL : cfg.model;
  const grace = cfg.grace_sec ?? CLAUDE_COMPLETE_GRACE_SEC;
  const args = ['--profile', cfg.profile ?? CLAUDE_COMPLETE_DEFAULT_PROFILE, '--timeout', pyStr(timeout)];
  if (model) args.push('--model', model);
  const tmp = mkdtempSync(join(tmpdir(), 'claude-call-'));
  try {
    const promptPath = join(tmp, 'prompt.txt');
    writeFileSync(promptPath, prompt, 'utf8');
    const fd = openSync(promptPath, 'r');
    try {
      return await new Promise<string>((resolve, reject) => {
        const child = spawn(bin, args, { env: { ...env }, stdio: [fd, 'pipe', 'pipe'] });
        const out: Buffer[] = [];
        const err: Buffer[] = [];
        child.stdout!.on('data', (b: Buffer) => out.push(b));
        child.stderr!.on('data', (b: Buffer) => err.push(b));
        let timedOut = false;
        const timer = setTimeout(() => {
          timedOut = true;
          child.kill('SIGKILL');
        }, (timeout + grace) * 1000);
        child.on('error', (e) => {
          clearTimeout(timer);
          reject(new ClaudeCompleteError(`claude-complete failed (rc=None): ${e.message}`, null));
        });
        child.on('close', (code) => {
          clearTimeout(timer);
          if (timedOut) {
            const e = new Error(`claude-complete timed out after ${timeout + grace}s`);
            e.name = 'TimeoutError';
            reject(e);
            return;
          }
          const stdout = Buffer.concat(out).toString('utf8');
          const stderr = Buffer.concat(err).toString('utf8');
          if (code !== 0) reject(new ClaudeCompleteError(failMessage(code, stdout, stderr), code));
          else resolve(pyStrip(stdout));
        });
      });
    } finally {
      closeSync(fd);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

/** vault_rag_core._anthropic_complete_str's prompt: system (stripped), `[role] content` per history message, the prompt; joined by a blank line. */
export function foldClaudePrompt(prompt: string, systemPrompt?: string | null, history?: readonly Record<string, unknown>[] | null): string {
  const parts: string[] = [];
  if (systemPrompt) parts.push(pyStrip(systemPrompt));
  for (const m of history ?? []) {
    const role = 'role' in m ? m.role : 'user';
    const content = 'content' in m ? m.content : '';
    parts.push(`[${pyStr(role)}] ${pyStr(content)}`);
  }
  parts.push(prompt);
  return parts.join('\n\n');
}

/**
 * A LightRAG `llm_model_func` over the bridge, as the vault builds it
 * (`_anthropic_complete_str`): `system_prompt` and `history_messages` fold
 * into the prompt, `timeout` (seconds) is passed on, every other keyword
 * (hashing_kv, keyword_extraction, stream, max_tokens, enable_cot, ...) is
 * dropped. Always returns a string: the bridge does not stream.
 */
export function makeClaudeCompleteLlm(cfg: ClaudeCompleteConfig = {}) {
  return async (prompt: string, opts: Record<string, unknown> = {}): Promise<string> => {
    const timeout = opts.timeout === undefined || opts.timeout === null ? (cfg.timeout ?? CLAUDE_COMPLETE_DEFAULT_TIMEOUT) : Math.trunc(Number(opts.timeout));
    const full = foldClaudePrompt(prompt, opts.system_prompt as string | null | undefined, opts.history_messages as Record<string, unknown>[] | null | undefined);
    return claude_complete(full, { ...cfg, timeout });
  };
}

/** _claude_call.effective_model: who answers a `model` call, asked of the bridge (`--describe-model`), never assumed. */
export async function describeClaudeModel(model: string | null = CLAUDE_COMPLETE_DEFAULT_MODEL, cfg: ClaudeCompleteConfig = {}): Promise<{ provider: string; model: string }> {
  const bin = resolveClaudeCompleteBin(cfg);
  const { code, stdout, stderr } = await new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(bin, ['--describe-model', model || CLAUDE_COMPLETE_DEFAULT_MODEL], { env: { ...(cfg.env ?? process.env) }, stdio: ['ignore', 'pipe', 'pipe'] });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    child.stdout.on('data', (b: Buffer) => out.push(b));
    child.stderr.on('data', (b: Buffer) => err.push(b));
    child.on('error', reject);
    child.on('close', (c) => resolve({ code: c, stdout: Buffer.concat(out).toString('utf8'), stderr: Buffer.concat(err).toString('utf8') }));
  });
  if (code !== 0) throw new ClaudeCompleteError(failMessage(code, stdout, stderr), code);
  const data = JSON.parse(stdout) as { provider?: unknown; model?: unknown };
  if (!data.provider || !data.model) throw new ClaudeCompleteError(`claude-complete --describe-model returned no pair: ${stdout.slice(0, 200)}`, code);
  return { provider: String(data.provider), model: String(data.model) };
}

// ---------------------------------------------------------------- stops

/**
 * Why a provider refuses every further call, as the vault ingest tells them
 * apart: `billing` (no credit: only a human topping up clears it),
 * `session_limit` (the Claude usage window is spent: it rolls over by
 * itself), `network` (no route). Retrying inside the same run is waste for
 * all three; the caller decides how long to stop.
 */
export type ProviderStop = 'billing' | 'session_limit' | 'network';

/** vault-rag-ingest.py's marker lists */
export const BILLING_MARKERS = ['insufficient_quota', 'credit_balance_exhausted', 'no credits remaining'] as const;
export const SESSION_LIMIT_MARKERS = ['hit your session limit', 'usage limit reached'] as const;
export const NETWORK_MARKERS = ['APIConnectionError', 'Connection error.', 'ConnectError'] as const;

function classifyText(text: string): ProviderStop | null {
  if (BILLING_MARKERS.some((m) => text.includes(m))) return 'billing';
  if (SESSION_LIMIT_MARKERS.some((m) => text.includes(m))) return 'session_limit';
  if (NETWORK_MARKERS.some((m) => text.includes(m))) return 'network';
  return null;
}

/**
 * Classify an error from either adapter. An OpenAI error is read by its own
 * `type` / `code` once a RetryError is unwrapped (the vault's #441: the
 * billing code never reached the log through tenacity's str()); `code` tells
 * an empty balance (`credit_balance_exhausted`) from a project spend limit.
 */
export function classifyProviderError(e: unknown): { kind: ProviderStop; code: string | null } | null {
  const real = unwrapRetryError(e);
  if (real instanceof APIStatusError && real.type === 'insufficient_quota') return { kind: 'billing', code: real.code };
  if (real instanceof APIConnectionError) return { kind: 'network', code: null };
  if (real instanceof ClaudeCompleteError) return real.kind ? { kind: real.kind, code: null } : null;
  const kind = classifyText(real instanceof Error ? `${real.name}: ${real.message}` : String(real));
  return kind ? { kind, code: null } : null;
}
