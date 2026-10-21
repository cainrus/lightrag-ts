#!/usr/bin/env node
// lightrag-ts command line: insert / query / delete / status over one working
// directory. Port-only (upstream ships a FastAPI server, not a CLI); its
// configuration is the subset of upstream's server environment (api/config.py,
// read after `.env` in the current directory, existing variables win) these
// commands need:
//
//   WORKING_DIR (./rag_storage), WORKSPACE ('')
//   LLM_BINDING            openai | claude-complete (port-only: the vault's bridge)
//   LLM_MODEL              openai: gpt-4o-mini; claude-complete: haiku
//   LLM_BINDING_HOST       openai base URL (https://api.openai.com/v1)
//   LLM_BINDING_API_KEY    openai key (else OPENAI_API_KEY)
//   CLAUDE_COMPLETE_BIN, CLAUDE_COMPLETE_PROFILE (home)
//   EMBEDDING_BINDING      openai
//   EMBEDDING_MODEL (text-embedding-3-small), EMBEDDING_DIM (1536),
//   EMBEDDING_SEND_DIM (false), EMBEDDING_TOKEN_LIMIT (unset: LightRAG's
//   embedding_token_limit None, as the vault; texts are cut at 8192 tokens either way),
//   EMBEDDING_BINDING_HOST, EMBEDDING_BINDING_API_KEY (else OPENAI_API_KEY)
//
// Everything else (MAX_ASYNC, MAX_PARALLEL_INSERT, CHUNK_SIZE, TOP_K,
// SUMMARY_LANGUAGE, ...) is read by the LightRAG class itself, as upstream's
// dataclass does.
//
// Exit codes: 0 done, 1 error, 2 usage, 3 a provider refuses for lack of
// credit (billing), 4 the Claude usage window is spent, 5 no network. The
// first such refusal stops an insert (StopGuard); several kinds in one run
// report the highest (billing > usage window > network).
import { readFileSync, realpathSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { QueryMode } from './base.ts';
import { makeQueryParam } from './base.ts';
import { getPipelineStatus, LightRAG, type LightRAGInit } from './lightrag.ts';
import { classifyProviderError, makeClaudeCompleteLlm, type ProviderStop } from './llm/claude.ts';
import { type EmbedCallOptions, makeEmbeddingFunc, openai_complete_if_cache, openai_embed_func } from './llm/openai.ts';
import { storeStats } from './store.ts';
import { getEnvValue } from './utils.ts';

export const EXIT = { ok: 0, error: 1, usage: 2, billing: 3, session_limit: 4, network: 5 } as const;
const STOP_EXIT: Record<ProviderStop, number> = { billing: EXIT.billing, session_limit: EXIT.session_limit, network: EXIT.network };
const MODES: readonly QueryMode[] = ['local', 'global', 'hybrid', 'naive', 'mix', 'bypass'];

const USAGE = `usage: lightrag-ts <command> [options]

  insert [FILE...]          insert files (no FILE or "-": stdin); per input, in order: --id ID, --file-path P
                            (default: the path relative to the current directory; stdin: unknown_source)
  query QUESTION            --mode local|global|hybrid|naive|mix|bypass (mix), --top-k N, --chunk-top-k N,
                            --only-context, --stream, --response-type T
  delete --doc-id ID        --delete-llm-cache to drop the document's cache entries too
  status                    documents by status, graph and vector counts (reads files only)

  common: --working-dir DIR (WORKING_DIR), --workspace W (WORKSPACE)`;

export type CliIO = { stdout: (s: string) => void; stderr: (s: string) => void; stdin: () => string; env: NodeJS.ProcessEnv; cwd?: () => string };

const defaultIO: CliIO = {
  stdout: (s) => process.stdout.write(s),
  stderr: (s) => process.stderr.write(s),
  stdin: () => readFileSync(0, 'utf8'),
  env: process.env,
};

/** Billing outranks a spent usage window, which outranks a lost network: the stop a run reports. */
const STOP_RANK: Record<ProviderStop, number> = { billing: 3, session_limit: 2, network: 1 };

/**
 * Stops a run at the first provider refusal, as the vault's ingest guard does:
 * the error is recorded, the pipeline is asked to cancel, and every later LLM
 * or embedding call fails at once with the recorded error instead of
 * spending its own retries (upstream keeps calling: each call retries and
 * fails on its own).
 */
export class StopGuard {
  stop: { kind: ProviderStop; error: unknown } | null = null;
  workspace = '';

  record(e: unknown): void {
    const kind = stopOf(e);
    if (!kind) return;
    if (!this.stop || STOP_RANK[kind] > STOP_RANK[this.stop.kind]) this.stop = { kind, error: e };
    getPipelineStatus(this.workspace).cancellation_requested = true;
  }

  wrap<A extends unknown[], R>(fn: (...a: A) => Promise<R>): (...a: A) => Promise<R> {
    return async (...a: A) => {
      if (this.stop) throw this.stop.error;
      try {
        return await fn(...a);
      } catch (e) {
        this.record(e);
        throw e;
      }
    };
  }
}

/** The LightRAG init this environment describes: the two functions and the storage location. */
export function lightragInitFromEnv(env: NodeJS.ProcessEnv, over: { working_dir?: string; workspace?: string } = {}, guard: StopGuard | null = null): LightRAGInit {
  const get = (k: string): string | undefined => (env[k] === undefined || env[k] === '' ? undefined : env[k]);
  const guarded = <A extends unknown[], R>(fn: (...a: A) => Promise<R>) => (guard ? guard.wrap(fn) : fn);
  // the adapter reads OPENAI_API_KEY / OPENAI_API_BASE from this environment, not from process.env
  const useBase64 = ['true', '1', 'yes'].includes((get('EMBEDDING_USE_BASE64') ?? 'true').toLowerCase());
  const llmBinding = get('LLM_BINDING') ?? 'openai';
  let llm_model_func: LightRAGInit['llm_model_func'];
  let llmModel: string;
  if (llmBinding === 'openai') {
    llmModel = get('LLM_MODEL') ?? 'gpt-4o-mini';
    const host = get('LLM_BINDING_HOST') ?? 'https://api.openai.com/v1';
    const apiKey = get('LLM_BINDING_API_KEY') ?? null;
    llm_model_func = guarded((prompt: string, opts: Record<string, unknown> = {}) => openai_complete_if_cache(llmModel, prompt, { ...opts, base_url: host, api_key: apiKey, env }));
  } else if (llmBinding === 'claude-complete') {
    llmModel = get('LLM_MODEL') ?? 'haiku';
    llm_model_func = guarded(
      makeClaudeCompleteLlm({
        model: llmModel,
        profile: get('CLAUDE_COMPLETE_PROFILE') ?? 'home',
        env,
        ...(get('CLAUDE_COMPLETE_BIN') ? { bin: get('CLAUDE_COMPLETE_BIN')! } : {}),
      }),
    );
  } else {
    throw new Error(`LLM_BINDING=${llmBinding} is not ported (openai, claude-complete)`);
  }

  const embBinding = get('EMBEDDING_BINDING') ?? 'openai';
  if (embBinding !== 'openai') throw new Error(`EMBEDDING_BINDING=${embBinding} is not ported (openai)`);
  const embModel = get('EMBEDDING_MODEL') ?? 'text-embedding-3-small';
  const embHost = get('EMBEDDING_BINDING_HOST') ?? 'https://api.openai.com/v1';
  const embKey = get('EMBEDDING_BINDING_API_KEY') ?? null;
  const intEnv = (k: string, d: number) => {
    const v = get(k);
    if (v === undefined) return d;
    const n = Number(v);
    if (!Number.isInteger(n)) throw new Error(`${k} must be an integer, got ${JSON.stringify(v)}`);
    return n;
  };
  // As the vault: the EmbeddingFunc carries no max_token_size (LightRAG's
  // embedding_token_limit stays None unless EMBEDDING_TOKEN_LIMIT is set),
  // while every call truncates at openai_embed's own 8192.
  const tokenLimit = get('EMBEDDING_TOKEN_LIMIT') === undefined ? null : intEnv('EMBEDDING_TOKEN_LIMIT', 0);
  const embedding_func = makeEmbeddingFunc({
    embedding_dim: intEnv('EMBEDDING_DIM', 1536),
    max_token_size: tokenLimit,
    send_dimensions: getEnvValue<boolean>('EMBEDDING_SEND_DIM', false, 'bool', false, env) === true,
    model_name: embModel,
    func: guarded((texts: string[], o: EmbedCallOptions) =>
      openai_embed_func(texts, {
        model: embModel,
        base_url: embHost,
        api_key: embKey,
        embedding_dim: o.embedding_dim ?? null,
        max_token_size: o.max_token_size ?? 8192,
        use_base64: useBase64,
        env,
      }),
    ),
  });

  return {
    working_dir: over.working_dir ?? get('WORKING_DIR') ?? './rag_storage',
    ...(over.workspace !== undefined ? { workspace: over.workspace } : {}),
    llm_model_func,
    llm_model_name: llmModel,
    embedding_func,
  };
}

/** The provider stop behind a failure, from the error itself or from a failed document's error_msg. */
function stopOf(e: unknown): ProviderStop | null {
  return classifyProviderError(e)?.kind ?? null;
}

async function withRag<T>(init: LightRAGInit, fn: (rag: LightRAG) => Promise<T>, guard: StopGuard | null = null): Promise<T> {
  const rag = new LightRAG(init);
  if (guard) guard.workspace = rag.workspace;
  await rag.initialize_storages();
  try {
    return await fn(rag);
  } finally {
    await rag.finalize_storages();
  }
}

export async function main(argv: string[], io: CliIO = defaultIO): Promise<number> {
  const [command, ...rest] = argv;
  if (!command || command === '--help' || command === '-h') {
    io.stdout(USAGE + '\n');
    return command ? EXIT.ok : EXIT.usage;
  }
  let parsed;
  try {
    parsed = parseArgs({
      args: rest,
      allowPositionals: true,
      options: {
        'working-dir': { type: 'string' },
        workspace: { type: 'string' },
        id: { type: 'string', multiple: true },
        'file-path': { type: 'string', multiple: true },
        mode: { type: 'string', default: 'mix' },
        'top-k': { type: 'string' },
        'chunk-top-k': { type: 'string' },
        'response-type': { type: 'string' },
        'only-context': { type: 'boolean', default: false },
        stream: { type: 'boolean', default: false },
        'doc-id': { type: 'string' },
        'delete-llm-cache': { type: 'boolean', default: false },
      },
    });
  } catch (e) {
    io.stderr(`${(e as Error).message}\n${USAGE}\n`);
    return EXIT.usage;
  }
  const { values: v, positionals } = parsed;
  const over = { ...(v['working-dir'] !== undefined ? { working_dir: v['working-dir'] } : {}), ...(v.workspace !== undefined ? { workspace: v.workspace } : {}) };
  const usage = (msg: string) => {
    io.stderr(`${msg}\n${USAGE}\n`);
    return EXIT.usage;
  };

  try {
    switch (command) {
      case 'status': {
        const dir = over.working_dir ?? io.env.WORKING_DIR ?? './rag_storage';
        const workspace = over.workspace ?? io.env.WORKSPACE ?? '';
        const s = storeStats(dir, workspace);
        io.stdout(JSON.stringify({ working_dir: dir, workspace, documents: s.docStatus, graph: s.graph, kv: s.kv, vdb: s.vdb }, null, 2) + '\n');
        return EXIT.ok;
      }
      case 'insert': {
        const files = positionals.length ? positionals : ['-'];
        const ids = v.id ?? null;
        if (ids && ids.length !== files.length) return usage(`--id given ${ids.length} times for ${files.length} input(s)`);
        const given = v['file-path'] ?? null;
        if (given && given.length !== files.length) return usage(`--file-path given ${given.length} times for ${files.length} input(s)`);
        // the vault stores a document under its path inside the vault: by default
        // the path relative to the current directory, so a/x.md and b/x.md stay apart
        const cwd = io.cwd?.() ?? process.cwd();
        const texts = files.map((f) => (f === '-' ? io.stdin() : readFileSync(resolve(cwd, f), 'utf8')));
        const paths = files.map((f, i) => given?.[i] ?? (f === '-' ? 'unknown_source' : relative(cwd, resolve(cwd, f))));
        const guard = new StopGuard();
        return await withRag(lightragInitFromEnv(io.env, over, guard), async (rag) => {
          let trackId: string;
          try {
            trackId = await rag.ainsert(texts, null, false, ids, paths);
          } catch (e) {
            if (!guard.stop) throw e;
            io.stderr(`${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}\nstopped: ${guard.stop.kind}\n`);
            return STOP_EXIT[guard.stop.kind];
          }
          const docs = await rag.aget_docs_by_track_id(trackId);
          const out: Record<string, { status: string; file_path: string; chunks_count?: number | null; error_msg?: string }> = {};
          let stop: ProviderStop | null = guard.stop?.kind ?? null;
          for (const [id, d] of docs) {
            out[id] = { status: d.status, file_path: d.file_path, ...(d.chunks_count !== undefined ? { chunks_count: d.chunks_count } : {}), ...(d.error_msg ? { error_msg: d.error_msg } : {}) };
            const k = d.error_msg ? stopOf(new Error(d.error_msg)) : null;
            if (k && (!stop || STOP_RANK[k] > STOP_RANK[stop])) stop = k;
          }
          io.stdout(JSON.stringify({ track_id: trackId, documents: out }, null, 2) + '\n');
          const failed = Object.values(out).filter((d) => d.status === 'failed').length;
          if (stop) {
            io.stderr(`stopped: ${stop}\n`);
            return STOP_EXIT[stop];
          }
          return failed ? EXIT.error : EXIT.ok;
        }, guard);
      }
      case 'query': {
        const question = positionals.join(' ');
        if (!question) return usage('query: a question is required');
        const mode = v.mode as QueryMode;
        if (!MODES.includes(mode)) return usage(`query: unknown mode ${mode}`);
        const num = (k: 'top-k' | 'chunk-top-k') => {
          const x = v[k];
          if (x === undefined) return undefined;
          const n = Number(x);
          if (!Number.isInteger(n)) throw new Error(`--${k} must be an integer`);
          return n;
        };
        const param = makeQueryParam({
          mode,
          only_need_context: v['only-context'],
          stream: v.stream,
          ...(num('top-k') !== undefined ? { top_k: num('top-k')! } : {}),
          ...(num('chunk-top-k') !== undefined ? { chunk_top_k: num('chunk-top-k')! } : {}),
          ...(v['response-type'] !== undefined ? { response_type: v['response-type'] } : {}),
        });
        return await withRag(lightragInitFromEnv(io.env, over), async (rag) => {
          const r = await rag.aquery(question, param);
          if (r === null) io.stdout('\n');
          else if (typeof r === 'string') io.stdout(r.endsWith('\n') ? r : r + '\n');
          else {
            for await (const chunk of r) io.stdout(chunk);
            io.stdout('\n');
          }
          return EXIT.ok;
        });
      }
      case 'delete': {
        const docId = v['doc-id'];
        if (!docId) return usage('delete: --doc-id is required');
        return await withRag(lightragInitFromEnv(io.env, over), async (rag) => {
          const r = await rag.adelete_by_doc_id(docId, v['delete-llm-cache']);
          io.stdout(JSON.stringify(r, null, 2) + '\n');
          return r.status === 'success' ? EXIT.ok : EXIT.error;
        });
      }
      default:
        return usage(`unknown command: ${command}`);
    }
  } catch (e) {
    const stop = stopOf(e);
    io.stderr(`${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}\n`);
    if (stop) {
      io.stderr(`stopped: ${stop}\n`);
      return STOP_EXIT[stop];
    }
    return EXIT.error;
  }
}

const isMain = (() => {
  try {
    return !!process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
})();
if (isMain) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (e: unknown) => {
      process.stderr.write(`${String(e)}\n`);
      process.exitCode = EXIT.error;
    },
  );
}
