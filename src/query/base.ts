// Port of the query half of lightrag/base.py: QueryParam, QueryResult,
// QueryContextResult. Kept out of src/base.ts while B2 edits that file; folds
// into it once both land.
import {
  DEFAULT_CHUNK_TOP_K,
  DEFAULT_HISTORY_TURNS,
  DEFAULT_MAX_ENTITY_TOKENS,
  DEFAULT_MAX_RELATION_TOKENS,
  DEFAULT_MAX_TOTAL_TOKENS,
  DEFAULT_TOP_K,
} from '../constants.ts';
import type { ChatMessage } from '../utils.ts';

export type QueryMode = 'local' | 'global' | 'hybrid' | 'naive' | 'mix' | 'bypass';

export type QueryLlmOptions = {
  system_prompt?: string | null;
  history_messages?: ChatMessage[];
  enable_cot?: boolean;
  stream?: boolean;
  keyword_extraction?: boolean;
  _priority?: number;
};
/** An LLM binding as kg_query calls it: a string, or an async iterator when streaming. */
export type QueryLlmFunc = (prompt: string, opts?: QueryLlmOptions) => Promise<string | AsyncIterable<string>>;

/** base.QueryParam. `null` stands for Python's None where upstream allows it. */
export type QueryParam = {
  mode: QueryMode;
  only_need_context: boolean;
  only_need_prompt: boolean;
  response_type: string | null;
  stream: boolean;
  top_k: number;
  chunk_top_k: number | null;
  max_entity_tokens: number;
  max_relation_tokens: number;
  max_total_tokens: number;
  hl_keywords: string[];
  ll_keywords: string[];
  conversation_history: ChatMessage[];
  /** deprecated upstream, kept for the dataclass shape */
  history_turns: number;
  model_func: QueryLlmFunc | null;
  user_prompt: string | null;
  enable_rerank: boolean;
  include_references: boolean;
};

/** Python's `int(os.getenv(name, str(default)))`, read once like the class attribute. */
function envInt(name: string, dflt: number): number {
  const v = process.env[name];
  if (v === undefined) return dflt;
  const t = v.trim();
  if (!/^[+-]?\d+(_\d+)*$/.test(t)) throw new Error(`invalid literal for int() with base 10: '${v}'`);
  return Number.parseInt(t.replaceAll('_', ''), 10);
}

/** Field defaults, evaluated at import like the dataclass's `os.getenv` defaults. */
const DEFAULTS = {
  top_k: envInt('TOP_K', DEFAULT_TOP_K),
  chunk_top_k: envInt('CHUNK_TOP_K', DEFAULT_CHUNK_TOP_K),
  max_entity_tokens: envInt('MAX_ENTITY_TOKENS', DEFAULT_MAX_ENTITY_TOKENS),
  max_relation_tokens: envInt('MAX_RELATION_TOKENS', DEFAULT_MAX_RELATION_TOKENS),
  max_total_tokens: envInt('MAX_TOTAL_TOKENS', DEFAULT_MAX_TOTAL_TOKENS),
  history_turns: envInt('HISTORY_TURNS', DEFAULT_HISTORY_TURNS),
  enable_rerank: (process.env.RERANK_BY_DEFAULT ?? 'true').toLowerCase() === 'true',
};

/** `QueryParam(**fields)` with upstream's defaults (fresh lists per instance). */
export function makeQueryParam(fields: Partial<QueryParam> = {}): QueryParam {
  return {
    mode: 'mix',
    only_need_context: false,
    only_need_prompt: false,
    response_type: 'Multiple Paragraphs',
    stream: false,
    top_k: DEFAULTS.top_k,
    chunk_top_k: DEFAULTS.chunk_top_k,
    max_entity_tokens: DEFAULTS.max_entity_tokens,
    max_relation_tokens: DEFAULTS.max_relation_tokens,
    max_total_tokens: DEFAULTS.max_total_tokens,
    hl_keywords: [],
    ll_keywords: [],
    conversation_history: [],
    history_turns: DEFAULTS.history_turns,
    model_func: null,
    user_prompt: null,
    enable_rerank: DEFAULTS.enable_rerank,
    include_references: false,
    ...fields,
  };
}

export type ReferenceEntry = { reference_id: string; file_path: string };

/** The dict convert_to_user_format builds, plus what callers add to it. */
export type QueryRawData = {
  status: string;
  message: string;
  data: {
    entities?: Record<string, unknown>[];
    relationships?: Record<string, unknown>[];
    chunks?: Record<string, unknown>[];
    references?: ReferenceEntry[];
  };
  metadata: Record<string, unknown>;
  [k: string]: unknown;
};

/** base.QueryResult */
export class QueryResult {
  content: string | null;
  response_iterator: AsyncIterable<string> | null;
  raw_data: QueryRawData | null;
  is_streaming: boolean;
  constructor(f: { content?: string | null; response_iterator?: AsyncIterable<string> | null; raw_data?: QueryRawData | null; is_streaming?: boolean } = {}) {
    this.content = f.content ?? null;
    this.response_iterator = f.response_iterator ?? null;
    this.raw_data = f.raw_data ?? null;
    this.is_streaming = f.is_streaming ?? false;
  }
  get reference_list(): ReferenceEntry[] {
    return this.raw_data ? (this.raw_data.data?.references ?? []) : [];
  }
  get metadata(): Record<string, unknown> {
    return this.raw_data ? (this.raw_data.metadata ?? {}) : {};
  }
}

/** base.QueryContextResult */
export class QueryContextResult {
  context: string;
  raw_data: QueryRawData;
  constructor(context: string, rawData: QueryRawData) {
    this.context = context;
    this.raw_data = rawData;
  }
  get reference_list(): ReferenceEntry[] {
    return this.raw_data.data?.references ?? [];
  }
}
