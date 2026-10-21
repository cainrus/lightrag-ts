// Port of lightrag/operate.py. B0 holds only the prompt construction of
// extract_entities (operate.py:2899-2960), which the parity harness needs to
// recompute LLM cache keys; the rest of the module lands in B2/B3.
import { DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE } from './constants.ts';
import { PROMPTS } from './prompt.ts';
import { llmCacheLookup, packUserAssToOpenaiMessages, pyFormat } from './utils.ts';

export type ExtractionPromptConfig = {
  /** addon_params.language */
  language?: string;
  /** addon_params.entity_types */
  entityTypes?: readonly string[];
};

export type ExtractionPrompts = {
  system: string;
  user: string;
  continueUser: string;
};

/** The three prompts extract_entities sends for one chunk. */
export function buildExtractionPrompts(content: string, cfg: ExtractionPromptConfig = {}): ExtractionPrompts {
  const language = cfg.language ?? DEFAULT_SUMMARY_LANGUAGE;
  const entityTypes = cfg.entityTypes ?? DEFAULT_ENTITY_TYPES;
  const exampleContextBase = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: entityTypes.join(', '),
    language,
  };
  const examples = pyFormat(PROMPTS.entity_extraction_examples.join('\n'), exampleContextBase);
  const contextBase = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: entityTypes.join(','),
    examples,
    language,
  };
  return {
    system: pyFormat(PROMPTS.entity_extraction_system_prompt, contextBase),
    user: pyFormat(PROMPTS.entity_extraction_user_prompt, { ...contextBase, input_text: content }),
    continueUser: pyFormat(PROMPTS.entity_continue_extraction_user_prompt, { ...contextBase, input_text: content }),
  };
}

/**
 * Cache keys of the initial extraction call and of the gleaning call for a
 * chunk. The gleaning key depends on the initial LLM answer (it is part of the
 * history), so it is computed from `initialResult` - typically the cached one.
 */
export function extractionCacheKeys(
  content: string,
  cfg: ExtractionPromptConfig = {},
  initialResult?: string,
): { initial: string; gleaning?: string } {
  const p = buildExtractionPrompts(content, cfg);
  const initial = llmCacheLookup(p.user, { systemPrompt: p.system, cacheType: 'extract' }).cacheKey;
  if (initialResult === undefined) return { initial };
  const history = packUserAssToOpenaiMessages(p.user, initialResult);
  const gleaning = llmCacheLookup(p.continueUser, {
    systemPrompt: p.system,
    historyMessages: history,
    cacheType: 'extract',
  }).cacheKey;
  return { initial, gleaning };
}
