// Port of lightrag/operate.py: chunking, extraction prompts and cache keys;
// extraction/merge land in B2, query in B3.
import { DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE } from './constants.ts';
import { ChunkTokenLimitExceededError } from './exceptions.ts';
import { PROMPTS } from './prompt.ts';
import { llmCacheLookup, packUserAssToOpenaiMessages, pyFormat, pyStrip, type Tokenizer } from './utils.ts';

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

export type TextChunk = { tokens: number; content: string; chunk_order_index: number };

/** operate.chunking_by_token_size */
export function chunkingByTokenSize(
  tokenizer: Tokenizer,
  content: string,
  splitByCharacter: string | null = null,
  splitByCharacterOnly = false,
  chunkOverlapTokenSize = 100,
  chunkTokenSize = 1200,
): TextChunk[] {
  const step = chunkTokenSize - chunkOverlapTokenSize;
  // Python's range(0, n, step): step 0 raises, a negative step yields nothing.
  if (step === 0) throw new Error('range() arg 3 must not be zero');
  const starts = (n: number) => (step > 0 ? Array.from({ length: Math.ceil(n / step) }, (_, i) => i * step) : []);
  const results: TextChunk[] = [];
  if (splitByCharacter) {
    const newChunks: [number, string][] = [];
    for (const chunk of content.split(splitByCharacter)) {
      const tokens = tokenizer.encode(chunk);
      if (tokens.length <= chunkTokenSize) {
        newChunks.push([tokens.length, chunk]);
        continue;
      }
      if (splitByCharacterOnly) {
        throw new ChunkTokenLimitExceededError(tokens.length, chunkTokenSize, Array.from(chunk).slice(0, 120).join(''));
      }
      for (const start of starts(tokens.length)) {
        newChunks.push([Math.min(chunkTokenSize, tokens.length - start), tokenizer.decode(tokens.slice(start, start + chunkTokenSize))]);
      }
    }
    newChunks.forEach(([len, chunk], index) => results.push({ tokens: len, content: pyStrip(chunk), chunk_order_index: index }));
    return results;
  }
  const tokens = tokenizer.encode(content);
  for (const [index, start] of starts(tokens.length).entries()) {
    results.push({
      tokens: Math.min(chunkTokenSize, tokens.length - start),
      content: pyStrip(tokenizer.decode(tokens.slice(start, start + chunkTokenSize))),
      chunk_order_index: index,
    });
  }
  return results;
}
