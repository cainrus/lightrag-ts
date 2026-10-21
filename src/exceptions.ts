// Port of lightrag/exceptions.py (the classes the ported modules raise).
import { pyStrip } from './utils.ts';

/** exceptions.ChunkTokenLimitExceededError (a ValueError upstream). */
export class ChunkTokenLimitExceededError extends Error {
  readonly chunkTokens: number;
  readonly chunkTokenLimit: number;
  readonly chunkPreview: string | null;

  constructor(chunkTokens: number, chunkTokenLimit: number, chunkPreview: string | null = null) {
    const preview = chunkPreview ? pyStrip(chunkPreview) : null;
    // Python slices by code point.
    const truncated = preview ? Array.from(preview).slice(0, 80).join('') : null;
    const note = truncated ? ` Preview: '${truncated}'` : '';
    super(`Chunk token length ${chunkTokens} exceeds chunk_token_size ${chunkTokenLimit}.${note}`);
    this.name = 'ChunkTokenLimitExceededError';
    this.chunkTokens = chunkTokens;
    this.chunkTokenLimit = chunkTokenLimit;
    this.chunkPreview = truncated;
  }
}

/**
 * An error named after the Python exception upstream raises at the same spot
 * (ValueError, IndexError, ...), so callers and the upstream-test adapter can
 * tell them apart the way Python code does.
 */
export function pyError(type: 'ValueError' | 'IndexError' | 'KeyError' | 'TypeError' | 'AttributeError', message: string): Error {
  const e = new Error(message);
  e.name = type;
  return e;
}

/** exceptions.PipelineCancelledException */
export class PipelineCancelledException extends Error {
  constructor(message = 'User cancelled') {
    super(message);
    this.name = 'PipelineCancelledException';
  }
}

/**
 * utils.create_prefixed_exception: same class, message `${prefix}: ${message}`.
 * JS errors carry one message, so this mutates and returns the original.
 */
export function createPrefixedException<E>(e: E, prefix: string): E {
  if (e instanceof Error) e.message = `${prefix}: ${e.message}`;
  return e;
}
