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
