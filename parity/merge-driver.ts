// The TS side of the merge comparisons, shared by parity/operate.ts (a store
// copy) and test/merge.test.ts (public fixtures). Mirrors cmd_merge in
// tools/upstream/operate_parity.py: same storages, same stand-ins for the LLM
// and the embedding model, same fixed clock.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import type { EmbeddingFunc } from '../src/base.ts';
import { DEFAULT_EMBEDDING_BATCH_NUM, DEFAULT_ENTITY_TYPES, DEFAULT_MAX_EXTRACT_INPUT_TOKENS } from '../src/constants.ts';
import { JsonKVStorage } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDBStorage } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from '../src/kg/networkx_impl.ts';
import { NameSpace } from '../src/namespace.ts';
import {
  type ChunkExtraction,
  defaultMergeConfig,
  edgeKey,
  type ExtractChunk,
  type ExtractedEntity,
  type ExtractedRelation,
  extractEntities,
  type MergeConfig,
  mergeNodesAndEdges,
} from '../src/operate.ts';
import { type LlmFunc, TiktokenTokenizer } from '../src/utils.ts';

/** A document to merge: chunk ids to extract from a store's cache, or extraction results as JSON. */
export type MergeDoc = {
  id: string;
  file_path: string;
  chunks?: string[];
  results?: { nodes: [string, ExtractedEntity[]][]; edges: [[string, string], ExtractedRelation[]][] }[];
};
export type MergeRec = { doc: string; skipped?: string; error?: string; messages?: string[] };

export const FIXED_NOW = 1760000000;
export const fixedClock = () => FIXED_NOW;

export class CacheMiss extends Error {
  override name = 'CacheMiss';
}

const SUMMARY_HEAD = '---Role---\nYou are a Knowledge Graph Specialist, proficient in data curation and synthesis.';
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;
/** md5 of Python's `s.encode("utf-8", errors="replace")` */
const md5Utf8Replace = (s: string) => createHash('md5').update(Buffer.from(s.replace(LONE_SURROGATE, '?'), 'utf8'));

/** merge_llm_stub(): a summary prompt gets a deterministic stand-in, anything else is a cache miss. */
export const mergeLlmStub: LlmFunc = async (prompt) => {
  if (prompt.startsWith(SUMMARY_HEAD)) return `Stub summary ${md5Utf8Replace(prompt).digest('hex')}`;
  throw new CacheMiss('cache miss: the parity run never calls an LLM');
};

/** stub_vector(): the first `dim` md5 bytes of the text, scaled to [-1, 1). */
export function stubEmbedding(dim: number): EmbeddingFunc {
  const f = async (texts: string[]) =>
    texts.map((t) => {
      const d = md5Utf8Replace(t).digest();
      return Float32Array.from({ length: dim }, (_, j) => (d[j]! - 128) / 128);
    });
  return Object.assign(f, { embedding_dim: dim });
}

export const MERGE_FILES = [
  'kv_store_llm_response_cache.json',
  'kv_store_entity_chunks.json',
  'kv_store_relation_chunks.json',
  'kv_store_full_entities.json',
  'kv_store_full_relations.json',
  'graph_chunk_entity_relation.graphml',
  'vdb_entities.json',
  'vdb_relationships.json',
];

const toExtraction = (r: NonNullable<MergeDoc['results']>[number]): ChunkExtraction => ({
  nodes: new Map(r.nodes.map(([k, v]) => [k, v])),
  edges: new Map(r.edges.map(([[s, t], v]) => [edgeKey(s, t), v])),
});

/** Merge `docs` one after another into the storages under `dir`, as cmd_merge does. */
export async function tsMergeDocs(
  dir: string,
  docs: readonly MergeDoc[],
  o: {
    vdb: boolean;
    dim: number;
    gc?: Partial<MergeConfig>;
    model?: string;
    maxConcurrency?: number;
    chunks?: Map<string, Record<string, unknown>>;
    /** every LLM cache key found (a hit), as the merge looks it up */
    onCacheHit?: (key: string) => void;
  },
): Promise<MergeRec[]> {
  const gcStore = {
    working_dir: dir,
    embedding_batch_num: DEFAULT_EMBEDDING_BATCH_NUM,
    vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 },
    enable_llm_cache: true,
    enable_llm_cache_for_entity_extract: true,
  };
  const kv = (namespace: string) => new JsonKVStorage({ namespace, global_config: gcStore, clock: fixedClock });
  const cache = kv(NameSpace.KV_STORE_LLM_RESPONSE_CACHE);
  const entityChunks = kv(NameSpace.KV_STORE_ENTITY_CHUNKS);
  const relationChunks = kv(NameSpace.KV_STORE_RELATION_CHUNKS);
  const fullEntities = kv(NameSpace.KV_STORE_FULL_ENTITIES);
  const fullRelations = kv(NameSpace.KV_STORE_FULL_RELATIONS);
  for (const s of [cache, entityChunks, relationChunks, fullEntities, fullRelations]) await s.initialize();
  if (o.onCacheHit) {
    const get = cache.get_by_id.bind(cache);
    cache.get_by_id = async (id: string) => {
      const r = await get(id);
      if (r) o.onCacheHit!(id);
      return r;
    };
  }
  const graph = new NetworkXStorage({ namespace: NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION, global_config: gcStore });
  const embedding = stubEmbedding(o.dim);
  const vdb = (namespace: string, meta: string[]) =>
    new NanoVectorDBStorage({ namespace, global_config: gcStore, embedding_func: embedding, meta_fields: meta, clock: fixedClock });
  const entVdb = o.vdb ? vdb(NameSpace.VECTOR_STORE_ENTITIES, ['entity_name', 'source_id', 'content', 'file_path']) : null;
  const relVdb = o.vdb ? vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS, ['src_id', 'tgt_id', 'source_id', 'content', 'file_path']) : null;

  const tok = new TiktokenTokenizer(o.model ?? 'gpt-4o-mini');
  const extractGc = {
    llm_model_func: mergeLlmStub,
    entity_extract_max_gleaning: 1,
    addon_params: { language: 'English', entity_types: DEFAULT_ENTITY_TYPES },
    tokenizer: tok,
    max_extract_input_tokens: DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    llm_model_max_async: 4,
  };
  const mergeGc = defaultMergeConfig(tok, mergeLlmStub, o.gc ?? {});
  const recs: MergeRec[] = [];
  for (const [n, d] of docs.entries()) {
    const rec: MergeRec = { doc: d.id };
    const results: ChunkExtraction[] = (d.results ?? []).map(toExtraction);
    try {
      for (const cid of d.chunks ?? []) {
        const [r] = await extractEntities(new Map([[cid, o.chunks!.get(cid) as ExtractChunk]]), extractGc, { llmResponseCache: cache });
        results.push(r!);
      }
    } catch (e) {
      rec.skipped = `${(e as Error).name}: ${(e as Error).message}`;
      recs.push(rec);
      continue;
    }
    const status = { history_messages: [] as string[], latest_message: '' };
    try {
      await mergeNodesAndEdges(results, graph, entVdb, relVdb, mergeGc, {
        fullEntitiesStorage: fullEntities,
        fullRelationsStorage: fullRelations,
        docId: d.id,
        pipelineStatus: status,
        llmResponseCache: cache,
        entityChunksStorage: entityChunks,
        relationChunksStorage: relationChunks,
        currentFileNumber: n + 1,
        totalFiles: docs.length,
        filePath: d.file_path,
        clock: fixedClock,
        ...(o.maxConcurrency === undefined ? {} : { maxConcurrency: o.maxConcurrency }),
      });
    } catch (e) {
      rec.error = `${(e as Error).name}: ${(e as Error).message}`;
    }
    rec.messages = status.history_messages;
    recs.push(rec);
  }
  for (const s of [cache, entityChunks, relationChunks, fullEntities, fullRelations, graph, entVdb, relVdb]) await s?.index_done_callback();
  return recs;
}

/** First byte where two files differ, with a little context, or null when identical. */
export function fileDiff(a: string, b: string): string | null {
  const ea = existsSync(a);
  const eb = existsSync(b);
  if (!ea || !eb) return ea === eb ? null : `${ea ? b : a} missing`;
  const x = readFileSync(a);
  const y = readFileSync(b);
  if (x.equals(y)) return null;
  let i = 0;
  while (i < x.length && i < y.length && x[i] === y[i]) i++;
  const ctx = (buf: Buffer) => JSON.stringify(buf.subarray(Math.max(0, i - 80), i + 120).toString('utf8'));
  return `${x.length} vs ${y.length} bytes, first difference at ${i}: expected=${ctx(x)} ts=${ctx(y)}`;
}

/** full_entities / full_relations: upstream lists a Python set, so only the members are comparable. */
export function setFileDiff(a: string, b: string, field: 'entity_names' | 'relation_pairs'): string | null {
  if (!existsSync(a) || !existsSync(b)) return existsSync(a) === existsSync(b) ? null : 'one side missing';
  const norm = (f: string) => {
    const o = JSON.parse(readFileSync(f, 'utf8')) as Record<string, Record<string, unknown>>;
    for (const v of Object.values(o)) v[field] = (v[field] as unknown[]).map((x) => JSON.stringify(x)).sort();
    return o;
  };
  const x = norm(a);
  const y = norm(b);
  if (JSON.stringify(Object.keys(x)) !== JSON.stringify(Object.keys(y))) return 'different doc keys or key order';
  for (const k of Object.keys(x)) {
    if (!isDeepStrictEqual(x[k], y[k])) return `${k}: expected=${JSON.stringify(x[k]).slice(0, 300)} ts=${JSON.stringify(y[k]).slice(0, 300)}`;
  }
  return null;
}

/** Every merge output file of `expected` vs `actual`: [file, difference | null]. */
export function mergeDirDiffs(expected: string, actual: string, o: { vdb: boolean; cacheAsMap?: boolean }): [string, string | null][] {
  const out: [string, string | null][] = [];
  for (const f of MERGE_FILES) {
    if (!o.vdb && f.startsWith('vdb_')) continue;
    const a = join(expected, f);
    const b = join(actual, f);
    if (f === 'kv_store_full_entities.json') out.push([f, setFileDiff(a, b, 'entity_names')]);
    else if (f === 'kv_store_full_relations.json') out.push([f, setFileDiff(a, b, 'relation_pairs')]);
    else if (f === 'kv_store_llm_response_cache.json' && o.cacheAsMap) {
      const m = (p: string) => Object.entries(JSON.parse(readFileSync(p, 'utf8')) as object).sort(([x], [y]) => (x < y ? -1 : 1));
      out.push([f, isDeepStrictEqual(m(a), m(b)) ? null : 'cache entries differ']);
    } else out.push([f, fileDiff(a, b)]);
  }
  return out;
}
