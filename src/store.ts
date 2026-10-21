// Read-only view over a whole LightRAG working directory written by the
// Python implementation (default storages: JsonKV, JsonDocStatus,
// NanoVectorDB, NetworkX). Not an upstream module: a port-only entry point
// that B1's storage classes will build on.
import { existsSync } from 'node:fs';
import { forEachJsonKv, kvStorePath } from './kg/json_kv_impl.ts';
import { graphmlPath, readGraphml } from './kg/networkx_impl.ts';
import { readNanoVectorDb, vdbPath } from './kg/nano_vector_db_impl.ts';
import { NameSpace } from './namespace.ts';

export const KV_NAMESPACES = [
  NameSpace.KV_STORE_FULL_DOCS,
  NameSpace.KV_STORE_TEXT_CHUNKS,
  NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
  NameSpace.KV_STORE_FULL_ENTITIES,
  NameSpace.KV_STORE_FULL_RELATIONS,
  NameSpace.KV_STORE_ENTITY_CHUNKS,
  NameSpace.KV_STORE_RELATION_CHUNKS,
  NameSpace.DOC_STATUS,
] as const;

export const VDB_NAMESPACES = [
  NameSpace.VECTOR_STORE_ENTITIES,
  NameSpace.VECTOR_STORE_RELATIONSHIPS,
  NameSpace.VECTOR_STORE_CHUNKS,
] as const;

export type StoreStats = {
  graph: { nodes: number; edges: number; edgeElements: number } | null;
  kv: Record<string, number | null>;
  docStatus: Record<string, number>;
  vdb: Record<string, { records: number; embeddingDim: number; matrixRows: number } | null>;
};

/** Count everything in a store, reading files strictly read-only and one at a time. */
export function storeStats(workingDir: string, workspace = '', log: (s: string) => void = () => {}): StoreStats {
  const out: StoreStats = { graph: null, kv: {}, docStatus: {}, vdb: {} };
  const gp = graphmlPath(workingDir, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION, workspace);
  if (existsSync(gp)) {
    const g = readGraphml(gp);
    out.graph = { nodes: g.nodes.size, edges: g.edges.size, edgeElements: g.edgeElements };
    log(`graph: ${g.nodes.size} nodes, ${g.edges.size} edges`);
  }
  for (const ns of KV_NAMESPACES) {
    const p = kvStorePath(workingDir, ns, workspace);
    if (!existsSync(p)) {
      out.kv[ns] = null;
      continue;
    }
    out.kv[ns] = forEachJsonKv<{ status?: string }>(p, (_k, v) => {
      if (ns === NameSpace.DOC_STATUS) {
        const s = v.status ?? '<none>';
        out.docStatus[s] = (out.docStatus[s] ?? 0) + 1;
      }
    });
    log(`kv ${ns}: ${out.kv[ns]}`);
  }
  for (const ns of VDB_NAMESPACES) {
    const p = vdbPath(workingDir, ns, workspace);
    if (!existsSync(p)) {
      out.vdb[ns] = null;
      continue;
    }
    const v = readNanoVectorDb(p, { matrix: 'skip' });
    out.vdb[ns] = { records: v.data.length, embeddingDim: v.embeddingDim, matrixRows: v.matrixBytes / 4 / v.embeddingDim };
    log(`vdb ${ns}: ${v.data.length} records, matrix rows ${out.vdb[ns]!.matrixRows}`);
  }
  return out;
}
