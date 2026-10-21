// Do graph edges/nodes and their vectors line up by id? Upstream ids:
// entity  = "ent-" + md5(entity_name)
// relation = make_relation_vdb_ids(src, tgt) (normalized order first, reverse as fallback)
//   node parity/check-vdb-ids.ts <working_dir>
import { join } from 'node:path';
import { readNanoVectorDb } from '../src/kg/nano_vector_db_impl.ts';
import { readGraphml } from '../src/kg/networkx_impl.ts';
import { computeMdhashId, makeRelationVdbIds } from '../src/utils.ts';

const dir = process.argv[2]!;
const g = readGraphml(join(dir, 'graph_chunk_entity_relation.graphml'));
const ents = new Set(readNanoVectorDb(join(dir, 'vdb_entities.json')).data.map((d) => d.__id__));
const rels = readNanoVectorDb(join(dir, 'vdb_relationships.json')).data;
const relIds = new Set(rels.map((d) => d.__id__));

let nodeHit = 0;
for (const n of g.nodes.keys()) if (ents.has(computeMdhashId(n, 'ent-'))) nodeHit++;

let normHit = 0;
let reverseHit = 0;
let none = 0;
const graphRelIds = new Set<string>();
for (const e of g.edges.values()) {
  const [norm, rev] = makeRelationVdbIds(e.source, e.target);
  graphRelIds.add(norm!);
  if (rev) graphRelIds.add(rev);
  if (relIds.has(norm!)) normHit++;
  else if (rev && relIds.has(rev)) reverseHit++;
  else none++;
}
// vectors that point at an edge missing from the graph
let vecWithoutEdge = 0;
for (const r of rels) if (!graphRelIds.has(r.__id__)) vecWithoutEdge++;
// id recomputed from the record's own src_id/tgt_id
let selfConsistent = 0;
for (const r of rels) {
  const ids = makeRelationVdbIds(String(r.src_id), String(r.tgt_id));
  if (ids.includes(r.__id__)) selfConsistent++;
}
console.log(
  JSON.stringify(
    {
      nodes: g.nodes.size,
      nodesWithVector: nodeHit,
      edges: g.edges.size,
      edgeVectorNormalizedId: normHit,
      edgeVectorReverseId: reverseHit,
      edgeWithoutVector: none,
      relVectors: rels.length,
      relVectorsWithoutEdge: vecWithoutEdge,
      relVectorIdMatchesOwnSrcTgt: selfConsistent,
    },
    null,
    2,
  ),
);
