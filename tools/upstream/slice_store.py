"""Copy a small, self-consistent slice of a real LightRAG store, using upstream writers.

    <venv>/bin/python -I tools/upstream/slice_store.py <src_working_dir> <dst_dir> [--docs 5]

Reads <src> only. Picks the first N `processed` docs and keeps exactly what
belongs to them: full_docs, doc_status, text_chunks, the LLM-cache entries of
their chunks, vdb_chunks rows, graph nodes/edges whose source_id cites one of
those chunks, and their entity/relation vectors. The output is PRIVATE data:
write it under fixtures/private/ (gitignored), never into a committed path.
"""

import argparse
import json
import os

import networkx as nx

from lightrag import utils
from lightrag.constants import GRAPH_FIELD_SEP

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--docs", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)


def kv(ns):
    return utils.load_json(os.path.join(a.src, f"kv_store_{ns}.json")) or {}


def put(ns, data):
    utils.write_json(data, os.path.join(a.dst, f"kv_store_{ns}.json"))


status = kv("doc_status")
docs = [k for k, v in status.items() if v.get("status") == "processed"][: a.docs]
chunk_ids = {c for d in docs for c in status[d].get("chunks_list", [])}

put("doc_status", {d: status[d] for d in docs})
put("full_docs", {k: v for k, v in kv("full_docs").items() if k in docs})
put("text_chunks", {k: v for k, v in kv("text_chunks").items() if k in chunk_ids})
put("llm_response_cache", {k: v for k, v in kv("llm_response_cache").items() if v.get("chunk_id") in chunk_ids})

g = nx.read_graphml(os.path.join(a.src, "graph_chunk_entity_relation.graphml"))


def cites(attrs):
    return any(s in chunk_ids for s in str(attrs.get("source_id", "")).split(GRAPH_FIELD_SEP))


nodes = {n for n, d in g.nodes(data=True) if cites(d)}
sub = nx.Graph()
sub.add_nodes_from((n, g.nodes[n]) for n in nodes)
sub.add_edges_from((u, v, d) for u, v, d in g.edges(data=True) if u in nodes and v in nodes and cites(d))
nx.write_graphml(sub, os.path.join(a.dst, "graph_chunk_entity_relation.graphml"))

ent_ids = {utils.compute_mdhash_id(n, prefix="ent-") for n in nodes}
rel_ids = {i for u, v in sub.edges() for i in utils.make_relation_vdb_ids(u, v)}


def vdb(ns, keep):
    with open(os.path.join(a.src, f"vdb_{ns}.json"), encoding="utf-8") as f:
        db = json.load(f)
    import base64

    import numpy as np

    dim = db["embedding_dim"]
    m = np.frombuffer(base64.b64decode(db["matrix"]), dtype=np.float32).reshape(-1, dim)
    idx = [i for i, d in enumerate(db["data"]) if d["__id__"] in keep]
    out = {
        "embedding_dim": dim,
        "data": [db["data"][i] for i in idx],
        "matrix": base64.b64encode(m[idx].tobytes()).decode(),
    }
    with open(os.path.join(a.dst, f"vdb_{ns}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return len(idx)


counts = {
    "docs": len(docs),
    "chunks": len(chunk_ids),
    "nodes": sub.number_of_nodes(),
    "edges": sub.number_of_edges(),
    "vdb_chunks": vdb("chunks", chunk_ids),
    "vdb_entities": vdb("entities", ent_ids),
    "vdb_relationships": vdb("relationships", rel_ids),
}
print(json.dumps(counts))
