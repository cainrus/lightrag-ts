"""Upstream side of the B1 storage parity checks. Runs in the rag-venv (lightrag-hku 1.4.16).

Every command reads a store directory and prints one JSON document on stdout;
parity/storages.ts runs the same operations with the TS storages and diffs.
Commands that write only ever write under the --out directory.

  graph-dump <dir>                      adjacency order, attrs, labels, degrees
  graph-rewrite <dir> --out <dir>       NetworkXStorage load + index_done_callback
  kv-dump <dir>                         every kv_store_*.json through JsonKVStorage / JsonDocStatusStorage
  vdb-dump <dir> <ns> <dim>             NanoVectorDBStorage: ids, meta, matrix digest, sample queries
  vdb-rewrite <dir> <ns> <dim> --out    load + save (index_done_callback) with upstream code
  graph-kg <dir> <label> <depth> <max>  get_knowledge_graph node/edge sets
  mutate <dir> <ops.json> <dim>         apply ops.json with upstream storages, persist (clock pinned)
  missing-rel-vectors <dir> <dim>       edges whose relationship vector is absent, as upstream sees it
"""

import asyncio
import hashlib
import time
import json
import os
import shutil
import sys

import numpy as np

from lightrag.kg.shared_storage import initialize_share_data
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus

initialize_share_data(1)

META_FIELDS = {
    "entities": {"entity_name", "source_id", "content", "file_path"},
    "relationships": {"src_id", "tgt_id", "source_id", "content", "file_path"},
    "chunks": {"full_doc_id", "content", "file_path"},
}


def cfg(d, **kw):
    return {
        "working_dir": d,
        "embedding_batch_num": 32,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": kw.get("threshold", 0.2)},
        "max_graph_nodes": kw.get("max_graph_nodes", 1000),
    }


def fake_vector(text, dim):
    # mirrored by fakeVector() in parity/storages.ts: deterministic, float64
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return [(int(h[(2 * i) % 32 : (2 * i) % 32 + 2], 16) - 127.5) / 127.5 + (i % 7) * 0.01 for i in range(dim)]


def emb(dim):
    async def f(texts, **kw):
        return np.array([fake_vector(t, dim) for t in texts], dtype=np.float64)

    return EmbeddingFunc(embedding_dim=dim, func=f)


async def graph_store(d):
    s = NetworkXStorage(namespace="chunk_entity_relation", workspace="", global_config=cfg(d), embedding_func=None)
    await s.initialize()
    return s


async def graph_dump(d):
    s = await graph_store(d)
    g = s._graph
    return {
        "nodes": list(g.nodes),
        "adj": {n: list(g.adj[n]) for n in g.nodes},
        "node_attrs": {n: g.nodes[n] for n in g.nodes},
        "edges": [[u, v, dd] for u, v, dd in g.edges(data=True)],
        "degrees": {n: g.degree(n) for n in g.nodes},
        "labels": await s.get_all_labels(),
        "popular": await s.get_popular_labels(50),
    }


async def graph_rewrite(d, out):
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(d, "graph_chunk_entity_relation.graphml"), out)
    s = await graph_store(out)
    await s.index_done_callback()
    return {"nodes": s._graph.number_of_nodes(), "edges": s._graph.number_of_edges()}


async def graph_kg(d, label, depth, max_nodes):
    s = await graph_store(d)
    kg = await s.get_knowledge_graph(label, int(depth), int(max_nodes))
    return {
        "nodes": sorted(n.id for n in kg.nodes),
        "edges": sorted(e.id for e in kg.edges),
        "is_truncated": kg.is_truncated,
        "node_props": {n.id: n.properties for n in kg.nodes},
    }


async def kv_dump(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("kv_store_") and f.endswith(".json")):
            continue
        ns = f[len("kv_store_") : -len(".json")]
        cls = JsonDocStatusStorage if ns == "doc_status" else JsonKVStorage
        s = cls(namespace=ns, workspace="", global_config=cfg(d), embedding_func=None)
        await s.initialize()
        keys = list(s._data.keys())
        rec = {
            "count": len(keys),
            "keys_sha": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "content_sha": hashlib.sha256(
                json.dumps(dict(s._data), ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "get_by_ids_first3": await s.get_by_ids(keys[:3] + ["missing-id"]),
            "filter_keys": sorted(await s.filter_keys(set(keys[:2] + ["missing-a", "missing-b"]))),
        }
        if ns == "doc_status":
            rec["status_counts"] = await s.get_all_status_counts()
            try:
                docs, total = await s.get_docs_paginated(None, 1, 10, "updated_at", "desc")
                rec["paginated"] = [[k, v.status] for k, v in docs]
                rec["paginated_total"] = total
            except Exception as e:
                rec["paginated"] = {"error": type(e).__name__}
            by_status = await s.get_docs_by_statuses(list(DocStatus))
            rec["by_statuses"] = sorted(by_status.keys())
        out[ns] = rec
    return out


async def vdb_store(d, ns, dim, threshold=0.2):
    s = NanoVectorDBStorage(
        namespace=ns,
        workspace="",
        global_config=cfg(d, threshold=threshold),
        embedding_func=emb(int(dim)),
        meta_fields=META_FIELDS.get(ns, set()),
    )
    await s.initialize()
    return s


async def vdb_dump(d, ns, dim, threshold=0.2):
    s = await vdb_store(d, ns, dim, float(threshold))
    st = await s.client_storage
    data = st["data"]
    m = st["matrix"]
    ids = [x["__id__"] for x in data]
    queries = []
    # deterministic probe vectors: a stored row, a mix of two rows, a fixed pseudo-random vector
    rng = np.random.default_rng(1234)
    probes = []
    if len(data):
        probes.append(m[0].astype(np.float64).tolist())
        probes.append((m[0] + m[len(data) // 2]).astype(np.float64).tolist())
    probes.append(rng.standard_normal(int(dim)).tolist())
    for q in probes:
        r = await s.query("", top_k=10, query_embedding=q)
        queries.append([[x["id"], float(x["distance"])] for x in r])
    return {
        "count": len(data),
        "ids_sha": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "meta_sha": hashlib.sha256(
            json.dumps([{k: v for k, v in x.items()} for x in data], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "matrix_shape": list(m.shape),
        "matrix_sha": hashlib.sha256(np.ascontiguousarray(m, dtype=np.float32).tobytes()).hexdigest(),
        "probes": probes,
        "queries": queries,
        "get_by_ids": [
            None if r is None else {k: v for k, v in r.items() if k != "vector"}
            for r in await s.get_by_ids(ids[:2] + ["missing-id"])
        ],
    }


async def vdb_rewrite(d, ns, dim, out):
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(d, f"vdb_{ns}.json"), out)
    s = await vdb_store(out, ns, dim)
    ok = await s.index_done_callback()
    return {"saved": ok}


async def mutate(d, ops_file, dim):
    ops = json.load(open(ops_file, encoding="utf-8"))
    clock = float(ops["clock"])
    time.time = lambda: clock
    for ns, o in ops.get("kv", {}).items():
        s = JsonKVStorage(namespace=ns, workspace="", global_config=cfg(d), embedding_func=None)
        await s.initialize()
        await s.upsert(o.get("upsert", {}))
        await s.delete(o.get("delete", []))
        await s.index_done_callback()
    if "doc_status" in ops:
        o = ops["doc_status"]
        s = JsonDocStatusStorage(namespace="doc_status", workspace="", global_config=cfg(d), embedding_func=None)
        await s.initialize()
        await s.upsert(o.get("upsert", {}))
        await s.delete(o.get("delete", []))
        await s.index_done_callback()
    if "graph" in ops:
        s = await graph_store(d)
        for op in ops["graph"]:
            name, args = op[0], op[1:]
            if name == "remove_edges":
                args = [[tuple(e) for e in args[0]]]
            await getattr(s, name)(*args)
        await s.index_done_callback()
    for ns, o in ops.get("vdb", {}).items():
        s = await vdb_store(d, ns, dim)
        await s.upsert(o.get("upsert", {}))
        await s.delete(o.get("delete", []))
        for name in o.get("delete_entity", []):
            await s.delete_entity(name)
        for name in o.get("delete_entity_relation", []):
            await s.delete_entity_relation(name)
        await s.index_done_callback()
    return {"ok": True}


async def missing_rel_vectors(d, dim):
    from lightrag.utils import make_relation_vdb_ids

    g = (await graph_store(d))._graph
    s = await vdb_store(d, "relationships", dim)
    # get_by_ids() is a membership filter over client.get(), a linear scan per call;
    # on a 100k-edge store that is hours, so test membership against the same ids directly
    stored = {dp["__id__"] for dp in (await s.client_storage)["data"]}
    missing = []
    total = 0
    for u, v in g.edges():
        total += 1
        if not any(i in stored for i in make_relation_vdb_ids(u, v)):
            missing.append([u, v])
    return {"edges": total, "missing": len(missing), "missing_sha": hashlib.sha256(json.dumps(missing, ensure_ascii=False).encode()).hexdigest(), "first": missing[:3]}


def main():
    a = sys.argv[1:]
    out = None
    if "--out" in a:
        i = a.index("--out")
        out = a[i + 1]
        a = a[:i] + a[i + 2 :]
    cmd, args = a[0], a[1:]
    fn = {
        "graph-dump": lambda: graph_dump(*args),
        "graph-rewrite": lambda: graph_rewrite(args[0], out),
        "graph-kg": lambda: graph_kg(*args),
        "kv-dump": lambda: kv_dump(*args),
        "vdb-dump": lambda: vdb_dump(*args),
        "vdb-rewrite": lambda: vdb_rewrite(args[0], args[1], args[2], out),
        "mutate": lambda: mutate(args[0], args[1], int(args[2])),
        "missing-rel-vectors": lambda: missing_rel_vectors(*args),
    }[cmd]
    res = asyncio.run(fn())
    json.dump(res, sys.stdout, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
