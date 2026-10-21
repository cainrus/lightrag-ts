"""Upstream side of the B3 query parity checks. Runs in the rag-venv (lightrag-hku 1.4.16).

  synthetic <out_dir>
      Write a small synthetic store (graph, three vdbs, text chunks; dim 16,
      clock pinned) with upstream's own storage classes, plus
      <out_dir>/probes.json: queries x modes x QueryParam / config variants.
      Nothing private: every string is made up here.

  run <working_dir> --probes FILE --out FILE [--workspace W]
      [--emb FILE] [--fake-dim N | --emb-from-store] [--emb-out FILE]
      Load the store with upstream storages and run each probe through
      operate.kg_query / operate.naive_query exactly as LightRAG.aquery_llm
      calls them. The LLM is a stub driven by the probe (keyword JSON, a hashed
      answer, a long answer, a stream), so nothing leaves the machine. Query
      embeddings come from --emb (text -> vector); a text not in it gets a
      deterministic vector (--fake-dim: md5-seeded, --emb-from-store: a stored
      chunk vector picked by md5), and every text used is written to --emb-out
      so the TS side replays the exact vectors. Output, one JSON line per probe:
      {id, result: {content, is_streaming, streamed, raw_data} | null,
       llm_calls: [...], cache_writes: [...], error}
      Nothing under <working_dir> is written (no index_done_callback).
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time

import numpy as np

from lightrag.kg.shared_storage import initialize_share_data

initialize_share_data(1)

from lightrag import operate, utils  # noqa: E402
from lightrag.base import QueryParam  # noqa: E402
from lightrag.constants import GRAPH_FIELD_SEP  # noqa: E402
from lightrag.kg.json_kv_impl import JsonKVStorage  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.networkx_impl import NetworkXStorage  # noqa: E402

PINNED_TIME = 1760000000
META_FIELDS = {
    "entities": {"entity_name", "source_id", "content", "file_path"},
    "relationships": {"src_id", "tgt_id", "source_id", "content", "file_path"},
    "chunks": {"full_doc_id", "content", "file_path"},
}
# keys of global_config (asdict(LightRAG)) the query path reads, at LightRAG's defaults
BASE_GC = {
    "kg_chunk_pick_method": "VECTOR",
    "related_chunk_number": 5,
    "min_rerank_score": 0.0,
    "rerank_model_func": None,
    "enable_llm_cache": True,
    "enable_llm_cache_for_entity_extract": True,
    "addon_params": {"language": "English"},
    "embedding_batch_num": 10,
    "max_graph_nodes": 1000,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
}


def f16(v):
    return np.asarray(v, dtype=np.float64).astype(np.float16).astype(np.float32)


def fake16(text, dim):
    # mirrored nowhere: the TS side only ever replays vectors written to --emb-out
    h = hashlib.md5(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return f16(rng.standard_normal(dim))


class Emb:
    def __init__(self, table, fallback, dim):
        self.table = {k: f16(v) for k, v in table.items()}
        self.fallback = fallback
        self.dim = dim
        self.used = {}

    def vec(self, text):
        v = self.table.get(text)
        if v is None:
            v = self.fallback(text)
            self.table[text] = v
        self.used[text] = v
        return v

    def func(self):
        async def embed(texts, **kw):
            return np.array([self.vec(t) for t in texts], dtype=np.float32)

        return utils.EmbeddingFunc(embedding_dim=self.dim, func=embed)


def cfg(working_dir, extra=None):
    gc = {"working_dir": working_dir, "tokenizer": utils.TiktokenTokenizer("gpt-4o-mini"), **json.loads(json.dumps({k: v for k, v in BASE_GC.items() if k != "rerank_model_func"}))}
    gc["rerank_model_func"] = None
    gc.update(extra or {})
    return gc


async def open_store(working_dir, workspace, gc, emb):
    g = NetworkXStorage(namespace="chunk_entity_relation", workspace=workspace, global_config=gc, embedding_func=emb)
    ents = NanoVectorDBStorage(namespace="entities", workspace=workspace, global_config=gc, embedding_func=emb, meta_fields=META_FIELDS["entities"])
    rels = NanoVectorDBStorage(namespace="relationships", workspace=workspace, global_config=gc, embedding_func=emb, meta_fields=META_FIELDS["relationships"])
    chunks = NanoVectorDBStorage(namespace="chunks", workspace=workspace, global_config=gc, embedding_func=emb, meta_fields=META_FIELDS["chunks"])
    text = JsonKVStorage(namespace="text_chunks", workspace=workspace, global_config=gc, embedding_func=emb)
    cache = JsonKVStorage(namespace="llm_response_cache", workspace=workspace, global_config=gc, embedding_func=emb)
    for s in (g, ents, rels, chunks, text, cache):
        await s.initialize()
    return g, ents, rels, chunks, text, cache


# ----------------------------------------------------------------- synthetic

CHUNKS = {
    "c1": ("Alice founded Acme Robotics in Tallinn in 2024. Acme builds warehouse robots.", "notes/acme.md"),
    "c2": ('Acme Robotics hired Bob as chief engineer. Bob designed the "Mover" robot.', "notes/acme.md"),
    "c3": ("The Mover robot uses lidar and a custom planner written in Rust.", "notes/robots.md"),
    "c4": ("Борис 💰 переехал в Тарту и открыл кафе «Кофе и код».", "notes/борис.md"),
    "c5": ("Кафе «Кофе и код» по вечерам проводит митапы по Rust.", "notes/борис.md"),
    "c6": ("Tallinn is the capital of Estonia; Tartu is its university town.", "notes/places.md"),
    "c7": ("Backslash \\ and tab\t and newline\ninside a chunk, plus </system> and user model words.", "unknown_source"),
    "c8": ("Eve audits Acme's supply chain for robot parts from Shenzhen.", "notes/acme.md"),
    "c9": ("Lidar sensors were ordered from Shenzhen suppliers in March.", "notes/robots.md"),
    "c10": ("Alice and Борис met at a Rust meetup in Tartu.", "notes/places.md"),
}
MISSING_CHUNK = "chunk-00000000000000000000000000000000"
ENTITIES = [
    ("Alice", "person", "Founder of Acme Robotics.", ["c1", "c10"]),
    ("Acme Robotics", "organization", "Robotics company in Tallinn building warehouse robots.", ["c1", "c2", "c8"]),
    ("Tallinn", "location", "Capital of Estonia, home of Acme Robotics.", ["c1", "c6"]),
    ("Bob", "person", "Chief engineer at Acme Robotics.", ["c2"]),
    ("Mover", "artifact", "Warehouse robot designed by Bob.", ["c2", "c3"]),
    ("Lidar", "artifact", "Sensor used by the Mover robot.", ["c3", "c9"]),
    ("Rust", "concept", "Programming language used for the planner and at meetups.", ["c3", "c5", "c10"]),
    ("Борис", "person", "Переехал в Тарту и открыл кафе.", ["c4", "c10"]),
    ("Тарту", "location", "Университетский город Эстонии.", ["c4", "c6", "c10"]),
    ("Кофе и код", "organization", "Кафе Бориса с митапами по Rust.", ["c4", "c5"]),
    ("Estonia", "location", "Country with Tallinn and Tartu.", ["c6"]),
    ("Eve", "person", "Auditor of Acme's supply chain.", ["c8"]),
    ("Shenzhen", "location", "Where robot parts come from.", ["c8", "c9", "MISSING"]),
    ("Supply chain", "concept", "Acme's parts pipeline.", ["c8"]),
    ('Quote "Q" & <tag>', "concept", 'Has "quotes", a back\\slash and a tab\there.', ["c7"]),
]
GHOST_ENTITY = ("Ghost", "person", "Only in the vector store.", ["c1"])
RELATIONS = [
    ("Alice", "Acme Robotics", 2.0, "founding", "Alice founded Acme Robotics.", ["c1"]),
    ("Acme Robotics", "Tallinn", 1.0, "headquarters", "Acme Robotics is based in Tallinn.", ["c1"]),
    ("Acme Robotics", "Bob", 1.5, "employment", "Acme hired Bob.", ["c2"]),
    ("Bob", "Mover", 1.0, "design", "Bob designed the Mover.", ["c2"]),
    ("Mover", "Lidar", 1.0, "sensor", "The Mover uses lidar.", ["c3"]),
    ("Mover", "Rust", 0.7, "software", "The Mover planner is written in Rust.", ["c3"]),
    ("Борис", "Тарту", 1.0, "переезд", "Борис переехал в Тарту.", ["c4"]),
    ("Борис", "Кофе и код", 2.0, "основание", "Борис открыл кафе.", ["c4"]),
    ("Кофе и код", "Rust", 1.0, "митапы", "В кафе проходят митапы по Rust.", ["c5"]),
    ("Tallinn", "Estonia", 3.0, "capital", "Tallinn is the capital of Estonia.", ["c6"]),
    ("Тарту", "Estonia", 1.0, "geography", "Tartu is in Estonia.", ["c6"]),
    ("Eve", "Acme Robotics", 1.0, "audit", "Eve audits Acme.", ["c8"]),
    ("Eve", "Supply chain", 1.0, "audit", "Eve audits the supply chain.", ["c8"]),
    ("Supply chain", "Shenzhen", 1.0, "sourcing", "Parts come from Shenzhen.", ["c8", "c9"]),
    ("Lidar", "Shenzhen", 1.0, "sourcing", "Lidar sensors come from Shenzhen.", ["c9", "MISSING"]),
    ("Alice", "Борис", 1.0, "meetup", "Alice met Борис at a meetup.", ["c10"]),
    ("Alice", "Rust", None, "meetup", "Alice attends Rust meetups (edge stored without weight).", ["c10"]),
    ('Quote "Q" & <tag>', "Alice", 1.0, "edge case", "Quoted entity relates to Alice.", ["c7"]),
]
GHOST_RELATION = ("Ghost", "Alice", 1.0, "phantom", "Only in the vector store.", ["c1"])


def chunk_id(key):
    return MISSING_CHUNK if key == "MISSING" else utils.compute_mdhash_id(CHUNKS[key][0], prefix="chunk-")


def norm_avg(vectors):
    return f16(np.mean(np.stack(vectors), axis=0))


async def build_synthetic(out):
    os.makedirs(out, exist_ok=True)
    dim = 16
    store_dir = os.path.join(out, "store")
    os.makedirs(store_dir, exist_ok=True)
    emb = Emb({}, lambda t: fake16(t, dim), dim)
    gc = cfg(store_dir)
    g, ents, rels, chunks_vdb, text, _cache = await open_store(store_dir, "", gc, emb.func())

    file_of = {}
    await text.upsert(
        {
            chunk_id(k): {"tokens": len(c), "content": c, "chunk_order_index": i, "full_doc_id": "doc-" + f, "file_path": f, "llm_cache_list": []}
            for i, (k, (c, f)) in enumerate(CHUNKS.items())
        }
    )
    await chunks_vdb.upsert({chunk_id(k): {"content": c, "full_doc_id": "doc-" + f, "file_path": f} for k, (c, f) in CHUNKS.items()})
    for k, (_, f) in CHUNKS.items():
        file_of[k] = f
    file_of["MISSING"] = "notes/missing.md"

    def fp(keys):
        return GRAPH_FIELD_SEP.join(dict.fromkeys(file_of[k] for k in keys))

    for name, typ, desc, ck in ENTITIES:
        await g.upsert_node(
            name,
            {"entity_id": name, "entity_type": typ, "description": desc, "source_id": GRAPH_FIELD_SEP.join(chunk_id(k) for k in ck), "file_path": fp(ck), "created_at": PINNED_TIME - 100},
        )
    for src, tgt, w, kw, desc, ck in RELATIONS:
        data = {"description": desc, "keywords": kw, "source_id": GRAPH_FIELD_SEP.join(chunk_id(k) for k in ck), "file_path": fp(ck), "created_at": PINNED_TIME - 50}
        if w is not None:
            data = {"weight": w, **data}
        await g.upsert_edge(src, tgt, data)
    ent_rows = {}
    for name, typ, desc, ck in ENTITIES + [GHOST_ENTITY]:
        ent_rows[utils.compute_mdhash_id(name, prefix="ent-")] = {
            "entity_name": name,
            "content": f"{name}\n{desc}",
            "source_id": GRAPH_FIELD_SEP.join(chunk_id(k) for k in ck),
            "file_path": fp(ck),
        }
    await ents.upsert(ent_rows)
    rel_rows = {}
    for src, tgt, _w, kw, desc, ck in RELATIONS + [GHOST_RELATION]:
        rel_rows[utils.make_relation_vdb_ids(src, tgt)[0]] = {
            "src_id": src,
            "tgt_id": tgt,
            "content": f"{kw}\t{src}\n{tgt}\n{desc}",
            "source_id": GRAPH_FIELD_SEP.join(chunk_id(k) for k in ck),
            "file_path": fp(ck),
        }
    await rels.upsert(rel_rows)
    for s in (g, ents, rels, chunks_vdb, text):
        await s.index_done_callback()

    # query-side embeddings: averages of stored vectors, so retrieval lands near known records
    ent_vec = {n: emb.vec(f"{n}\n{d}") for n, _t, d, _c in ENTITIES}
    rel_vec = {(s, t): emb.vec(f"{kw}\t{s}\n{t}\n{d}") for s, t, _w, kw, d, _c in RELATIONS}
    chunk_vec = {k: emb.vec(c) for k, (c, _f) in CHUNKS.items()}
    table = {}

    def target(text_, kind, names):
        src = {"entity": ent_vec, "relation": rel_vec, "chunk": chunk_vec}[kind]
        table[text_] = norm_avg([src[n] for n in names]).tolist()

    probes = make_probes(target)
    with open(os.path.join(out, "probes.json"), "w", encoding="utf-8") as f:
        json.dump({"embeddings": table, "probes": probes}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(out, "py-cases.json"), "w", encoding="utf-8") as f:
        json.dump(python_cases(), f, indent=1)  # ascii: a lone surrogate case must survive the file
        f.write("\n")


def python_cases():
    """Python semantics the query port reproduces, recorded from the interpreter itself."""
    import json_repair

    floats = [0.1, 0.5, 1.5, 2.5, 1e-05, 0.0001, 1.5e-07, 1e16, 1e15, 123456789.125, 1e22, 1.7976931348623157e308, 5e-324, -0.0, -2.75, 1 / 3]
    hash_args = [
        ["mix", "что мы решили?", "Multiple Paragraphs", 20, 20, 6000, 8000, 30000, "a, b", "c", "", True],
        ["naive", "q", None, 40, None, 6000, 8000, 30000, "", False],
        # non-integral floats only: JS cannot tell 1e16 (float, str "1e+16") from an int
        ["local", "q", "Bullet Points", 0.5, 1e-05, 1.25e-300, -2.75],
        ["mix", "emoji 💰 and lone \ud800 surrogate", "English"],
    ]
    strings = ["plain", 'q"uote\\back', "tab\tnl\ncr\r\x00\x1f\x7f", "Кириллица «ё» 💰   ", "﻿bom"]
    hl, ll = '"high_level_keywords": ["a", "b"]', '"low_level_keywords": ["c"]'
    repair_inputs = [
        '{"a": [1]}', '```json\n{"a": 1}\n```', 'Here: {"a": 1} done', "garbage", "", "[1, 2]", "{}", "null", '```\n{"b": "x"}\n```',
        # trailing / doubled / missing commas
        '{"high_level_keywords": ["a", "b",], "low_level_keywords": ["c"],}', '{"a": ["x",, "y"]}', '{"a": [1, 2] "b": [3]}',
        '{"a": true, "b": null, "c": 1.5,}', "[1, 2,]", '```json\n{"a": [1,]}\n```',
        # cut off by max_tokens
        "{" + hl + ", " + '"low_level_keywords": ["c"', "{" + '"high_level_keywords": ["a", "b', "{" + hl + ', "low_level_keywords": [',
        "{" + hl + ', "low_level_keywords"', "{" + hl + ', "low_', "{" + hl + ', "low_level_keywords": ', '{"a": ["x"]', "{", "[",
        '{"a": "unterminated', 'Keywords: {"a": ["x", "y"',
        # several values, prose around
        '{"a": 1} {"b": 2}', '{"a": 1}\n{"b": 2}', "{" + hl + ", " + ll + "}\n{" + '"high_level_keywords": ["x", "y"], "low_level_keywords": ["z"]}',
        "{" + hl + ", " + ll + "}\n{" + '"high_level_keywords": ["x"], "low_level_keywords": ["z"]}', '{"a": 1}, {"a": 2}',
        '{"a": ["x", "y"]} trailing prose', '{"a": ["x"]}}', "Sure!\n\n{" + hl + ", " + ll + "}\n\nHope this helps.",
        # quoting
        "{'a': ['x', 'y']}", '{a: ["x"]}', '{"a": "line\nbreak"}', '{"a": "esc \\" \\\\ \\n \\u00e9"}', '  {"a": ["ё", "💰"]}  ',
        '{"a": [x, y]}', '{"a": True, "b": None}', '{"a": 1} // done', '"just a string"', '{"a": -1.5e3, "b": .5}',
    ]
    return {
        "float_repr": [[x, repr(x)] for x in floats],
        "round": [[x, round(x)] for x in [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, 2.4999999999999996, 2.5000000000000004, 4.0, 4.6]],
        "args_hash": [[a, utils.compute_args_hash(*a)] for a in hash_args],
        "json_dumps": [[s, json.dumps(s), json.dumps(s, ensure_ascii=False)] for s in strings]
        + [[{"high_level_keywords": ["Ёж", "x"], "low_level_keywords": []}, json.dumps({"high_level_keywords": ["Ёж", "x"], "low_level_keywords": []}), None]],
        "json_repair": [[s, json_repair.loads(s)] for s in repair_inputs],
        "weighted_polling": [
            [items, mx, mn, utils.pick_by_weighted_polling([{"sorted_chunks": c} for c in items], mx, mn)]
            for items, mx, mn in [
                ([["a", "b", "c"]], 2, 1),
                ([["a", "b", "c"], ["d"], ["e", "f"], ["g", "h", "i"]], 5, 1),
                ([["a1", "a2", "a3", "a4", "a5", "a6"], ["b1", "b2", "b3"], [], ["d1", "d2", "d3", "d4"], ["e1"]], 4, 1),
                ([["x"] * 0, ["y1", "y2"]], 3, 1),
                ([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"], ["i", "j"], ["k", "l"]], 3, 2),
                ([["a", "b", "c"]], -1, 1),
            ]
        ],
    }


def kw(hl, ll):
    return json.dumps({"high_level_keywords": hl, "low_level_keywords": ll}, ensure_ascii=False)


def make_probes(target):
    Q1 = "Who founded Acme Robotics and what does it build?"
    Q1KW = kw(["Company founding", "Robotics industry"], ["Alice", "Acme Robotics"])
    target(Q1, "chunk", ["c1", "c2"])
    target("Alice, Acme Robotics", "entity", ["Alice", "Acme Robotics"])
    target("Company founding, Robotics industry", "relation", [("Alice", "Acme Robotics"), ("Acme Robotics", "Tallinn")])

    Q2 = "Что открыл Борис в Тарту?"
    Q2KW = kw(["Открытие бизнеса", "Переезд"], ["Борис", "Тарту", "Кофе и код"])
    target(Q2, "chunk", ["c4", "c5"])
    target("Борис, Тарту, Кофе и код", "entity", ["Борис", "Тарту", "Кофе и код"])
    target("Открытие бизнеса, Переезд", "relation", [("Борис", "Кофе и код"), ("Борис", "Тарту")])

    Q3 = "Where do lidar parts come from?"
    Q3KW = kw(["Supply chain", "Sourcing"], ["Lidar", "Shenzhen", "Mover"])
    target(Q3, "chunk", ["c9", "c8"])
    target("Lidar, Shenzhen, Mover", "entity", ["Lidar", "Shenzhen", "Mover"])
    target("Supply chain, Sourcing", "relation", [("Lidar", "Shenzhen"), ("Supply chain", "Shenzhen")])

    Q4 = "Rust meetups?"
    target(Q4, "chunk", ["c5", "c10"])
    QLONG = "Please tell me absolutely everything about every single robot that exists anywhere"
    target(QLONG, "chunk", ["c3"])
    QQ = 'What is "Quote" with a back\\slash?'
    target(QQ, "chunk", ["c7"])
    target('Quote "Q" & <tag>, Alice', "entity", ['Quote "Q" & <tag>', "Alice"])
    target("Edge cases", "relation", [('Quote "Q" & <tag>', "Alice")])

    probes = []

    def add(pid, query, mode, variant="full", param=None, gc=None, kwr=Q1KW, answer="hash", system_prompt=None, plain=False):
        param = dict(param or {})
        # a unique user_prompt makes the answer a cache miss, so the LLM call itself is compared
        if variant == "full" and not plain and "user_prompt" not in param:
            param["user_prompt"] = f"[{pid}]"
        probes.append(
            {"id": pid, "query": query, "mode": mode, "variant": variant, "param": param, "gc": gc or {}, "kw_response": kwr, "answer": answer, "system_prompt": system_prompt}
        )

    def fresh(suffix, chunks=("c1", "c2")):
        q = f"{Q1} ({suffix})"
        target(q, "chunk", list(chunks))
        return q

    for mode in ["local", "global", "hybrid", "mix", "naive"]:
        for variant in ["context", "prompt", "full"]:
            add(f"q1-{mode}-{variant}", Q1, mode, variant, plain=True)
    add("q1-mix-repeat-cache-hit", Q1, "mix", plain=True)
    add("q1-naive-repeat-cache-hit", Q1, "naive", plain=True)
    add("q1-mix-both-flags", Q1, "mix", "both")
    for mode in ["local", "global", "hybrid", "mix", "naive"]:
        add(f"q2-{mode}-full", Q2, mode, kwr=Q2KW, plain=True)
        add(f"q3-{mode}-context", Q3, mode, "context", kwr=Q3KW)
    for mode in ["hybrid", "mix", "naive"]:
        add(f"q1-{mode}-topk3-chunk2", Q1, mode, param={"top_k": 3, "chunk_top_k": 2})
        add(f"q1-{mode}-chunk-top-k-none", Q1, mode, param={"chunk_top_k": None})
        add(f"q1-{mode}-user-prompt", Q1, mode, param={"user_prompt": "Answer in one line.", "response_type": "Bullet Points"})
        add(f"q1-{mode}-response-type-none", Q1, mode, param={"response_type": None})
        add(f"q1-{mode}-response-type-empty", Q1, mode, param={"response_type": ""})
        add(f"q1-{mode}-history", Q1, mode, param={"conversation_history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
        add(f"q1-{mode}-no-rerank", Q1, mode, param={"enable_rerank": False})
        add(f"q1-{mode}-rerank", Q1, mode, gc={"rerank": "reverse", "min_rerank_score": 0.5})
        add(f"q1-{mode}-rerank-no-filter", Q1, mode, "context", gc={"rerank": "reverse"})
        add(f"q1-{mode}-rerank-filters-all", Q1, mode, "context", gc={"rerank": "reverse", "min_rerank_score": 0.99})
        add(f"q1-{mode}-long-answer", Q1, mode, answer="long")
        add(f"q1-{mode}-stream", Q1, mode, param={"stream": True}, answer="stream")
        for total in [300, 450, 600, 750, 900] if mode == "naive" else [1300, 1500, 1700, 1900]:
            add(f"q1-{mode}-total-{total}", Q1, mode, "context", param={"max_total_tokens": total})
    for total in [1000, 700, 500]:
        add(f"q1-mix-no-kg-total-{total}", Q1, "mix", "context", param={"max_entity_tokens": 0, "max_relation_tokens": 0, "max_total_tokens": total})
    add("q1-mix-empty-dataset-full", Q1, "mix", param={"max_entity_tokens": 0, "max_relation_tokens": 0, "max_total_tokens": 500})
    for mode in ["local", "global", "hybrid", "mix"]:
        add(f"q1-{mode}-entity-tokens-40", Q1, mode, "context", param={"max_entity_tokens": 40, "max_relation_tokens": 60})
        add(f"q1-{mode}-entity-tokens-0", Q1, mode, "context", param={"max_entity_tokens": 0, "max_relation_tokens": 0})
        add(f"q3-{mode}-weight", Q3, mode, "context", gc={"kg_chunk_pick_method": "WEIGHT", "related_chunk_number": 2}, kwr=Q3KW)
        add(f"q1-{mode}-weight-1", Q1, mode, "context", gc={"kg_chunk_pick_method": "WEIGHT", "related_chunk_number": 1})
        add(f"q1-{mode}-vector-1", Q1, mode, "context", gc={"related_chunk_number": 1})
        add(f"q1-{mode}-unknown-pick", Q1, mode, "context", gc={"kg_chunk_pick_method": "NONE"})
        add(f"q1-{mode}-sys-template", Q1, mode, "context", gc={"system_prompt_template": "TEMPLATE {response_type} {user_prompt} {context_data} " * 40})
        add(f"q1-{mode}-top-k-2", Q1, mode, "context", param={"top_k": 2})
        add(f"qq-{mode}-full", QQ, mode, kwr=kw(["Edge cases"], ['Quote "Q" & <tag>', "Alice"]))
        add(f"q1-{mode}-given-keywords", Q1, mode, "context", param={"hl_keywords": [], "ll_keywords": ["Борис"]})
    add("q1-kg-custom-system", Q1, "mix", system_prompt="CUSTOM {response_type} | {user_prompt} | {context_data}")
    add("q1-naive-custom-system", Q1, "naive", system_prompt="CUSTOM {response_type} | {user_prompt} | {content_data}")
    # keyword edge cases: each on its own query text, so no earlier probe cached its keywords
    add("kw-local-only-hl", fresh("only hl"), "local", "context", kwr=kw(["Company founding"], []))
    add("kw-global-only-ll", fresh("only ll"), "global", "context", kwr=kw([], ["Alice"]))
    add("kw-mix-empty", Q4, "mix", kwr=kw([], []))
    add("kw-local-garbage", Q4, "local", kwr="no keywords here")
    add("kw-mix-empty-long-query", QLONG, "mix", kwr=kw([], []))
    add("kw-hybrid-think-fence", fresh("think", ("c9",)), "hybrid", kwr="<think>hmm</think>\n```json\n" + Q3KW + "\n```")
    add("kw-global-prose-json", fresh("prose", ("c9",)), "global", kwr="Sure! Here they are: " + Q3KW + " Hope it helps.")
    add("kw-local-string-value", fresh("string"), "local", "context", kwr=json.dumps({"high_level_keywords": "Sourcing", "low_level_keywords": ["Lidar"]}))
    add("kw-local-list-json", fresh("list"), "local", kwr="[1, 2]")
    add("kw-mix-only-hl-key", fresh("missing ll key"), "mix", kwr=json.dumps({"high_level_keywords": ["Robotics industry"]}))
    add("kw-naive-ignores-keywords", fresh("naive"), "naive", kwr="[1, 2]")
    add("empty-query", "", "mix")
    add("empty-query-naive", "", "naive")
    return probes


# ----------------------------------------------------------------------- run

STATE = {}


async def stub_llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, enable_cot=False, stream=False, _priority=None, **kw):
    STATE["calls"].append(
        {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "history_messages": history_messages,
            "keyword_extraction": keyword_extraction,
            "enable_cot": enable_cot,
            "stream": stream,
            "_priority": _priority,
        }
    )
    p = STATE["probe"]
    if keyword_extraction:
        return p["kw_response"]
    digest = hashlib.md5(((system_prompt or "") + "\n" + prompt).encode("utf-8")).hexdigest()
    if p["answer"] == "stream":

        async def gen():
            for piece in ["stub ", "stream ", digest]:
                yield piece

        return gen()
    if p["answer"] == "long":
        return (system_prompt or "") + " user said: model answer for " + prompt + " <system>" + digest + "</system>"
    return "STUB " + digest


async def stub_rerank(query, documents, top_n=None):
    n = len(documents)
    out = [{"index": n - 1 - k, "relevance_score": round(0.95 - 0.1 * k, 2)} for k in range(n)]
    return out[:top_n] if top_n else out


def make_param(p):
    q = dict(p["param"])
    q["mode"] = p["mode"]
    if p["variant"] in ("context", "both"):
        q["only_need_context"] = True
    if p["variant"] in ("prompt", "both"):
        q["only_need_prompt"] = True
    return QueryParam(**q)


async def run(a):
    data = json.load(open(a.probes, encoding="utf-8"))
    probes = data["probes"]
    table = data.get("embeddings", {})
    if a.emb:
        table.update(json.load(open(a.emb, encoding="utf-8")))
    working_dir = a.dir
    gc = cfg(working_dir, {"llm_model_func": stub_llm})
    base = dict(gc)
    store_vec = None
    if a.emb_from_store:

        def pick(text):
            ids, vecs = store_vec
            i = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % len(ids)
            return f16(vecs[i])

        fallback = pick
    else:
        fallback = lambda t: fake16(t, a.fake_dim)  # noqa: E731
    dim = a.fake_dim
    if a.emb_from_store:
        with open(os.path.join(working_dir, a.workspace, "vdb_chunks.json"), encoding="utf-8") as f:
            dim = json.load(f)["embedding_dim"]
    emb = Emb(table, fallback, dim)
    g, ents, rels, chunks_vdb, text, cache = await open_store(working_dir, a.workspace, gc, emb.func())
    if a.emb_from_store:
        ids = sorted(chunks_vdb._client._NanoVectorDB__storage["data"][i]["__id__"] for i in range(len(chunks_vdb._client._NanoVectorDB__storage["data"])))
        got = await chunks_vdb.get_vectors_by_ids(ids)
        store_vec = (ids, [got[i] for i in ids])

    writes = []
    orig_upsert = cache.upsert

    async def recording_upsert(d):
        writes.append(json.loads(json.dumps(d)))
        await orig_upsert(d)

    cache.upsert = recording_upsert
    t0 = time.perf_counter()
    with open(a.out, "w", encoding="utf-8") as out:
        for p in probes:
            for k in list(gc):
                if k not in base:
                    del gc[k]
            gc.update(base)
            extra = dict(p["gc"])
            if extra.pop("rerank", None) == "reverse":
                gc["rerank_model_func"] = stub_rerank
            gc.update(extra)
            STATE["probe"] = p
            STATE["calls"] = []
            writes.clear()
            rec = {"id": p["id"], "result": None, "error": None}
            try:
                param = make_param(p)
                if p["mode"] == "naive":
                    r = await operate.naive_query(p["query"].strip(), chunks_vdb, param, gc, hashing_kv=cache, system_prompt=p["system_prompt"])
                else:
                    r = await operate.kg_query(p["query"].strip(), g, ents, rels, text, param, gc, hashing_kv=cache, system_prompt=p["system_prompt"], chunks_vdb=chunks_vdb)
                if r is not None:
                    streamed = None
                    if r.is_streaming:
                        streamed = "".join([c async for c in r.response_iterator])
                    rec["result"] = {"content": r.content, "is_streaming": r.is_streaming, "streamed": streamed, "raw_data": r.raw_data}
            except Exception as e:  # recorded, compared with the TS error class
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["llm_calls"] = STATE["calls"]
            rec["cache_writes"] = list(writes)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.emb_out:
        with open(a.emb_out, "w", encoding="utf-8") as f:
            json.dump({k: [float(x) for x in v] for k, v in emb.used.items()}, f, ensure_ascii=False)
            f.write("\n")
    print(json.dumps({"probes": len(probes), "seconds": round(time.perf_counter() - t0, 1), "embedded_texts": len(emb.used)}), file=sys.stderr)


def main():
    time.time = lambda: float(PINNED_TIME)  # storage timestamps and vdb __created_at__
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("--workspace", default="")
    r.add_argument("--probes", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--emb")
    r.add_argument("--emb-out")
    r.add_argument("--fake-dim", type=int, default=16)
    r.add_argument("--emb-from-store", action="store_true")
    a = ap.parse_args()
    if a.cmd == "synthetic":
        asyncio.run(build_synthetic(a.out))
    else:
        asyncio.run(run(a))


if __name__ == "__main__":
    main()
