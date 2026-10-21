"""Generate committed parity fixtures by running UPSTREAM code on synthetic input.

    <venv>/bin/python -I tools/upstream/make_fixtures.py test/fixtures

Writes:
  upstream-cases.json  - sanitize/hash outputs of upstream utils for tricky strings,
                         and the exact LLM-cache entries upstream extract_entities
                         writes for synthetic chunks (fake LLM, fake KV; no API calls)
  synthetic-store/     - a tiny working dir in the default storage formats:
                         graphml via networkx, vdb via nano-vectordb + LightRAG's
                         float16/zlib vector field, kv via upstream write_json
All inputs are synthetic: nothing from a private store goes into the repo.
"""

import asyncio
import base64
import json
import os
import sys
import zlib

import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

import lightrag
from lightrag import operate, utils
from lightrag.prompt import PROMPTS

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

TRICKY = [
    "plain ascii",
    "  leading and trailing spaces \n",
    "　ideographic space and \x1c file separator\x1f",
    "﻿BOM is not whitespace in Python",
    "emoji 💰⏱📉 and 👑💅 and 🔥, 😏, 😇",
    "html &amp; &lt;tag&gt; &quot;q&quot; &#39; &#x1F600; &#128512;",
    "loose refs &amp &notit; &notin; &ampx &#0; &#x80; &#xD800; &#1114112; &#99999999999999999999;",
    "control \x00\x01\x07\x08 kept \t\n\r removed \x0b\x0c\x7f",
    "lone surrogate \ud800 here and \udfff there",
    "nonchars ￾ ￿",
    "Кириллица: «Ёжик в тумане» — тест",
    "braces {not_a_field} {{double}}",
]


def sanitize_cases():
    out = []
    for s in TRICKY:
        out.append(
            {
                "input": s,
                "sanitized": utils.sanitize_text_for_encoding(s),
                "args_hash": utils.compute_args_hash(s),
                "strip": s.strip(),
            }
        )
    return out


class FakeKV:
    def __init__(self, global_config):
        self.global_config = global_config
        self.data = {}

    async def get_by_id(self, k):
        return self.data.get(k)

    async def get_by_ids(self, ks):
        return [self.data.get(k) for k in ks]

    async def upsert(self, d):
        self.data.update(d)


class FakeTokenizer:
    def encode(self, s):
        return list(range(len(s)))


CHUNKS = {
    "chunk-synthetic-1": {
        "content": "Alice founded Acme Robotics in Tallinn in 2024. Acme builds warehouse robots.",
        "file_path": "notes/acme.md",
    },
    "chunk-synthetic-2": {
        "content": "  Борис 💰 переехал в Тарту &amp; открыл кафе «Кофе {и} код».\x07\n",
        "file_path": "notes/борис.md",
    },
    "chunk-synthetic-3": {
        "content": "Edge case: &#x1F600; html refs, 　 wide spaces \ud800 and a {brace}.",
        "file_path": "notes/edge.md",
    },
}


async def extraction_cache(language, entity_types):
    calls = []

    async def fake_llm(prompt, system_prompt=None, history_messages=None, **kw):
        n = len(calls)
        calls.append(prompt)
        # deterministic answer that varies per call so gleaning history is non-trivial
        d = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        return (
            f"entity{d}Thing {n}{d}concept{d}Synthetic entity number {n} 💡.\n"
            f"{PROMPTS['DEFAULT_COMPLETION_DELIMITER']}"
        )

    cfg = {
        "llm_model_func": fake_llm,
        "entity_extract_max_gleaning": 1,
        "addon_params": {"language": language, "entity_types": entity_types},
        "tokenizer": FakeTokenizer(),
        "max_extract_input_tokens": 10**9,
        "enable_llm_cache": True,
        "enable_llm_cache_for_entity_extract": True,
    }
    kv = FakeKV(cfg)
    await operate.extract_entities(CHUNKS, cfg, llm_response_cache=kv)
    return kv.data


async def main():
    cases = {
        "lightrag_version": lightrag.__version__,
        "sanitize": sanitize_cases(),
        "chunks": CHUNKS,
        "extraction": [
            {
                "language": "English",
                "entity_types": None,
                "cache": await extraction_cache("English", PROMPTS and lightrag.constants.DEFAULT_ENTITY_TYPES),
            },
            {
                "language": "Russian",
                "entity_types": ["person", "place", "cafe"],
                "cache": await extraction_cache("Russian", ["person", "place", "cafe"]),
            },
        ],
    }
    with open(os.path.join(OUT, "upstream-cases.json"), "w", encoding="utf-8") as f:
        # ensure_ascii=True keeps lone surrogates representable in the file
        json.dump(cases, f, ensure_ascii=True, indent=1)
    write_store(os.path.join(OUT, "synthetic-store"))


def write_store(d):
    os.makedirs(d, exist_ok=True)
    g = nx.Graph()
    g.add_node("Alice", entity_id="Alice", entity_type="person", description="Founder <of> Acme & co", source_id="chunk-synthetic-1", file_path="notes/acme.md", created_at=1779035540, truncate="")
    g.add_node("Acme Robotics", entity_id="Acme Robotics", entity_type="organization", description="Builds\nrobots", source_id="chunk-synthetic-1", file_path="notes/acme.md", created_at=1779035541, truncate="")
    g.add_node("Тарту", entity_id="Тарту", entity_type="location", description="Город 💰", source_id="chunk-synthetic-2", file_path="notes/борис.md", created_at=1779035542, truncate="")
    g.add_edge("Alice", "Acme Robotics", weight=2.5, description="founded", keywords="founder,company", source_id="chunk-synthetic-1", file_path="notes/acme.md", created_at=1779035543, truncate="")
    g.add_edge("Тарту", "Alice", weight=1.0, description="visited", keywords="travel", source_id="chunk-synthetic-2", file_path="notes/борис.md", created_at=1779035544, truncate="")
    nx.write_graphml(g, os.path.join(d, "graph_chunk_entity_relation.graphml"))

    kv = {
        "full_docs": {"doc-1": {"content": "Alice founded Acme.", "file_path": "notes/acme.md"}},
        "text_chunks": {k: {**v, "tokens": 10, "chunk_order_index": 0, "full_doc_id": "doc-1"} for k, v in CHUNKS.items()},
        "doc_status": {"doc-1": {"status": "processed", "chunks_count": 3, "file_path": "notes/acme.md"}},
        "llm_response_cache": {},
    }
    for ns, data in kv.items():
        utils.write_json(data, os.path.join(d, f"kv_store_{ns}.json"))

    rng = np.random.default_rng(7)
    dim = 8
    for ns, ids in {"entities": ["ent-a", "ent-b", "ent-c"], "relationships": ["rel-a"], "chunks": list(CHUNKS)}.items():
        path = os.path.join(d, f"vdb_{ns}.json")
        if os.path.exists(path):
            os.remove(path)
        db = NanoVectorDB(dim, storage_file=path)
        rows = []
        for i, _id in enumerate(ids):
            v = rng.standard_normal(dim).astype(np.float32)
            # same encoding as lightrag/kg/nano_vector_db_impl.py:131-135
            enc = base64.b64encode(zlib.compress(v.astype(np.float16).tobytes())).decode("utf-8")
            rows.append({"__id__": _id, "__vector__": v, "vector": enc, "content": f"{ns} {i} ✓", "__created_at__": 1779035540 + i})
        db.upsert(rows)
        db.save()


asyncio.run(main())
