"""Upstream side of the B4 pipeline parity: a scenario run through upstream's LightRAG class.

    <venv>/bin/python -I tools/upstream/pipeline_parity.py <spec.json> --out <records.jsonl>

The spec names a working dir (already seeded by the caller), the LightRAG
config, the stand-ins for the LLM and the embedding model, and a list of steps
(insert, delete, query, edits, ...); parity/pipeline.ts runs the same spec
through the TS LightRAG and compares every record and every file.

To make one upstream run reproducible, and comparable, this harness changes
upstream's process, not its code:

- clocks: time.time() and datetime.now() are FIXED_NOW in the lightrag
  modules (created_at, cache and vdb stamps, doc-status times);
- sets iterate in insertion order: `set` in the lightrag modules is replaced
  by an insertion-ordered set. Upstream iterates sets of strings in several
  places (list(set) into full_entities, ", ".join(set(keywords)) in the
  rebuild, set-ordered deletes, ...), whose order follows the per-process
  string hash seed; insertion order is one of the orders a run can produce,
  and the one TS uses;
- one task at a time: max_parallel_insert=1 and operate's semaphores at 1, so
  merge and rebuild writes land in task order (see SCOPE.md, B2/B4).
"""

import argparse
import asyncio
import builtins
import collections.abc
import dataclasses
import datetime as _dt
import hashlib
import json
import sys
import time as _time
import types

FIXED_NOW = 1760000000
_CLOCKED = {
    "lightrag.lightrag",
    "lightrag.operate",
    "lightrag.utils",
    "lightrag.utils_graph",
    "lightrag.kg.json_kv_impl",
    "lightrag.kg.json_doc_status_impl",
    "lightrag.kg.nano_vector_db_impl",
}
_ORDERED_SET_MODULES = [
    "lightrag.lightrag",
    "lightrag.operate",
    "lightrag.utils",
    "lightrag.utils_graph",
    "lightrag.kg.json_kv_impl",
    "lightrag.kg.json_doc_status_impl",
]


class OrderedSet(collections.abc.MutableSet):
    """A set that iterates in insertion order (a dict underneath)."""

    def __init__(self, iterable=()):
        self._d = dict.fromkeys(iterable)

    def __contains__(self, x):
        return x in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def add(self, x):
        self._d[x] = None

    def discard(self, x):
        self._d.pop(x, None)

    def update(self, *its):
        for it in its:
            for x in it:
                self._d[x] = None

    def copy(self):
        return OrderedSet(self._d)

    def union(self, *its):
        out = self.copy()
        out.update(*its)
        return out

    def difference(self, *its):
        drop = builtins.set().union(*its)
        return OrderedSet(x for x in self._d if x not in drop)

    def intersection(self, *its):
        keep = builtins.set(self._d).intersection(*its)
        return OrderedSet(x for x in self._d if x in keep)

    def issubset(self, other):
        return all(x in other for x in self._d)

    def issuperset(self, other):
        return all(x in self._d for x in other)

    def symmetric_difference(self, other):
        other = OrderedSet(other)
        return OrderedSet([x for x in self._d if x not in other] + [x for x in other if x not in self._d])

    def difference_update(self, *its):
        for it in its:
            for x in it:
                self._d.pop(x, None)

    def intersection_update(self, *its):
        keep = builtins.set(self._d).intersection(*its)
        self._d = dict.fromkeys(x for x in self._d if x in keep)

    def symmetric_difference_update(self, other):
        self._d = self.symmetric_difference(other)._d

    def clear(self):
        self._d.clear()

    def pop(self):
        if not self._d:
            raise KeyError("pop from an empty set")
        k = next(iter(self._d))
        del self._d[k]
        return k

    def remove(self, x):
        del self._d[x]

    def __repr__(self):
        return "{" + ", ".join(map(repr, self._d)) + "}" if self._d else "set()"

    __hash__ = None


def patch_process(sequential=True):
    real_time = _time.time

    def fake_time():
        return float(FIXED_NOW) if sys._getframe(1).f_globals.get("__name__") in _CLOCKED else real_time()

    _time.time = fake_time

    class FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime.fromtimestamp(FIXED_NOW, tz)

    import importlib

    for name in _ORDERED_SET_MODULES:
        importlib.import_module(name).set = OrderedSet
    import lightrag.lightrag as lr

    lr.datetime = FixedDatetime
    if sequential:
        from lightrag import operate

        shim = types.ModuleType("asyncio_sequential")
        shim.__dict__.update(asyncio.__dict__)
        shim.Semaphore = lambda value=1: asyncio.Semaphore(1)

        async def ordered_wait(fs, *args, **kwargs):
            # extract_entities collects chunk results from asyncio.wait's `done`
            # set, whose order follows task ids; task creation order instead
            fs = list(fs)
            done, pending = await asyncio.wait(fs, *args, **kwargs)
            return [t for t in fs if t in done], [t for t in fs if t in pending]

        shim.wait = ordered_wait
        operate.asyncio = shim


# ----------------------------------------------------------------- stand-ins


def md5_hex(s):
    return hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()


def md5_bytes(s):
    return hashlib.md5(s.encode("utf-8", errors="replace")).digest()


def stub_vector(text, dim):
    """B2's stub (md5 bytes scaled to [-1, 1)), extended past 16 dims with md5 of f"{i}:{text}"."""
    raw = b"".join(md5_bytes(text if i == 0 else f"{i}:{text}") for i in range((dim + 15) // 16))
    return [(raw[j] - 128) / 128.0 for j in range(dim)]


SUMMARY_PROMPT_HEAD = "---Role---\nYou are a Knowledge Graph Specialist, proficient in data curation and synthesis."
VOCAB = ["Alice Moreau", "Bob Tanaka", "Carol Diaz", "Acme Corp", "Globex", "Paris", "Berlin", "Quantum Engine", "Project Atlas", "Red River", "Old Library", "Night Train"]
TYPES = ["person", "organization", "location", "artifact", "concept"]
KWS = ["collaboration", "conflict", "travel", "ownership", "research", "funding"]


class CacheMiss(Exception):
    pass


def synth_extraction(prompt, glean):
    d = md5_bytes(prompt)
    h = md5_hex(prompt)
    n = 1 if glean else 2 + d[0] % 3
    names = []
    for i in range(6):
        nm = VOCAB[d[1 + i] % len(VOCAB)]
        if nm not in names:
            names.append(nm)
        if len(names) == n:
            break
    lines = []
    for i, nm in enumerate(names):
        t = TYPES[d[7 + i] % len(TYPES)]
        extra = " with more detail" if glean else ""
        lines.append(f"entity<|#|>{nm}<|#|>{t}<|#|>{nm} is mentioned in passage {h[i * 4:i * 4 + 8]}{extra}.")
    for a, b in zip(names, names[1:]):
        lines.append(f"relation<|#|>{a}<|#|>{b}<|#|>{KWS[d[12] % len(KWS)]}, {KWS[d[13] % len(KWS)]}<|#|>{a} and {b} are linked in passage {h[:8]}.")
    lines.append("<|COMPLETE|>")
    return "\n".join(lines)


def synth_keywords(prompt):
    d = md5_bytes(prompt)
    return json.dumps({"high_level_keywords": [KWS[d[0] % 6], KWS[d[1] % 6]], "low_level_keywords": [VOCAB[d[2] % 12], VOCAB[d[3] % 12]]})


def make_llm(kind):
    """kind "synthetic": answers every prompt; "cache": a store replay, where extraction and keywords must be cache hits."""

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, stream=False, **kw):
        if prompt.startswith(SUMMARY_PROMPT_HEAD):
            return "Stub summary " + md5_hex(prompt)
        if keyword_extraction:
            # a store's keyword entries exist for the modes and params it was asked with; the rest get stand-ins
            return synth_keywords(prompt)
        if system_prompt and "<|#|>" in system_prompt:
            if kind == "synthetic":
                return synth_extraction(prompt, bool(history_messages))
            raise CacheMiss("cache miss: the parity run never calls an LLM")
        return "STUB " + md5_hex((system_prompt or "") + "\n" + prompt)

    return llm


# ----------------------------------------------------------------- run


def jsonable(v):
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return jsonable(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items() if k != "response_iterator"}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if hasattr(v, "value") and isinstance(getattr(v, "value"), str):  # str enums
        return v.value
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


async def run(spec, out_path):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.base import QueryParam
    from lightrag.kg.shared_storage import get_namespace_data
    from lightrag.utils import EmbeddingFunc, logger

    logger.setLevel("ERROR")
    dim = spec["dim"]

    async def embed(texts, **kw):
        return np.array([stub_vector(t, dim) for t in texts], dtype=np.float32)

    config = dict(spec.get("config", {}))
    config["max_parallel_insert"] = 1
    rag = LightRAG(
        working_dir=spec["working_dir"],
        workspace=spec.get("workspace", ""),
        embedding_func=EmbeddingFunc(embedding_dim=dim, func=embed),
        llm_model_func=make_llm(spec["llm"]),
        **config,
    )
    await rag.initialize_storages()
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, step in enumerate(spec["steps"]):
            op = step["op"]
            rec = {"i": i, "op": op, "result": None, "error": None}
            del status["history_messages"][:]
            try:
                if op == "insert":
                    r = await rag.ainsert(step["docs"], step.get("split_by_character"), step.get("split_by_character_only", False), step.get("ids"), step.get("file_paths"), step["track_id"])
                elif op == "enqueue":
                    r = await rag.apipeline_enqueue_documents(step["docs"], step.get("ids"), step.get("file_paths"), step["track_id"])
                elif op == "process":
                    r = await rag.apipeline_process_enqueue_documents(step.get("split_by_character"), step.get("split_by_character_only", False))
                elif op == "error_docs":
                    r = await rag.apipeline_enqueue_error_documents(step["files"], step["track_id"])
                elif op == "custom_kg":
                    r = await rag.ainsert_custom_kg(step["kg"], step.get("full_doc_id"))
                elif op == "delete":
                    r = await rag.adelete_by_doc_id(step["doc_id"], step.get("delete_llm_cache", False))
                elif op in ("query", "query_data", "aquery"):
                    param = QueryParam(**step.get("param", {}))
                    if op == "query":
                        r = await rag.aquery_llm(step["query"], param, step.get("system_prompt"))
                    elif op == "aquery":
                        r = await rag.aquery(step["query"], param, step.get("system_prompt"))
                    else:
                        r = await rag.aquery_data(step["query"], param)
                elif op == "delete_entity":
                    r = await rag.adelete_by_entity(step["name"])
                elif op == "delete_relation":
                    r = await rag.adelete_by_relation(step["src"], step["tgt"])
                elif op == "edit_entity":
                    r = await rag.aedit_entity(step["name"], step["data"], step.get("allow_rename", True), step.get("allow_merge", False))
                elif op == "edit_relation":
                    r = await rag.aedit_relation(step["src"], step["tgt"], step["data"])
                elif op == "create_entity":
                    r = await rag.acreate_entity(step["name"], step["data"])
                elif op == "create_relation":
                    r = await rag.acreate_relation(step["src"], step["tgt"], step["data"])
                elif op == "merge_entities":
                    r = await rag.amerge_entities(step["sources"], step["target"], step.get("strategy"), step.get("target_data"))
                elif op == "entity_info":
                    r = await rag.get_entity_info(step["name"], step.get("include_vector_data", False))
                elif op == "relation_info":
                    r = await rag.get_relation_info(step["src"], step["tgt"], step.get("include_vector_data", False))
                elif op == "counts":
                    r = await rag.get_processing_status()
                else:
                    raise ValueError(f"unknown op {op}")
                rec["result"] = jsonable(r)
            except Exception as e:  # noqa: BLE001 - the record carries it
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["history"] = list(status["history_messages"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    await rag.finalize_storages()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrent", action="store_true", help="keep upstream's own concurrency (no sequential shim)")
    a = p.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    patch_process(sequential=not a.concurrent)
    asyncio.run(run(spec, a.out))


if __name__ == "__main__":
    sys.exit(main())
