"""Upstream side of the B2 operate parity checks. Runs in the rag-venv (lightrag-hku 1.4.16).

Every command reads a store directory and writes JSON Lines to --out (one
record per doc / chunk); parity/operate.ts produces the same records with the
TS port and diffs them. Nothing under <dir> is written.

  chunk <dir> --out <file> [--model gpt-4o-mini] [--size 1200] [--overlap 100]
        [--split CHAR] [--split-only] [--sample N]
      chunking_by_token_size for every doc in kv_store_full_docs.json:
      {doc, tokens_md5, n_tokens, chunks: [{id, tokens, content}] | error}

  extract <dir> --out <file> [--language English] [--gleaning 1] [--limit N]
      extract_entities for every chunk of kv_store_text_chunks.json, one chunk
      per call, replayed from kv_store_llm_response_cache.json; the LLM is a
      stub that raises, so a cache miss is an error record, never a call:
      {chunk, nodes: [[name, [entity]]], edges: [[[src, tgt], [relation]]],
       llm_cache_list} | {chunk, error}

  parse <dir> --in <file> --out <file>
      _process_extraction_result over {result} lines (a fuzz corpus the TS side
      builds from cached responses): {nodes, edges}
"""

import argparse
import asyncio
import hashlib
import json
import sys

from lightrag.constants import DEFAULT_ENTITY_TYPES, DEFAULT_MAX_EXTRACT_INPUT_TOKENS
from lightrag.operate import chunking_by_token_size, extract_entities
from lightrag.utils import TiktokenTokenizer, compute_mdhash_id


def load_kv(d, name):
    with open(f"{d}/kv_store_{name}.json", encoding="utf-8") as f:
        return json.load(f)


def tokens_md5(tokens):
    return hashlib.md5(",".join(map(str, tokens)).encode()).hexdigest()


def cmd_chunk(a):
    tok = TiktokenTokenizer(a.model)
    docs = load_kv(a.dir, "full_docs")
    ids = sorted(docs)
    if a.sample:
        ids = ids[:: max(1, len(ids) // a.sample)][: a.sample]
    split = a.split.encode().decode("unicode_escape") if a.split else None
    with open(a.out, "w", encoding="utf-8") as out:
        for did in ids:
            content = docs[did]["content"]
            rec = {"doc": did}
            try:
                t = tok.encode(content)
                rec["n_tokens"] = len(t)
                rec["tokens_md5"] = tokens_md5(t)
            except Exception as e:  # special-token text: tiktoken refuses it
                rec["encode_error"] = type(e).__name__
            try:
                chunks = chunking_by_token_size(tok, content, split, a.split_only, a.overlap, a.size)
                rec["chunks"] = [
                    {"id": compute_mdhash_id(c["content"], prefix="chunk-"), "tokens": c["tokens"],
                     "index": c["chunk_order_index"], "content": c["content"]}
                    for c in chunks
                ]
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"docs": len(ids)}))


class CacheMiss(Exception):
    pass


async def no_llm(prompt, system_prompt=None, **kw):
    raise CacheMiss("cache miss: the parity run never calls an LLM")


class MemKV:
    """The slice of BaseKVStorage that the cache path and update_chunk_cache_list touch."""

    def __init__(self, data, global_config=None):
        self.data = data
        self.global_config = global_config or {}

    async def get_by_id(self, id):
        return self.data.get(id)

    async def upsert(self, d):
        self.data.update(d)


def extract_config(a, tok):
    return {
        "llm_model_func": no_llm,
        "entity_extract_max_gleaning": a.gleaning,
        "addon_params": {"language": a.language, "entity_types": DEFAULT_ENTITY_TYPES},
        "tokenizer": tok,
        "max_extract_input_tokens": DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
        "llm_model_max_async": 4,
    }


def cmd_extract(a):
    tok = TiktokenTokenizer(a.model)
    cache = MemKV(load_kv(a.dir, "llm_response_cache"), {"enable_llm_cache_for_entity_extract": True})
    chunks = load_kv(a.dir, "text_chunks")
    ids = sorted(chunks)[: a.limit or None]
    gc = extract_config(a, tok)

    async def run():
        with open(a.out, "w", encoding="utf-8") as out:
            for cid in ids:
                rec = {"chunk": cid}
                # a private copy: update_chunk_cache_list appends to it
                store = MemKV({cid: json.loads(json.dumps(chunks[cid]))})
                try:
                    [(nodes, edges)] = await extract_entities(
                        {cid: chunks[cid]}, gc, llm_response_cache=cache, text_chunks_storage=store
                    )
                    rec["nodes"] = [[k, v] for k, v in nodes.items()]
                    rec["edges"] = [[list(k), v] for k, v in edges.items()]
                    rec["llm_cache_list"] = store.data[cid].get("llm_cache_list")
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    asyncio.run(run())
    print(json.dumps({"chunks": len(ids)}))


def cmd_parse(a):
    from lightrag.operate import _process_extraction_result

    async def run():
        with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
            for line in f:
                case = json.loads(line)
                nodes, edges = await _process_extraction_result(case["result"], "chunk-x", 7, "f.md")
                out.write(json.dumps({"nodes": [[k, v] for k, v in nodes.items()],
                                      "edges": [[list(k), v] for k, v in edges.items()]}, ensure_ascii=False) + "\n")

    asyncio.run(run())
    print("{}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("parse")
    pa.add_argument("dir")
    pa.add_argument("--in", dest="inp", required=True)
    pa.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("dir")
    e.add_argument("--out", required=True)
    e.add_argument("--model", default="gpt-4o-mini")
    e.add_argument("--language", default="English")
    e.add_argument("--gleaning", type=int, default=1)
    e.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("chunk")
    c.add_argument("dir")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default="gpt-4o-mini")
    c.add_argument("--size", type=int, default=1200)
    c.add_argument("--overlap", type=int, default=100)
    c.add_argument("--split")
    c.add_argument("--split-only", action="store_true")
    c.add_argument("--sample", type=int, default=0)
    a = p.parse_args()
    {"chunk": cmd_chunk, "extract": cmd_extract, "parse": cmd_parse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
