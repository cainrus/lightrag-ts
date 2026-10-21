"""Upstream side of the B2 operate parity checks. Runs in the rag-venv (lightrag-hku 1.4.16).

Every command reads a store directory and writes JSON Lines to --out (one
record per doc / chunk); parity/operate.ts produces the same records with the
TS port and diffs them. Nothing under <dir> is written.

  chunk <dir> --out <file> [--model gpt-4o-mini] [--size 1200] [--overlap 100]
        [--split CHAR] [--split-only] [--sample N]
      chunking_by_token_size for every doc in kv_store_full_docs.json:
      {doc, tokens_md5, n_tokens, chunks: [{id, tokens, content}] | error}
"""

import argparse
import hashlib
import json
import sys

from lightrag.operate import chunking_by_token_size
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


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
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
    {"chunk": cmd_chunk}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
