# Scope

Port target: `lightrag-hku` **1.4.16** (`lightrag.__version__`, api 0291).

The port is "drop-in" at the **store** level: a working dir written by Python
LightRAG with the default storages must load in lightrag-ts, and vice versa,
and the same input must produce the same LLM calls (hence the same cache keys).

## Ported (or to be ported), file by file

| upstream | here | step | notes |
|---|---|---|---|
| `prompt.py` | `src/prompt.ts` | B0 ✓ | generated verbatim, `tools/upstream/gen.ts` |
| `constants.py` | `src/constants.ts` | B0 ✓ | generated |
| `namespace.py` | `src/namespace.ts` | B0 ✓ | |
| `types.py` | `src/types.ts` | B0 ✓ | pydantic models -> types |
| `utils.py` | `src/utils.ts` | B0 partial | hashing, cache keys, sanitize, Python-compat helpers; rest in B2/B3 |
| `operate.py` | `src/operate.ts` | B0 partial | extraction prompt building only; extraction/merge B2, query B3 |
| `kg/json_kv_impl.py` | `src/kg/json_kv_impl.ts` | B1 ✓ | byte-identical `write_json`, atomic |
| `kg/json_doc_status_impl.py` | `src/kg/json_doc_status_impl.ts` | B1 ✓ | file_path sort: `lower()`, not pypinyin |
| `kg/nano_vector_db_impl.py` + nano-vectordb 0.0.4.3 | `src/kg/nano_vector_db_impl.ts` | B1 ✓ | numpy-exact normalize, byte-identical save, streamed |
| `kg/networkx_impl.py` + networkx 3.6 GraphML IO | `src/kg/networkx_impl.ts` | B1 ✓ | Map-based nx.Graph (`NxGraph`), graphology export |
| `base.py` | `src/base.ts` | B1 ✓ storage contracts | QueryParam and results in B3 |
| `lightrag.py` | `src/lightrag.ts` | B4 | pipeline, doc status, insert/delete |
| `utils_graph.py` | `src/utils_graph.ts` | B4 | |
| `llm/openai.py` (subset) | `src/llm/` | B5 | OpenAI-compatible completion + embeddings |

Port-only modules: `src/internal/json_stream.ts` (streaming JSON: vdb files
outgrow V8's max string length), `src/internal/py_json.ts` (`json.dump`
byte format, atomic temp+rename writes), `src/internal/numpy.ts` (numpy's
pairwise float32 sum, so normalized matrices match bit for bit),
`src/internal/html5_entities.ts` (CPython's `html.unescape` tables),
`src/store.ts` (read-only store view).

## Storages (B1): what "compatible" means here

Checked by `parity/storages.ts` against upstream on a store copy: every file
the TS storages write is byte-identical to what upstream writes for the same
state (KV, GraphML, vdb including the zlib `vector` field), after a plain
rewrite and after the same sequence of upserts/deletes on both sides.

Deliberate differences:

- **Writes are atomic** (temp file + fsync + rename). Upstream writes in place;
  a crash mid-write leaves a truncated file.
- **No cross-process flags.** `shared_storage` (namespace locks, update flags
  for multi-worker servers) is not ported: one process owns a store. KV and
  doc-status keep a dirty flag; `mark_dirty()` is port-only.
- **Graph core is not graphology.** graphology keeps neighbours in plain
  objects, so integer-like ids ("10", "2024") iterate first; networkx keeps
  insertion order, and that order reaches `get_node_edges`, `Graph.edges()`
  and the GraphML file. `NxGraph` mirrors nx.Graph, including the MultiGraph
  -> Graph rebuild `read_graphml` does (neighbour order after loading is not
  file order). `NxGraph.toGraphology()` exports for graphology tooling.
- **int vs float.** JS numbers do not carry the distinction. Graph attributes
  typed `double` in the file, and edge `weight` always, are written as Python
  floats; JSON values that are integral are written as ints (no KV value in the
  real store is an integral float).
- **`get_knowledge_graph` order.** Upstream iterates `graph.subgraph()`, whose
  order follows a Python set (varies per run); here it is graph order. Node and
  edge sets match.
- **Query scores** agree with numpy to ~1e-15 (BLAS summation order differs);
  ids and order match on the checked probes.
- `get_docs_paginated(sort_field="file_path")` sorts by `lower()`; upstream
  uses pypinyin when installed (CJK paths order differently).

## Not ported

- `kg/` backends other than the defaults: postgres, neo4j, mongo, milvus,
  qdrant, redis, faiss, memgraph, opensearch.
- `kg/shared_storage.py` multiprocess locking: Node runs one process; an
  in-process lock replaces it. Two processes on one store stay unsupported, as
  with upstream's default storages without its server.
- `api/` (FastAPI server, web UI, Ollama emulation), `evaluation/`, `tools/`,
  `rerank.py`, LLM bindings other than OpenAI-compatible.
- Deprecated code under `kg/deprecated/`.

## Known parity hazards

- **Tokenizer**: chunking uses tiktoken token counts; the TS side needs a
  byte-identical tiktoken encoder (B2). Chunk ids are `md5(content)`, so any
  boundary difference changes every downstream id.
- **Python semantics**: `str.format`, `str.strip`, `html.unescape`, `json.dumps`
  separators, code-point string ordering, `str()` of lists/floats in
  `compute_args_hash` (query cache keys hash the query params, B3).
- **Summary cache keys** depend on the order descriptions are merged (B2).

## Tracking upstream

1. The pinned version is the one in `src/prompt.ts` (`PROMPTS_VERSION`).
2. On an upstream release: install it into a venv, rerun
   `tools/upstream/dump.py` + `gen.ts`, and run the test suite - generated
   prompts or constants that changed show up as a diff.
3. Diff upstream `operate.py`, `utils.py`, `lightrag.py` and the four default
   storages between the two tags (`git diff v1.4.16..vX -- lightrag/<file>`);
   each changed function maps to its same-named TS function.
4. Regenerate parity fixtures (`tools/upstream/make_fixtures.py`) and rerun
   `parity/replay-extract.ts` on a real store.
