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
| `operate.py` query half (`kg_query` .. `naive_query`) | `src/query.ts` | B3 ✓ | keywords + cache, local/global/hybrid/mix/naive, context builder; folds into `operate.ts` after B2 |
| `base.py` QueryParam, QueryResult, QueryContextResult | `src/query/base.ts` | B3 ✓ | folds into `base.ts` after B2 |
| `utils.py` query helpers (`process_chunks_unified`, `pick_by_*`, rerank, references, ...) | `src/query/utils.ts` | B3 ✓ | plus `compute_args_hash` with Python `str()`; folds into `utils.ts` after B2 |
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

## Query (B3): what "compatible" means here

Same query, same store, same QueryParam: the same LLM calls (prompts byte for
byte), the same context / prompt strings, the same `raw_data`, the same cache
entries. Checked three ways:

- `test/query.test.ts`: 136 probes over the five modes on a synthetic store
  (`tools/upstream/query_parity.py synthetic` + `run`): QueryParam and config
  variants, token budgets that cut entities, relations and chunks, WEIGHT /
  VECTOR / unknown chunk pick, rerank with score filter, stream, custom system
  prompts, keyword JSON in fences, prose, lists, empty. Upstream's
  `tests/test_batch_embeddings.py` is ported alongside.
- `parity/query.ts keys <store>` (read-only): every `*:keywords:*` and
  `*:query:*` cache entry of a real store, its key recomputed from the entry's
  query and `queryparam`; for kg modes the keywords are replayed through
  `extract_keywords_only` from the cache, so the whole key path runs.
- `parity/query-context.ts` on a store copy: each cached query in every mode
  through upstream and TS; query vectors are stored chunk vectors (no live
  embedding call), the LLM is a stub or the cached answer.

On the live vault store (21.10.2026: 77 keywords + 49 query entries): every
key recomputed (49/49 through the chained path, 3 of them via upstream's
"empty keywords -> the query is the ll keyword" branch); on its copy, 77
queries x 5 modes x {context, full} = 770 probes, 768 identical. The other 2
are one local-mode query whose entity search has an exact score tie
(0.6364680528640747 on both sides): nano-vectordb ranks with `np.argsort`,
which is not stable, so the order of tied records is numpy's, not reproducible
here (ties keep storage order in TS). With float32 query vectors (what an
embedding binding returns) scores agree to one float32 ulp (6e-8 measured), so
a near-tie can flip the same way.

Query cache keys hash the parameters, not the prompt (`compute_args_hash` over
mode, query, response_type, top_k, ..., joined keywords, user_prompt,
enable_rerank), so `str()` of None / bool / float matters: `computeArgsHashPy`
prints them the Python way.

Contract for B4 (what `LightRAG.aquery*` must hand the query functions):

- `global_config` = `lightragQueryConfig({ llm_model_func, ... })`: tokenizer
  (tiktoken gpt-4o-mini), `kg_chunk_pick_method`, `related_chunk_number`,
  `min_rerank_score` (LightRAG passes 0.0; process_chunks_unified alone falls
  back to 0.5 when the key is missing), `addon_params.language` (part of the
  keywords cache key), `enable_llm_cache`, `rerank_model_func`, each with
  LightRAG's env override.
- `text_chunks_db` = `queryTextChunks(kvStorage, globalConfig, embeddingFunc)`.
  Upstream reads `global_config` and `embedding_func` off the storage object;
  TS storages carry neither. Without an embedding function VECTOR chunk
  picking falls back to WEIGHT (upstream does the same); the TS side also
  emits one process warning (`LIGHTRAG_TS_VECTOR_PICK_WITHOUT_EMBEDDING`).
- `hashing_kv` = the llm_response_cache storage with that same config.
- `src/query/base.ts` loads `.env` from the working directory on import
  (`process.loadEnvFile`, existing variables win), as base.py's
  `load_dotenv(override=False)` does before the QueryParam defaults read
  TOP_K, CHUNK_TOP_K, MAX_*_TOKENS, HISTORY_TURNS, RERANK_BY_DEFAULT. Node's
  .env syntax is close to python-dotenv's, not identical (multi-line values,
  `${VAR}` expansion).

`json_repair.loads` (upstream reads keyword answers with it) is a lenient
reader here, checked against json_repair 0.61.6 on 44 recorded inputs: valid
JSON, prose around the JSON, fenced blocks, trailing / doubled / missing
commas, output cut off mid-string / mid-array / after a key (closed, a bare key
dropped, `"key":` at the end gives `""`), several values (a list, or the
newest when shaped like the previous one, json_repair's update rule),
single-quoted strings, unquoted keys and words, True/False, comments. Where
upstream crashes on the result (a list, e.g. two differently shaped objects:
AttributeError on `.get`), TS raises the same AttributeError. Text with no
JSON gives `""` and a warning through `log`. Not ported: json_repair's
heuristics for unescaped quotes inside strings, parenthesised tuples and
schema-guided repair.

Deliberate differences:

- `extract_keywords_only` with no cache storage skips the save (upstream raises
  AttributeError); `llm_model_func` is looked up when first called, not at the
  start of `kg_query` (upstream raises KeyError even for `only_need_context`).
- `pick_by_vector_similarity` collects candidates in first-seen order; upstream
  iterates a Python set (hash-seeded per process). Only ties in similarity and
  the error fallback depend on it.
- `cosine_similarity` runs in double precision; upstream divides by a float32
  norm of the query, a factor common to every score of one call.
- `_apply_token_truncation` does not format `created_at` (upstream formats it
  with `time.localtime`, then drops the field before use).
- An integral float has no JS form: `json.dumps(1.0)` is `"1.0"` in Python,
  `"1"` here (graph weights such as 1.0 in `raw_data`, a rerank score of 1.0
  in a chunk's truncation key). Values compare equal; strings differ.
- The `bypass` mode and the `aquery*` dispatch live in `lightrag.py` (B4).

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
