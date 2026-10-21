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
| `kg/json_kv_impl.py` | `src/kg/json_kv_impl.ts` | B0 reader, B1 storage | |
| `kg/json_doc_status_impl.py` | `src/kg/json_doc_status_impl.ts` | B1 | |
| `kg/nano_vector_db_impl.py` + nano-vectordb 0.0.4.3 | `src/kg/nano_vector_db_impl.ts` | B0 reader, B1 storage | float16/zlib `vector` field and float32 `matrix` |
| `kg/networkx_impl.py` | `src/kg/networkx_impl.ts` | B0 reader, B1 storage | graphology in memory, GraphML IO compatible with networkx 3.x |
| `base.py` | `src/base.ts` | B1 | |
| `lightrag.py` | `src/lightrag.ts` | B4 | pipeline, doc status, insert/delete |
| `utils_graph.py` | `src/utils_graph.ts` | B4 | |
| `llm/openai.py` (subset) | `src/llm/` | B5 | OpenAI-compatible completion + embeddings |

Port-only modules: `src/internal/json_stream.ts` (streaming JSON: vdb files
outgrow V8's max string length), `src/internal/html5_entities.ts` (CPython's
`html.unescape` tables), `src/store.ts` (read-only store view).

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
