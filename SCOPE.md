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
| `utils.py` | `src/utils.ts` | B0, B2 partial | hashing, cache keys, sanitize, LLM cache path, tokenizer, extraction parsing, source-id/merge helpers, Python-compat helpers; query helpers in B3 |
| `operate.py` | `src/operate.ts` | B2 ✓ ingest half | chunking, extraction, merge and summaries; `rebuild_knowledge_from_chunks` (delete path) B4, query B3 |
| `exceptions.py` | `src/exceptions.ts` | B2 | the classes the ported code raises |
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

The `vector` field of a record TS embeds itself is deflated with pako (a port
of vanilla zlib) since B2. Node's own `zlib` is Chromium's fork, whose match
finder picks different matches: on the live store it reproduced CPython's
`zlib.compress` output for 104 of 8944 real embeddings, pako for all 8944.
B1's checks did not see it because a rewrite carries the stored string over.
The match holds against CPython linked with vanilla zlib (the rag-venv's
1.2.12); a CPython built on zlib-ng deflates differently, and TS would differ
from it the same way Node's zlib does.

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

## Operate (B2): what "compatible" means here

`parity/operate.ts` runs each piece against upstream's own code
(`tools/upstream/operate_parity.py` in the rag-venv) on a copy of a real store:

- **chunk**: token ids, chunk ids, token counts and content for every document,
  plus sampled window/split variants; the stored `kv_store_text_chunks` is
  reproduced for every document that has chunks.
- **extract**: replayed from the store's LLM cache with an LLM that raises:
  zero misses on the chunks that have entries, parsed entities/relations,
  timestamps and `llm_cache_list` equal for every chunk; 20000 fuzzed answers
  parse the same.
- **merge**: both sides merge whole documents (each chunk extracted from the
  cache, in chunk order) into fresh storages, one document after another, with
  a fixed clock, a summary stand-in for cache misses and a hash embedding.
  Every file they write is byte-identical - GraphML, entity/relation chunk
  lists, LLM cache, both vdb files - and so is every pipeline status line per
  document. `full_entities` / `full_relations` are compared as sets (below).
  A summary prompt that hits the store's own cache is one TS rebuilt byte for
  byte from what upstream sent months earlier.
- `test/merge.test.ts` does the same on public input (parsed prompt examples)
  under three configs: defaults; KEEP with tiny source-id and file-path caps;
  FIFO with tiny caps and summary sizes that force map-reduce.
- `npm run test:upstream` runs upstream's own pytest files against TS (d11,
  see README).

Deliberate differences:

- **Merge writes land in input order.** Upstream runs the per-entity and
  per-relation merges as concurrent asyncio tasks (8 at a time); the order in
  which their writes reach the graph and vdb files follows the event loop and,
  in production, LLM and embedding latency. TS runs the LLM work concurrently
  but commits each merge's writes in input order, so its result is upstream's
  with those tasks run one at a time - which is what the checks compare against
  (`--sequential` on the upstream side). Upstream's own concurrent run differs
  from its sequential one in node/edge order; attributes can differ too when
  two relations create the same missing entity (the first to commit sets its
  description). Measured below.
- **Concurrent merges into one graph run one after another**: every merge
  writer holds `withGraphLock(graph)` for the whole call, so documents merged
  at the same time are merged in arrival order. Upstream holds a keyed lock per
  entity / relation name, so such documents interleave name by name; TS's
  result equals merging them in call order.
- **Vector writes are applied at the end of each phase** (entities, then
  relations): the queued contents are embedded in batches of
  `embedding_batch_num`, at most `embedding_func_max_async` (8) calls at once,
  then the upserts and deletes replay in order; the vdb ends up as after
  upstream's one-record-at-a-time writes. Consequences: `__created_at__` is
  the clock at the flush, seconds after the record's merge; the embedding
  service sees fewer, larger calls than upstream's one record per call (a
  provider whose vectors depend on the batch would differ slightly). A vdb
  object without `embedding_func` gets plain `upsert(data)` calls instead.
- **Failure mid-merge**: when entity or relation k fails, the merges before it
  still commit (graph, chunk lists) and their vectors are written; nothing of k
  is, and the next phase does not start. Merges after k that are in flight stop
  at their next LLM call and are not reported - the status gets one
  `Error processing ...` line, for k. Upstream keeps whatever its tasks
  finished, logs every task that raised, and records a failing merge's chunk
  list before its LLM call. If the embedding call fails, the phase's graph and
  chunk-list writes are already in place without vectors; a retry of the
  document then meets its own descriptions as existing ones and may summarize
  more than a clean run (upstream: the same for its finished tasks).
- **LLM concurrency**: the merge calls `llm_model_func` with `_priority: 8`.
  `defaultMergeConfig` wraps the function in `priorityLimitAsyncFuncCall`
  (`llm_model_max_async`, lower `_priority` first) as `LightRAG.__post_init__`
  does; upstream's per-call timeouts, queue limit and health checks are not
  ported. A hand-built `MergeConfig` must pass an already limited function.
  Cancellation is also checked between the LLM calls of a map-reduce summary.
- `extract_entities` returns results in chunk order; upstream collects them
  from an asyncio `done` set (undefined order).
- `full_entities` / `full_relations` list names and pairs in insertion order;
  upstream writes `list(set)`, whose order depends on the per-process string
  hash seed. The members are the same.
- With concurrent summaries the LLM cache gains its new keys in completion
  order (upstream too); one at a time they come in upstream's order.
- No Python logging: the functions take an optional `log(level, msg)` with
  upstream's messages (the test adapter forwards it to `lightrag.operate.logger`).
- A failing relation's error prefix is the Python repr of its sorted key pair;
  `pyReprStr` covers quotes and common escapes, not every non-printable
  character.
- `pipeline_status_lock` and shared-storage's cross-process locks are not
  ported: one process; within it `withGraphLock` replaces the keyed locks.
- Not in B2: `rebuild_knowledge_from_chunks` and the rest of the delete path
  (B4), `utils_graph` (B4), query (B3).

Numbers (a copy of the live personal-vault store, 21.10.2026; upstream
lightrag-hku 1.4.16 on CPython 3.12.1):

- Full replay: all 4082 processed documents (4967 chunks), in doc-status order,
  onto an empty store. 4044 merged, 38 skipped by both sides (chunks with no
  extract entry in the cache), 3389 merges needed an LLM summary. All eight
  files byte-identical: GraphML 78.6 MB, vdb_entities 33.5 MB and
  vdb_relationships 46.6 MB (dim 8), entity/relation chunk lists 12.5/20.2 MB,
  LLM cache 341 MB with 2053 stand-in summaries; 62962 status lines equal.
  Upstream 1686 s, TS 516 s, both one merge at a time.
- Summary cache entries: of the replay's 3389 summary calls, 1336 were
  answered by distinct entries of the store's 5334 (TS rebuilt the prompt
  upstream sent months earlier, byte for byte) and 2053 went to the stand-in.
  Every stand-in prompt names an entity or relation the store has summaries
  for (2053/2053): upstream summarized it too, at a different graph state. The
  replay's document order is not the historical one (two documents in flight,
  re-ingests, failed documents), and once a stand-in text replaces a real
  summary in the graph, every later summary of that name misses as well.
- First 500 documents: TS at its default concurrency equals upstream run
  sequentially, file for file (the cache compared as a map). Upstream's own
  concurrent run against its sequential one: GraphML bytes differ (order),
  0 of 6923 nodes and 0 of 7934 edges differ in attributes. Of the 99 LLM
  summaries these documents needed, 93 came from the store's own cache
  entries and 6 from the stand-in.
- Onto the store's own graph (107 MB GraphML, its chunk lists and document
  indexes): 200 processed documents merged a second time - existing nodes
  with real `<SEP>`-joined descriptions, full chunk lists, FIFO caps -
  138 LLM merges, all files byte-identical, TS sequential and concurrent;
  and the one not-processed document whose chunks are all stored.
- KEEP with caps 3/2 and two file paths, 500 documents: byte-identical (the
  caps keep every list under the summary threshold; test/merge.test.ts covers
  KEEP with summaries).
- Extraction re-checked on the 21.10 snapshot (18640 cache entries): no miss
  on the 6590 chunks with entries, 6628/6628 chunks equal.

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

- **Tokenizer**: chunking uses tiktoken token counts, so chunk boundaries (and
  every `md5(content)` id downstream) depend on a byte-identical encoder:
  js-tiktoken, pinned, checked on every document of a real store (B2).
- **Python semantics**: `str.format`, `str.strip`, `html.unescape`, `json.dumps`
  separators, code-point string ordering, `str()` of lists/floats in
  `compute_args_hash` (query cache keys hash the query params, B3).
- **Summary cache keys** depend on the order descriptions are merged, i.e. on
  the merge order: deterministic here, scheduling-dependent upstream (B2).
- **Python's `sum()`** of floats (edge weights) uses Neumaier compensation
  since 3.12; `pySumFloats` reproduces it, a plain `+` can differ in the last bit.

## Tracking upstream

1. The pinned version is the one in `src/prompt.ts` (`PROMPTS_VERSION`).
2. On an upstream release: install it into a venv, rerun
   `tools/upstream/dump.py` + `gen.ts`, and run the test suite - generated
   prompts or constants that changed show up as a diff.
3. Diff upstream `operate.py`, `utils.py`, `lightrag.py` and the four default
   storages between the two tags (`git diff v1.4.16..vX -- lightrag/<file>`);
   each changed function maps to its same-named TS function.
4. Regenerate parity fixtures (`tools/upstream/make_fixtures.py`,
   `parity/operate.ts --phase fixture,merge-fixture`) and rerun
   `parity/replay-extract.ts` and `parity/operate.ts` on a real store.
5. Bump the tag in `tools/upstream/upstream-tests.json`, add upstream's new
   test files to it and run `npm run test:upstream`; a new test in a listed
   file stops the run until it is marked `ts` or skipped with a reason.
