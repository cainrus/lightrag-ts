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
| `utils.py` | `src/utils.ts` | B0, B2 partial | hashing, cache keys, sanitize, LLM cache path, tokenizer, extraction parsing, source-id/merge helpers, Python-compat helpers (`pyStr`, `pyFormat`, `pyJsonDumps` with ensure_ascii, `computeArgsHash` over Python `str()`), shared by ingest and query |
| `operate.py` | `src/operate.ts` | B2 ✓ ingest half, B4 ✓ rebuild | chunking, extraction, merge and summaries; `rebuild_knowledge_from_chunks` (delete path, B4); query in `src/query.ts` (B3) |
| `exceptions.py` | `src/exceptions.ts` | B2 | the classes the ported code raises |
| `operate.py` query half (`kg_query` .. `naive_query`) | `src/query.ts` | B3 ✓ | keywords + cache, local/global/hybrid/mix/naive, context builder; its own module beside `operate.ts` (the ingest half), which would otherwise pass 2500 lines |
| `utils.py` helpers only the query uses (`process_chunks_unified`, `pick_by_*`, rerank, references, `json_repair.loads`) | `src/query/utils.ts` | B3 ✓ | |
| `kg/json_kv_impl.py` | `src/kg/json_kv_impl.ts` | B1 ✓ | byte-identical `write_json`, atomic |
| `kg/json_doc_status_impl.py` | `src/kg/json_doc_status_impl.ts` | B1 ✓ | file_path sort: `lower()`, not pypinyin |
| `kg/nano_vector_db_impl.py` + nano-vectordb 0.0.4.3 | `src/kg/nano_vector_db_impl.ts` | B1 ✓ | numpy-exact normalize, byte-identical save, streamed |
| `kg/networkx_impl.py` + networkx 3.6 GraphML IO | `src/kg/networkx_impl.ts` | B1 ✓ | Map-based nx.Graph (`NxGraph`), graphology export |
| `base.py` | `src/base.ts` | B1 ✓ storage contracts, B3 ✓ QueryParam and results | |
| `lightrag.py` | `src/lightrag.ts` | B4 ✓ | the `LightRAG` class: config, storages, pipeline and doc status, delete with rebuild, queries, editing entry points; async API only |
| `utils_graph.py` | `src/utils_graph.ts` | B4 ✓ | delete / edit / create / merge entities and relations |
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
- Not in B2: `rebuild_knowledge_from_chunks` and the rest of the delete path,
  `utils_graph` (both B4, see below), query (B3).

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

## Query (B3): what "compatible" means here

Same query, same store, same QueryParam: the same LLM calls (prompts byte for
byte), the same context / prompt strings, the same `raw_data`, the same cache
entries. Checked three ways:

- `test/query.test.ts`: 136 probes over the five modes on a synthetic store
  (`tools/upstream/query_parity.py synthetic` + `run`): QueryParam and config
  variants, token budgets that cut entities, relations and chunks, WEIGHT /
  VECTOR / unknown chunk pick, rerank with score filter, stream, custom system
  prompts, keyword JSON in fences, prose, lists, empty. Upstream's
  `tests/test_batch_embeddings.py` runs unchanged through the pytest bridge
  (`npm run test:upstream`; the adapter passes mocks and numpy arrays).
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
enable_rerank), so `str()` of None / bool / float matters: `computeArgsHash` prints
them the Python way (strings, all the extract and summary keys hash, pass
unchanged).

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
- `src/base.ts` loads `.env` from the working directory on import
  (`process.loadEnvFile`, existing variables win), as base.py's
  `load_dotenv(override=False)` does before the QueryParam defaults read
  TOP_K, CHUNK_TOP_K, MAX_*_TOKENS, HISTORY_TURNS, RERANK_BY_DEFAULT. Node's
  .env syntax is close to python-dotenv's, not identical (multi-line values,
  `${VAR}` expansion).
- **Side effect, kept for parity:** importing the library writes the .env
  variables into `process.env` of the whole process, as `import lightrag` does
  in Python - unexpected for an npm package, but those variables change query
  cache keys. `LIGHTRAG_TS_NO_DOTENV=1` turns it off.

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
- The `bypass` mode and the `aquery*` dispatch live in `lightrag.py`: ported in B4.

## Pipeline (B4): what "compatible" means here

`src/lightrag.ts` is the `LightRAG` class over the four default storages:
the dataclass fields with upstream's defaults (class-level `os.getenv` defaults
read at import, `default_factory` ones per instance), `__post_init__`
(tokenizer, wrapped LLM and embedding functions, storages),
`initialize_storages` / `finalize_storages`, the document pipeline
(`ainsert`, `apipeline_enqueue_documents`, `apipeline_process_enqueue_documents`,
`apipeline_enqueue_error_documents`, `ainsert_custom_chunks`, `ainsert_custom_kg`),
`adelete_by_doc_id` with `rebuild_knowledge_from_chunks` (in `src/operate.ts`),
`aquery` / `aquery_llm` / `aquery_data` (bypass mode, unknown mode), the
migration helpers and the editing entry points over `src/utils_graph.ts`.

`parity/pipeline.ts` runs one scenario (a list of steps: inserts, deletes,
queries, edits) through upstream's `LightRAG` (`tools/upstream/pipeline_parity.py`)
and through the TS class, from the same starting files, with the same stand-ins
(an LLM stub, a hash embedding, a fixed clock); each step's return value and its
pipeline messages are compared, then every file both stores end with, byte for
byte. TS runs twice: one task at a time (`llm_model_max_async=1`,
`max_parallel_insert=1`), which must equal upstream file for file and message
for message, and at the spec's own concurrency, which must give the same files
(the LLM cache compared as a map: its new keys follow completion order) and the
same results. In that second run the LLM and embedding stand-ins answer after
a seeded 0-4 ms delay (md5 of the input), so concurrent calls finish out of
call order as a provider's latency makes them; `test/lightrag.test.ts` checks
that completions really were reordered.

To get one reproducible upstream run the harness changes upstream's process,
not its code, and the port follows the same order:

- **Clocks**: `time.time()` and `datetime.now()` are fixed in the lightrag
  modules (doc-status times, `created_at`, cache and vdb stamps). TS takes the
  clock as the port-only `now` option.
- **Sets iterate in insertion order.** Upstream iterates Python sets of
  strings in several places: `list(set)` into full_entities / full_relations,
  `", ".join(set(keywords))` and `max(set(types), key=count)` in the rebuild,
  `_merge_attributes`' keys and `join_unique` values, the ids it deletes. Their
  order follows the per-process string hash seed, so two upstream runs differ.
  The widest of them is `filter_keys` (`set(keys) - set(self._data)` in the
  doc-status storage, called by `apipeline_enqueue_documents`, lightrag.py
  ~1451/1501): it decides the order of the documents of one `ainsert`, and
  with it the order their merges append descriptions (the input of every
  summary, hence its cache key), which source ids KEEP / FIFO cut, and which
  file paths survive `max_file_paths`. So upstream's own result for a
  multi-document insert depends on PYTHONHASHSEED; the harness pins it to
  the caller's order, which is what TS does.
  The harness replaces `set` in the lightrag modules by an insertion-ordered set;
  TS uses insertion order everywhere. With that, full_entities / full_relations
  match byte for byte too (B2 compared them as sets).
- **`asyncio.wait`'s `done` set**: extract_entities returns the chunk results
  in the order of a set of tasks (by task id, i.e. memory address), and the
  merge consumes them in that order. The harness hands them back in task order;
  TS returns them in chunk order. Upstream's own run, without this, is not
  reproducible (`node parity/pipeline.ts upstream-concurrent`, 21.10.2026):
  three runs of the synthetic scenario with operate's own concurrency (clock
  and set order still pinned) each differ from the sequential run in 8 of 12
  files (GraphML, both entity / relation vdb files, the entity / relation
  chunk lists, full_entities / full_relations, the LLM cache) and in 31-32 of
  76 step results; 10-12 of the 16 nodes and 4-5 of the 49 edges carry other
  attributes (descriptions, summaries). The three runs also differ from each
  other (two pairs in 8 files, one pair in 1).
- **One task at a time** upstream (`max_parallel_insert=1`, operate's
  semaphores at 1), as for B2.

Deliberate differences:

- **Documents in flight do not interleave.** Upstream processes
  `max_parallel_insert` documents at once and lets their chunk writes and their
  merges (entity by entity, under keyed locks) land in scheduling order. TS
  chunks and extracts concurrently too, but a document's stage-1 writes (doc
  status, chunk vectors, chunk records) wait for the previous document's, and
  its merge waits for the previous document's merge: the files equal upstream's
  with one document at a time, whatever the parallelism.
- **The rebuild commits in input order**: the reads and LLM summaries of
  `rebuild_knowledge_from_chunks` run `llm_model_max_async * 2` at once, each
  rebuild's writes (chunk list, node, vectors) apply in order, entities first.
  A failing rebuild is reported and skipped, as upstream does. A rebuilt
  relation whose two endpoints are both missing creates them in the order of
  the set display `{src, tgt}` upstream (operate.py:1510), i.e. string-hash
  order, which varies per process; TS creates `src` first. The harness pins
  upstream to that order: `pipeline_parity.py` recompiles
  `_rebuild_single_relationship` with an ordered pair (a set display is not
  reached by the `set` replacement above).
- **Tracebacks**: where upstream appends `traceback.format_exc()` to the
  pipeline history, TS appends the JS stack (the comparisons skip both).
- **Duplicate records** quote the original's status as upstream's f-string
  prints it: `DocStatus.PROCESSED` when this process wrote it (a DocStatus member
  in memory), `processed` when it was read from the file. TS tracks which.
- Only the async API: `insert`, `query`, `delete_by_doc_id`, ... (event-loop
  wrappers) are not ported. `aexport_data` (pandas / Excel export) is not
  ported. Storages are the four defaults; another storage name is an error.
- The LLM function gets upstream's keyword arguments as one options object
  (`hashing_kv` and `llm_model_kwargs` merged in, as `functools.partial` does);
  `priority_limit_async_func_call`'s timeouts and queue limits are not ported
  (B2).
- `LightRAG.adelete_by_entity`, `adelete_by_relation`, `acreate_entity` and
  `acreate_relation` pass no chunk-tracking stores to utils_graph, as upstream;
  `aedit_*` and `amerge_entities` do.
- utils_graph takes `withGraphLock(graph)` (the lock every graph writer of the
  port takes) instead of upstream's keyed locks per entity name.
- `extract_entities` with no chunks raises ValueError `Set of Tasks/Futures is
  empty.` (what `asyncio.wait([])` raises upstream): an empty document ends
  FAILED with that message on both sides.

Upstream's `compute_incremental_chunk_ids` docstring example disagrees with its
code (the code keeps a kept chunk in place); the port follows the code.

Ported but not compared with upstream yet: `check_and_migrate_data` (seeding
chunk tracking and full_entities / full_relations for a store from before
they existed) - none of the checked stores needs it.

The upstream test bridge (d11) now runs all of `test_description_api_validation.py`
against TS, the six utils_graph tests included: functions a test monkeypatches
on the Python module (`utils_graph._edit_entity_impl`, `get_entity_info`) go to
TS as hooks (`utilsGraphHooks`), called back where the Python code looks them up.

Numbers:

- **Synthetic scenario** (`node parity/pipeline.ts synthetic`, public, also
  `test/lightrag.test.ts` against upstream's digests in
  `test/fixtures/pipeline-synthetic.json`): 76 steps - 14 documents in 5
  inserts (fresh, a duplicate, custom ids, a blank document, error records),
  queries in all five modes plus bypass, an unknown mode, `aquery_data` and
  `aquery` (three rounds), 4 deletions (with and without their cache, one
  missing) that rebuild shared entities and relations, the editing API
  (edit, rename, merge by rename, create, relation edit / delete, entity
  delete), a custom KG, custom chunks, a re-insert, track-id and id lookups.
  Every result and message equal, all 12
  files byte-identical, TS one task at a time and at its default concurrency.
- **Rebuild without cached extractions** (`node parity/pipeline.ts nocache`,
  public, fixtures `test/fixtures/pipeline-nocache-*.json` checked by
  `test/lightrag.test.ts`): entities a custom KG created (its chunk has no
  `llm_cache_list`) and documents then extended, the documents deleted until
  only the custom chunk is left; and `enable_llm_cache_for_entity_extract=false`,
  where no chunk records its extraction. Both reach upstream's "No LLM cache
  IDs found" and "No cached extraction results found, cannot rebuild"
  branches (1 and 2 rebuilds); 12 and 9 steps, every result and message
  equal, all 12 / 11 files byte-identical, TS sequential and at default
  concurrency.
- **Store replay** (`node parity/pipeline.ts store <copy> ...`, a copy of the
  live personal-vault store, 21.10.2026): 300 processed documents (in
  doc-status order, the store's own ids) in 15 inserts of 20, a duplicate
  re-insert, 4 cached queries x 5 modes x {context, full}, 30 deletions (every
  other one with its LLM cache) and 8 more queries: 95 steps, every result and
  message equal; all 12 files byte-identical (GraphML 4.4 MB, LLM cache 12.9 MB,
  both vdb files), TS sequential and at default concurrency. The run read 651
  of the store's cache entries (570 extraction, 47 summaries, 4 keywords); 67
  summaries and 12 keyword sets went to the stand-ins. The deletions rebuilt
  166 entities and 23 relations from cached extractions. Upstream 95 s, TS
  60 s (each including a per-document save of every store). 60 documents in 6
  inserts with 8 deletions: the same, 52 steps.
- **Deletions on the full store copy** (`parity/pipeline.ts store-delete`):
  4 processed documents deleted from
  a copy of the whole live store (4082 processed documents, GraphML 107 MB,
  vdb files 0.8 + 0.9 GB at dim 1536, LLM cache 333 MB), two of them with their
  cache entries: 42 entities and 60 relations deleted, 21 entities rebuilt
  (their descriptions from the store's cache or the summary stand-in); all 12
  files byte-identical, TS sequential and default. Upstream 153 s, TS 32 s.
- **Queries through LightRAG**: B3's probes, run through `aquery_llm` instead of
  kg_query / naive_query, build the same records: on the synthetic store
  (`test/lightrag.test.ts`) 132 of 136 probes (the other 4 set
  `system_prompt_template`, which is not a LightRAG field) and on a copy of the live store
  (`parity/query-context.ts run ... --via lightrag`) 120 of 120: 12 cached
  queries x 5 modes x {context, full}, 24 per mode, query vectors taken from
  stored chunk vectors as in B3; the same 120 through kg_query directly also
  match. In the store replay above the 48 aquery_llm queries also equal
  upstream's LightRAG, bypass and the unknown mode in the synthetic scenario.

## Not ported

- `kg/` backends other than the defaults: postgres, neo4j, mongo, milvus,
  qdrant, redis, faiss, memgraph, opensearch.
- `kg/shared_storage.py` multiprocess locking: Node runs one process; an
  in-process lock replaces it. Two processes on one store stay unsupported, as
  with upstream's default storages without its server.
- `LightRAG`'s sync wrappers (`insert`, `query`, ...) and `aexport_data`
  (pandas / Excel export).
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
