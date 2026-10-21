# lightrag-ts

TypeScript port of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) **v1.4.16**
(MIT). LightRAG is a graph-based retrieval-augmented generation library: it
extracts entities and relations from documents with an LLM, keeps them in a
knowledge graph plus vector indexes, and answers queries over both.

Goal: a drop-in TypeScript implementation that reads and writes the **same
on-disk store** as the Python library (default storages: JSON KV, NanoVectorDB,
NetworkX GraphML), so an existing store moves between the two without
re-ingesting anything.

> Status: early. Done: the four default storages (read and write, byte-compatible
> with upstream), prompt templates, LLM-cache key computation, and the ingest
> half of `operate.py`: chunking, entity/relation extraction, merge and
> summaries, byte-identical to upstream on a real store; the five query modes
> (local, global, hybrid, mix, naive) with keyword extraction and context
> building; the `LightRAG` class with its document pipeline, deletion with
> rebuild, queries and the entity/relation editing API; OpenAI-compatible LLM
> and embedding functions, an LLM function over the `claude-complete` CLI
> bridge, and a command line. See [SCOPE.md](SCOPE.md).

## What works today

- Reading a full Python-written store, including vector files larger than V8's
  maximum string length (streaming JSON reader; a 2.1 GB store with 62.7k
  entities and 105.6k relations loads in ~6 s, peak RSS ~0.8 GB).
- Byte-exact reconstruction of upstream's entity-extraction prompts and their
  LLM-cache keys, including the Python-specific parts: `html.unescape`,
  `str.strip`, `str.format`, `json.dumps` separators.

- Chunking (`chunkingByTokenSize` over a tiktoken-compatible tokenizer),
  extraction (`extractEntities`) and merging into the graph and vector stores
  (`mergeNodesAndEdges`, map-reduce summaries): the same chunks, records,
  graph, vectors and LLM cache entries as upstream for the same input.
- Queries: `kgQuery` (local / global / hybrid / mix) and `naiveQuery` build
  the same keyword prompt, context, final prompt and cache keys as upstream
  (`lightragQueryConfig` and `queryTextChunks` give them the config LightRAG
  would): on a real store 768 of 770 query/mode probes identical, the other two
  an exact score tie in the vector search; every cached query key recomputed.
- The `LightRAG` class (`src/lightrag.ts`, upstream's method and field names):
  `ainsert` / `apipeline_enqueue_documents` / `apipeline_process_enqueue_documents`
  with doc status, duplicates, cancellation and `max_parallel_insert`;
  `adelete_by_doc_id` with the rebuild of entities and relations other
  documents still cite; `aquery`, `aquery_llm`, `aquery_data` (bypass
  included); `aedit_entity`, `amerge_entities` and the rest of utils_graph.
  The same scenario through upstream's `LightRAG` and this one ends with
  byte-identical store files (`node parity/pipeline.ts synthetic`, and on a
  copy of a real store `parity/pipeline.ts store`).

  ```ts
  import { LightRAG, makeQueryParam } from 'lightrag-ts';
  const rag = new LightRAG({ working_dir: './rag_storage', llm_model_func, embedding_func });
  await rag.initialize_storages();
  await rag.ainsert(['Some text ...'], null, false, null, ['notes/a.md']);
  const answer = await rag.aquery('What is in my notes?', makeQueryParam({ mode: 'mix' }));
  await rag.finalize_storages();
  ```

  `llm_model_func(prompt, { system_prompt, history_messages, keyword_extraction, stream, ... })`
  returns the completion (or an async iterator when streaming);
  `embedding_func(texts)` returns one vector per text and carries
  `embedding_dim`.
- LLM and embedding functions (`src/llm/`): `openai_complete_if_cache`,
  `openai_complete`, `openai_embed` from upstream's `llm/openai.py`, over
  `fetch` (no `openai` package), with both of upstream's retry layers; a
  server sees the same request bodies byte for byte (checked against
  recordings of upstream). `makeClaudeCompleteLlm` calls the `claude-complete`
  bridge the way the personal vault does; `classifyProviderError` tells an
  empty balance, a spent Claude usage window and a lost network apart.

  ```ts
  import { LightRAG, makeEmbeddingFunc, openai_complete_if_cache, openai_embed_func } from 'lightrag-ts';
  const rag = new LightRAG({
    working_dir: './rag_storage',
    llm_model_name: 'gpt-4o-mini',
    llm_model_func: (prompt, opts) => openai_complete_if_cache('gpt-4o-mini', prompt, opts),
    embedding_func: makeEmbeddingFunc({
      embedding_dim: 1536,
      max_token_size: 8192,
      func: (texts, o) => openai_embed_func(texts, { model: 'text-embedding-3-small', ...o }),
    }),
  });
  ```
- Writing it back: `JsonKVStorage`, `JsonDocStatusStorage`,
  `NanoVectorDBStorage` and `NetworkXStorage` follow upstream's storage API
  and persist files byte-identical to what upstream writes for the same state
  (`node parity/storages.ts <store>`, which works on a copy).

## Command line

```sh
lightrag-ts insert notes/*.md          # stored as notes/<name>.md; stdin: ... | lightrag-ts insert --file-path note.md
lightrag-ts insert ../x.md             # paths are relative to the current directory: stored as ../x.md
lightrag-ts query "What did I decide about X?" --mode mix
lightrag-ts delete --doc-id doc-0123... [--delete-llm-cache]
lightrag-ts status                     # documents by status, graph and vector counts
```

(`npm run cli -- <command>` from a checkout.) Configuration comes from the
environment, `.env` in the current directory first, with upstream's server
variable names: `WORKING_DIR`, `WORKSPACE`, `LLM_BINDING` (`openai` or
`claude-complete`), `LLM_MODEL`, `LLM_BINDING_HOST`, `LLM_BINDING_API_KEY`,
`EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_SEND_DIM`,
`EMBEDDING_TOKEN_LIMIT` (unset: texts are still cut at 8192 tokens), `EMBEDDING_BINDING_HOST`, `EMBEDDING_BINDING_API_KEY`
(keys fall back to `OPENAI_API_KEY`), and whatever the `LightRAG` class reads
itself (`MAX_ASYNC`, `MAX_PARALLEL_INSERT`, `CHUNK_SIZE`, `TOP_K`, ...). The
personal vault's setup, for example:

```sh
WORKING_DIR=~/.claude-home/rag WORKSPACE=personal-vault \
LLM_BINDING=claude-complete LLM_MODEL=haiku \
EMBEDDING_MODEL=text-embedding-3-large EMBEDDING_DIM=1536 EMBEDDING_SEND_DIM=true \
MAX_ASYNC=4 EMBEDDING_FUNC_MAX_ASYNC=4 MAX_PARALLEL_INSERT=2 \
lightrag-ts status
```

Exit codes: 0 done, 1 error, 2 usage, 3 no credit (OpenAI
`insufficient_quota`), 4 the Claude usage window is spent, 5 no network. The
first such refusal stops an insert; when several kinds meet, billing wins, then
the usage window.

## Parity approach

Upstream caches every LLM call under `md5(prompt)`. If the TS port builds the
same prompt, it finds the same cache entry, so a whole ingest can be replayed
against a real store's `kv_store_llm_response_cache.json` with **zero API
calls** and the outputs compared with what Python produced.

```sh
node parity/replay-extract.ts <working_dir>   # read-only
```

On a real 6.6k-chunk store: 6590/6590 chunks with cache entries hit both the
initial extraction key and the gleaning key.

The merge is replayed the same way (`node parity/operate.ts <store> --phase merge`):
both sides merge whole documents from the cache into fresh stores and every file
they write is compared byte for byte.

The whole class is compared the same way (`node parity/pipeline.ts`): a list
of steps (inserts, deletions, queries, edits) runs through upstream's
`LightRAG` (`tools/upstream/pipeline_parity.py`) and through the TS one with
the same LLM and embedding stand-ins and a fixed clock; every step's result,
every pipeline message and every store file must match.

Fixtures under `test/fixtures/` are generated by running upstream code on
synthetic input (`tools/upstream/make_fixtures.py`) or on upstream's own prompt
examples (`parse-cases.json`, `merge-cases/`: `parity/operate.ts --phase
fixture,merge-fixture`); no private data is committed.

## Upstream's own tests

`npm run test:upstream` runs upstream's pytest files, unchanged, at the pinned
tag (taken from a LightRAG checkout, `LIGHTRAG_UPSTREAM`) against lightrag-ts:
`tools/upstream/lightrag_ts_adapter.py` swaps each ported function for a call
into TS through a Node bridge, while the tokenizers, mocks and stub storages the
tests define stay Python and are called back. `tools/upstream/upstream-tests.json`
says which tests run against TS and why the others are skipped; a test marked
`ts` that never reaches TS fails. Upstream's next release brings its new tests
along with the down-merge. Plain values cross the bridge as JSON copies: a
`pipeline_status` dict is synced back when the call returns, so a
`cancellation_requested` that a Python callback sets in the middle of a TS
call is not seen by TS.

## Layout

`src/` mirrors `lightrag/` file by file (`operate.py` -> `src/operate.ts`,
`kg/networkx_impl.py` -> `src/kg/networkx_impl.ts`, ...), so a reviewer can
diff a module against its upstream counterpart. Port-only code lives in
`src/internal/` and `src/store.ts`.

`src/prompt.ts` and `src/constants.ts` are generated from upstream by
`tools/upstream/dump.py` + `tools/upstream/gen.ts`; do not edit them by hand.

## Development

Node >= 22.18 (runs `.ts` directly for tests and scripts).

```sh
npm install
npm test          # node --test, includes upstream-parity fixtures
npm run test:upstream   # upstream's pytest files against TS (needs a venv, see the runner)
npm run typecheck
npm run build     # tsc -> dist/ (JS + .d.ts)
```

## License

MIT. Original work © 2025 LightRAG Team; see [LICENSE](LICENSE).
