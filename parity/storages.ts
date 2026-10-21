// B1 storage parity: the TS storages against upstream's, on a copy of a store.
//
//   node parity/storages.ts <store-dir> [--work <dir>] [--dim 1536] [--no-mutate] [--vdb entities,chunks,relationships]
//
// The store is copied into <work>/orig first; nothing is ever written to <store-dir>.
//   A read     TS and upstream load the same files: graph adjacency order, attrs, degrees,
//              labels; KV counts/ids/get_by_ids/filter_keys/status counts/pagination;
//              vector ids, get_by_ids and top-10 queries (scores to 1e-6).
//   B rewrite  TS loads and persists every file -> upstream loads the TS output and must
//              dump exactly what it dumps for the original; files are compared byte-wise
//              with upstream's own rewrite.
//   C mutate   the same ops (upserts, deletes, node/edge removal, vector upserts with a
//              deterministic fake embedding) applied by both sides on separate copies;
//              upstream dumps both results, and files are compared byte-wise.
//   D kg       get_knowledge_graph node/edge sets for a few labels.
//   E          relationship edges without a vector (workflow-3361): same set on both sides.
// Exit code 1 on any mismatch. Upstream side: tools/upstream/storage_parity.py in the rag-venv.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { JsonDocStatusStorage } from '../src/kg/json_doc_status_impl.ts';
import { JsonKVStorage } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDBStorage } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from '../src/kg/networkx_impl.ts';
import type { EmbeddingFunc, GlobalConfig, KvRecord } from '../src/base.ts';
import { makeRelationVdbIds } from '../src/utils.ts';

const PY = process.env.LIGHTRAG_PY ?? join(process.env.HOME!, '.claude-home/rag-venv/bin/python');
const PY_TOOL = resolve(import.meta.dirname, '../tools/upstream/storage_parity.py');
const META_FIELDS: Record<string, string[]> = {
  entities: ['entity_name', 'source_id', 'content', 'file_path'],
  relationships: ['src_id', 'tgt_id', 'source_id', 'content', 'file_path'],
  chunks: ['full_doc_id', 'content', 'file_path'],
};
const GRAPH = 'graph_chunk_entity_relation.graphml';

const argv = process.argv.slice(2);
const opt = (name: string) => {
  const i = argv.indexOf(name);
  return i < 0 ? undefined : argv.splice(i, 2)[1];
};
const flag = (name: string) => {
  const i = argv.indexOf(name);
  return i < 0 ? false : (argv.splice(i, 1), true);
};
const DIM = Number(opt('--dim') ?? 1536);
const WORK = opt('--work') ?? mkdtempSync(join(tmpdir(), 'lightrag-parity-'));
const VDB_ONLY = opt('--vdb')?.split(',');
const NO_MUTATE = flag('--no-mutate');
const SRC = argv[0];
if (!SRC) throw new Error('usage: storages.ts <store-dir> [--work dir] [--dim N] [--no-mutate] [--vdb a,b]');

let failures = 0;
const t0 = Date.now();
const log = (s: string) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${s}`);
function check(name: string, ok: boolean, detail = ''): void {
  if (!ok) failures++;
  log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` - ${detail}` : ''}`);
}

function py(...args: string[]): any {
  try {
    const out = execFileSync(PY, ['-I', PY_TOOL, ...args], { maxBuffer: 1 << 30, stdio: ['ignore', 'pipe', 'pipe'] });
    return JSON.parse(out.toString('utf8'));
  } catch (e) {
    const err = (e as { stderr?: Buffer }).stderr?.toString('utf8') ?? '';
    throw new Error(`upstream ${args[0]} failed:\n${err.split('\n').filter((l) => !l.startsWith('INFO')).slice(-8).join('\n')}`);
  }
}

const cfg = (dir: string): GlobalConfig => ({
  working_dir: dir,
  embedding_batch_num: 32,
  vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 },
  max_graph_nodes: 1000,
});

/** fake_vector() in storage_parity.py */
function fakeVector(text: string, dim: number): number[] {
  const h = createHash('md5').update(text, 'utf8').digest('hex');
  return Array.from({ length: dim }, (_, i) => (parseInt(h.slice((2 * i) % 32, ((2 * i) % 32) + 2), 16) - 127.5) / 127.5 + (i % 7) * 0.01);
}
const emb: EmbeddingFunc = Object.assign(async (texts: string[]) => texts.map((t) => fakeVector(t, DIM)), { embedding_dim: DIM });

const storeFiles = (dir: string) => readdirSync(dir).filter((f) => /^(kv_store_.*|vdb_.*)\.json$/.test(f) || f === GRAPH);
const kvNamespaces = (dir: string) =>
  storeFiles(dir)
    .filter((f) => f.startsWith('kv_store_'))
    .map((f) => f.slice('kv_store_'.length, -'.json'.length))
    .sort();
const vdbNamespaces = (dir: string) =>
  storeFiles(dir)
    .filter((f) => f.startsWith('vdb_'))
    .map((f) => f.slice('vdb_'.length, -'.json'.length))
    .filter((ns) => !VDB_ONLY || VDB_ONLY.includes(ns));

function copyStore(from: string, to: string): void {
  mkdirSync(to, { recursive: true });
  for (const f of storeFiles(from)) copyFileSync(join(from, f), join(to, f));
}
const sameBytes = (a: string, b: string) => existsSync(a) && existsSync(b) && readFileSync(a).equals(readFileSync(b));

// ---------------------------------------------------------------- TS dumps

const kvStore = async (dir: string, ns: string) => {
  const s = ns === 'doc_status' ? new JsonDocStatusStorage({ namespace: ns, global_config: cfg(dir) }) : new JsonKVStorage({ namespace: ns, global_config: cfg(dir) });
  await s.initialize();
  return s;
};

async function tsGraphDump(dir: string) {
  const s = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(dir) });
  const g = s.graph;
  return {
    nodes: [...g.nodes.keys()],
    adj: Object.fromEntries([...g.nodes.keys()].map((n) => [n, [...g.adj.get(n)!.keys()]])),
    node_attrs: Object.fromEntries(g.nodes),
    edges: [...g.edges()].map(([u, v, d]) => [u, v, d]),
    degrees: Object.fromEntries([...g.nodes.keys()].map((n) => [n, g.degree(n)])),
    labels: await s.get_all_labels(),
    popular: await s.get_popular_labels(50),
  };
}

async function tsKvDump(dir: string) {
  const out: Record<string, unknown> = {};
  for (const ns of kvNamespaces(dir)) {
    const s = await kvStore(dir, ns);
    const keys = [...s.keys()];
    const rec: Record<string, unknown> = {
      count: keys.length,
      keys_sha: createHash('sha256').update(keys.join('\n')).digest('hex'),
      get_by_ids_first3: await s.get_by_ids([...keys.slice(0, 3), 'missing-id']),
      filter_keys: [...(await s.filter_keys([...keys.slice(0, 2), 'missing-a', 'missing-b']))].sort(),
    };
    if (s instanceof JsonDocStatusStorage) {
      rec.status_counts = await s.get_all_status_counts();
      try {
        const [docs, total] = await s.get_docs_paginated(null, 1, 10, 'updated_at', 'desc');
        rec.paginated = docs.map(([k, v]) => [k, v.status]);
        rec.paginated_total = total;
      } catch (e) {
        rec.paginated = { error: (e as Error).name };
      }
      const byStatus = await s.get_docs_by_statuses(['pending', 'processed', 'failed', 'processing', 'preprocessed']);
      rec.by_statuses = [...byStatus.keys()].sort();
    }
    out[ns] = rec;
  }
  return out;
}

const vdbStore = (dir: string, ns: string) =>
  new NanoVectorDBStorage({ namespace: ns, global_config: cfg(dir), embedding_func: emb, meta_fields: META_FIELDS[ns] ?? [] });

// ---------------------------------------------------------------- compare helpers

/** JSON round trip on the TS side: Python's 1.0 and JS's 1 both parse to 1. */
const norm = (v: unknown) => JSON.parse(JSON.stringify(v));

function firstDiff(a: any, b: any, path = ''): string {
  if (isDeepStrictEqual(a, b)) return '';
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const d = firstDiff(a[k], b[k], `${path}.${k}`);
      if (d) return d;
    }
    if (Array.isArray(a) && Array.isArray(b) && a.length !== b.length) return `${path}: length ${a.length} vs ${b.length}`;
  }
  return `${path}: ${JSON.stringify(a)?.slice(0, 160)} vs ${JSON.stringify(b)?.slice(0, 160)}`;
}

function compareDumps(name: string, a: unknown, b: unknown): void {
  const d = firstDiff(norm(a), norm(b));
  check(name, d === '', d);
}

function compareQueries(name: string, pyQ: [string, number][][], tsQ: [string, number][][]): void {
  let maxErr = 0;
  let orderOk = true;
  pyQ.forEach((q, i) => {
    const t = tsQ[i]!;
    if (q.length !== t.length) orderOk = false;
    q.forEach(([id, s], j) => {
      if (t[j]?.[0] !== id) orderOk = false;
      maxErr = Math.max(maxErr, Math.abs(s - (t[j]?.[1] ?? Infinity)));
    });
  });
  check(name, orderOk && maxErr < 1e-6, `ids ${orderOk ? 'same order' : 'DIFFER'}, max |score diff| ${maxErr.toExponential(2)}`);
}

// ---------------------------------------------------------------- run

const ORIG = join(WORK, 'orig');
log(`work dir ${WORK}`);
copyStore(SRC, ORIG);
const hasGraph = existsSync(join(ORIG, GRAPH));

// A: read parity
if (hasGraph) {
  const pyG = py('graph-dump', ORIG);
  compareDumps('A graph: nodes, adjacency order, attrs, edges(), degrees, labels', pyG, await tsGraphDump(ORIG));
}
const pyKv = py('kv-dump', ORIG);
const tsKv = await tsKvDump(ORIG);
for (const ns of Object.keys(pyKv)) {
  const { content_sha: _c, keys_sha: _k, ...rest } = pyKv[ns];
  const { keys_sha: _k2, ...tsRest } = tsKv[ns] as Record<string, unknown>;
  compareDumps(`A kv ${ns}: ${pyKv[ns].count} records, ids/get_by_ids/filter_keys${ns === 'doc_status' ? '/counts/pages' : ''}`, { ...rest, k: _k }, { ...tsRest, k: _k2 });
}
const pyVdb: Record<string, any> = {};
for (const ns of vdbNamespaces(ORIG)) {
  pyVdb[ns] = py('vdb-dump', ORIG, ns, String(DIM));
  const s = vdbStore(ORIG, ns);
  const ids = s.db.data.map((d) => d.__id__);
  const tsDump = {
    count: s.db.length,
    ids_sha: createHash('sha256').update(ids.join('\n')).digest('hex'),
    get_by_ids: await s.get_by_ids([...ids.slice(0, 2), 'missing-id']),
  };
  compareDumps(`A vdb ${ns}: ${tsDump.count} rows, ids, get_by_ids`, { count: pyVdb[ns].count, ids_sha: pyVdb[ns].ids_sha, get_by_ids: pyVdb[ns].get_by_ids }, tsDump);
  const tsQ: [string, number][][] = [];
  for (const p of pyVdb[ns].probes as number[][]) tsQ.push((await s.query('', 10, p)).map((r) => [r.id, r.distance]));
  compareQueries(`A vdb ${ns}: top-10 cosine queries (${pyVdb[ns].probes.length} probes, threshold 0.2)`, pyVdb[ns].queries, tsQ);
}

// B: TS rewrite -> upstream reads it
const TSRT = join(WORK, 'ts-rewrite');
copyStore(ORIG, TSRT);
for (const ns of kvNamespaces(TSRT)) {
  const s = await kvStore(TSRT, ns);
  s.mark_dirty();
  await s.index_done_callback();
}
if (hasGraph) await new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(TSRT) }).index_done_callback();
for (const ns of vdbNamespaces(TSRT)) await vdbStore(TSRT, ns).index_done_callback();
log('B TS rewrote every file');

if (hasGraph) {
  compareDumps('B graph: upstream reads TS output == upstream reads original', py('graph-dump', ORIG), py('graph-dump', TSRT));
  const pyRt = join(WORK, 'py-rewrite');
  py('graph-rewrite', ORIG, '--out', pyRt);
  check('B graph: TS file byte-identical to upstream rewrite', sameBytes(join(pyRt, GRAPH), join(TSRT, GRAPH)));
}
compareDumps('B kv: upstream reads TS output == original (content sha incl.)', pyKv, py('kv-dump', TSRT));
for (const ns of kvNamespaces(ORIG)) {
  check(`B kv ${ns}: TS file byte-identical to original`, sameBytes(join(ORIG, `kv_store_${ns}.json`), join(TSRT, `kv_store_${ns}.json`)));
}
for (const ns of vdbNamespaces(ORIG)) {
  const after = py('vdb-dump', TSRT, ns, String(DIM));
  // Upstream renormalizes the matrix on every load and float32 normalize is not
  // idempotent bit for bit, so upstream's own load+save changes some rows: the
  // matrix is compared with upstream's rewrite, everything else with the original.
  const { matrix_sha: _m0, probes: _p0, queries: _q0, ...origRest } = pyVdb[ns];
  const { matrix_sha: _m1, probes: _p1, queries: _q1, ...afterRest } = after;
  compareDumps(`B vdb ${ns}: upstream reads TS output == original (count, ids, meta incl. vector fields)`, origRest, afterRest);
  const pyRt = join(WORK, `py-rewrite-${ns}`);
  py('vdb-rewrite', ORIG, ns, String(DIM), '--out', pyRt);
  const afterPy = py('vdb-dump', pyRt, ns, String(DIM));
  compareDumps(`B vdb ${ns}: upstream reads TS output == upstream reads its own rewrite (matrix sha, queries)`, afterPy, after);
  if (afterPy.matrix_sha !== pyVdb[ns].matrix_sha) log(`note vdb ${ns}: upstream's own load+save changes the matrix bits (renormalization)`);
  check(`B vdb ${ns}: TS file byte-identical to upstream rewrite`, sameBytes(join(pyRt, `vdb_${ns}.json`), join(TSRT, `vdb_${ns}.json`)));
}

// C: same mutations on both sides
if (!NO_MUTATE) {
  const ops = await buildOps(ORIG);
  const opsFile = join(WORK, 'ops.json');
  writeFileSync(opsFile, JSON.stringify(ops, null, 1));
  const PYM = join(WORK, 'py-mutate');
  const TSM = join(WORK, 'ts-mutate');
  copyStore(ORIG, PYM);
  copyStore(ORIG, TSM);
  py('mutate', PYM, opsFile, String(DIM));
  await tsMutate(TSM, ops);
  log(`C applied ${JSON.stringify(opsSummary(ops))} on both sides`);
  if (hasGraph) compareDumps('C graph after ops: upstream dump of TS result == of upstream result', py('graph-dump', PYM), py('graph-dump', TSM));
  compareDumps('C kv after ops: upstream dump equal', py('kv-dump', PYM), py('kv-dump', TSM));
  for (const ns of Object.keys(ops.vdb)) {
    const a = py('vdb-dump', PYM, ns, String(DIM));
    const b = py('vdb-dump', TSM, ns, String(DIM));
    compareDumps(`C vdb ${ns} after ops: upstream dump equal (meta incl. vector fields, matrix sha)`, a, b);
  }
  for (const f of storeFiles(PYM)) {
    const same = sameBytes(join(PYM, f), join(TSM, f));
    check(`C ${f} after ops: byte-identical`, same);
  }
}

// D: knowledge graph subsets
if (hasGraph) {
  const s = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(ORIG) });
  const pop = await s.get_popular_labels(300);
  const labels = ['*', pop[0], pop[Math.floor(pop.length / 2)], pop[pop.length - 1]].filter((x): x is string => !!x);
  for (const [label, depth, max] of labels.flatMap((l) => [[l, 2, 50], [l, 3, 1000]] as const)) {
    const p = py('graph-kg', ORIG, label, String(depth), String(max));
    const kg = await s.get_knowledge_graph(label, depth, max);
    compareDumps(`D knowledge graph "${label.slice(0, 30)}" depth ${depth} max ${max}: ${p.nodes.length} nodes, ${p.edges.length} edges`, p, {
      nodes: kg.nodes.map((n) => n.id).sort(cmpPy),
      edges: kg.edges.map((e) => e.id).sort(cmpPy),
      is_truncated: kg.is_truncated,
      node_props: Object.fromEntries(kg.nodes.map((n) => [n.id, n.properties])),
    });
  }
}

// E: edges without a relationship vector (workflow-3361)
if (hasGraph && vdbNamespaces(ORIG).includes('relationships')) {
  const p = py('missing-rel-vectors', ORIG, String(DIM));
  const g = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(ORIG) });
  const v = vdbStore(ORIG, 'relationships');
  const missing: [string, string][] = [];
  let total = 0;
  for (const [u, w] of g.graph.edges()) {
    total++;
    const got = await v.get_by_ids(makeRelationVdbIds(u, w));
    if (got.every((x) => x === null)) missing.push([u, w]);
  }
  const ts = { edges: total, missing: missing.length, missing_sha: pySha(missing), first: missing.slice(0, 3) };
  compareDumps(`E edges without a relationship vector: ${p.missing} of ${p.edges}, same edges both sides`, p, ts);
}

log(failures ? `${failures} FAILED` : 'all parity checks passed');
process.exit(failures ? 1 : 0);

// ---------------------------------------------------------------- ops

type Ops = {
  clock: number;
  kv: Record<string, { upsert: Record<string, KvRecord>; delete: string[] }>;
  doc_status?: { upsert: Record<string, KvRecord>; delete: string[] };
  graph?: unknown[][];
  vdb: Record<string, { upsert: Record<string, KvRecord>; delete: string[]; delete_entity: string[]; delete_entity_relation: string[] }>;
};

async function buildOps(dir: string): Promise<Ops> {
  const ops: Ops = { clock: 1700000000, kv: {}, vdb: {} };
  for (const ns of kvNamespaces(dir)) {
    const keys = [...(await kvStore(dir, ns)).keys()];
    const fresh: KvRecord = { content: `parity record for ${ns}`, n: 3, nested: { a: [1, 'x'] } };
    if (ns === 'doc_status') {
      ops.doc_status = {
        upsert: {
          'doc-parity-new': {
            status: 'pending',
            content_summary: 'parity',
            content_length: 6,
            file_path: 'parity.md',
            created_at: '2026-10-21T00:00:00+00:00',
            updated_at: '2026-10-21T00:00:00+00:00',
          },
        },
        delete: keys.slice(1, 2),
      };
      continue;
    }
    const upsert: Record<string, KvRecord> = { [`parity-new-${ns}`]: fresh };
    if (keys[0]) upsert[keys[0]] = { content: 'replaced by parity run' };
    ops.kv[ns] = { upsert, delete: [...keys.slice(1, 3), 'missing-key'] };
  }
  if (existsSync(join(dir, GRAPH))) {
    const g = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(dir) }).graph;
    const nodes = [...g.nodes.keys()];
    const edges = [...g.edges()].map(([u, v]) => [u, v]);
    const a = nodes[0]!;
    ops.graph = [
      ['upsert_node', '10', { entity_id: '10', entity_type: 'number', description: 'an integer-like id', created_at: 1700000000 }],
      ['upsert_node', 'Parity <&> "node"', { entity_id: 'Parity <&> "node"', description: 'line\nbreak\ttab', truncate: '' }],
      ['upsert_edge', a, '10', { weight: 2.5, description: 'parity edge', keywords: 'k', source_id: 'chunk-x', created_at: 1700000000 }],
      ['upsert_edge', '10', 'Parity <&> "node"', { weight: 0.1, description: 'second' }],
      ['upsert_edge', 'Parity <&> "node"', 'Parity <&> "node"', { weight: 1e-7, description: 'self loop' }],
      ...(edges[0] ? [['upsert_edge', edges[0][1], edges[0][0], { weight: 7.25 }]] : []),
      ...(nodes[2] ? [['delete_node', nodes[2]]] : []),
      ['delete_node', 'no such node'],
      ['remove_edges', edges.slice(3, 5).concat([['no', 'edge']])],
      ['upsert_nodes_batch', [['batch-1', { entity_id: 'batch-1' }], ['2', { entity_id: '2' }]]],
      ['upsert_edges_batch', [['batch-1', '2', { weight: 3.5 }], ['2', a, { weight: 0.5 }]]],
    ];
  }
  for (const ns of vdbNamespaces(dir)) {
    const s = vdbStore(dir, ns);
    const data = s.db.data;
    const meta = (id: string, extra: KvRecord): KvRecord => ({
      content: `vector parity ${ns} ${id}`,
      entity_name: `Parity ${id}`,
      src_id: 'Parity A',
      tgt_id: 'Parity B',
      source_id: 'chunk-parity',
      full_doc_id: 'doc-parity',
      file_path: 'parity.md',
      not_a_meta_field: 'dropped',
      ...extra,
    });
    const upsert: Record<string, KvRecord> = { [`${ns}-parity-new-1`]: meta('1', {}), [`${ns}-parity-new-2`]: meta('2', {}) };
    if (data[0]) upsert[data[0].__id__] = meta('existing', { content: `replaced ${data[0].__id__}` });
    ops.vdb[ns] = {
      upsert,
      delete: data.slice(1, 2).map((d) => d.__id__),
      delete_entity: ns === 'entities' && data[3] ? [String(data[3].entity_name)] : [],
      delete_entity_relation: ns === 'relationships' && data[4] ? [String(data[4].src_id)] : [],
    };
  }
  return ops;
}

function opsSummary(ops: Ops) {
  return { kv: Object.keys(ops.kv).length, doc_status: !!ops.doc_status, graph: ops.graph?.length ?? 0, vdb: Object.keys(ops.vdb) };
}

async function tsMutate(dir: string, ops: Ops): Promise<void> {
  // upstream reads time.time() inside upsert; both sides pin it
  const clock = () => ops.clock;
  for (const [ns, o] of Object.entries(ops.kv)) {
    const s = new JsonKVStorage({ namespace: ns, global_config: cfg(dir), clock });
    await s.initialize();
    await s.upsert(structuredClone(o.upsert));
    await s.delete(o.delete);
    await s.index_done_callback();
  }
  if (ops.doc_status) {
    const s = new JsonDocStatusStorage({ namespace: 'doc_status', global_config: cfg(dir), clock });
    await s.initialize();
    await s.upsert(structuredClone(ops.doc_status.upsert));
    await s.delete(ops.doc_status.delete);
    await s.index_done_callback();
  }
  if (ops.graph) {
    const s = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg(dir) });
    for (const [name, ...args] of structuredClone(ops.graph)) await (s as any)[name as string](...args);
    await s.index_done_callback();
  }
  for (const [ns, o] of Object.entries(ops.vdb)) {
    const s = new NanoVectorDBStorage({ namespace: ns, global_config: cfg(dir), embedding_func: emb, meta_fields: META_FIELDS[ns] ?? [], clock });
    await s.upsert(structuredClone(o.upsert));
    await s.delete(o.delete);
    for (const n of o.delete_entity) await s.delete_entity(n);
    for (const n of o.delete_entity_relation) await s.delete_entity_relation(n);
    await s.index_done_callback();
  }
}

function cmpPy(a: string, b: string): number {
  const x = [...a].map((c) => c.codePointAt(0)!);
  const y = [...b].map((c) => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i]! - y[i]!;
  return x.length - y.length;
}

/** sha256 of Python json.dumps(obj, ensure_ascii=False) for lists of string pairs */
function pySha(pairs: [string, string][]): string {
  const s = '[' + pairs.map(([a, b]) => `[${JSON.stringify(a)}, ${JSON.stringify(b)}]`).join(', ') + ']';
  return createHash('sha256').update(s).digest('hex');
}
