// Read-only smoke of the TS storages on a real store: load every file with the
// storage class, report counts, load time and the process RSS, one fresh process
// per file so each RSS is that file's own. Nothing here persists: no
// index_done_callback, upsert or drop is called, and the only storage path that
// writes on load (legacy llm-cache migration) is checked to be a no-op first.
//
//   node parity/live-smoke.ts <store-dir> [--max-rss-mb 1024]
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { JsonDocStatusStorage } from '../src/kg/json_doc_status_impl.ts';
import { JsonKVStorage, migrateLegacyCacheStructure, readJsonKv } from '../src/kg/json_kv_impl.ts';
import { NanoVectorDBStorage } from '../src/kg/nano_vector_db_impl.ts';
import { NetworkXStorage } from '../src/kg/networkx_impl.ts';
import type { EmbeddingFunc, GlobalConfig } from '../src/base.ts';
import { makeRelationVdbIds } from '../src/utils.ts';

const args = process.argv.slice(2);
const oneMode = args[0] === '--one';
const dirArg = oneMode ? args[1] : args[0];
if (!dirArg) throw new Error('usage: live-smoke.ts <store-dir> [--max-rss-mb N]');
const dir: string = dirArg;
const maxRss = args.includes('--max-rss-mb') ? Number(args[args.indexOf('--max-rss-mb') + 1]) : 1024;
const cfg: GlobalConfig = { working_dir: dir, vector_db_storage_cls_kwargs: { cosine_better_than_threshold: 0.2 } };
const noEmbed: EmbeddingFunc = Object.assign(
  async () => {
    throw new Error('no embeddings in a smoke run');
  },
  { embedding_dim: 1536 },
);
const rssMb = () => Math.round(process.memoryUsage().rss / 2 ** 20);

async function one(target: string): Promise<Record<string, unknown>> {
  const t = Date.now();
  const [kind, ns] = target.split(':') as [string, string];
  if (kind === 'graph') {
    const s = new NetworkXStorage({ namespace: ns, global_config: cfg });
    const g = s.graph;
    const intLike = [...g.nodes.keys()].filter((n) => /^(0|[1-9]\d*)$/.test(n)).length;
    const top = (await s.get_popular_labels(1))[0]!;
    const kg = await s.get_knowledge_graph(top, 2, 200);
    return { nodes: g.nodes.size, edges: g.numberOfEdges, intLikeNodeIds: intLike, kgFromTop: [kg.nodes.length, kg.edges.length, kg.is_truncated], ms: Date.now() - t, rssMb: rssMb() };
  }
  if (kind === 'kv') {
    const file = join(dir, `kv_store_${ns}.json`);
    if (ns.endsWith('_cache') && migrateLegacyCacheStructure(readJsonKv(file)).count > 0) throw new Error(`${file} would be migrated (written) on load; refusing`);
    const s = ns === 'doc_status' ? new JsonDocStatusStorage({ namespace: ns, global_config: cfg }) : new JsonKVStorage({ namespace: ns, global_config: cfg });
    await s.initialize();
    const out: Record<string, unknown> = { records: s.size, ms: Date.now() - t, rssMb: rssMb() };
    if (s instanceof JsonDocStatusStorage) out.statusCounts = await s.get_all_status_counts();
    return out;
  }
  const s = new NanoVectorDBStorage({ namespace: ns, global_config: cfg, embedding_func: noEmbed, meta_fields: [] });
  const loadMs = Date.now() - t;
  const db = s.db;
  const out: Record<string, unknown> = { rows: db.length, loadMs, rssMb: rssMb() };
  if (db.length) {
    const tq = Date.now();
    const r = await s.query('', 10, Array.from(db.row(0)));
    out.selfQuery = { top: r[0]?.id === db.data[0]?.__id__, score: r[0]?.distance, ms: Date.now() - tq };
  }
  if (ns === 'relationships') {
    // workflow-3361: edges whose vector is absent; storages just return null for them
    const g = new NetworkXStorage({ namespace: 'chunk_entity_relation', global_config: cfg }).graph;
    let edges = 0;
    let missing = 0;
    for (const [u, v] of g.edges()) {
      edges++;
      if ((await s.get_by_ids(makeRelationVdbIds(u, v))).every((x) => x === null)) missing++;
    }
    out.edgesWithoutVector = `${missing} of ${edges}`;
    out.rssMbWithGraph = rssMb();
  }
  return out;
}

if (oneMode) {
  console.log(JSON.stringify(await one(args[2]!)));
} else {
  const targets: string[] = [];
  const files = readdirSync(dir);
  if (existsSync(join(dir, 'graph_chunk_entity_relation.graphml'))) targets.push('graph:chunk_entity_relation');
  for (const f of files) if (/^kv_store_.*\.json$/.test(f)) targets.push(`kv:${f.slice(9, -5)}`);
  for (const f of files) if (/^vdb_.*\.json$/.test(f)) targets.push(`vdb:${f.slice(4, -5)}`);
  let bad = 0;
  for (const target of targets) {
    const out = JSON.parse(
      execFileSync(process.execPath, ['--max-old-space-size=4096', import.meta.filename, '--one', dir, target], { maxBuffer: 1 << 26 }).toString(),
    );
    const vdbOver = target.startsWith('vdb:') && out.rssMb > maxRss;
    if (vdbOver) bad++;
    console.log(`${vdbOver ? 'OVER' : 'ok  '} ${target.padEnd(28)} ${JSON.stringify(out)}`);
  }
  console.log(bad ? `${bad} vector store load(s) over ${maxRss} MB RSS` : `vector store loads under ${maxRss} MB RSS`);
  process.exit(bad ? 1 : 0);
}
