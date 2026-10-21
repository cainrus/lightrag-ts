// Port of lightrag/kg/networkx_impl.py (NetworkXStorage) and of the networkx
// 3.x pieces it relies on: nx.Graph adjacency order, read_graphml, write_graphml.
//
// Why not graphology as the core: graphology keeps a node's neighbours in a
// plain object, so integer-like ids ("10", "2024") iterate first. networkx keeps
// insertion order, and get_node_edges / Graph.edges() order leaks into GraphML
// output and query results. NxGraph below is a Map-based nx.Graph; graphology is
// available as an export (`toGraphology`).
//
// read_graphml detail that matters: networkx parses into a MultiGraph and
// converts it with nx.Graph(G), which rebuilds the adjacency in Graph.edges()
// order. A node's neighbour order after loading is therefore not the file order;
// `loadNxGraph` replays that conversion.
import { closeSync, existsSync, mkdirSync, openSync, readSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import { UndirectedGraph } from 'graphology';
import { SaxesParser } from 'saxes';
import type { BaseGraphStorage, DropResult, EdgeData, GlobalConfig, NodeData, StorageOptions } from '../base.ts';
import { edgePairKey, workspaceDir } from '../base.ts';
import { atomicWrite, pyFloatRepr } from '../internal/py_json.ts';
import type { KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode } from '../types.ts';
import { pyCompare, pyStrip } from '../utils.ts';

export type GraphAttrs = Record<string, string | number | boolean>;

/** `<working_dir>[/<workspace>]/graph_<namespace>.graphml` */
export function graphmlPath(workingDir: string, namespace: string, workspace = ''): string {
  return join(workspace ? join(workingDir, workspace) : workingDir, `graph_${namespace}.graphml`);
}

export function edgeKey(source: string, target: string, directed = false): string {
  if (!directed && target < source) [source, target] = [target, source];
  return `${source}\u0000${target}`;
}

/** Python-type tag of a numeric attribute: JS numbers do not carry int vs float. */
export type AttrScope = 'node' | 'edge';
const floatKey = (scope: AttrScope, name: string) => `${scope}\u0000${name}`;

/**
 * nx.Graph: insertion-ordered nodes, insertion-ordered adjacency, one attribute
 * object shared by both directions of an edge.
 */
export class NxGraph {
  readonly nodes = new Map<string, GraphAttrs>();
  readonly adj = new Map<string, Map<string, GraphAttrs>>();
  /**
   * Attributes whose numbers are Python floats (GraphML "double"/"float").
   * Upstream always writes edge `weight` as a float.
   */
  readonly floatAttrs = new Set<string>([floatKey('edge', 'weight')]);

  isFloat(scope: AttrScope, name: string): boolean {
    return this.floatAttrs.has(floatKey(scope, name));
  }
  markFloat(scope: AttrScope, name: string): void {
    this.floatAttrs.add(floatKey(scope, name));
  }

  addNode(id: string, attrs: GraphAttrs = {}): void {
    let a = this.nodes.get(id);
    if (!a) {
      a = {};
      this.nodes.set(id, a);
      this.adj.set(id, new Map());
    }
    Object.assign(a, attrs);
  }

  addEdge(u: string, v: string, attrs: GraphAttrs = {}): void {
    if (!this.nodes.has(u)) this.addNode(u);
    if (!this.nodes.has(v)) this.addNode(v);
    const au = this.adj.get(u)!;
    const dd = au.get(v) ?? {};
    Object.assign(dd, attrs);
    au.set(v, dd);
    this.adj.get(v)!.set(u, dd);
  }

  hasNode(id: string): boolean {
    return this.nodes.has(id);
  }
  hasEdge(u: string, v: string): boolean {
    return this.adj.get(u)?.has(v) ?? false;
  }
  getEdge(u: string, v: string): GraphAttrs | undefined {
    return this.adj.get(u)?.get(v);
  }
  /** nx degree: a self-loop counts twice. */
  degree(id: string): number {
    const nbrs = this.adj.get(id);
    if (!nbrs) throw new Error(`KeyError: '${id}'`);
    return nbrs.size + (nbrs.has(id) ? 1 : 0);
  }
  neighbors(id: string): IterableIterator<string> {
    const nbrs = this.adj.get(id);
    if (!nbrs) throw new Error(`NetworkXError: The node ${id} is not in the graph.`);
    return nbrs.keys();
  }

  removeNode(id: string): void {
    const nbrs = this.adj.get(id);
    if (!nbrs) throw new Error(`NetworkXError: The node ${id} is not in the graph.`);
    for (const u of [...nbrs.keys()]) this.adj.get(u)!.delete(id);
    this.adj.delete(id);
    this.nodes.delete(id);
  }

  removeEdge(u: string, v: string): void {
    if (!this.hasEdge(u, v)) throw new Error(`NetworkXError: The edge ${u}-${v} is not in the graph`);
    this.adj.get(u)!.delete(v);
    if (u !== v) this.adj.get(v)!.delete(u);
  }

  get numberOfEdges(): number {
    let n = 0;
    for (const _ of this.edges()) n++;
    return n;
  }

  /** nx Graph.edges(data=True) order: nodes in order, each neighbour not yet visited as a node. */
  *edges(nbunch?: Iterable<string>): Generator<[string, string, GraphAttrs]> {
    const seen = new Set<string>();
    const nodes = nbunch ?? this.nodes.keys();
    for (const n of nodes) {
      const nbrs = this.adj.get(n);
      if (!nbrs) continue;
      for (const [nbr, dd] of nbrs) if (!seen.has(nbr)) yield [n, nbr, dd];
      seen.add(n);
    }
  }

  /** nx.Graph(G) on a graph: same nodes, adjacency rebuilt in edges() order. */
  rebuilt(): NxGraph {
    const g = new NxGraph();
    for (const k of this.floatAttrs) g.floatAttrs.add(k);
    for (const id of this.nodes.keys()) g.addNode(id);
    for (const [u, v, dd] of this.edges()) g.addEdge(u, v, dd);
    for (const [id, a] of this.nodes) Object.assign(g.nodes.get(id)!, a);
    return g;
  }

  toGraphology(): UndirectedGraph {
    const g = new UndirectedGraph({ allowSelfLoops: true });
    for (const [id, a] of this.nodes) g.addNode(id, { ...a });
    for (const [u, v, dd] of this.edges()) g.addEdge(u, v, { ...dd });
    return g;
  }
}

// ---------------------------------------------------------------- GraphML read

type KeyDef = { for: string; name: string; type: string };

/** networkx GraphML.python_type for the types it knows */
function convert(type: string, raw: string): string | number | boolean {
  switch (type) {
    case 'int':
    case 'long':
    case 'integer':
    case 'float':
    case 'double':
      return Number(raw);
    case 'boolean': {
      const b = { true: true, false: false, '0': false, '1': true }[raw.toLowerCase()];
      if (b === undefined) throw new Error(`KeyError: '${raw.toLowerCase()}'`);
      return b;
    }
    default:
      return raw;
  }
}

const isFloatType = (t: string) => t === 'float' || t === 'double';

type GraphmlHandlers = {
  graph(directed: boolean): void;
  key(def: KeyDef): void;
  node(id: string, attrs: GraphAttrs): void;
  edge(source: string, target: string, attrs: GraphAttrs): void;
};

/** SAX pass over a networkx-written GraphML file. */
function parseGraphml(file: string, h: GraphmlHandlers): void {
  const keys = new Map<string, KeyDef>();
  let current: { kind: 'node'; id: string; attrs: GraphAttrs } | { kind: 'edge'; source: string; target: string; attrs: GraphAttrs } | null =
    null;
  let currentKey: KeyDef | undefined;
  let text = '';

  const p = new SaxesParser({ xmlns: false });
  p.on('opentag', (t) => {
    const a = t.attributes as Record<string, string>;
    switch (t.name) {
      case 'key': {
        const def = { for: a.for ?? 'all', name: a['attr.name'] ?? a.id!, type: a['attr.type'] ?? 'string' };
        keys.set(a.id!, def);
        h.key(def);
        break;
      }
      case 'graph':
        h.graph(a.edgedefault === 'directed');
        break;
      case 'node':
        current = { kind: 'node', id: a.id!, attrs: {} };
        break;
      case 'edge':
        current = { kind: 'edge', source: a.source!, target: a.target!, attrs: {} };
        break;
      case 'data':
        currentKey = keys.get(a.key!);
        if (!currentKey) throw new Error(`${file}: <data key="${a.key}"> without a <key> definition`);
        text = '';
        break;
    }
  });
  p.on('text', (s) => {
    text += s;
  });
  p.on('closetag', (t) => {
    switch (t.name) {
      case 'data':
        // networkx decode_data_elements: an element without text becomes "" whatever the key type
        if (current && currentKey) current.attrs[currentKey.name] = text === '' ? '' : convert(currentKey.type, text);
        currentKey = undefined;
        text = '';
        break;
      case 'node':
        if (current?.kind === 'node') h.node(current.id, current.attrs);
        current = null;
        break;
      case 'edge':
        if (current?.kind === 'edge') h.edge(current.source, current.target, current.attrs);
        current = null;
        break;
    }
  });

  const fd = openSync(file, 'r');
  const buf = Buffer.allocUnsafe(1 << 22);
  const dec = new StringDecoder('utf8');
  try {
    for (;;) {
      const n = readSync(fd, buf, 0, buf.length, null);
      if (n === 0) break;
      p.write(dec.write(buf.subarray(0, n)));
    }
    p.write(dec.end());
    p.close();
  } finally {
    closeSync(fd);
  }
}

/** nx.read_graphml for an undirected, non-multi graph, as LightRAG writes it. */
export function loadNxGraph(file: string): NxGraph {
  const g = new NxGraph();
  let parallel = false;
  parseGraphml(file, {
    graph(directed) {
      if (directed) throw new Error(`${file}: directed GraphML is not supported (LightRAG writes nx.Graph)`);
    },
    key(def) {
      if (isFloatType(def.type) && (def.for === 'node' || def.for === 'edge')) g.markFloat(def.for, def.name);
    },
    node(id, attrs) {
      g.addNode(id, attrs);
    },
    edge(s, t, attrs) {
      if (g.hasEdge(s, t)) parallel = true;
      g.addEdge(s, t, attrs);
    },
  });
  // networkx would return a MultiGraph here; LightRAG never writes one
  if (parallel) throw new Error(`${file}: parallel edges (networkx would load a MultiGraph)`);
  return g.rebuilt();
}

export type GraphmlGraph = {
  directed: boolean;
  nodes: Map<string, GraphAttrs>;
  /** keyed by `edgeKey(source, target)` */
  edges: Map<string, { source: string; target: string; attrs: GraphAttrs }>;
  /** <edge> elements seen in the file */
  edgeElements: number;
};

/** B0 read-only view, now backed by the same parser as the storage. */
export function readGraphml(file: string): GraphmlGraph {
  let edgeElements = 0;
  const g = new NxGraph();
  parseGraphml(file, {
    graph(directed) {
      if (directed) throw new Error(`${file}: directed GraphML is not supported`);
    },
    key() {},
    node(id, attrs) {
      g.addNode(id, attrs);
    },
    edge(s, t, attrs) {
      edgeElements++;
      g.addEdge(s, t, attrs);
    },
  });
  const edges: GraphmlGraph['edges'] = new Map();
  for (const [source, target, attrs] of g.edges()) edges.set(edgeKey(source, target), { source, target, attrs });
  return { directed: false, nodes: g.nodes, edges, edgeElements };
}

// --------------------------------------------------------------- GraphML write

function escCdata(s: string): string {
  if (!/[&<>]/.test(s)) return s;
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escAttr(s: string): string {
  if (!/[&<>"\r\n\t]/.test(s)) return s;
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\r/g, '&#13;')
    .replace(/\n/g, '&#10;')
    .replace(/\t/g, '&#09;');
}

/** (xml type, Python str(value)) as networkx GraphMLWriter.add_data gets them */
function pyTyped(g: NxGraph, scope: AttrScope, name: string, v: string | number | boolean): [string, string] {
  if (typeof v === 'string') return ['string', v];
  if (typeof v === 'boolean') return ['boolean', v ? 'True' : 'False'];
  if (g.isFloat(scope, name) || !Number.isInteger(v)) return ['double', pyFloatRepr(v)];
  return ['long', String(v)];
}

/**
 * nx.write_graphml(G, path) with networkx's ElementTree writer (lxml absent in
 * the upstream venv): key ids in first-use order, key elements prepended, the
 * writer's own indent(), ElementTree escaping, `' />'` for empty elements.
 * Written atomically.
 */
export function writeNxGraphml(g: NxGraph, file: string): void {
  // first pass: key ids, in the order GraphMLWriter.add_data creates them
  const keyIds = new Map<string, string>();
  const keyEls: string[] = [];
  const keyFor = (scope: AttrScope, name: string, type: string): string => {
    const k = `${name}\u0000${type}\u0000${scope}`;
    let id = keyIds.get(k);
    if (id === undefined) {
      id = `d${keyIds.size}`;
      keyIds.set(k, id);
      keyEls.push(`<key id="${id}" for="${scope}" attr.name="${escAttr(name)}" attr.type="${type}" />`);
    }
    return id;
  };
  for (const a of g.nodes.values()) for (const [k, v] of Object.entries(a)) keyFor('node', k, pyTyped(g, 'node', k, v)[0]);
  for (const [, , a] of g.edges()) for (const [k, v] of Object.entries(a)) keyFor('edge', k, pyTyped(g, 'edge', k, v)[0]);

  atomicWrite(file, (w) => {
    w.write(`<?xml version='1.0' encoding='utf-8'?>\n`);
    w.write(
      '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' +
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
    );
    for (let i = keyEls.length - 1; i >= 0; i--) w.write(`\n  ${keyEls[i]}`);
    const datas = (scope: AttrScope, a: GraphAttrs): string => {
      let s = '';
      for (const [k, v] of Object.entries(a)) {
        const [type, text] = pyTyped(g, scope, k, v);
        const id = keyIds.get(`${k}\u0000${type}\u0000${scope}`)!;
        s += text === '' ? `\n      <data key="${id}" />` : `\n      <data key="${id}">${escCdata(text)}</data>`;
      }
      return s;
    };
    const elem = (open: string, close: string, body: string) => (body ? `\n    ${open}>${body}\n    ${close}` : `\n    ${open} />`);
    let any = false;
    for (const [id, a] of g.nodes) {
      if (!any) w.write('\n  <graph edgedefault="undirected">');
      any = true;
      w.write(elem(`<node id="${escAttr(id)}"`, '</node>', datas('node', a)));
    }
    for (const [u, v, a] of g.edges()) {
      if (!any) w.write('\n  <graph edgedefault="undirected">');
      any = true;
      w.write(elem(`<edge source="${escAttr(u)}" target="${escAttr(v)}"`, '</edge>', datas('edge', a)));
    }
    // indent() gives the root element a tail of "\n" too
    w.write(any ? '\n  </graph>\n</graphml>\n' : '\n  <graph edgedefault="undirected" />\n</graphml>\n');
  });
}

// ------------------------------------------------------------------- storage

export class NetworkXStorage implements BaseGraphStorage {
  readonly namespace: string;
  readonly workspace: string;
  readonly global_config: GlobalConfig;
  readonly graphml_file: string;
  graph: NxGraph;

  static load_nx_graph(file: string): NxGraph | null {
    return existsSync(file) ? loadNxGraph(file) : null;
  }
  static write_nx_graph(graph: NxGraph, file: string): void {
    writeNxGraphml(graph, file);
  }

  constructor(opts: StorageOptions) {
    this.namespace = opts.namespace;
    this.workspace = opts.workspace ?? '';
    this.global_config = opts.global_config;
    mkdirSync(workspaceDir(opts), { recursive: true });
    this.graphml_file = graphmlPath(this.global_config.working_dir, this.namespace, this.workspace);
    // upstream loads in __post_init__, i.e. on construction
    this.graph = NetworkXStorage.load_nx_graph(this.graphml_file) ?? new NxGraph();
  }

  async initialize(): Promise<void> {}
  async finalize(): Promise<void> {}

  async has_node(node_id: string): Promise<boolean> {
    return this.graph.hasNode(node_id);
  }
  async has_edge(s: string, t: string): Promise<boolean> {
    return this.graph.hasEdge(s, t);
  }
  /** The live attribute dict, as upstream returns it. */
  async get_node(node_id: string): Promise<NodeData | null> {
    return this.graph.nodes.get(node_id) ?? null;
  }
  async node_degree(node_id: string): Promise<number> {
    return this.graph.hasNode(node_id) ? this.graph.degree(node_id) : 0;
  }
  async edge_degree(src_id: string, tgt_id: string): Promise<number> {
    return (await this.node_degree(src_id)) + (await this.node_degree(tgt_id));
  }
  async get_edge(s: string, t: string): Promise<EdgeData | null> {
    return this.graph.getEdge(s, t) ?? null;
  }
  async get_node_edges(source_node_id: string): Promise<[string, string][] | null> {
    if (!this.graph.hasNode(source_node_id)) return null;
    return [...this.graph.edges([source_node_id])].map(([u, v]) => [u, v]);
  }

  async upsert_node(node_id: string, node_data: NodeData): Promise<void> {
    this.graph.addNode(node_id, node_data);
  }
  async upsert_edge(s: string, t: string, edge_data: EdgeData): Promise<void> {
    this.graph.addEdge(s, t, edge_data);
  }
  async upsert_nodes_batch(nodes: [string, NodeData][]): Promise<void> {
    for (const [id, d] of nodes) this.graph.addNode(id, d);
  }
  async has_nodes_batch(node_ids: string[]): Promise<Set<string>> {
    return new Set(node_ids.filter((id) => this.graph.hasNode(id)));
  }
  async upsert_edges_batch(edges: [string, string, EdgeData][]): Promise<void> {
    for (const [s, t, d] of edges) this.graph.addEdge(s, t, d);
  }

  // BaseGraphStorage default batch implementations (base.py), one call at a time
  async get_nodes_batch(node_ids: string[]): Promise<Map<string, NodeData>> {
    const out = new Map<string, NodeData>();
    for (const id of node_ids) {
      const n = await this.get_node(id);
      if (n !== null) out.set(id, n);
    }
    return out;
  }
  async node_degrees_batch(node_ids: string[]): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    for (const id of node_ids) out.set(id, await this.node_degree(id));
    return out;
  }
  async edge_degrees_batch(pairs: [string, string][]): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    for (const [s, t] of pairs) out.set(edgePairKey(s, t), await this.edge_degree(s, t));
    return out;
  }
  async get_edges_batch(pairs: { src: string; tgt: string }[]): Promise<Map<string, EdgeData>> {
    const out = new Map<string, EdgeData>();
    for (const { src, tgt } of pairs) {
      const e = await this.get_edge(src, tgt);
      if (e !== null) out.set(edgePairKey(src, tgt), e);
    }
    return out;
  }
  async get_nodes_edges_batch(node_ids: string[]): Promise<Map<string, [string, string][]>> {
    const out = new Map<string, [string, string][]>();
    for (const id of node_ids) out.set(id, (await this.get_node_edges(id)) ?? []);
    return out;
  }

  async delete_node(node_id: string): Promise<void> {
    if (this.graph.hasNode(node_id)) this.graph.removeNode(node_id);
  }
  async remove_nodes(nodes: string[]): Promise<void> {
    for (const n of nodes) if (this.graph.hasNode(n)) this.graph.removeNode(n);
  }
  async remove_edges(edges: [string, string][]): Promise<void> {
    for (const [s, t] of edges) if (this.graph.hasEdge(s, t)) this.graph.removeEdge(s, t);
  }

  async get_all_labels(): Promise<string[]> {
    return [...new Set(this.graph.nodes.keys())].sort(pyCompare);
  }

  /** Degree-descending; Python's sort(reverse=True) keeps node order for ties. */
  private nodesByDegree(): [string, number][] {
    return [...this.graph.nodes.keys()].map((n) => [n, this.graph.degree(n)] as [string, number]).sort((a, b) => b[1] - a[1]);
  }

  async get_popular_labels(limit = 300): Promise<string[]> {
    return this.nodesByDegree()
      .slice(0, limit)
      .map(([n]) => n);
  }

  async search_labels(query: string, limit = 50): Promise<string[]> {
    const q = pyStrip(pyLower(query));
    if (!q) return [];
    const matches: [string, number][] = [];
    for (const node of this.graph.nodes.keys()) {
      const lower = pyLower(node);
      if (!lower.includes(q)) continue;
      let score: number;
      if (lower === q) score = 1000;
      else if (lower.startsWith(q)) score = 500;
      else {
        score = 100 - [...node].length;
        if (lower.includes(` ${q}`) || lower.includes(`_${q}`)) score += 50;
      }
      matches.push([node, score]);
    }
    matches.sort((a, b) => b[1] - a[1] || pyCompare(a[0], b[0]));
    return matches.slice(0, limit).map(([n]) => n);
  }

  /**
   * Upstream builds the result from `graph.subgraph(nodes)`, whose iteration
   * order follows a Python set when the selection is under half the graph, i.e.
   * it varies between Python runs. Here nodes and edges follow graph order; the
   * node and edge sets are what is compared with upstream.
   */
  async get_knowledge_graph(node_label: string, max_depth = 3, max_nodes: number | null = null): Promise<KnowledgeGraph> {
    const cfgMax = this.global_config.max_graph_nodes ?? 1000;
    const limit = max_nodes === null ? cfgMax : Math.min(max_nodes, cfgMax);
    const g = this.graph;
    const result: KnowledgeGraph = { nodes: [], edges: [], is_truncated: false };
    let selected: string[];
    if (node_label === '*') {
      const sorted = this.nodesByDegree();
      if (sorted.length > limit) result.is_truncated = true;
      selected = sorted.slice(0, limit).map(([n]) => n);
    } else {
      if (!g.hasNode(node_label)) return result;
      const bfs: string[] = [];
      const visited = new Set<string>();
      let queue: [string, number, number][] = [[node_label, 0, g.degree(node_label)]];
      let qi = 0;
      let unexplored = false;
      while (qi < queue.length && bfs.length < limit) {
        const depth = queue[qi]![1];
        const level: [string, number, number][] = [];
        while (qi < queue.length && queue[qi]![1] === depth) level.push(queue[qi++]!);
        level.sort((a, b) => b[2] - a[2]);
        for (const [node, d] of level) {
          if (!visited.has(node)) {
            visited.add(node);
            bfs.push(node);
            const unvisited = [...g.neighbors(node)].filter((n) => !visited.has(n));
            if (d < max_depth) for (const n of unvisited) queue.push([n, d + 1, g.degree(n)]);
            else if (unvisited.length) unexplored = true;
          }
          if (bfs.length >= limit) break;
        }
        if (qi > 4096 && qi * 2 > queue.length) {
          queue = queue.slice(qi);
          qi = 0;
        }
      }
      if ((qi < queue.length && bfs.length >= limit) || unexplored) {
        if (bfs.length >= limit) result.is_truncated = true;
      }
      selected = bfs;
    }
    const keep = new Set(selected);
    const order = [...g.nodes.keys()].filter((n) => keep.has(n));
    for (const n of order) {
      result.nodes.push({ id: n, labels: [n], properties: { ...g.nodes.get(n)! } } satisfies KnowledgeGraphNode);
    }
    const seen = new Set<string>();
    const visitedNodes = new Set<string>();
    for (const n of order) {
      for (const [nbr, dd] of g.adj.get(n)!) {
        if (!keep.has(nbr) || visitedNodes.has(nbr)) continue;
        let [s, t] = [n, nbr];
        if (pyCompare(s, t) > 0) [s, t] = [t, s];
        const id = `${s}-${t}`;
        if (seen.has(id)) continue;
        seen.add(id);
        result.edges.push({ id, type: 'DIRECTED', source: s, target: t, properties: { ...dd } } satisfies KnowledgeGraphEdge);
      }
      visitedNodes.add(n);
    }
    return result;
  }

  async get_all_nodes(): Promise<NodeData[]> {
    return [...this.graph.nodes].map(([id, a]) => ({ ...a, id }));
  }
  async get_all_edges(): Promise<EdgeData[]> {
    return [...this.graph.edges()].map(([u, v, a]) => ({ ...a, source: u, target: v }));
  }

  async index_done_callback(): Promise<boolean> {
    NetworkXStorage.write_nx_graph(this.graph, this.graphml_file);
    return true;
  }

  async drop(): Promise<DropResult> {
    try {
      rmSync(this.graphml_file, { force: true });
      this.graph = new NxGraph();
      return { status: 'success', message: 'data dropped' };
    } catch (e) {
      return { status: 'error', message: String((e as Error).message ?? e) };
    }
  }
}

/** Python str.lower() for the ids LightRAG produces; JS differs only on a few special casings. */
function pyLower(s: string): string {
  return s.toLowerCase();
}
