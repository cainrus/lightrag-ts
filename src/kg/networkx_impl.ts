// Port of lightrag/kg/networkx_impl.py - B0: GraphML reader compatible with
// what networkx 3.x `write_graphml` emits for LightRAG's undirected nx.Graph.
// The in-memory graph (graphology) and the writer land in B1.
import { closeSync, openSync, readSync } from 'node:fs';
import { join } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import { SaxesParser } from 'saxes';

export type GraphAttrs = Record<string, string | number | boolean>;

export type GraphmlGraph = {
  directed: boolean;
  nodes: Map<string, GraphAttrs>;
  /** keyed by `edgeKey(source, target)`; for undirected graphs parallel edges collapse like nx.Graph */
  edges: Map<string, { source: string; target: string; attrs: GraphAttrs }>;
  /** <edge> elements seen in the file (before collapsing) */
  edgeElements: number;
};

/** `<working_dir>[/<workspace>]/graph_<namespace>.graphml` */
export function graphmlPath(workingDir: string, namespace: string, workspace = ''): string {
  return join(workspace ? join(workingDir, workspace) : workingDir, `graph_${namespace}.graphml`);
}

export function edgeKey(source: string, target: string, directed = false): string {
  if (!directed && target < source) [source, target] = [target, source];
  return `${source}\u0000${target}`;
}

type KeyDef = { for: string; name: string; type: string; default?: string };

function convert(type: string, raw: string): string | number | boolean {
  switch (type) {
    case 'int':
    case 'long':
    case 'float':
    case 'double':
      return Number(raw);
    case 'boolean':
      // networkx: {"true": True, "false": False, "0": False, "1": True}
      return raw.toLowerCase() === 'true' || raw === '1';
    default:
      return raw;
  }
}

export function readGraphml(file: string): GraphmlGraph {
  const keys = new Map<string, KeyDef>();
  const g: GraphmlGraph = { directed: false, nodes: new Map(), edges: new Map(), edgeElements: 0 };
  let current: { kind: 'node'; id: string; attrs: GraphAttrs } | { kind: 'edge'; source: string; target: string; attrs: GraphAttrs } | null = null;
  let currentKey: KeyDef | undefined;
  let keyDefault: KeyDef | undefined;
  let text = '';

  const p = new SaxesParser({ xmlns: false });
  p.on('opentag', (t) => {
    const a = t.attributes as Record<string, string>;
    switch (t.name) {
      case 'key':
        keyDefault = { for: a.for ?? 'all', name: a['attr.name'] ?? a.id!, type: a['attr.type'] ?? 'string' };
        keys.set(a.id!, keyDefault);
        break;
      case 'default':
        text = '';
        break;
      case 'graph':
        g.directed = a.edgedefault === 'directed';
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
      case 'default':
        if (keyDefault) keyDefault.default = text;
        break;
      case 'data':
        if (current && currentKey) current.attrs[currentKey.name] = convert(currentKey.type, text);
        currentKey = undefined;
        break;
      case 'node':
        if (current?.kind === 'node') g.nodes.set(current.id, withDefaults(current.attrs, keys, 'node'));
        current = null;
        break;
      case 'edge':
        if (current?.kind === 'edge') {
          g.edgeElements++;
          // networkx adds endpoints that have no <node> element
          for (const id of [current.source, current.target]) if (!g.nodes.has(id)) g.nodes.set(id, {});
          const k = edgeKey(current.source, current.target, g.directed);
          const prev = g.edges.get(k);
          // nx.Graph.add_edge on an existing edge updates its attribute dict
          const attrs = { ...(prev?.attrs ?? {}), ...withDefaults(current.attrs, keys, 'edge') };
          g.edges.set(k, { source: prev?.source ?? current.source, target: prev?.target ?? current.target, attrs });
        }
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
  return g;
}

function withDefaults(attrs: GraphAttrs, keys: Map<string, KeyDef>, kind: 'node' | 'edge'): GraphAttrs {
  for (const k of keys.values()) {
    if (k.default !== undefined && (k.for === kind || k.for === 'all') && !(k.name in attrs)) {
      attrs[k.name] = convert(k.type, k.default);
    }
  }
  return attrs;
}
