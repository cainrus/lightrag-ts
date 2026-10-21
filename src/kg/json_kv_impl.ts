// Port of lightrag/kg/json_kv_impl.py - B0: file layout and a read-only loader.
// The storage class (upsert/delete/index_done_callback) lands in B1.
import { join } from 'node:path';
import { iterateJsonObject } from '../internal/json_stream.ts';

/** `<working_dir>[/<workspace>]/kv_store_<namespace>.json`, as JsonKVStorage.__post_init__ builds it. */
export function kvStorePath(workingDir: string, namespace: string, workspace = ''): string {
  return join(workspace ? join(workingDir, workspace) : workingDir, `kv_store_${namespace}.json`);
}

/** Load a kv_store_*.json file. Streams the file, so it is not bounded by V8's max string length. */
export function readJsonKv<V = Record<string, unknown>>(file: string): Map<string, V> {
  const out = new Map<string, V>();
  for (const [k, v] of iterateJsonObject(file)) out.set(k, v as V);
  return out;
}

/** Visit every record without keeping the whole store in memory. */
export function forEachJsonKv<V = Record<string, unknown>>(file: string, fn: (key: string, value: V) => void): number {
  let n = 0;
  for (const [k, v] of iterateJsonObject(file)) {
    fn(k, v as V);
    n++;
  }
  return n;
}
