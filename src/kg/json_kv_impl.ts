// Port of lightrag/kg/json_kv_impl.py (JsonKVStorage).
// One process owns the store: upstream's shared_storage (namespace locks, cross-
// process update flags) collapses to a `dirty` flag, see SCOPE.md.
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import type { BaseKVStorage, Clock, DropResult, GlobalConfig, KvRecord, StorageOptions } from '../base.ts';
import { systemClock, workspaceDir } from '../base.ts';
import { iterateJsonObject } from '../internal/json_stream.ts';
import { writeJsonPy } from '../internal/py_json.ts';
import { generateCacheKey } from '../utils.ts';

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

/** upstream utils.load_json: null when the file is missing. */
export function loadJsonKv(file: string): Map<string, KvRecord> | null {
  return existsSync(file) ? readJsonKv<KvRecord>(file) : null;
}

/** Python truthiness of a stored value (`if result:`): an empty dict is falsy. */
export function pyTruthy(v: unknown): boolean {
  if (v === null || v === undefined || v === false || v === 0 || v === '') return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return true;
}

export type JsonStorageOptions = StorageOptions & { clock?: Clock };

/** Shared file handling of JsonKVStorage and JsonDocStatusStorage (both persist via write_json). */
export abstract class JsonFileStorage {
  readonly namespace: string;
  readonly workspace: string;
  readonly global_config: GlobalConfig;
  readonly file_name: string;
  protected data: Map<string, KvRecord> = new Map();
  protected dirty = false;
  protected clock: Clock;
  private initialized = false;

  constructor(opts: JsonStorageOptions) {
    this.namespace = opts.namespace;
    this.workspace = opts.workspace ?? '';
    this.global_config = opts.global_config;
    this.clock = opts.clock ?? systemClock;
    const dir = workspaceDir(opts);
    mkdirSync(dir, { recursive: true });
    this.file_name = kvStorePath(this.global_config.working_dir, this.namespace, this.workspace);
  }

  protected abstract readonly className: string;

  async initialize(): Promise<void> {
    if (this.initialized) return;
    let loaded = loadJsonKv(this.file_name) ?? new Map<string, KvRecord>();
    loaded = this.migrate(loaded);
    for (const [k, v] of loaded) this.data.set(k, v);
    this.initialized = true;
  }

  protected migrate(loaded: Map<string, KvRecord>): Map<string, KvRecord> {
    return loaded;
  }

  async finalize(): Promise<void> {}

  protected assertReady(): void {
    if (!this.initialized) throw new Error(`StorageNotInitializedError: ${this.className}`);
  }

  /** Port-only: make the next index_done_callback write (upstream flips a shared update flag). */
  mark_dirty(): void {
    this.dirty = true;
  }

  /** Persist when dirty. Returns whether a write happened (upstream returns None). */
  async index_done_callback(): Promise<boolean> {
    if (!this.dirty) return false;
    this.persist(this.data);
    this.dirty = false;
    return true;
  }

  protected persist(data: Map<string, KvRecord>): void {
    const { sanitized } = writeJsonPy(data, this.file_name);
    // upstream reloads the cleaned file into memory after a sanitizing write
    if (sanitized) {
      const cleaned = loadJsonKv(this.file_name);
      if (cleaned) this.data = cleaned;
    }
  }

  async filter_keys(keys: Iterable<string>): Promise<Set<string>> {
    this.assertReady();
    const out = new Set<string>();
    for (const k of keys) if (!this.data.has(k)) out.add(k);
    return out;
  }

  async delete(ids: string[]): Promise<void> {
    this.assertReady();
    let any = false;
    for (const id of ids) if (this.data.delete(id)) any = true;
    if (any) this.dirty = true;
  }

  async is_empty(): Promise<boolean> {
    this.assertReady();
    return this.data.size === 0;
  }

  async drop(): Promise<DropResult> {
    try {
      this.data.clear();
      this.dirty = true;
      await this.index_done_callback();
      return { status: 'success', message: 'data dropped' };
    } catch (e) {
      return { status: 'error', message: String((e as Error).message ?? e) };
    }
  }

  /** Number of records (Python `len(self._data)`). */
  get size(): number {
    return this.data.size;
  }

  keys(): IterableIterator<string> {
    return this.data.keys();
  }
}

export class JsonKVStorage extends JsonFileStorage implements BaseKVStorage {
  protected readonly className = 'JsonKVStorage';

  protected override migrate(loaded: Map<string, KvRecord>): Map<string, KvRecord> {
    if (!this.namespace.endsWith('_cache')) return loaded;
    const { data: migrated, count } = migrateLegacyCacheStructure(loaded);
    if (count === 0) return migrated;
    // upstream persists the migration immediately
    const { sanitized } = writeJsonPy(migrated, this.file_name);
    return sanitized ? (loadJsonKv(this.file_name) ?? migrated) : migrated;
  }

  async get_by_id(id: string): Promise<KvRecord | null> {
    this.assertReady();
    const result = this.data.get(id);
    if (result === undefined) return null;
    if (!pyTruthy(result)) return result;
    return withDefaults(result, id);
  }

  async get_by_ids(ids: string[]): Promise<(KvRecord | null)[]> {
    this.assertReady();
    return ids.map((id) => {
      const d = this.data.get(id);
      return d !== undefined && pyTruthy(d) ? withDefaults(d, id) : null;
    });
  }

  /** Mutates the given records like upstream does (timestamps, `_id`, `llm_cache_list`). */
  async upsert(data: Record<string, KvRecord>): Promise<void> {
    const entries = Object.entries(data);
    if (entries.length === 0) return;
    const now = this.clock();
    this.assertReady();
    for (const [k, v] of entries) {
      if (this.namespace.endsWith('text_chunks') && !('llm_cache_list' in v)) v.llm_cache_list = [];
      if (this.data.has(k)) v.update_time = now;
      else {
        v.create_time = now;
        v.update_time = now;
      }
      v._id = k;
    }
    for (const [k, v] of entries) this.data.set(k, v);
    this.dirty = true;
  }

  override async finalize(): Promise<void> {
    if (this.namespace.endsWith('_cache')) await this.index_done_callback();
  }
}

function withDefaults(d: KvRecord, id: string): KvRecord {
  const r: KvRecord = { ...d };
  if (!('create_time' in r)) r.create_time = 0;
  if (!('update_time' in r)) r.update_time = 0;
  r._id = id;
  return r;
}

/** JsonKVStorage._migrate_legacy_cache_structure: `{mode: {hash: entry}}` -> `{mode:type:hash: entry}`. */
export function migrateLegacyCacheStructure(data: Map<string, KvRecord>): { data: Map<string, KvRecord>; count: number } {
  if (data.size === 0) return { data, count: 0 };
  const first = data.keys().next().value as string;
  if (first.includes(':') && first.split(':').length === 3) return { data, count: 0 };
  const out = new Map<string, KvRecord>();
  let count = 0;
  for (const [key, value] of data) {
    const vals = value && typeof value === 'object' && !Array.isArray(value) ? Object.values(value) : null;
    // Python all() over an empty dict is True: an empty mode dict is dropped
    const legacy = vals !== null && vals.every((v) => v !== null && typeof v === 'object' && !Array.isArray(v) && 'return' in v);
    if (legacy) {
      for (const [hash, entry] of Object.entries(value as Record<string, KvRecord>)) {
        const type = (entry.cache_type as string | undefined) ?? 'extract';
        out.set(generateCacheKey(key, type, hash), entry);
        count++;
      }
    } else out.set(key, value);
  }
  return { data: out, count };
}
