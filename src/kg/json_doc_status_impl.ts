// Port of lightrag/kg/json_doc_status_impl.py (JsonDocStatusStorage).
// Same file as a KV store (kv_store_doc_status.json, write_json indent=2).
//
// Deviation: get_docs_paginated(sort_field="file_path") sorts by `lower()`;
// upstream uses pypinyin when installed, so CJK file paths order differently.
import type { DocProcessingStatus, DocStatusValue, KvRecord } from '../base.ts';
import { DOC_STATUS_VALUES, PyTypeError, makeDocProcessingStatus } from '../base.ts';
import { pyCompare } from '../utils.ts';
import { JsonFileStorage, pyTruthy } from './json_kv_impl.ts';

export class JsonDocStatusStorage extends JsonFileStorage {
  protected readonly className = 'JsonDocStatusStorage';

  async get_by_ids(ids: string[]): Promise<(KvRecord | null)[]> {
    this.assertReady();
    return ids.map((id) => {
      const d = this.data.get(id);
      return d !== undefined && pyTruthy(d) ? { ...d } : null;
    });
  }

  /** Upstream returns the live dict, not a copy. */
  async get_by_id(id: string): Promise<KvRecord | null> {
    return this.data.get(id) ?? null;
  }

  async get_status_counts(): Promise<Record<string, number>> {
    this.assertReady();
    const counts: Record<string, number> = Object.fromEntries(DOC_STATUS_VALUES.map((s) => [s, 0]));
    for (const doc of this.data.values()) {
      const s = doc.status as string;
      // counts[doc["status"]] += 1 raises KeyError on an unknown status
      if (!Object.hasOwn(counts, s)) throw new Error(`KeyError: '${s}'`);
      counts[s]!++;
    }
    return counts;
  }

  async get_all_status_counts(): Promise<Record<string, number>> {
    const counts = await this.get_status_counts();
    counts.all = Object.values(counts).reduce((a, b) => a + b, 0);
    return counts;
  }

  async get_docs_by_status(status: DocStatusValue): Promise<Map<string, DocProcessingStatus>> {
    return this.get_docs_by_statuses([status]);
  }

  async get_docs_by_statuses(statuses: DocStatusValue[]): Promise<Map<string, DocProcessingStatus>> {
    const out = new Map<string, DocProcessingStatus>();
    if (statuses.length === 0) return out;
    const want = new Set<string>(statuses);
    for (const [k, v] of this.data) {
      if (!want.has(v.status as string)) continue;
      try {
        out.set(k, makeDocProcessingStatus(normalizeDoc(v)));
      } catch (e) {
        // upstream catches (KeyError, TypeError) here and skips the document
        if (!(e instanceof PyTypeError)) throw e;
      }
    }
    return out;
  }

  /** Upstream catches only KeyError here, so a malformed document raises (TypeError). */
  async get_docs_by_track_id(track_id: string): Promise<Map<string, DocProcessingStatus>> {
    const out = new Map<string, DocProcessingStatus>();
    for (const [k, v] of this.data) {
      if (v.track_id === track_id) out.set(k, makeDocProcessingStatus(normalizeDoc(v)));
    }
    return out;
  }

  /** Marks `chunks_list`, stores, then persists immediately (upstream calls index_done_callback). */
  async upsert(data: Record<string, KvRecord>): Promise<void> {
    const entries = Object.entries(data);
    if (entries.length === 0) return;
    this.assertReady();
    for (const [, v] of entries) if (!('chunks_list' in v)) v.chunks_list = [];
    for (const [k, v] of entries) this.data.set(k, v);
    this.dirty = true;
    await this.index_done_callback();
  }

  async get_docs_paginated(
    status_filter: DocStatusValue | null = null,
    page = 1,
    page_size = 50,
    sort_field = 'updated_at',
    sort_direction = 'desc',
  ): Promise<[[string, DocProcessingStatus][], number]> {
    if (page < 1) page = 1;
    if (page_size < 10) page_size = 10;
    else if (page_size > 200) page_size = 200;
    if (!['created_at', 'updated_at', 'id', 'file_path'].includes(sort_field)) sort_field = 'updated_at';
    if (!['asc', 'desc'].includes(sort_direction.toLowerCase())) sort_direction = 'desc';

    const all: { id: string; doc: DocProcessingStatus; key: string }[] = [];
    for (const [id, v] of this.data) {
      if (status_filter !== null && v.status !== status_filter) continue;
      const doc = makeDocProcessingStatus(normalizeDoc(v));
      let key: string;
      if (sort_field === 'id') key = id;
      else if (sort_field === 'file_path') key = doc.file_path ? doc.file_path.toLowerCase() : '';
      else key = (doc[sort_field as 'created_at' | 'updated_at'] ?? '') as string;
      all.push({ id, doc, key });
    }
    const desc = sort_direction.toLowerCase() === 'desc';
    // Python's sort(reverse=True) stays stable: equal keys keep insertion order
    all.sort((a, b) => (desc ? -pyCompare(a.key, b.key) : pyCompare(a.key, b.key)));
    const start = (page - 1) * page_size;
    return [all.slice(start, start + page_size).map((x) => [x.id, x.doc]), all.length];
  }

  async get_doc_by_file_path(file_path: string): Promise<KvRecord | null> {
    this.assertReady();
    for (const v of this.data.values()) if (v.file_path === file_path) return v;
    return null;
  }
}

/** The shared preamble of every DocProcessingStatus(**data) call upstream. */
function normalizeDoc(v: KvRecord): KvRecord {
  const d: KvRecord = { ...v };
  delete d.content;
  if (!pyTruthy(d.file_path)) d.file_path = 'no-file-path';
  if (!('metadata' in d)) d.metadata = {};
  if (!('error_msg' in d)) d.error_msg = null;
  return d;
}
