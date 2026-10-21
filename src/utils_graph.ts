// Port of lightrag/utils_graph.py: delete, edit, create and merge entities and
// relations by hand, keeping the graph, both vector stores and the chunk
// tracking stores in step.
//
// Locks: upstream takes get_storage_keyed_lock over the entity names; here the
// whole operation runs under withGraphLock(graph), the lock every graph writer
// of this port takes (operate.ts). Internal helpers are looked up through
// `utilsGraphHooks`, the module-attribute lookup Python tests monkeypatch.
import type { BaseGraphStorage, BaseKVStorage, BaseVectorStorage, DeletionResult, EdgeData, NodeData } from './base.ts';
import { GRAPH_FIELD_SEP } from './constants.ts';
import { pyError } from './exceptions.ts';
import { pyTruthy } from './kg/json_kv_impl.ts';
import { withGraphLock } from './operate.ts';
import { computeIncrementalChunkIds, computeMdhashId, makeRelationChunkKey, makeRelationVdbIds, pyCompare, pyFloatParse, pyStr, pyStrip } from './utils.ts';

export type GraphLog = (level: 'debug' | 'info' | 'warning' | 'error', msg: string) => void;
type Opts = { log?: GraphLog };

/** The graph methods utils_graph calls. */
export type EditGraph = Pick<BaseGraphStorage, 'has_node' | 'get_node' | 'upsert_node' | 'get_node_edges' | 'get_edge' | 'upsert_edge' | 'delete_node' | 'has_edge' | 'remove_edges' | 'index_done_callback'>;
/** The vector store methods utils_graph calls. */
export type EditVdb = Pick<BaseVectorStorage, 'upsert' | 'delete' | 'get_by_id' | 'index_done_callback'> & Partial<Pick<BaseVectorStorage, 'delete_entity' | 'delete_entity_relation'>>;
/** The chunk-tracking store methods utils_graph calls. */
export type EditKv = Pick<BaseKVStorage, 'get_by_id' | 'upsert' | 'delete' | 'index_done_callback'>;

type Dict = Record<string, unknown>;
const noLog: GraphLog = () => {};
const sorted2 = (a: string, b: string): [string, string] => [a, b].sort(pyCompare) as [string, string];
const splitIds = (s: unknown) => String(s ?? '').split(GRAPH_FIELD_SEP).filter((c) => c);
const get = (d: Dict, k: string, dflt: unknown): unknown => (k in d ? d[k] : dflt);
const errMsg = (e: unknown) => (e as Error)?.message ?? String(e);
/** Python `float(x)` for the values a weight can hold. */
function pyFloat(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  const f = typeof v === 'string' ? pyFloatParse(v) : null;
  if (f === null) throw pyError('ValueError', `could not convert string to float: ${pyStr(v)}`);
  return f;
}

/** Runs under the graph's write lock (none when there is no graph object, as in the upstream tests' stubs). */
function locked<T>(graph: EditGraph | null | undefined, fn: () => Promise<T>): Promise<T> {
  return graph && typeof graph === 'object' ? withGraphLock(graph as BaseGraphStorage, fn) : fn();
}

function requireNonEmptyDescription(description: unknown, operation: string, objectType: string): void {
  if (description === null || description === undefined || !pyStrip(pyStr(description))) {
    throw pyError('ValueError', `${objectType[0]!.toUpperCase()}${objectType.slice(1).toLowerCase()} description cannot be empty for ${operation} operation`);
  }
}

/** utils_graph._persist_graph_updates */
async function persistGraphUpdates(...storages: ({ index_done_callback(): Promise<unknown> } | null | undefined)[]): Promise<void> {
  await Promise.all(storages.filter((s) => s !== null && s !== undefined).map((s) => s!.index_done_callback()));
}

/** utils_graph.adelete_by_entity */
export async function adeleteByEntity(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb,
  entityName: string,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  { log = noLog }: Opts = {},
): Promise<DeletionResult> {
  return locked(graph, async () => {
    try {
      if (!(await graph.has_node(entityName))) {
        log('warning', `Entity '${entityName}' not found.`);
        return { status: 'not_found', doc_id: entityName, message: `Entity '${entityName}' not found.`, status_code: 404, file_path: null };
      }
      const edges = await graph.get_node_edges(entityName);
      const related = edges ? edges.length : 0;
      if (entityChunksStorage) {
        await entityChunksStorage.delete([entityName]);
        log('info', `Entity Delete: removed chunk tracking for \`${entityName}\``);
      }
      if (relationChunksStorage && edges && edges.length) {
        const keys = edges.map(([s, t]) => makeRelationChunkKey(...sorted2(s, t)));
        if (keys.length) {
          await relationChunksStorage.delete(keys);
          log('info', `Entity Delete: removed chunk tracking for ${keys.length} relations`);
        }
      }
      await entitiesVdb.delete_entity!(entityName);
      await relationshipsVdb.delete_entity_relation!(entityName);
      await graph.delete_node(entityName);
      const message = `Entity Delete: remove '${entityName}' and its ${related} relations`;
      log('info', message);
      await persistGraphUpdates(entitiesVdb, relationshipsVdb, graph, entityChunksStorage, relationChunksStorage);
      return { status: 'success', doc_id: entityName, message, status_code: 200, file_path: null };
    } catch (e) {
      const message = `Error while deleting entity '${entityName}': ${errMsg(e)}`;
      log('error', message);
      return { status: 'fail', doc_id: entityName, message, status_code: 500, file_path: null };
    }
  });
}

/** utils_graph.adelete_by_relation */
export async function adeleteByRelation(
  graph: EditGraph,
  relationshipsVdb: EditVdb,
  sourceEntity: string,
  targetEntity: string,
  relationChunksStorage: EditKv | null = null,
  { log = noLog }: Opts = {},
): Promise<DeletionResult> {
  const relationStr = `${sourceEntity} -> ${targetEntity}`;
  if (pyCompare(sourceEntity, targetEntity) > 0) [sourceEntity, targetEntity] = [targetEntity, sourceEntity];
  return locked(graph, async () => {
    try {
      if (!(await graph.has_edge(sourceEntity, targetEntity))) {
        const message = `Relation from '${sourceEntity}' to '${targetEntity}' does not exist`;
        log('warning', message);
        return { status: 'not_found', doc_id: relationStr, message, status_code: 404, file_path: null };
      }
      if (relationChunksStorage) {
        await relationChunksStorage.delete([makeRelationChunkKey(...sorted2(sourceEntity, targetEntity))]);
        log('info', `Relation Delete: removed chunk tracking for \`${sourceEntity}\`~\`${targetEntity}\``);
      }
      await relationshipsVdb.delete([computeMdhashId(sourceEntity + targetEntity, 'rel-'), computeMdhashId(targetEntity + sourceEntity, 'rel-')]);
      await graph.remove_edges([[sourceEntity, targetEntity]]);
      const message = `Relation Delete: \`${sourceEntity}\`~\`${targetEntity}\` deleted successfully`;
      log('info', message);
      await persistGraphUpdates(relationshipsVdb, graph, relationChunksStorage);
      return { status: 'success', doc_id: relationStr, message, status_code: 200, file_path: null };
    } catch (e) {
      const message = `Error while deleting relation from '${sourceEntity}' to '${targetEntity}': ${errMsg(e)}`;
      log('error', message);
      return { status: 'fail', doc_id: relationStr, message, status_code: 500, file_path: null };
    }
  });
}

/** utils_graph._edit_entity_impl: the edit without taking the lock. */
async function editEntityImpl(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb,
  entityName: string,
  updatedData: Dict,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  log: GraphLog = noLog,
): Promise<Dict> {
  const newEntityName = (get(updatedData, 'entity_name', entityName) as string) ?? entityName;
  const isRenaming = newEntityName !== entityName;
  const originalEntityName = entityName;

  if (!(await graph.has_node(entityName))) throw pyError('ValueError', `Entity '${entityName}' does not exist`);
  const nodeData = ((await graph.get_node(entityName)) ?? {}) as Dict;
  if (isRenaming && (await graph.has_node(newEntityName))) {
    throw pyError('ValueError', `Entity name '${newEntityName}' already exists, cannot rename`);
  }
  const newNodeData: Dict = { ...nodeData, ...updatedData };
  newNodeData.entity_id = newEntityName;
  delete newNodeData.entity_name;

  const relationsToUpdate: [string, string, Dict][] = [];
  if (isRenaming) {
    log('info', `Entity Edit: renaming \`${entityName}\` to \`${newEntityName}\``);
    await graph.upsert_node(newEntityName, newNodeData as NodeData);
    const relationsToDelete: string[] = [];
    const edges = await graph.get_node_edges(entityName);
    for (const [source, target] of edges ?? []) {
      const edgeData = (await graph.get_edge(source, target)) as Dict | null;
      if (!edgeData || !pyTruthy(edgeData)) continue;
      relationsToDelete.push(computeMdhashId(source + target, 'rel-'), computeMdhashId(target + source, 'rel-'));
      if (source === entityName) {
        await graph.upsert_edge(newEntityName, target, edgeData as EdgeData);
        relationsToUpdate.push([newEntityName, target, edgeData]);
      } else {
        await graph.upsert_edge(source, newEntityName, edgeData as EdgeData);
        relationsToUpdate.push([source, newEntityName, edgeData]);
      }
    }
    await graph.delete_node(entityName);
    await entitiesVdb.delete([computeMdhashId(entityName, 'ent-')]);
    await relationshipsVdb.delete(relationsToDelete);
    for (const [src, tgt, edgeData] of relationsToUpdate) {
      const [ns, nt] = sorted2(src, tgt);
      const description = get(edgeData, 'description', '');
      const keywords = get(edgeData, 'keywords', '');
      await relationshipsVdb.upsert({
        [computeMdhashId(ns + nt, 'rel-')]: {
          content: `${ns}\t${nt}\n${pyStr(keywords)}\n${pyStr(description)}`,
          src_id: ns,
          tgt_id: nt,
          source_id: get(edgeData, 'source_id', ''),
          description,
          keywords,
          weight: pyFloat(get(edgeData, 'weight', 1.0)),
        },
      });
    }
    entityName = newEntityName;
  } else {
    await graph.upsert_node(entityName, newNodeData as NodeData);
  }

  const description = get(newNodeData, 'description', '');
  await entitiesVdb.upsert({
    [computeMdhashId(entityName, 'ent-')]: {
      content: `${entityName}\n${pyStr(description)}`,
      entity_name: entityName,
      source_id: get(newNodeData, 'source_id', ''),
      description,
      entity_type: get(newNodeData, 'entity_type', ''),
    },
  });

  if (entityChunksStorage || relationChunksStorage) {
    if (entityChunksStorage) {
      const storageKey = isRenaming ? originalEntityName : entityName;
      const stored = await entityChunksStorage.get_by_id(storageKey);
      const hasStored = !!stored && typeof stored === 'object' && pyTruthy(stored) && pyTruthy(stored.chunk_ids);
      const oldIds = splitIds(get(nodeData, 'source_id', ''));
      const newIds = splitIds(get(newNodeData, 'source_id', ''));
      const changed = !sameSet(oldIds, newIds);
      if (changed || !hasStored || isRenaming) {
        let existing: string[] = hasStored ? ((stored!.chunk_ids as string[]) ?? []).filter((c) => c) : [];
        if (!existing.length) existing = [...oldIds];
        const updated = computeIncrementalChunkIds(existing, oldIds, newIds);
        if (isRenaming) await entityChunksStorage.delete([originalEntityName]);
        await entityChunksStorage.upsert({ [entityName]: { chunk_ids: updated, count: updated.length } });
        log('info', `Entity Edit: find ${updated.length} chunks related to \`${entityName}\``);
      }
    }
    if (isRenaming && relationChunksStorage && relationsToUpdate.length) {
      for (const [src, tgt, edgeData] of relationsToUpdate) {
        const oldSrc = src === entityName ? originalEntityName : src;
        const oldTgt = tgt === entityName ? originalEntityName : tgt;
        const oldKey = makeRelationChunkKey(...sorted2(oldSrc, oldTgt));
        const newKey = makeRelationChunkKey(...sorted2(src, tgt));
        if (oldKey === newKey) continue;
        const old = await relationChunksStorage.get_by_id(oldKey);
        const ids =
          old && typeof old === 'object' && !Array.isArray(old) ? ((old.chunk_ids as string[] | undefined) ?? []).filter((c) => c) : splitIds(get(edgeData, 'source_id', ''));
        await relationChunksStorage.delete([oldKey]);
        if (ids.length) await relationChunksStorage.upsert({ [newKey]: { chunk_ids: ids, count: ids.length } });
      }
      log('info', `Entity Edit: migrate ${relationsToUpdate.length} relations after rename`);
    }
  }

  await persistGraphUpdates(entitiesVdb, relationshipsVdb, graph, entityChunksStorage, relationChunksStorage);
  log('info', `Entity Edit: \`${entityName}\` successfully updated`);
  return utilsGraphHooks.getEntityInfo(graph, entitiesVdb, entityName, true);
}

const sameSet = (a: readonly string[], b: readonly string[]) => {
  const x = new Set(a);
  const y = new Set(b);
  return x.size === y.size && [...x].every((v) => y.has(v));
};

/** utils_graph.aedit_entity */
export async function aeditEntity(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb,
  entityName: string,
  updatedData: Dict,
  allowRename = true,
  allowMerge = false,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  { log = noLog }: Opts = {},
): Promise<Dict> {
  if ('description' in updatedData) requireNonEmptyDescription(updatedData.description, 'edit', 'entity');
  const newEntityName = (get(updatedData, 'entity_name', entityName) as string) ?? entityName;
  const isRenaming = newEntityName !== entityName;
  const summary: Dict = {
    merged: false,
    merge_status: 'not_attempted',
    merge_error: null,
    operation_status: 'success',
    target_entity: null,
    final_entity: isRenaming ? newEntityName : entityName,
    renamed: isRenaming,
  };
  return locked(graph, async () => {
    try {
      if (isRenaming && !allowRename) throw pyError('ValueError', 'Entity renaming is not allowed. Set allow_rename=True to enable this feature');
      if (isRenaming && (await graph.has_node(newEntityName))) {
        if (!allowMerge) throw pyError('ValueError', `Entity name '${newEntityName}' already exists, cannot rename`);
        log('info', `Entity Edit: \`${entityName}\` will be merged into \`${newEntityName}\``);
        let nonNameApplied = false;
        const nonName = Object.fromEntries(Object.entries(updatedData).filter(([k]) => k !== 'entity_name'));
        if (Object.keys(nonName).length) {
          try {
            log('info', 'Entity Edit: applying non-name updates before merge');
            await utilsGraphHooks.editEntityImpl(graph, entitiesVdb, relationshipsVdb, entityName, nonName, entityChunksStorage, relationChunksStorage, log);
            nonNameApplied = true;
          } catch (e) {
            log('error', `Entity Edit: non-name updates failed: ${errMsg(e)}`);
            throw e;
          }
        }
        try {
          const merged = await utilsGraphHooks.mergeEntitiesImpl(graph, entitiesVdb, relationshipsVdb, [entityName], newEntityName, null, null, entityChunksStorage, relationChunksStorage, log);
          Object.assign(summary, { merged: true, merge_status: 'success', merge_error: null, operation_status: 'success', target_entity: newEntityName, final_entity: newEntityName });
          return { ...merged, operation_summary: summary };
        } catch (e) {
          log('error', `Entity Edit: merge failed: ${errMsg(e)}`);
          Object.assign(summary, {
            merged: false,
            merge_status: 'failed',
            merge_error: errMsg(e),
            operation_status: nonNameApplied ? 'partial_success' : 'failure',
            target_entity: newEntityName,
            final_entity: entityName,
          });
          const info = await utilsGraphHooks.getEntityInfo(graph, entitiesVdb, entityName, true);
          return { ...info, operation_summary: summary };
        }
      }
      const result = await utilsGraphHooks.editEntityImpl(graph, entitiesVdb, relationshipsVdb, entityName, updatedData, entityChunksStorage, relationChunksStorage, log);
      summary.operation_status = 'success';
      return { ...result, operation_summary: summary };
    } catch (e) {
      log('error', `Error while editing entity '${entityName}': ${errMsg(e)}`);
      throw e;
    }
  });
}

/** utils_graph.aedit_relation */
export async function aeditRelation(
  graph: EditGraph,
  _entitiesVdb: EditVdb | null,
  relationshipsVdb: EditVdb,
  sourceEntity: string,
  targetEntity: string,
  updatedData: Dict,
  relationChunksStorage: EditKv | null = null,
  { log = noLog }: Opts = {},
): Promise<Dict> {
  if ('description' in updatedData) requireNonEmptyDescription(updatedData.description, 'edit', 'relation');
  if (pyCompare(sourceEntity, targetEntity) > 0) [sourceEntity, targetEntity] = [targetEntity, sourceEntity];
  return locked(graph, async () => {
    try {
      if (!(await graph.has_edge(sourceEntity, targetEntity))) {
        throw pyError('ValueError', `Relation from '${sourceEntity}' to '${targetEntity}' does not exist`);
      }
      const edgeData = ((await graph.get_edge(sourceEntity, targetEntity)) ?? {}) as Dict;
      await relationshipsVdb.delete([computeMdhashId(sourceEntity + targetEntity, 'rel-'), computeMdhashId(targetEntity + sourceEntity, 'rel-')]);
      log('debug', `Relation Delete: delete vdb for \`${sourceEntity}\`~\`${targetEntity}\``);
      const newEdgeData: Dict = { ...edgeData, ...updatedData };
      await graph.upsert_edge(sourceEntity, targetEntity, newEdgeData as EdgeData);
      const description = get(newEdgeData, 'description', '');
      const keywords = get(newEdgeData, 'keywords', '');
      await relationshipsVdb.upsert({
        [computeMdhashId(sourceEntity + targetEntity, 'rel-')]: {
          content: `${sourceEntity}\t${targetEntity}\n${pyStr(keywords)}\n${pyStr(description)}`,
          src_id: sourceEntity,
          tgt_id: targetEntity,
          source_id: get(newEdgeData, 'source_id', ''),
          description,
          keywords,
          weight: pyFloat(get(newEdgeData, 'weight', 1.0)),
        },
      });
      if (relationChunksStorage) {
        const key = makeRelationChunkKey(sourceEntity, targetEntity);
        const stored = await relationChunksStorage.get_by_id(key);
        const hasStored = !!stored && typeof stored === 'object' && pyTruthy(stored) && pyTruthy(stored.chunk_ids);
        const oldIds = splitIds(get(edgeData, 'source_id', ''));
        const newIds = splitIds(get(newEdgeData, 'source_id', ''));
        if (!sameSet(oldIds, newIds) || !hasStored) {
          let existing: string[] = hasStored ? ((stored!.chunk_ids as string[]) ?? []).filter((c) => c) : [];
          if (!existing.length) existing = [...oldIds];
          const updated = computeIncrementalChunkIds(existing, oldIds, newIds);
          await relationChunksStorage.upsert({ [key]: { chunk_ids: updated, count: updated.length } });
          log('info', `Relation Delete: update chunk tracking for \`${sourceEntity}\`~\`${targetEntity}\``);
        }
      }
      await persistGraphUpdates(relationshipsVdb, graph, relationChunksStorage);
      log('info', `Relation Delete: \`${sourceEntity}\`~\`${targetEntity}\`' successfully updated`);
      return utilsGraphHooks.getRelationInfo(graph, relationshipsVdb, sourceEntity, targetEntity, true);
    } catch (e) {
      log('error', `Error while editing relation from '${sourceEntity}' to '${targetEntity}': ${errMsg(e)}`);
      throw e;
    }
  });
}

/** utils_graph.acreate_entity. `clock`: `int(time.time())` for created_at (port-only). */
export async function acreateEntity(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb | null,
  entityName: string,
  entityData: Dict,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  { log = noLog, clock = () => Math.floor(Date.now() / 1000) }: Opts & { clock?: () => number } = {},
): Promise<Dict> {
  requireNonEmptyDescription(entityData.description, 'create', 'entity');
  return locked(graph, async () => {
    try {
      if (await graph.has_node(entityName)) throw pyError('ValueError', `Entity '${entityName}' already exists`);
      const nodeData: Dict = {
        entity_id: entityName,
        entity_type: get(entityData, 'entity_type', 'UNKNOWN'),
        description: get(entityData, 'description', ''),
        source_id: get(entityData, 'source_id', 'manual_creation'),
        file_path: get(entityData, 'file_path', 'manual_creation'),
        created_at: clock(),
      };
      await graph.upsert_node(entityName, nodeData as NodeData);
      const description = get(nodeData, 'description', '');
      await entitiesVdb.upsert({
        [computeMdhashId(entityName, 'ent-')]: {
          content: `${entityName}\n${pyStr(description)}`,
          entity_name: entityName,
          source_id: get(nodeData, 'source_id', ''),
          description,
          entity_type: get(nodeData, 'entity_type', ''),
          file_path: get(entityData, 'file_path', 'manual_creation'),
        },
      });
      if (entityChunksStorage) {
        const ids = splitIds(get(nodeData, 'source_id', ''));
        if (ids.length) {
          await entityChunksStorage.upsert({ [entityName]: { chunk_ids: ids, count: ids.length } });
          log('info', `Entity Create: tracked ${ids.length} chunks for \`${entityName}\``);
        }
      }
      await persistGraphUpdates(entitiesVdb, relationshipsVdb, graph, entityChunksStorage, relationChunksStorage);
      log('info', `Entity Create: '${entityName}' successfully created`);
      return utilsGraphHooks.getEntityInfo(graph, entitiesVdb, entityName, true);
    } catch (e) {
      log('error', `Error while creating entity '${entityName}': ${errMsg(e)}`);
      throw e;
    }
  });
}

/** utils_graph.acreate_relation */
export async function acreateRelation(
  graph: EditGraph,
  _entitiesVdb: EditVdb | null,
  relationshipsVdb: EditVdb,
  sourceEntity: string,
  targetEntity: string,
  relationData: Dict,
  relationChunksStorage: EditKv | null = null,
  { log = noLog, clock = () => Math.floor(Date.now() / 1000) }: Opts & { clock?: () => number } = {},
): Promise<Dict> {
  requireNonEmptyDescription(relationData.description, 'create', 'relation');
  return locked(graph, async () => {
    try {
      const sourceExists = await graph.has_node(sourceEntity);
      const targetExists = await graph.has_node(targetEntity);
      if (!sourceExists) throw pyError('ValueError', `Source entity '${sourceEntity}' does not exist`);
      if (!targetExists) throw pyError('ValueError', `Target entity '${targetEntity}' does not exist`);
      if (await graph.has_edge(sourceEntity, targetEntity)) {
        throw pyError('ValueError', `Relation from '${sourceEntity}' to '${targetEntity}' already exists`);
      }
      const edgeData: Dict = {
        description: get(relationData, 'description', ''),
        keywords: get(relationData, 'keywords', ''),
        source_id: get(relationData, 'source_id', 'manual_creation'),
        weight: pyFloat(get(relationData, 'weight', 1.0)),
        file_path: get(relationData, 'file_path', 'manual_creation'),
        created_at: clock(),
      };
      await graph.upsert_edge(sourceEntity, targetEntity, edgeData as EdgeData);
      if (pyCompare(sourceEntity, targetEntity) > 0) [sourceEntity, targetEntity] = [targetEntity, sourceEntity];
      const description = get(edgeData, 'description', '');
      const keywords = get(edgeData, 'keywords', '');
      await relationshipsVdb.upsert({
        [computeMdhashId(sourceEntity + targetEntity, 'rel-')]: {
          content: `${pyStr(keywords)}\t${sourceEntity}\n${targetEntity}\n${pyStr(description)}`,
          src_id: sourceEntity,
          tgt_id: targetEntity,
          source_id: get(edgeData, 'source_id', ''),
          description,
          keywords,
          weight: get(edgeData, 'weight', 1.0),
          file_path: get(relationData, 'file_path', 'manual_creation'),
        },
      });
      if (relationChunksStorage) {
        const ids = splitIds(get(edgeData, 'source_id', ''));
        if (ids.length) {
          await relationChunksStorage.upsert({ [makeRelationChunkKey(...sorted2(sourceEntity, targetEntity))]: { chunk_ids: ids, count: ids.length } });
          log('info', `Relation Create: tracked ${ids.length} chunks for \`${sourceEntity}\`~\`${targetEntity}\``);
        }
      }
      await persistGraphUpdates(relationshipsVdb, graph, relationChunksStorage);
      log('info', `Relation Create: \`${sourceEntity}\`~\`${targetEntity}\` successfully created`);
      return utilsGraphHooks.getRelationInfo(graph, relationshipsVdb, sourceEntity, targetEntity, true);
    } catch (e) {
      log('error', `Error while creating relation from '${sourceEntity}' to '${targetEntity}': ${errMsg(e)}`);
      throw e;
    }
  });
}

/** utils_graph._merge_entities_impl: the merge without taking the lock. */
async function mergeEntitiesImpl(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb,
  sourceEntities: string[],
  targetEntity: string,
  mergeStrategy: Record<string, string> | null = null,
  targetEntityData: Dict | null = null,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  log: GraphLog = noLog,
): Promise<Dict> {
  const defaultStrategy: Record<string, string> = { description: 'concatenate', entity_type: 'keep_first', source_id: 'join_unique', file_path: 'join_unique' };
  let strategy = defaultStrategy;
  if (mergeStrategy && Object.keys(mergeStrategy).length) {
    log('warning', 'Entity Merge: merge_strategy parameter is deprecated and will be ignored in a future release.');
    strategy = { ...defaultStrategy, ...mergeStrategy };
  }
  const overrides = targetEntityData ?? {};

  // 1. sources must exist
  const sourcesData = new Map<string, Dict>();
  for (const name of sourceEntities) {
    if (!(await graph.has_node(name))) throw pyError('ValueError', `Source entity '${name}' does not exist`);
    sourcesData.set(name, ((await graph.get_node(name)) ?? {}) as Dict);
  }
  // 2. target
  const targetExists = await graph.has_node(targetEntity);
  let existingTarget: Dict = {};
  if (targetExists) existingTarget = ((await graph.get_node(targetEntity)) ?? {}) as Dict;
  // 3. merged attributes, explicit values on top
  const merged = mergeAttributes([...sourcesData.values(), ...(targetExists ? [existingTarget] : [])], strategy, false);
  for (const [k, v] of Object.entries(overrides)) merged[k] = v;

  // 4. relations of the sources (and the target)
  const allRelations: [string, string, Dict][] = [];
  const toCollect = [...sourceEntities];
  if (targetExists && !sourceEntities.includes(targetEntity)) toCollect.push(targetEntity);
  for (const name of toCollect) {
    for (const [src, tgt] of (await graph.get_node_edges(name)) ?? []) {
      if (src === name) allRelations.push([src, tgt, ((await graph.get_edge(src, tgt)) ?? {}) as Dict]);
    }
  }

  // 5. target node
  merged.entity_id = targetEntity;
  await graph.upsert_node(targetEntity, merged as NodeData);
  log('info', targetExists ? `Entity Merge: Updated target '${targetEntity}'` : `Entity Merge: created target '${targetEntity}'`);

  // 6. relations re-pointed at the target, duplicates merged, chunk lists collected
  const updates = new Map<string, { graph_src: string; graph_tgt: string; norm_src: string; norm_tgt: string; data: Dict }>();
  const relationsToDelete: string[] = [];
  const chunkTracking = new Map<string, string[]>();
  const oldKeysToDelete: string[] = [];
  for (const [src, tgt, edgeData] of allRelations) {
    relationsToDelete.push(computeMdhashId(src + tgt, 'rel-'), computeMdhashId(tgt + src, 'rel-'));
    let oldKey = '';
    if (relationChunksStorage) {
      oldKey = makeRelationChunkKey(src, tgt);
      oldKeysToDelete.push(oldKey);
    }
    const newSrc = sourceEntities.includes(src) ? targetEntity : src;
    const newTgt = sourceEntities.includes(tgt) ? targetEntity : tgt;
    if (newSrc === newTgt) {
      log('info', `Entity Merge: skipping \`${src}\`~\`${tgt}\` to avoid self-loop`);
      continue;
    }
    const [ns, nt] = sorted2(newSrc, newTgt);
    const relationKey = `${ns}|${nt}`;
    if (relationChunksStorage) {
      const storageKey = makeRelationChunkKey(ns, nt);
      const stored = await relationChunksStorage.get_by_id(oldKey);
      const ids =
        stored !== null && typeof stored === 'object' && !Array.isArray(stored)
          ? ((stored.chunk_ids as string[] | undefined) ?? []).filter((c) => c)
          : splitIds(get(edgeData, 'source_id', ''));
      const list = chunkTracking.get(storageKey) ?? [];
      chunkTracking.set(storageKey, list);
      const have = new Set(list);
      for (const c of ids) {
        if (have.has(c)) continue;
        have.add(c);
        list.push(c);
      }
    }
    const existing = updates.get(relationKey);
    if (existing) {
      existing.data = mergeAttributes(
        [existing.data, edgeData],
        { description: 'concatenate', keywords: 'join_unique_comma', source_id: 'join_unique', file_path: 'join_unique', weight: 'max' },
        true,
      );
      log('debug', `Entity Merge: deduplicating relation \`${ns}\`~\`${nt}\``);
    } else updates.set(relationKey, { graph_src: newSrc, graph_tgt: newTgt, norm_src: ns, norm_tgt: nt, data: { ...edgeData } });
  }
  log('info', `Entity Merge: updatign ${updates.size} relations`);
  for (const r of updates.values()) {
    await graph.upsert_edge(r.graph_src, r.graph_tgt, r.data as EdgeData);
    log('info', `Entity Merge: updating relation \`${r.graph_src}\`~\`${r.graph_tgt}\``);
  }
  if (relationChunksStorage && allRelations.length) {
    if (oldKeysToDelete.length) await relationChunksStorage.delete(oldKeysToDelete);
    if (chunkTracking.size) {
      const upd: Record<string, Dict> = {};
      for (const [k, ids] of chunkTracking) upd[k] = { chunk_ids: ids, count: ids.length };
      await relationChunksStorage.upsert(upd);
      log('info', `Entity Merge: ${chunkTracking.size} relation chunk tracking records updated`);
    }
  }

  // 7. relation vectors
  log('debug', `Entity Merge: deleting ${relationsToDelete.length} relations from vdb`);
  await relationshipsVdb.delete(relationsToDelete);
  for (const r of updates.values()) {
    const d = r.data;
    const description = get(d, 'description', '');
    const keywords = get(d, 'keywords', '');
    await relationshipsVdb.upsert({
      [computeMdhashId(r.norm_src + r.norm_tgt, 'rel-')]: {
        content: `${pyStr(keywords)}\t${r.norm_src}\n${r.norm_tgt}\n${pyStr(description)}`,
        src_id: r.norm_src,
        tgt_id: r.norm_tgt,
        source_id: get(d, 'source_id', ''),
        description,
        keywords,
        weight: pyFloat(get(d, 'weight', 1.0)),
        file_path: get(d, 'file_path', ''),
      },
    });
    log('debug', `Entity Merge: updating vdb \`${r.norm_src}\`~\`${r.norm_tgt}\``);
  }
  log('info', `Entity Merge: ${updates.size} relations in vdb updated`);

  // 8. entity vector
  const description = get(merged, 'description', '');
  await entitiesVdb.upsert({
    [computeMdhashId(targetEntity, 'ent-')]: {
      content: `${targetEntity}\n${pyStr(description)}`,
      entity_name: targetEntity,
      source_id: get(merged, 'source_id', ''),
      description,
      entity_type: get(merged, 'entity_type', ''),
      file_path: get(merged, 'file_path', ''),
    },
  });
  log('info', `Entity Merge: updating vdb \`${targetEntity}\``);

  // 9. entity chunk lists: sources first, then the target
  if (entityChunksStorage) {
    const lists: string[][] = [];
    const toProcess = sourceEntities.filter((e) => e !== targetEntity);
    if (targetExists) toProcess.push(targetEntity);
    for (const name of toProcess) {
      const stored = await entityChunksStorage.get_by_id(name);
      if (stored && typeof stored === 'object' && pyTruthy(stored)) {
        const ids = ((stored.chunk_ids as string[] | undefined) ?? []).filter((c) => c);
        if (ids.length) lists.push(ids);
      }
    }
    const mergedIds: string[] = [];
    const seen = new Set<string>();
    for (const l of lists) for (const c of l) if (!seen.has(c)) (seen.add(c), mergedIds.push(c));
    const keysToDelete = sourceEntities.filter((e) => e !== targetEntity);
    if (keysToDelete.length) await entityChunksStorage.delete(keysToDelete);
    if (mergedIds.length) {
      await entityChunksStorage.upsert({ [targetEntity]: { chunk_ids: mergedIds, count: mergedIds.length } });
      log('info', `Entity Merge: find ${mergedIds.length} chunks related to '${targetEntity}'`);
    }
  }

  // 10. sources go
  for (const name of sourceEntities) {
    if (name === targetEntity) {
      log('warning', `Entity Merge: source entity'${name}' is same as target entity`);
      continue;
    }
    log('info', `Entity Merge: deleting '${name}' from KG and vdb`);
    await graph.delete_node(name);
    await entitiesVdb.delete([computeMdhashId(name, 'ent-')]);
  }

  await persistGraphUpdates(entitiesVdb, relationshipsVdb, graph, entityChunksStorage, relationChunksStorage);
  log('info', `Entity Merge: successfully merged ${sourceEntities.length} entities into '${targetEntity}'`);
  return utilsGraphHooks.getEntityInfo(graph, entitiesVdb, targetEntity, true);
}

/** utils_graph.amerge_entities */
export async function amergeEntities(
  graph: EditGraph,
  entitiesVdb: EditVdb,
  relationshipsVdb: EditVdb,
  sourceEntities: string[],
  targetEntity: string,
  mergeStrategy: Record<string, string> | null = null,
  targetEntityData: Dict | null = null,
  entityChunksStorage: EditKv | null = null,
  relationChunksStorage: EditKv | null = null,
  { log = noLog }: Opts = {},
): Promise<Dict> {
  return locked(graph, async () => {
    try {
      return await utilsGraphHooks.mergeEntitiesImpl(graph, entitiesVdb, relationshipsVdb, sourceEntities, targetEntity, mergeStrategy, targetEntityData, entityChunksStorage, relationChunksStorage, log);
    } catch (e) {
      log('error', `Error merging entities: ${errMsg(e)}`);
      throw e;
    }
  });
}

/**
 * utils_graph._merge_attributes. Upstream collects the keys and the join_unique
 * items in Python sets; first-seen order stands in for their iteration order.
 */
export function mergeAttributes(dataList: readonly Dict[], mergeStrategy: Record<string, string>, filterNoneOnly = false): Dict {
  const merged: Dict = {};
  const allKeys = new Set<string>();
  for (const d of dataList) for (const k of Object.keys(d)) allKeys.add(k);
  for (const key of allKeys) {
    const values = dataList.map((d) => d[key]).filter((v) => (filterNoneOnly ? v !== null && v !== undefined : pyTruthy(v)));
    if (!values.length) continue;
    const s = mergeStrategy[key] ?? 'keep_first';
    if (s === 'concatenate') merged[key] = values.map(pyStr).join(GRAPH_FIELD_SEP);
    else if (s === 'keep_last') merged[key] = values[values.length - 1];
    else if (s === 'join_unique') {
      const items = new Set<string>();
      for (const v of values) for (const it of pyStr(v).split(GRAPH_FIELD_SEP)) items.add(it);
      merged[key] = [...items].join(GRAPH_FIELD_SEP);
    } else if (s === 'join_unique_comma') {
      const items = new Set<string>();
      for (const v of values) for (const it of pyStr(v).split(',')) if (pyStrip(it)) items.add(pyStrip(it));
      merged[key] = [...items].sort(pyCompare).join(',');
    } else if (s === 'max') {
      try {
        merged[key] = Math.max(...values.map(pyFloat));
      } catch {
        merged[key] = values[0];
      }
    } else merged[key] = values[0];
  }
  return merged;
}

/** utils_graph.get_entity_info */
async function getEntityInfo(graph: Pick<EditGraph, 'get_node'>, entitiesVdb: Pick<EditVdb, 'get_by_id'>, entityName: string, includeVectorData = false): Promise<Dict> {
  const nodeData = await graph.get_node(entityName);
  const result: Dict = { entity_name: entityName, source_id: pyTruthy(nodeData) ? (nodeData!.source_id ?? null) : null, graph_data: nodeData ?? null };
  if (includeVectorData) result.vector_data = await entitiesVdb.get_by_id(computeMdhashId(entityName, 'ent-'));
  return result;
}

/** utils_graph.get_relation_info: the normalized vdb id first, then the reverse one. */
async function getRelationInfo(
  graph: Pick<EditGraph, 'get_edge'>,
  relationshipsVdb: Pick<EditVdb, 'get_by_id'>,
  srcEntity: string,
  tgtEntity: string,
  includeVectorData = false,
): Promise<Dict> {
  const edgeData = await graph.get_edge(srcEntity, tgtEntity);
  const result: Dict = { src_entity: srcEntity, tgt_entity: tgtEntity, source_id: pyTruthy(edgeData) ? (edgeData!.source_id ?? null) : null, graph_data: edgeData ?? null };
  if (includeVectorData) {
    let vectorData: Dict | null = null;
    for (const id of makeRelationVdbIds(srcEntity, tgtEntity)) {
      vectorData = await relationshipsVdb.get_by_id(id);
      if (vectorData !== null) break;
    }
    result.vector_data = vectorData;
  }
  return result;
}

/**
 * The helpers the public functions call, looked up at call time like Python
 * module attributes: replacing one here (as upstream's tests monkeypatch
 * `utils_graph._edit_entity_impl`) changes what aeditEntity & co. call.
 */
export const utilsGraphHooks = { editEntityImpl, mergeEntitiesImpl, getEntityInfo, getRelationInfo };
export { editEntityImpl, getEntityInfo, getRelationInfo, mergeEntitiesImpl };
