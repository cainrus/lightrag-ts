// Port of lightrag/namespace.py. Values must not change: they are file names on disk.

export const NameSpace = {
  KV_STORE_FULL_DOCS: 'full_docs',
  KV_STORE_TEXT_CHUNKS: 'text_chunks',
  KV_STORE_LLM_RESPONSE_CACHE: 'llm_response_cache',
  KV_STORE_FULL_ENTITIES: 'full_entities',
  KV_STORE_FULL_RELATIONS: 'full_relations',
  KV_STORE_ENTITY_CHUNKS: 'entity_chunks',
  KV_STORE_RELATION_CHUNKS: 'relation_chunks',

  VECTOR_STORE_ENTITIES: 'entities',
  VECTOR_STORE_RELATIONSHIPS: 'relationships',
  VECTOR_STORE_CHUNKS: 'chunks',

  GRAPH_STORE_CHUNK_ENTITY_RELATION: 'chunk_entity_relation',

  DOC_STATUS: 'doc_status',
} as const;

export type NameSpaceValue = (typeof NameSpace)[keyof typeof NameSpace];

export function isNamespace(namespace: string, base: string | Iterable<string>): boolean {
  if (typeof base === 'string') return namespace.endsWith(base);
  for (const ns of base) if (isNamespace(namespace, ns)) return true;
  return false;
}
