// Port of lightrag/types.py (pydantic models -> plain types).

export type GPTKeywordExtractionFormat = {
  high_level_keywords: string[];
  low_level_keywords: string[];
};

export type KnowledgeGraphNode = {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
};

export type KnowledgeGraphEdge = {
  id: string;
  type: string | null;
  source: string;
  target: string;
  properties: Record<string, unknown>;
};

export type KnowledgeGraph = {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  is_truncated: boolean;
};
