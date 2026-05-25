export interface Coords3D {
  x2d: number;
  y2d: number;
  x3d: number;
  y3d: number;
  z3d: number;
}

export interface Neighbor {
  label: string;
  cosine_similarity: number;
}

export interface SemanticNode {
  id?: number;
  token_id: number;
  label: string;
  cluster: string;
  color: string;
  coords: Coords3D;
  top5_neighbors: Neighbor[];
}

export interface SemanticMeta {
  model: string;
  embedding_dim: number;
  total_tokens: number;
  multi_token_terms?: string[];
  umap_params?: {
    n_neighbors: number;
    min_dist: number;
    metric: string;
  };
}

export interface SemanticSpace {
  meta: SemanticMeta;
  nodes: SemanticNode[];
}
