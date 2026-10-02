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

export type ModelKey = 'qwen' | 'gemma' | 'gemma2' | 'bert';
export type Algorithm = 'umap' | 'tsne' | 'tsne_p30' | 'pca' | 'isomap' | 'pca_nc';
export type ColorLayer = 'category' | 'heat';

export interface SemanticNode {
  id?: number;
  token_id: number;
  label: string;
  /** reference partition: content_noun | content_verb | function_word */
  cluster: string;
  color: string;
  coords: Coords3D;
  top5_neighbors: Neighbor[];
  /** node ids ordered by high-dimensional rank (rank 1..K_max) */
  hd_neighbors?: number[];
  /** node ids ordered by projected rank (rank 1..K_max) */
  ld_neighbors?: number[];
  /** B_NX,i(K) keyed by K */
  b_nx_local?: Record<string, number>;
}

export interface CorankingCurvePoint {
  K: number;
  N: number;
  trustworthiness: number;
  continuity: number;
  U_P: number;
  U_N: number;
  U_X: number;
  Q_NX: number;
  B_NX: number;
  LCMC: number;
  R_NX: number;
  B_TC: number;
}

export interface SemanticMeta {
  model: string;
  model_key?: ModelKey;
  algorithm?: Algorithm;
  params?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  embedding_dim: number;
  total_tokens: number;
  dataset_hash?: string;
  dataset_frozen?: boolean;
  dataset_note?: string | null;
  reduced_sample?: boolean;
  K_main?: number;
  K_curve?: number[];
  metrics?: CorankingCurvePoint[];
  silhouette_projected?: number;
  silhouette_projected_2d?: number;
  category_colors?: Record<string, string>;
}

export interface SemanticSpace {
  meta: SemanticMeta;
  nodes: SemanticNode[];
}

export interface ProbeSets {
  selected: number;
  K: number;
  true: Set<number>;
  false: Set<number>;
  missing: Set<number>;
}

/** Probe for node i: T = N_K ∩ n_K, F = n_K \ N_K, M = N_K \ n_K (plan 9.7). */
export function probeSets(node: SemanticNode, K: number): ProbeSets | null {
  if (node.id === undefined || !node.hd_neighbors || !node.ld_neighbors) return null;
  const hi = new Set(node.hd_neighbors.slice(0, K));
  const lo = new Set(node.ld_neighbors.slice(0, K));
  return {
    selected: node.id,
    K,
    true: new Set([...lo].filter((j) => hi.has(j))),
    false: new Set([...lo].filter((j) => !hi.has(j))),
    missing: new Set([...hi].filter((j) => !lo.has(j))),
  };
}
