export interface DatasetSummary {
  dataset_id: string;
  cell_count: number;
  gene_count: number;
}

export interface EmbeddingPoint {
  cell_id: string;
  x: number;
  y: number;
  z: number;
  cluster: number;
  annotation?: string | null;
}

export interface MetadataPayload {
  gene_columns: string[];
  original_clusters?: number[];
}

export interface AnnotationPayload {
  dataset_id: string;
  cell_id: string;
  label: string;
}

export interface InferenceResult {
  dataset_id: string;
  cell_id?: string;
  predicted_type: string;
  latent_vector: number[];
}

export interface MLMetadata {
  latent_dim: number;
  input_dim: number;
  classes: string[];
}
