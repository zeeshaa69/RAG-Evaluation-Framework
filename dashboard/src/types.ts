export interface MetricResult {
  metric: string;
  score: number | null;
  method: string;
  evidence: Record<string, unknown>;
  heuristic: boolean;
  note: string;
}

export interface RetrievedRow {
  doc_id: string;
  rank: number;
  score: number;
  relevant: boolean;
  text: string;
}

export interface Failure {
  category: string;
  severity: string;
  description: string;
  evidence: Record<string, unknown>;
  heuristic: boolean;
}

export interface SampleRow {
  sample_id: string;
  question: string;
  expected_answer: string | null;
  relevant_docs: string[];
  metadata: Record<string, unknown>;
  retrieved: RetrievedRow[];
  answer: string;
  metrics: Record<string, MetricResult>;
  failures: Failure[];
}

export interface MetricSummary {
  mean: number | null;
  std: number | null;
  n: number;
  n_total: number;
  not_applicable: number;
  method: string;
  heuristic: boolean;
}

export interface Manifest {
  run_id: string;
  name: string;
  created_at: string;
  eval_version: string;
  framework_version: string;
  config_hash: string;
  dataset: {
    name: string;
    version: string;
    corpus_hash: string;
    samples_hash: string;
    num_samples: number;
    num_documents: number;
  };
  retriever: { type: string; name: string; params: Record<string, unknown> };
  generator: { type: string; model_id: string; params: Record<string, unknown> };
  embedder: string;
  top_k: number;
  k_values: number[];
  metrics: string[];
  seed: number;
  python: string;
  platform: string;
  duration_seconds: number;
}

export interface DatasetQuality {
  name?: string;
  version?: string;
  num_samples?: number;
  num_documents?: number;
  samples_with_expected_answer?: number;
  samples_with_relevant_docs?: number;
  samples_multi_relevant?: number;
  documents_never_relevant?: number;
  relevant_docs_per_sample?: Distribution;
  document_length_words?: Distribution;
  answer_length_words?: Distribution;
  duplicate_documents?: string[][];
  answer_support_mean?: number | null;
  low_answer_support?: { sample_id: string; coverage: number }[];
  note?: string;
}

export interface Distribution {
  min: number;
  median: number;
  max: number;
  mean: number;
}

export interface Run {
  manifest: Manifest;
  config: Record<string, unknown>;
  summary: {
    num_samples: number;
    metrics: Record<string, MetricSummary>;
    failure_counts: Record<string, number>;
  };
  samples: SampleRow[];
  dataset_quality: DatasetQuality;
}

export interface Bundle {
  generated_at: string;
  runs: Run[];
}
