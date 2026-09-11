export type InferenceSource = "unavailable" | "mock" | "pytorch";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  inference_source: InferenceSource | string;
}

export interface ModelInfoResponse {
  checkpoint_id: string | null;
  dataset_version: string | null;
  trained_on_split: string | null;
  headline_metrics: Record<string, number> | null;
  num_classes: number;
  input_modalities: string[];
  model_loaded: boolean;
  inference_source: InferenceSource | string;
  message: string;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
