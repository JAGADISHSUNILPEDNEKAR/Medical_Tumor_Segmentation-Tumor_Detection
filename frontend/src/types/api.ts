export type InferenceSource = "unavailable" | "mock" | "pytorch";

export type CaseStatus = "CREATED" | "UPLOADING" | "VALIDATING" | "READY" | "FAILED" | string;

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

export interface ModalityFileInfo {
  modality: string;
  original_filename: string;
  size_bytes: number;
  shape: number[] | null;
  spacing: number[] | null;
  warnings: string[];
}

export interface SpatialChecks {
  shape_consistent: boolean | null;
  affine_consistent: boolean | null;
  spacing_consistent: boolean | null;
  notes: string[];
}

export interface CaseValidationReport {
  modalities: Record<string, boolean>;
  spatial: SpatialChecks;
  warnings: string[];
  ready_for_prediction: boolean;
  ready_for_evaluation: boolean;
}

export interface CaseResponse {
  case_id: string;
  status: CaseStatus;
  created_at: string;
  modalities_present: string[];
  has_ground_truth: boolean;
  total_bytes: number;
  files: ModalityFileInfo[];
  validation: CaseValidationReport | null;
  error: string | null;
  detail: string | null;
  inference: string;
}

export interface CaseCreateResponse {
  case_id: string;
  status: CaseStatus;
  created_at: string;
  modalities_present: string[];
  has_ground_truth: boolean;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly caseId: string | null;

  constructor(message: string, status: number, code: string | null = null, caseId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.caseId = caseId;
  }
}

export function parseApiError(status: number, body: unknown): ApiError {
  if (typeof body === "object" && body !== null) {
    const record = body as Record<string, unknown>;
    const detail = typeof record.detail === "string" ? record.detail : `Request failed (${status}).`;
    const code = typeof record.error === "string" ? record.error : null;
    const caseId = typeof record.case_id === "string" ? record.case_id : null;
    return new ApiError(detail, status, code, caseId);
  }
  return new ApiError(`Request failed (${status}).`, status);
}

export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | string;

export interface JobStatusResponse {
  job_id: string;
  case_id: string;
  job_type: string;
  status: JobStatus;
  progress: number;
  inference_source: InferenceSource | string;
  model_version: string | null;
  result_id: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

/** Per-region volumetrics. Keyed by the backend's human-readable region name. */
export interface RegionMeasurement {
  /** BraTS label value (1 = NCR, 2 = ED, 4 = ET). Authoritative for display colour. */
  label: number;
  voxel_count: number;
  volume_mm3: number;
  volume_cm3: number;
}

export interface Measurements {
  /** True while results come from MockInferenceService. */
  synthetic: boolean;
  description: string;
  foreground_voxels: number;
  foreground_volume_mm3: number;
  foreground_volume_cm3: number;
  regions: Record<string, RegionMeasurement>;
}

export interface ResultResponse {
  result_id: string;
  job_id: string;
  case_id: string;
  status: string;
  inference_source: InferenceSource | string;
  model_version: string | null;
  segmentation: { available: boolean };
  measurements: Measurements;
  evaluation: { available: boolean; dice: number | null; hd95_mm: number | null };
  created_at: string;
}
