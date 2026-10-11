# Tumor Segmentation Workbench API

This document describes the REST API for the Tumor Segmentation Workbench. The API is versioned under `/api/v1` and is built with FastAPI.

## Base Path
`/api/v1`

## Endpoints Summary

1. `GET /health`
2. `GET /model/info`
3. `POST /cases`
4. `GET /cases/{case_id}`
5. `POST /cases/{case_id}/files/{modality}`
6. `POST /cases/{case_id}/complete`
7. `GET /cases/{case_id}/artifacts/{artifact}`
8. `POST /cases/{case_id}/predict`
9. `POST /predict`
10. `POST /evaluate`
11. `GET /jobs/{job_id}`
12. `GET /results/{result_id}`
13. `GET /results/{result_id}/measurements`
14. `GET /results/{result_id}/report`

---

## Health & Model Info

### `GET /health`
Returns the operational status of the API and the inference backend.
- **Response `200 OK`**:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "inference_source": "pytorch"
  }
  ```

### `GET /model/info`
Returns detailed information about the loaded inference model, configuration, and device parameters.
- **Response `200 OK`**: (ModelInfoResponse)

---

## Case Lifecycle

A case represents a single study (a patient session) requiring multiple MRI modalities.

### `POST /cases`
Create a new, empty case.
- **Response `201 Created`**:
  ```json
  {
    "case_id": "uuid",
    "status": "CREATED",
    "created_at": "timestamp",
    "modalities_present": [],
    "has_ground_truth": false
  }
  ```

### `GET /cases/{case_id}`
Retrieve the current status of a case.
- **Response `200 OK`**: CaseResponse
- **Errors**: `404 CASE_NOT_FOUND`

### `POST /cases/{case_id}/files/{modality}`
Upload a single NIfTI file for a specific modality (`t1`, `t1ce`, `t2`, `flair`, `seg`).
- **Request**: `multipart/form-data` with `file`
- **Response `200 OK`**: CaseResponse (updated `modalities_present`)
- **Errors**: `404 CASE_NOT_FOUND`, `400 INVALID_MODALITY`, `413 CASE_TOO_LARGE`, `422 INVALID_NIFTI`, `409 DUPLICATE_MODALITY`

### `POST /cases/{case_id}/complete`
Finalize the case upload phase. Validates that all required modalities (T1, T1ce, T2, FLAIR) are present and shape-aligned.
- **Response `200 OK`**: CaseResponse (status becomes `READY`)
- **Errors**: `404 CASE_NOT_FOUND`, `409 INVALID_CASE_STATE`, `400 MISSING_MODALITY`, `422 MISMATCHED_SHAPES`

### `GET /cases/{case_id}/artifacts/{artifact}`
Download an artifact associated with a case (e.g., `t1.nii.gz`, `segmentation.nii.gz`).
- **Response `200 OK`**: Streamed binary NIfTI file (application/gzip).
- **Errors**: `404 ARTIFACT_NOT_FOUND`, `400 INVALID_ARTIFACT`

---

## Inference (Prediction & Evaluation)

### `POST /predict`
Upload all files, create a case, validate it, and enqueue it for prediction in a single shot.
- **Request**: `multipart/form-data` with `t1`, `t1ce`, `t2`, `flair`
- **Response `202 Accepted`**:
  ```json
  {
    "job_id": "uuid",
    "case_id": "uuid",
    "status": "QUEUED",
    "inference": "pytorch",
    "message": "Job queued for trained 3D U-Net inference. Poll GET /api/v1/jobs/{job_id} for status."
  }
  ```
- **Errors**: `503 MODEL_UNAVAILABLE`, `400 MISSING_MODALITY`, `413 CASE_TOO_LARGE`, `422 INVALID_NIFTI`

### `POST /evaluate`
Similar to `/predict`, but requires a ground truth segmentation (`seg`) to generate quantitative metrics.
- **Request**: `multipart/form-data` with `t1`, `t1ce`, `t2`, `flair`, `seg`
- **Response `202 Accepted`**: PredictAcceptedResponse

### `POST /cases/{case_id}/predict`
Start inference for an already uploaded and `READY` case.
- **Response `202 Accepted`**: PredictAcceptedResponse
- **Errors**: `503 MODEL_UNAVAILABLE`, `409 INVALID_CASE_STATE`

---

## Job Lifecycle

### `GET /jobs/{job_id}`
Poll the status of an async inference job.
- **Lifecycle**: `QUEUED` → `RUNNING` → `COMPLETED` | `FAILED`
- **Response `200 OK`**:
  ```json
  {
    "job_id": "uuid",
    "case_id": "uuid",
    "status": "COMPLETED",
    "progress": 100,
    "result_id": "uuid",
    "inference_source": "pytorch"
  }
  ```
- **Errors**: `404 JOB_NOT_FOUND`

---

## Results

### `GET /results/{result_id}`
Get the complete result payload for a finished inference job, including measurements and provenance.
- **Response `200 OK`**: ResultResponse
- **Errors**: `404 RESULT_NOT_FOUND`

### `GET /results/{result_id}/measurements`
Get only the volumetric measurements portion of the result.
- **Response `200 OK`**: MeasurementsInfo schema.

### `GET /results/{result_id}/report`
Get the structured data needed to render the clinical report.
- **Response `200 OK`**: ReportResponse schema.

---

## Error Format
All application errors follow the `AppError` schema (HTTP 4xx/5xx):
```json
{
  "error": {
    "code": "MISSING_MODALITY",
    "detail": "Required modality 'T1' is missing.",
    "case_id": "uuid"
  }
}
```
