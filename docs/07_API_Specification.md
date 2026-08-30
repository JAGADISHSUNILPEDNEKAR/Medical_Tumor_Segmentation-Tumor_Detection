# Document 7 — API Specification

Base path: `/api/v1`

## 1. Health Check
* **Endpoint:** `GET /api/v1/health`
* **Authorization:** None
* **Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## 2. Model Info
* **Endpoint:** `GET /api/v1/model/info`
* **Authorization:** None (V1: session-gated)
* **Response (200 OK):**
```json
{
  "checkpoint_id": "unet3d-v1-ep180",
  "dataset_version": "brats-2021-v1",
  "trained_on_split": "official_train",
  "headline_metrics": {"mean_dice": 0.81, "mean_hd95_mm": 4.2},
  "num_classes": 4,
  "input_modalities": ["T1", "T1ce", "T2", "FLAIR"]
}
```

## 3. Submit Case for Segmentation
* **Endpoint:** `POST /api/v1/predict`
* **Authorization:** None (V1: session-gated)
* **Request:** `multipart/form-data` with 4 named file fields (`t1`, `t1ce`, `t2`, `flair`)
* **Response (202 Accepted):**
```json
{
  "job_id": "job_9f1a3b21",
  "status": "queued"
}
```
* **Errors:** `400` missing modality; `413` file too large; `422` unparsable NIfTI.

## 4. Submit Case for Evaluation
* **Endpoint:** `POST /api/v1/evaluate`
* **Authorization:** None (V1: session-gated)
* **Request:** As `/predict`, plus a `seg` ground-truth file field.
* **Response (202 Accepted):**
```json
{
  "job_id": "job_9f1a3b22",
  "status": "queued"
}
```
* **Errors:** As `/predict`, plus `400` if `seg` shape doesn't match the image modalities.

## 5. Poll / Fetch Job Result
* **Endpoint:** `GET /api/v1/results/{id}`
* **Authorization:** None (V1: session-gated; only the owning session may fetch)
* **Response (200 OK, done):**
```json
{
  "job_id": "job_9f1a3b22",
  "status": "done",
  "case_id": "case_7d2e1a90",
  "mask_url": "/api/v1/cases/case_7d2e1a90/mask",
  "mesh_url": "/api/v1/cases/case_7d2e1a90/mesh",
  "metrics": {
    "dice": {"necrotic": 0.71, "edema": 0.84, "enhancing": 0.79, "mean": 0.78},
    "hd95_mm": {"necrotic": 5.1, "edema": 3.2, "enhancing": 4.0, "mean": 4.1}
  },
  "checkpoint_id": "unet3d-v1-ep180",
  "created_at": "2026-08-27T10:00:00Z"
}
```
* **Response (200 OK, running):** `{"job_id": "...", "status": "running"}`
* **Response (200 OK, failed):** `{"job_id": "...", "status": "failed", "error": "oom", "detail": "..."}`
* **Errors:** `404` unknown job ID.

## 6. List Cases (V1 — History Screen)
* **Endpoint:** `GET /api/v1/cases`
* **Authorization:** Session-gated
* **Response (200 OK):** Paginated array of `CaseSummary` objects.

## 7. Aggregate Dashboard Metrics
* **Endpoint:** `GET /api/v1/metrics`
* **Authorization:** Session-gated
* **Response (200 OK):** Aggregate Dice/HD95 distributions, per-class breakdown, inference-time statistics across all/filtered evaluated cases.
