# Medical Image Segmentation & Tumor Detection

Research / decision-support prototype for BraTS-format brain MRI tumor sub-region segmentation.

**This is not a medical device. It is not clinically validated. It must not be used for diagnosis or treatment decisions.**

The trained 3D U-Net is developed in a separate notebook. This repository currently contains **Phase 2**: API health, case creation, four-modality NIfTI upload, spatial validation, and isolated storage. Inference is not started.

## Current status

| Capability | Status |
|---|---|
| Frontend shell | Phase 1 |
| Backend health | Phase 1 |
| Model info stub | Phase 1 (`model_loaded: false`, no headline metrics) |
| Case upload + NIfTI validation | Phase 2 |
| Mock or real inference | Not implemented |

## Requirements

- Python 3.12+
- Node.js 20+
- Docker (optional)

## Environment

Copy `.env.example` to `.env` and adjust if needed. Do not commit secrets.

| Variable | Purpose |
|---|---|
| `ENVIRONMENT` | `development` / later `production` |
| `LOG_LEVEL` | Backend log level |
| `CORS_ORIGINS` | Comma-separated browser origins |
| `VITE_API_URL` | Browser API origin. Empty = same-origin / Vite proxy |
| `MODEL_PATH` | Reserved. Unused until a real checkpoint is integrated |
| `MODEL_VERSION` | Reserved |
| `UPLOAD_DIR` / `RESULTS_DIR` | Isolated case files (not web-served) |
| `DATABASE_URL` | SQLite URL for case metadata |
| `MAX_UPLOAD_SIZE_MB` | Limit for the **entire case** (sum of files), default 500 |

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- Health: `GET /api/v1/health` → `{ "status": "ok", "model_loaded": false, "inference_source": "unavailable" }`
- Upload: http://localhost:3000/upload
- Case API: `POST /api/v1/cases`, `POST /api/v1/cases/{id}/files/{modality}`, `POST /api/v1/cases/{id}/complete`
- PRD ingest (store + validate only): `POST /api/v1/predict`, `POST /api/v1/evaluate` — returns 201, `job_id: null`

```bash
cd backend
source .venv/bin/activate
pytest
```

## Frontend

```bash
cd frontend
npm install
npm run dev
npm test
npm run build
```

App: http://localhost:3000 — use **Upload MRI** to assign T1 / T1ce / T2 / FLAIR (and optional `seg`).

## Docker

```bash
docker compose up --build
```

- Web: http://localhost:3000 (nginx proxies `/api` to the backend)
- API: http://localhost:8000/docs

CPU-only. No GPU image in Phase 1.

## Inference boundary

`backend/app/inference/base.py` defines `InferenceService`. A mock implementation and a real PyTorch checkpoint loader are later phases. The UI must not assume a model is loaded.

## Documentation

Product requirements: `docs/Complete_PRD.md`. Architecture notes: `ARCHITECTURE.md`.
