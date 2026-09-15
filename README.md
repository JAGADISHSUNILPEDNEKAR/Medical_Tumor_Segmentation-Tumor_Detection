# Medical Image Segmentation & Tumor Detection

Research / decision-support prototype for BraTS-format brain MRI tumor sub-region segmentation.

**This is not a medical device. It is not clinically validated. It must not be used for diagnosis or treatment decisions.**

The trained 3D U-Net is developed in a separate notebook. This repository currently contains **Phase 4**: API health, case creation, four-modality NIfTI upload, spatial validation, isolated storage, async job execution behind a replaceable inference boundary, and in-browser 2D/3D visualization.

**The only inference implementation is `MockInferenceService`**, which synthesizes a geometric ellipsoid mask. No trained checkpoint is loaded and no real prediction is produced. Every screen that shows a result is marked as synthetic.

### Phase 4: Medical Image Visualization [✓]

Built an in-browser MRI visualization suite without relying on external medical servers.
- **In-Memory NIfTI Engine**: Parses `.nii.gz` volumes using `nifti-reader-js`, preserving spatial affine geometry and scaling data without resampling.
- **Multiplanar Slicing**: Extracts and normalizes arbitrary Axial, Coronal, and Sagittal slices natively via HTML Canvas rendering for smooth performance.
- **3D Tumor Extraction**: A custom marching cubes engine generates lightweight, WebGL-ready triangle meshes directly from the segmentation voxel masks.
- **Interactive Tooling**: Crosshair synchronization across 2D planes, overlay opacity and per-region visibility, modality toggling (T1, T1ce, T2, FLAIR), keyboard slice step-through, a case metadata panel, and interactive 3D OrbitControls (via `react-three-fiber`).

## Current status

| Capability | Status |
|---|---|
| Frontend shell | Phase 1 |
| Backend health | Phase 1 |
| Model info stub | Phase 1 (`model_loaded: false`, no headline metrics) |
| Case upload + NIfTI validation | Phase 2 |
| Async job queue + result endpoints | Phase 3 |
| Mock inference (synthetic, non-clinical) | Phase 3 |
| 2D multiplanar viewer + 3D tumor mesh | Phase 4 |
| Measurements UI | Phase 4 (volumes only; report + export are Phase 5) |
| Real model inference | Not implemented (Phase 6) |
| Dice / HD95 evaluation | Not implemented (`/evaluate` stores `seg` but computes no metrics) |

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
- Health: `GET /api/v1/health` → `{ "status": "ok", "model_loaded": false, "inference_source": "mock" }`
- Upload: http://localhost:3000/upload
- Case API: `POST /api/v1/cases`, `POST /api/v1/cases/{id}/files/{modality}`, `POST /api/v1/cases/{id}/complete`
- Inference: `POST /api/v1/cases/{id}/predict` → 202 with a `job_id` (409 if the case is not READY)
- Polling: `GET /api/v1/jobs/{job_id}`, then `GET /api/v1/results/{result_id}`
- Viewer artifacts: `GET /api/v1/cases/{id}/artifacts/{t1|t1ce|t2|flair|segmentation}`
- PRD ingest (upload + validate + enqueue in one call): `POST /api/v1/predict`, `POST /api/v1/evaluate` — 202 with a `job_id`

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

App: http://localhost:3000 — use **Upload MRI** to assign T1 / T1ce / T2 / FLAIR (and optional `seg`),
run the analysis, then open the viewer.

### Viewer controls

| Input | Action |
|---|---|
| `J` / `L` | Step back / forward through the active plane |
| `I` / `K` | Cycle which plane the keyboard steps |
| Click a plane | Sync the crosshair across the other two, and make that plane active |
| Drag / scroll / right-drag in 3D | Rotate / zoom / pan the tumor mesh |

## Docker

```bash
docker compose up --build
```

- Web: http://localhost:3000 (nginx proxies `/api` to the backend)
- API: http://localhost:8000/docs

CPU-only. No GPU image yet — GPU deployment is deferred to Phase 6, when a real
checkpoint exists. The backend image is `python:3.12-slim`; keep the code importable
on 3.12 (no reliance on PEP 649 deferred annotations).

## Inference boundary

`backend/app/inference/base.py` defines the `InferenceService` protocol. `MockInferenceService`
satisfies it today; a `RealBraTSInferenceService` loading `MODEL_PATH` is Phase 6. Swapping
implementations touches one construction site in `app/main.py` — no route, schema, or
frontend change. The UI must not assume a model is loaded.

Mock output is deterministic and clearly synthetic: a centered ellipsoid with concentric
ED / NCR / ET shells, `synthetic: true` on every measurement payload, and
`inference_source: "mock"` on every job and result.

## Documentation

Product requirements: `docs/Complete_PRD.md`. Architecture notes: `ARCHITECTURE.md`.
