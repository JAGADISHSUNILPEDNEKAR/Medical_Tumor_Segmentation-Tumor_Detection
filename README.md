# Medical Image Segmentation & Tumor Detection

Research / decision-support prototype for BraTS-format brain MRI tumor sub-region segmentation.

**This is not a medical device. It is not clinically validated. It must not be used for diagnosis or treatment decisions.**

The 3D U-Net is trained in a separate notebook. This repository contains **Phase 6**: API health, case creation, four-modality NIfTI upload, spatial validation, isolated storage, async job execution behind a replaceable inference boundary, in-browser 2D/3D visualization, volumetric tumor measurements, ground-truth metric evaluation, diagnostic clinical reports, and **real PyTorch inference with the trained 3D U-Net**.

Two inference backends are selectable at runtime via `INFERENCE_BACKEND`:

- `mock` (default) — `MockInferenceService` synthesizes a geometric ellipsoid mask. No checkpoint, no PyTorch. Every screen that shows such a result is marked synthetic.
- `pytorch` — `RealBraTSInferenceService` runs the trained checkpoint. Results carry `inference_source: "pytorch"` and `synthetic: false`.

No trained checkpoint is committed to this repository (`models/*.pth` is gitignored). With `INFERENCE_BACKEND=pytorch` and no usable `MODEL_PATH`, the API starts, reports `model_loaded: false` / `inference_source: "unavailable"`, and returns **503** on ingest rather than falling back to synthetic output.

### Phase 6: Real PyTorch Model Integration [✓]

Connected the trained research pipeline to the production application with documented notebook parity.
- **Exact architecture reconstruction**: the notebook's custom `UNet3D` — 5 encoder stages, channel schedule `[32, 64, 128, 256, 320]`, 18,774,756 parameters, two 3×3×3 `Conv3d(bias=False)` → `InstanceNorm3d(affine=True)` → `LeakyReLU(0.01)` per block, strided-conv downsampling, transposed-conv upsampling. Not nnU-Net, not MONAI, not SegResNet (ADR-001).
- **Strict checkpoint loading**: reads the notebook's `save_checkpoint` payload format, verifies its recorded model/patch/preprocessing fingerprint, and always loads weights with `strict=True`. Missing, corrupt, and incompatible checkpoints each fail with a specific message; none are silently tolerated.
- **Preprocessing parity**: per-case foreground-only z-score per modality with background pinned at 0, union-foreground bounding-box crop with a 4-voxel margin, channels ordered `[T1, T1ce, T2, FLAIR]`. No resampling and no reorientation — the notebook measured neither as needed.
- **Gaussian sliding-window inference**: 128³ patches, overlap 0.5 (stride 64), separable Gaussian importance map (σ = 0.125 × dim), weighted accumulation and normalization, end-only padding that inverts by a plain crop.
- **Post-processing and export**: per-class connected-component filtering (min 50 voxels), internal `{0,1,2,3}` → BraTS `{0,1,2,4}` restoration, inverse crop back to the original volume, and a `segmentation.nii.gz` carrying the input's affine, spacing, and qform/sform codes.
- **Verified parity**: `scripts/notebook_parity_check.py` runs a verbatim transcription of the notebook's inference path alongside the production modules and compares preprocessed voxels, normalization statistics, crop box, probability map, labels, and exported mask. Bit-exact at the real trained geometry (240×240×155 volume, 128³ patches).

- **Verified against the trained checkpoint**: `best_model.pth` (epoch 19, 18,774,756 parameters) loads with `strict=True` into the reconstructed network, and its recorded configuration fingerprint `af663316e8bd` matches this service's derived fingerprint exactly — confirming the transcribed preprocessing, patch, and model configuration field-for-field.

Run it:

```bash
INFERENCE_BACKEND=pytorch MODEL_PATH=models/best_model.pth uvicorn app.main:app
```

See [docs/Phase6_Real_Inference.md](docs/Phase6_Real_Inference.md) for the full parity table, configuration reference, checkpoint verification, and known limitations.

### Phase 4: Medical Image Visualization [✓]

Built an in-browser MRI visualization suite without relying on external medical servers.
- **In-Memory NIfTI Engine**: Parses `.nii.gz` volumes using `nifti-reader-js`, preserving spatial affine geometry and scaling data without resampling.
- **Multiplanar Slicing**: Extracts and normalizes arbitrary Axial, Coronal, and Sagittal slices natively via HTML Canvas rendering for smooth performance.
- **3D Tumor Extraction**: A custom marching cubes engine generates lightweight, WebGL-ready triangle meshes directly from the segmentation voxel masks.
- **Interactive Tooling**: Crosshair synchronization across 2D planes, overlay opacity and per-region visibility, modality toggling (T1, T1ce, T2, FLAIR), keyboard slice step-through, a case metadata panel, and interactive 3D OrbitControls (via `react-three-fiber`).

### Phase 5: Reports, Measurements & Results [✓]

Built a clinical results presentation and quantitative measurement pipeline consuming segmentation results.
- **Volumetric Extraction**: `MeasurementService` calculates exact physical tumor volumes (cm³ and mm³), voxel counts, 3D axis-aligned bounding boxes, and physical centroids for WT (Whole Tumor), TC (Tumor Core), and ET (Enhancing Tumor) preserving NIfTI affine spacing.
- **Quantitative Metrics Evaluation**: `MetricsService` computes Dice Similarity Coefficients and 95th-percentile Hausdorff Distance (HD95) using surface boundary extraction via binary erosion, with graceful handling of clinically empty mask edge cases.
- **Model Provenance**: Explicit tracking of model version, checkpoint identifier, inference timestamp, and synthetic disclaimer flags.
- **Clinical Report Page**: Dedicated, printable diagnostic report view (`/cases/:caseId/jobs/:jobId/report`) with patient/scan metadata, volumetric summaries, comparative ground-truth evaluation, and non-diagnostic disclaimers.

## Current status

| Capability | Status |
|---|---|
| Frontend shell | Phase 1 |
| Backend health | Phase 1 |
| Model info | Phase 6 (`model_loaded` reflects a real loaded checkpoint; `headline_metrics` always null — none are registered) |
| Case upload + NIfTI validation | Phase 2 |
| Async job queue + result endpoints | Phase 3 |
| Mock inference (synthetic, non-clinical) | Phase 3 |
| 2D multiplanar viewer + 3D tumor mesh | Phase 4 |
| Measurements UI | Phase 5 (volumetric summary, regional breakdown, printable report) |
| Real model inference | Phase 6 (`INFERENCE_BACKEND=pytorch`, trained 3D U-Net, notebook parity verified) |
| Dice / HD95 evaluation | Phase 5 (evaluated against ground truth segmentation) |

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
| `INFERENCE_BACKEND` | `mock` (default) or `pytorch` |
| `MODEL_PATH` | Path to the trained `best_model.pth`. Required for `pytorch` |
| `MODEL_VERSION` | Optional label. Empty = derived from the checkpoint's digest and epoch |
| `INFERENCE_DEVICE` | `auto` (default) / `cpu` / `cuda` / `cuda:0` |
| `INFERENCE_STRICT_FINGERPRINT` | `true` (default). Never relaxes `state_dict` strictness |
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
  With `INFERENCE_BACKEND=pytorch` and a loaded checkpoint: `{ "status": "ok", "model_loaded": true, "inference_source": "pytorch" }`
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

The compose file mounts `./models` read-only at `/models` and passes
`INFERENCE_BACKEND`, `MODEL_PATH`, `MODEL_VERSION` and `INFERENCE_DEVICE` through
from the environment, so real inference runs with:

```bash
INFERENCE_BACKEND=pytorch MODEL_PATH=/models/best_model.pth docker compose up --build
```

The backend image is `python:3.12-slim` and installs the **CPU** torch wheel — it
starts and serves without a GPU, but full-scale 128³ inference on CPU is not
practical (see `docs/Phase6_Real_Inference.md` §9). A CUDA base image is still
required for GPU deployment. Keep the code importable on 3.12 (no reliance on
PEP 649 deferred annotations).

## Inference boundary

`backend/app/inference/base.py` defines the `InferenceService` protocol.
`MockInferenceService` and `RealBraTSInferenceService` both satisfy it, and
`app/inference/factory.py` picks between them from `INFERENCE_BACKEND`. Swapping
implementations touches one construction site in `app/main.py` — no route, schema, or
frontend change. The UI must not assume a model is loaded.

Mock output is deterministic and clearly synthetic: a centered ellipsoid with concentric
ED / NCR / ET shells, `synthetic: true` on every measurement payload, and
`inference_source: "mock"` on every job and result.

## Documentation

Product requirements: `docs/Complete_PRD.md`. Architecture notes: `ARCHITECTURE.md`.
