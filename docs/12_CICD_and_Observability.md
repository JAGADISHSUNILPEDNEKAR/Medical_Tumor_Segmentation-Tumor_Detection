# Document 12 — CI/CD, Observability & Deployment

## 1. CI/CD Pipeline Design

```
[Developer Push] ──► [GitHub Actions Triggered]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Lint & Code Style]        [PyTest Unit Suite]
   (Flake8 / Black / ESLint)  (Preprocessing, Metrics, API)
             │                           │
             └─────────────┬─────────────┘
                           ▼
              [Build Docker Container]
                           │
                           ▼
          [Deploy to Single-GPU Server]
```

### GitHub Actions Workflow Configuration
* Automatic runs on `push` to `main` or `pull_request`.
* Steps:
  1. Checkout repository code.
  2. Set up Python (backend/ML) and Node.js (frontend) environments.
  3. Install PyTorch (CPU mode, for fast CI execution) and backend dependencies.
  4. Run the PyTest suite (`tests/unit`, `tests/integration`).
  5. Run frontend tests via Vitest/Jest.

## 2. Observability & Monitoring
* **API Metrics:** Request latency (p95, p99), HTTP status code distribution, throughput (requests/sec).
* **Compute Metrics:** GPU VRAM utilization (peak allocated, to debug OOM errors), GPU/CPU/memory utilization.
* **ML Inference Metrics:** Per-stage processing time (preprocessing, inference, evaluation, mesh export), per job.
* **Structured Logging:** JSON logs including request ID, job/inference ID, model/checkpoint version, and timestamp — never uploaded voxel data or patient-identifying metadata.

## 3. Deployment Architecture

```
                       ┌─────────────────────────┐
                       │      NGINX REVERSE      │
                       │     PROXY (PORT 80/443) │
                       └────────────┬────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           │                                                 │
┌──────────▼──────────────┐                       ┌──────────▼──────────────┐
│  Frontend Container     │                       │  Backend Container      │
│  (React Static Build)   │                       │  (FastAPI + PyTorch)    │
└─────────────────────────┘                       └──────────┬──────────────┘
                                                             │
                                                  ┌──────────▼──────────────┐
                                                  │  Database Container     │
                                                  │  (SQLite / PostgreSQL)  │
                                                  └─────────────────────────┘
```

* **Local development:** `docker-compose` bringing up backend + a dev-mode frontend against a local SQLite DB and local file storage; CPU inference acceptable for dev iteration if no GPU is available.
* **GPU deployment:** hosting target not fixed by this document — a single cloud GPU VM (stopped outside active demo/eval windows to control cost), a lab-provided GPU box, or a managed inference endpoint are all acceptable; verify actual GPU resources before pinning the Dockerfile's CUDA/PyTorch base image.
