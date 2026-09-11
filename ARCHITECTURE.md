# Architecture — Phase 1

Research / decision-support prototype. **Not a diagnostic device. Not clinically validated.**

## Boundary

The production application serves cases, visualization, and reports. Training stays in a separate notebook/research pipeline. This repository's web stack must not train models.

```mermaid
flowchart LR
    User[User] --> FE[React SPA]
    FE -->|REST /api/v1| BE[FastAPI]
    BE --> Inf[InferenceService protocol]
    Inf -.-> Mock[MockInferenceService later]
    Inf -.-> Real[RealBraTSInferenceService later]
```

Phase 1 implements the left side only: frontend shell, FastAPI process, configuration, health, and the `InferenceService` interface. No upload, jobs, mock predictions, or checkpoint loading.

## Runtime (Phase 1)

```mermaid
flowchart TB
    Browser[Browser :3000]
    Browser -->|dev proxy or nginx /api| API[FastAPI :8000]
    API --> Health["GET /api/v1/health"]
    API --> Info["GET /api/v1/model/info"]
    Health --> Status["model_loaded: false\ninference_source: unavailable"]
```

## Replaceable inference

`backend/app/inference/base.py` defines:

```text
InferenceService.predict(case) -> result
```

The frontend never knows whether a future result came from a mock, local PyTorch, or a remote GPU server. Phase 1 does not register an implementation.

## Configuration

All paths and origins come from environment variables (`pydantic-settings`). `MODEL_PATH` is reserved for Phase 6. Setting it in Phase 1 does **not** load weights and does **not** change `model_loaded`.

## What is intentionally absent

- MRI upload and NIfTI validation
- SQLite / PostgreSQL
- Job queue and mock segmentation
- 2D/3D viewers and reports
- `ml/` training code
