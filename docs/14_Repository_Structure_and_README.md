# Document 14 — Repository Structure & README

## 1. GitHub Repository Structure

```
medical-image-segmentation-tumor-detection/
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── api/            # route definitions, versioned (v1/)
│   │   ├── core/           # config loading, logging setup
│   │   ├── models/         # Pydantic request/response schemas
│   │   ├── db/              # DB models + migrations
│   │   ├── services/       # job orchestration, calls into ml/
│   │   └── main.py
│   └── tests/
├── ml/                      # ML pipeline — independently executable, no FastAPI dependency
│   ├── data/                # dataset loading, splitting, fingerprinting
│   ├── preprocessing/       # resample / normalize / crop stages
│   ├── model/               # 3D U-Net definition
│   ├── training/            # training loop, compound Dice+CE loss
│   ├── inference/           # sliding-window inference
│   ├── evaluation/          # Dice/HD95
│   ├── postprocessing/      # connected-component filtering, mesh export
│   └── tests/
├── frontend/                # React application
│   ├── src/
│   │   ├── pages/           # Landing, Upload, Analysis, Results, History
│   │   ├── components/      # SliceViewer, MeshViewer3D, Dashboard widgets
│   │   ├── api/             # Axios client
│   │   └── App.tsx
│   └── tests/
├── models/                  # checkpoint artifacts (git-ignored; referenced by manifest)
├── data/                    # local dataset cache (git-ignored)
├── configs/                 # dataset.yaml, model.yaml, training.yaml, evaluation.yaml, api.yaml
├── scripts/                 # CLI entry points: preprocess.py, train.py, evaluate.py
├── docker/                  # Dockerfiles for backend, frontend, and a combined dev compose file
├── docs/                    # ARCHITECTURE.md, DATASET.md, API.md, EVALUATION.md, MODEL_CARD.md
├── tests/                   # end-to-end / integration tests spanning multiple modules
├── .github/workflows/ci.yml
├── docker-compose.yml
├── LICENSE
└── README.md
```

## 2. Production README

```markdown
# Medical Image Segmentation & Tumor Detection

> A reproducible 3D deep-learning pipeline for brain tumor sub-region segmentation on BraTS MRI, conceptually grounded in nnU-Net, wrapped in a FastAPI service and a React + react-three-fiber 3D viewer.

## Overview
Manual delineation of tumor sub-regions in multi-modal MRI is slow and inter-observer variable. This project trains a custom 3D U-Net on the BraTS dataset, evaluates it with the field-standard Dice and Hausdorff95 metrics, and delivers the result through a full-stack app so it is inspectable, not just a notebook metric.

> **This is a research prototype. It is not a medical device and must not be used for diagnosis or treatment decisions.**

## Tech Stack
- **Frontend:** React, Tailwind CSS, react-three-fiber, Axios
- **Backend:** FastAPI, Python
- **AI/ML:** PyTorch, nibabel, SimpleITK, scikit-image
- **Database:** SQLite (MVP) / PostgreSQL (V1 upgrade path)

## Quick Start (Local Setup via Docker)

1. **Clone Repository:**
   ```bash
   git clone https://github.com/org/medical-image-segmentation-tumor-detection.git
   cd medical-image-segmentation-tumor-detection
   ```

2. **Launch System Containers:**
   ```bash
   docker-compose up --build -d
   ```

3. **Access Services:**
   - Web Application Dashboard: `http://localhost:3000`
   - Interactive OpenAPI Documentation: `http://localhost:8000/docs`

## License
Distributed under the MIT License. See `LICENSE` for details.
```
