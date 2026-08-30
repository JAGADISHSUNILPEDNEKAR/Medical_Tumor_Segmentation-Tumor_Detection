# Document 13 — Cost Analysis, Roadmap & Team Responsibilities

## 1. Cost Analysis (Monthly Estimate)

| Category | Component Description | Provider / Specification | Estimated Monthly Cost (USD) |
| :--- | :--- | :--- | :--- |
| **Compute / GPU** | Single GPU Server Instance (3D volumes need more VRAM than 2D CV tasks) | Dedicated GPU Server (e.g., RTX 4090 / 1x A10) | $220.00 |
| **Database** | SQLite (file-based, MVP) → PostgreSQL (V1) | Co-located on compute host (Dockerized) | $0.00 |
| **Storage** | Object / Disk Storage for NIfTI cases, masks & checkpoints | 250 GB NVMe Storage Block | $18.00 |
| **Networking** | Bandwidth egress | 500 GB Data Transfer | $10.00 |
| **Total** | | | **~$248.00 / month** |

*Illustrative estimate — per Document 1's risk register, train in short checkpointed sessions and run the GPU inference endpoint only during active demo/grading windows to control actual spend.*

## 2. Project Roadmap (3 Months)
* **Month 1 — Foundations:** Study nnU-Net paper & 3D U-Net fundamentals; confirm the exact BraTS release/access; build the preprocessing pipeline (v1); scaffold FastAPI + React; smoke-test a baseline training run.
* **Month 2 — Core ML + API:** Full training & tuning; sliding-window inference; Dice/HD95 evaluation implementation; FastAPI endpoints (`predict`/`evaluate`/`results`); frontend upload & analysis screens.
* **Month 3 — Integration + Polish:** Full frontend/backend integration; 3D visualization (mesh pipeline + viewer); dashboard + history; unit/integration/E2E testing; documentation (README/ARCHITECTURE/MODEL_CARD/etc.); deployment; demo & final presentation prep.

## 3. Team Responsibilities (1–2 Person Team)
* **Role A — AI/ML Lead & Backend Architect:**
  * BraTS dataset acquisition, fingerprinting, and preprocessing pipeline implementation.
  * Custom 3D U-Net design, training loop, compound Dice+CE loss, and checkpoint management.
  * FastAPI async job orchestration and Dice/HD95 evaluation module.
* **Role B — Full-Stack & Visualization Lead:**
  * React SPA: upload flow, 2D slice viewer, react-three-fiber 3D mesh viewer.
  * Results Dashboard and (V1) case history screen.
  * Docker Compose containerization, CI/CD pipeline setup, and documentation.

*For a solo team, both role sets are carried by the same person across the 3-month timeline above.*
