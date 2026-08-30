# Document 4 — Technical Requirements Document (TRD)

## 1. Technical Goals
* Build a 3D segmentation pipeline whose preprocessing, training, and evaluation methodology is conceptually grounded in nnU-Net, without depending on the official nnU-Net framework as a library.
* Achieve a deployable end-to-end pipeline (preprocessing + sliding-window inference + post-processing) targeted under ~1–2 minutes per case on a single target GPU.

## 2. Technology Selection & Rationale
* **Frontend: React + Tailwind CSS + react-three-fiber + Axios**
  * *Why:* react-three-fiber gives idiomatic React integration for rendering a marching-cubes surface mesh with per-sub-region color, at far lower setup cost than a purpose-built medical volume renderer (vtk.js), which is reserved for a V1/stretch upgrade if full volumetric ray-casting is ever needed.
* **Backend: FastAPI (Python)**
  * *Why:* Async background execution for inference/evaluation jobs that are too slow to run synchronously in a single HTTP request; native OpenAPI docs; clean integration with the PyTorch inference module.
* **AI/ML: PyTorch + a custom 3D U-Net (nnU-Net-inspired) + nibabel/SimpleITK**
  * *Why:* A custom, from-scratch, config-driven 3D U-Net satisfies the learning objective of implementing the architecture/loss/training loop directly, avoids debugging a large third-party framework's internals under a 3-month deadline, and keeps full control over inference latency and output format for a smooth live demo.
* **Database: SQLite (MVP) with a documented PostgreSQL upgrade path**
  * *Why:* Zero-ops, file-based, sufficient for a single-instance academic deployment; upgrade path exists if concurrent multi-user load ever requires it.

## 3. System Non-Functional Technical Targets
* **Model Inference Latency:** End-to-end (preprocessing + sliding-window inference + post-processing) targeted under ~1–2 minutes per case — a measured, reported number, not a guaranteed SLA.
* **Determinism:** Same checkpoint + same input + fixed seed → identical output mask on repeat runs (no test-time-augmentation randomness by default).
* **API Behavior:** Every inference-triggering endpoint returns a job ID immediately (`202 Accepted`); results are fetched via a polling `GET /results/{id}` endpoint.

## 4. Functional Technical Requirements Mapping
| Requirement ID | Technical Requirement | Architectural Mapping |
| :--- | :--- | :--- |
| **TR-001** | Validate and preprocess a 4-modality BraTS case (orientation, resampling, foreground-only normalization). | Preprocessing Module — `nibabel`/SimpleITK pipeline (Document 9). |
| **TR-002** | Execute a custom 3D U-Net via sliding-window inference with Gaussian-weighted stitching. | PyTorch Inference Module wrapped in FastAPI background tasks. |
| **TR-003** | Compute Dice and Hausdorff95 per class and in aggregate, with documented edge-case handling. | Evaluation Module — physical-spacing-aware surface-distance computation. |
| **TR-004** | Extract a decimated 3D surface mesh from the predicted mask. | Visualization Module — `scikit-image.measure.marching_cubes`, exported as glTF/GLB or vertex/face JSON. |
