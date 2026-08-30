# Document 15 — ADRs, Traceability Matrix, Interview Prep & Quality Score

## 1. Architecture Decision Records (ADRs)

### ADR-001: Custom 3D U-Net vs. Official nnU-Net Framework
* **Status:** Accepted
* **Context:** Needed a 3D segmentation architecture and training methodology for BraTS tumor sub-region classification, within a 3-month, single-GPU, small-team constraint.
* **Decision:** Implement a custom 3D U-Net conceptually grounded in nnU-Net's ideas (fingerprinting, resampling, compound loss, sliding-window inference), without depending on the official nnU-Net codebase.
* **Rationale:** The course's learning objective is 3D U-Net theory and implementation — delegating to the official framework would satisfy the deliverable but not the learning goal, and the framework's own setup/convention overhead does not fit a 3-month timeline that also has to build a full-stack app.
* **Consequences:** The team owns every methodology decision directly (and can debug it), at the cost of not automatically benefiting from nnU-Net's automatic architecture/config search or official 5-fold ensembling.

### ADR-002: react-three-fiber vs. vtk.js for 3D Visualization
* **Status:** Accepted
* **Context:** Needed to render a rotatable/zoomable tumor mesh in the browser, integrated with the required React frontend stack.
* **Decision:** Use react-three-fiber (Three.js + React bindings) rendering a marching-cubes-extracted surface mesh, rather than vtk.js's full volumetric ray-casting.
* **Rationale:** react-three-fiber is idiomatic within the required React stack and sufficient for "2D slices for detail + 3D mesh for tumor shape," which satisfies every stated UX requirement (rotate, zoom, pan, toggle, distinguish sub-regions) at much lower implementation cost than vtk.js's medical-imaging-specific but heavier-learning-curve API.
* **Consequences:** Full volumetric ray-casting of the raw MRI intensity volume is out of scope for MVP; reserved as a V1/stretch upgrade if needed.

## 2. Traceability Matrix

| Business Requirement (BR) | Feature Description | User Story | Technical Requirement (TR) | API Endpoint | Component / Class | Database Entity | Test Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BR-001** | 4-Modality Case Ingestion | US-01 | TR-001 | `POST /predict` | `PreprocessingModule` | `CASES` | TEST-04 |
| **BR-002** | 3D Deep Learning Tumor Segmentation | US-01 | TR-002 | `POST /predict` | `InferenceModule` | `JOBS` | TEST-03 |
| **BR-003** | Quantitative Evaluation Engine | US-03 | TR-003 | `POST /evaluate` | `EvaluationModule` | `RESULTS`, `METRICS` | TEST-01 |
| **BR-004** | 2D + 3D Visual Review Interface | US-02 | TR-004 | `GET /results/{id}` | `VizExportStage` | `RESULTS` | TEST-05 |
| **BR-005** | Reproducible Research Pipeline | US-01/02/03 | TR-001–004 | `GET /model/info` | `CheckpointRegistry` | `CHECKPOINTS` | TEST-05 |

## 3. Interview Preparation & Viva Questions

### Product & Technical Questions
* **Q: Why ground the model in nnU-Net's methodology without using the official framework?**
  * *Answer:* The learning objective is implementing 3D U-Net theory directly — a custom, config-driven pipeline is fully understood, debuggable, and reproducible by a small team within a 3-month window, while the field-validated methodology (fingerprinting, compound loss, sliding-window inference) is still adopted conceptually.
* **Q: Why report HD95 instead of raw Hausdorff distance as the primary boundary metric?**
  * *Answer:* Raw Hausdorff distance is dominated by a single worst-case outlier voxel, making it unstable for model comparison; HD95 discards the top 5% most extreme distances while still penalizing genuine boundary disagreement, which is why it is the field-standard metric on BraTS leaderboards.

### Viva Defense Questions
* **Beginner:** What framework was used to build the REST API? (*FastAPI*)
* **Intermediate:** Why is InstanceNorm preferred over BatchNorm for this task? (*3D volumetric training uses very small batch sizes — 1–4 patches — at which BatchNorm's running statistics become unreliable; InstanceNorm normalizes per-sample, independent of batch size.*)
* **Advanced:** How is a train/inference preprocessing mismatch avoided when adopting nnU-Net's ideas without its framework? (*A single, versioned `configs/dataset_fingerprint.yaml` is computed once at fingerprinting time and reused identically at both training and inference time, so there is only one source of truth for resampling/normalization parameters.*)

## 4. Final Project Quality Score

```
+-------------------------------------------------------------+
|               PROJECT EVALUATION SCORECARD                  |
+-------------------------------------------------------------+
|  Dimension                     Score   Weight   Weighted    |
+-------------------------------------------------------------+
|  1. Business Value              8 / 10   1.0       8.0      |
|  2. Problem Clarity              9 / 10   1.0       9.0      |
|  3. UX Specification            8 / 10   1.0       8.0      |
|  4. Technical Complexity        10 / 10   1.0      10.0      |
|  5. Architecture                9 / 10   1.0       9.0      |
|  6. Database & Data Design      8 / 10   1.0       8.0      |
|  7. Multimodal AI Integration   10 / 10   1.0      10.0      |
|  8. Security Design             7 / 10   1.0       7.0      |
|  9. Testing & CI/CD             8 / 10   1.0       8.0      |
| 10. Cost & Deploy Realism        8 / 10   1.0       8.0      |
+-------------------------------------------------------------+
|  TOTAL SCORE                              85.0 / 100 (8.5/10)|
+-------------------------------------------------------------+
```
