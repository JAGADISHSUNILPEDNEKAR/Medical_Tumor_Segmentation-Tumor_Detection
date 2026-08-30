# Document 2 — Product Requirements Document (PRD)

## Product Vision & Goals
Deliver a transparent, reproducible visual segmentation research tool that turns raw multi-modal BraTS MRI volumes into an inspectable 3D tumor visualization and a quantitative Dice/HD95 evaluation report — for radiologists, oncologists, and researchers alike.

## User Stories & Acceptance Criteria

### US-01: Case Ingestion & Preprocessing
* **As a** Radiologist or Researcher,
* **I want to** upload a 4-modality BraTS-format MRI case (T1, T1ce, T2, FLAIR),
* **So that** the system can validate, preprocess, and prepare it for segmentation.
  * **Acceptance Criteria:**
    * Client-side validation catches a missing modality before (or alongside) the server round-trip, naming the specific missing file.
    * Server-side validation checks NIfTI structure, shape, and affine consistency across all 4 modalities before parsing proceeds further.
    * Upload progress indicator shown for realistic BraTS case file sizes.

### US-02: 3D Visual Review of Segmentation
* **As a** Radiologist,
* **I want to** view 2D axial/coronal/sagittal slices with the segmentation overlay, and a rotatable 3D tumor mesh,
* **So that** I can visually verify sub-region boundaries rather than trust a single number.
  * **Acceptance Criteria:**
    * 2D slice viewer supports all 3 anatomical planes with an overlay toggle.
    * 3D mesh (marching-cubes, decimated to a bounded triangle budget) supports rotate/zoom/pan and per-sub-region visibility toggles.

### US-03: Automated Dice/HD95 Evaluation Reporting
* **As a** Researcher or Oncologist,
* **I want the system to** automatically compute Dice and Hausdorff95 per class and in aggregate against ground truth,
* **So that** I get a field-standard, reproducible quantitative report without manual metric computation.
  * **Acceptance Criteria:**
    * Per-class and mean Dice/HD95 reported with case-level aggregation (mean across cases, not a pooled voxel count).
    * Empty-mask and absent-class edge cases explicitly flagged, never silently zeroed.
    * Report includes evaluation split, case count, checkpoint ID, and a timestamp.

## Feature Priorities (MoSCoW)
* **Must Have:** BraTS preprocessing pipeline, custom 3D U-Net + sliding-window inference, Dice/HD95 evaluation module, 2D slice + 3D mesh viewer, FastAPI async job API, non-clinical-use disclaimer on every result screen.
* **Should Have:** Tumor volume estimation (mm³/cm³) surfaced in the UI, polished aggregate dashboard (Dice/HD95 distributions, per-class table), basic auth/session handling.
* **Could Have:** Case history screen with persisted past results, model versioning UI (compare two checkpoints).
* **Won't Have (for MVP):** Growth-tracking across timepoints, confidence/uncertainty visualization, official nnU-Net framework integration, DICOM/PACS/EHR ingestion.
