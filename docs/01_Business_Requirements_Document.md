# Document 1 — Business Requirements Document (BRD)

## 1. Executive Summary
Manual delineation of tumor sub-regions in multi-modal MRI is slow, inter-observer variable, and a bottleneck in both clinical workflow and research. The **Medical Image Segmentation & Tumor Detection System** trains and demonstrates a custom 3D U-Net — conceptually grounded in nnU-Net's self-configuring methodology — on the BraTS dataset, then delivers the result through a full-stack app so it is inspectable, not just a notebook metric. The product's value is **transparency and reproducibility of a segmentation research pipeline**, not clinical decision-making authority.

## 2. Problem Statement
* **Current State:** Tumor sub-region delineation is performed manually by radiologists, slice by slice, with known inter-observer variability.
* **Affected Stakeholders:** Radiologists, oncologists, medical AI researchers, ML engineers/student evaluators.
* **Bottlenecks:** Slow manual annotation, lack of a reproducible research pipeline, no transparent way to inspect model quality beyond a single notebook metric.
* **Business/Academic Impact:** Segmentation research that isn't wrapped in a usable, reproducible system is hard to demo, hard to trust, and hard for a second person to reproduce.

## 3. Vision
To build a transparent, reproducible 3D segmentation research pipeline — from raw NIfTI volumes to a trained model to an inspectable full-stack application — that a radiologist, oncologist, or researcher can use to see *and* quantify candidate tumor sub-regions, without ever presenting itself as a diagnostic authority.

## 4. Objectives (SMART, derived from Goals & Success Criteria)
* **Model Correctness:** Train a 3D U-Net to convergence on the official BraTS training split, producing non-degenerate masks (not all-background) on held-out cases.
* **Evaluation Rigor:** Compute per-class and mean Dice and Hausdorff95 correctly, per case then aggregated, with every documented edge case (empty mask, absent class) handled explicitly rather than silently zeroed.
* **Demonstrability:** Run the full pipeline — upload → analyze → 3D view → metrics — live in a 5–10 minute demo without manual patching.
* **Reproducibility:** A second person can retrain/re-evaluate from config files and a README without asking the original author questions.

## 5. Target Personas
* **Persona 1: Radiologist (Reviewer)**
  * *Goals:* Fast visual localization of candidate tumor sub-regions to speed up manual review.
  * *Pain Points:* Needs to trust — but independently verify — anything a model proposes; should never infer the mask is diagnostic truth needing no verification.
  * *Tech Skill:* Intermediate web application user.
* **Persona 2: Oncologist**
  * *Goals:* Tumor volume and rough sub-region breakdown to support discussion, not decision.
  * *Pain Points:* Must not mistake a volume trend across scans for a validated clinical progression assessment (e.g., RANO criteria).
* **Persona 3: Medical AI Researcher / ML Engineer**
  * *Goals:* Reproducible Dice/HD95 numbers, checkpoint provenance, ability to re-run evaluation and cross-check against config files.
  * *Tech Skill:* Advanced, metric-literate.

## 6. User Journey
1. **Landing & Disclaimer:** User opens the app, reads the persistent non-clinical-use disclaimer.
2. **Case Upload:** User uploads (or selects) a 4-modality BraTS-format case (T1, T1ce, T2, FLAIR).
3. **Automated Pipeline:** System validates and preprocesses the volumes, runs sliding-window 3D U-Net inference, and (if ground truth is present) computes Dice/HD95.
4. **Visual Review:** User inspects 2D axial/coronal/sagittal slices with the segmentation overlay, then rotates/zooms/pans a 3D tumor mesh, toggling sub-regions on and off.
5. **Quantitative Report:** User reviews per-class and mean Dice/HD95, tumor volume, and the model/checkpoint version on the Results Dashboard.

## 7. Business Use Cases
* **UC-01:** Segment tumor sub-regions (necrotic core, edema, enhancing tumor) from a 4-modality MRI case.
* **UC-02:** Evaluate a segmentation against ground truth using Dice and Hausdorff95, per class and in aggregate.
* **UC-03:** Visually and quantitatively review a case via a 2D slice viewer and an interactive 3D tumor mesh.

## 8. Functional Business Requirements
| Requirement ID | Requirement Description | Priority | Business Justification | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **BR-001** | 4-Modality Case Ingestion | High | Enable ingest of BraTS-format NIfTI cases with structural validation. | Reject cases missing any of T1/T1ce/T2/FLAIR with a specific error naming the missing modality. |
| **BR-002** | 3D Deep Learning Tumor Segmentation | High | Produce non-degenerate tumor sub-region masks via a custom 3D U-Net. | Trained checkpoint's validation Dice exceeds the trivial all-background baseline by a non-trivial margin. |
| **BR-003** | Quantitative Evaluation Engine | High | Compute field-standard Dice/HD95 metrics against ground truth. | Per-class and mean Dice/HD95 reported, matching the documented aggregation and edge-case rules exactly. |
| **BR-004** | 2D + 3D Visual Review Interface | High | Let reviewers inspect the mask against the anatomy, not just trust a number. | Users can toggle overlay visibility and rotate/zoom/pan a decimated 3D mesh at interactive frame rates. |
| **BR-005** | Reproducible Research Pipeline | Medium | Ensure the pipeline is independently rerunnable and auditable. | A second person can retrain/re-evaluate from `configs/` and the README without asking questions. |

## 9. Non-Functional Business Requirements
* **Performance:** End-to-end inference (preprocessing + sliding-window inference + post-processing) targeted under ~1–2 minutes per case on the target GPU (measured, not a guaranteed SLA).
* **Availability:** Single-instance academic/demo deployment; concurrent inference requests queued FIFO rather than parallel-executed and OOM-crashed.
* **Security & Privacy:** No real patient data — only public, de-identified BraTS research data; opaque non-guessable case/result IDs; no PHI ever logged.

## 10. Success Metrics
* Every documented functional requirement (§10 of the TRD) demonstrable live within a 5–10 minute demo.
* Dice and HD95 computed correctly and cross-checked against a hand-computed or independent reference implementation at least once.
* A second person can reproduce training/evaluation from configs and the README alone.

## 11. Assumptions
* A single GPU with enough memory for at least a modest 3D patch (e.g., 128³ at a reasonable batch size) is available for training.
* The team has, or will confirm, legitimate access to a BraTS release under its stated license terms.
* SQLite is sufficient for the metadata DB given the academic/demo scale of the project, with a documented upgrade path to PostgreSQL.

## 12. Constraints
* **Compute:** Single mid-tier GPU for both training and local inference serving.
* **Duration:** 3-month development cycle, sized for a 1–2 person engineering team (see Document 13).
* **Scope:** BraTS-defined MRI sequences only (T1, T1ce, T2, FLAIR) — no arbitrary DICOM ingestion, no PACS/EHR integration, no regulatory submission artifacts.

## 13. Risks & Mitigations
| Risk | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| GPU memory limitations block training at a useful patch size | Medium | High | Mixed-precision training, patch-based training by design, tunable patch size/channel width. |
| Class imbalance yields a degenerate all-background model | Medium | High | Compound Dice + Cross-Entropy loss with foreground-oversampled patch sampling. |
| Metric implementation errors undermine every downstream claim | Low–Medium | High | Unit tests against hand-computed synthetic cases; explicit empty-mask edge-case handling. |
| Scope creep (chasing full official nnU-Net integration, over-building auth/security) | Medium | High | Explicit MVP/V1/Stretch boundaries with a stated priority ordering; cut V1/Stretch first. |

## 14. MVP Scope
* **In Scope:** BraTS-format 4-modality preprocessing; custom 3D U-Net training; sliding-window inference; Dice + HD95 evaluation; FastAPI service with async job handling; React app (landing/disclaimer, upload, 2D slice + 3D mesh analysis workspace, results screen); README + architecture doc + model card; unit tests for preprocessing/metrics + one API integration path.
* **Out of Scope (MVP):** Multi-institution data harmonization, non-BraTS/raw-DICOM ingestion, clinical workflow (PACS/EHR) integration, production multi-tenant auth, regulatory submission artifacts.

## 15. Future Scope
* Case history with persisted past results, polished aggregate dashboard, basic auth/session handling (V1).
* Tumor volume estimation surfaced in the UI (V1); growth-tracking scaffold across timepoints, model versioning/comparison UI, confidence/uncertainty visualization (Stretch) — explicitly never presented as a validated clinical progression tool.
