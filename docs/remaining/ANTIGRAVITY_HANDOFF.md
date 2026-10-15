# Antigravity Session Handoff: Clinical Roadmap Review & Architectural Assessment

**Date:** 2026-10-12  
**Conversation ID:** `04f349fd-f34e-4e40-a912-66c6e370acb1`  
**Repository:** `ojt-5-hf-brain-tumor-classification`  
**Target Document Reviewed:** [`roadmap.md`](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md)  
**Status:** Complete session record & engineering handoff.

---

## 1. Session Integrity & Context Access Statement

All messages, user prompts, assistant evaluations, and metadata in this document were retrieved directly and verbatim from the system chronological logs (`transcript_full.jsonl`). No earlier parts of this conversation session were inaccessible or truncated.

---

## 2. Chronological Conversation Record

### 2.1 Exchange 1: Roadmap Completion & Clinical Utility Audit

#### User Prompt 1:
> @[/Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md] are all this features done and the platform supports everything that a radiologists and oncologists need?

#### Assistant Response 1:
**No**, not all features in [roadmap.md](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md) are completed, and **the platform does not yet support everything that radiologists and neuro-oncologists need** in a clinical environment.

As stated in the roadmap itself (§1):
> *"research / decision-support prototype, not a medical device, not clinically validated."*

---

### Part 1: Are all the features in `roadmap.md` done?

While significant progress has been made on the **P0**, **P1**, and early **P2** priorities, several features are **partially implemented, missing, or explicitly deferred**:

#### 1. What is Implemented and Completed
* **P0 Diagnostics & Metrics:**
  * Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) composite volumes ([P0-1](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L61)).
  * 3D maximal diameters, in-plane major/minor axes, and bidimensional product ([P0-2](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L62)).
  * ET presence flag, necrotic core fraction, edema-to-core ratio ([P0-3](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L63)).
  * True scanner-space centroid coordinate correction ([P0-4](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L64)).
  * Composite Dice / HD95 evaluation ([P0-5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L65)).
  * Case worklist view and API endpoint ([P0-7](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L67)).
  * Domain-shift input similarity detection ([P0-9](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L69)).
* **P1 Longitudinal Architecture:**
  * Subject entities with study dates and timepoints ([P1-1](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L82)).
  * Longitudinal tumor volume tracking panel ([P1-2](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L84)).
  * Prior vs. current side-by-side comparative viewer ([P1-3](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L87)).
  * RANO 2010 response assessment calculator ([P1-4](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L88), detailed in [docs/rano_response_assessment.md](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/docs/rano_response_assessment.md)).
  * Multimodal treatment timeline (surgery, radiotherapy courses, chemotherapy, steroid tracking) ([P1-5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L94)).
* **P2 Clinical Context:**
  * WHO CNS 2021 molecular & pathology records (IDH, 1p/19q, MGMT, ATRX, etc.) ([§5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L105)).
  * Model vs. pathology reconciliation and cohort checks ([§5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L106)).
  * Extent of resection calculations with 72-hour timing validation ([§5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L107)).
  * 2D caliper distance and ROI region measurements saved to case records ([§5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L109)).
  * Key-image reference bookmarking ([§5](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L109)).

---

#### 2. What is Still Incomplete or Remaining in the Roadmap
| Feature | Status in Roadmap (§9) | Gap in Codebase |
|---|---|---|
| **Exports (P0-6)** | **Partially done** | Measurements CSV and segmentation NIfTI download are shipped, but **server-side PDF generation is deferred** (currently relies on browser `window.print()`). |
| **Data Retention (P0-8)** | **Partially done** | Case deletion with file removal is implemented, but **automated scheduled retention sweeps** do not exist. |
| **Contour Correction (P2)** | **Partially done** | Supports uploading an externally corrected NIfTI mask, but **in-viewer interactive brush/eraser editing is not built**. |
| **Audit Trail (P2)** | **Partially done** | Shipped as an append-only *Activity History* ([docs/activity_history.md](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/docs/activity_history.md)); does not log read events, lacks role separation, and is not tamper-resistant. |
| **2×2 Multi-Sequence Compare (P2)** | **Not done** | The viewer shows orthogonal planes (Axial/Coronal/Sagittal) of **one sequence at a time**; simultaneous side-by-side view of T1ce, T2, FLAIR, and T1 is absent. |
| **Tumor Board Packet (P2)** | **Not done** | Consolidated one-click export (anonymized key slices + volume changes + treatment history PDF) is not implemented. |
| **W/L Presets & Intensity Readout (P2)** | **Not done** | Standard radiological Window/Level presets and calibrated cursor intensity readouts are missing. |
| **User Roles / RBAC (P2)** | **Not done** | No permission tiering (e.g., attending radiologist vs. resident vs. oncologist vs. researcher). |
| **P3 Infrastructure Items** | **Deferred** | DICOM / DICOMweb / PACS ingestion, DICOM SEG, RTSTRUCT, HL7-FHIR reporting, and Postgres/cloud storage are deferred. |

---

### Part 2: Does the platform support everything radiologists and oncologists need?

**No.** Real-world neuro-radiology and neuro-oncology workflows require several fundamental capabilities that are outside the prototype's current capabilities:

#### 1. Ingestion & Imaging Formats (The Largest Practical Blocker)
* **No Raw DICOM or PACS Connectivity:** Hospital scanners produce multi-slice DICOM series across hospital PACS/VNA systems. Radiologists do not manually download, skull-strip, co-register, and convert scans to four `.nii.gz` files to upload to a browser.
* **No Preprocessing Pipeline for Clinical Scans:** The ML models assume BraTS-standard inputs (1 mm isotropic, skull-stripped, co-registered to SRI24 space, LPS orientation). Unprocessed clinical DICOM scans have variable slice thickness (e.g., 5 mm axial FLAIR), skull tissue, and non-isotropic resolution. Feeding raw scans into the model produces invalid segmentations.
* **No RTSTRUCT or DICOM-SEG Export:** Radiation oncologists and neurosurgeons cannot import NIfTI files directly into Treatment Planning Systems (TPS like Eclipse, RayStation) or surgical neuronavigation systems (Brainlab, StealthStation). They require **DICOM RTSTRUCT** or **DICOM-SEG**.

#### 2. Neuroradiology Workflow Needs
* **Simultaneous Multi-Sequence Sync (2×2 or 4-up):** Radiologists do not assess tumors using one sequence in isolation. Distinguishing true contrast enhancement from intrinsic T1 hyperintensity (subacute blood, proteinaceous fluid, melanin) requires looking at T1-pre and T1-post simultaneously, while FLAIR and T2 are needed in parallel to evaluate non-enhancing tumor vs. vasogenic edema.
* **Direct In-Viewer Slice-by-Slice Contour Editing:** Radiologists will not sign off on an AI segmentation without the ability to touch up contours in seconds using brush, erase, and smart interpolation tools directly inside the reading viewer.
* **Hanging Protocols & Multi-Monitor Support:** High-throughput reading rooms depend on multi-display setups, standardized hanging protocols, and calibrated DICOM Part 14 Grayscale Standard Display Function (GSDF) monitors.
* **Advanced MRI Sequences:** Distinguishing radiation necrosis or pseudoprogression from true tumor recurrence often requires **Perfusion MRI (rCBV via DSC/DCE)**, **Diffusion (DWI/ADC)**, or **MR Spectroscopy**, none of which are handled by this platform.

#### 3. Neuro-Oncology & Clinical Trial Workflow Needs
* **RANO 2.0 / iRANO Standards:** The platform implements RANO 2010 (2D perpendicular diameters for high-grade glioma). Current clinical practice and trials are adopting RANO 2.0 (incorporating volumetric criteria, non-enhancing disease components, and updated post-radiotherapy timelines) and iRANO (for immunotherapies).
* **EHR & LIS Interoperability (HL7 / FHIR):** Oncologists need reports pushed directly into electronic health records (e.g., Epic/Cerner) via HL7 ORU or FHIR `DiagnosticReport`, and molecular pathology imported directly from lab systems rather than entered by hand.
* **Multidisciplinary Tumor Board (MDT) Packet:** The ability to generate presentation-ready, de-identified case digests with key image cross-sections, volumetric response plots, and steroid timelines for weekly tumor boards.

#### 4. Compliance and Regulatory Status
* **Not Cleared as a Medical Device:** The platform has no FDA 510(k) clearance, De Novo classification, or CE-MDR marking. It cannot be used for primary diagnostic or clinical decision-making.
* **Single-Worker Concurrency & HIPAA:** The backend operates on an in-process SQLite database with a single-worker queue, which cannot serve concurrent departmental readers or satisfy enterprise HIPAA/audit security mandates.

---

### Summary

The platform has a strong foundation for **volumetric tracking, RANO calculation, domain-shift safety checks, and longitudinal reporting**, but it is still a **research and decision-support prototype**. 

To be considered a tool ready for daily clinical reading and oncology workflows, it would still need:
1. Native DICOM / PACS / DICOMweb ingestion and automatic clinical preprocessing.
2. DICOM-RT / RTSTRUCT export for radiation oncology.
3. 2×2 synchronized multi-sequence viewing.
4. Interactive in-viewer contour modification tools.
5. Tumor board export packets and enterprise multi-user RBAC.

---

### 2.2 Exchange 2: Feasibility & Safety Analysis of "Do Not Build" Features

#### User Context & Action:
*The user inspected lines 147–160 of [`roadmap.md`](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md):*
```markdown
147: | Survival or prognosis prediction | Unsupported by any model or evaluation here; direct medical-claim risk |
148: | Per-case "confidence %" | No calibration study below 0.8; a number would imply certainty the project has measured as absent |
149: | Age/sex covariates in the grade model | Defensible, but requires a fresh pre-registration and retrain — not a UI change |
150: | Perfusion / MRS / ADC-DWI features | Outside the four-modality input contract |
151: | Automatic modality classification | Slot-based identity is honest; this needs its own validated model |
152: 
153: ---
154: 
155: ## 9. Progress
156: 
157: | Item | State |
158: |---|---|
159: | P0-1 WT/TC/ET composite volumes | **done** |
160: | P0-2 Maximal diameters | **done** |
```

#### User Prompt 2:
> Can all of these features be integrated in the website?

#### Assistant Response 2:
Technically, **yes, any of these features *could* be coded and shown on the website**. 

However, the reason lines 147–151 of [roadmap.md](file:///Users/jagadishsunilpednekar/ojt-5-hf-brain-tumor-classification/roadmap.md#L147-L151) are explicitly titled **"Do not build"** is not a UI limitation. It is a deliberate **scientific, legal, and medical safety boundary**. 

Displaying an unvalidated or misleading number in a clinical neuro-oncology tool creates direct risk of patient harm and regulatory liability.

Here is an analysis of each feature: why it was excluded, what it would take to actually build it, and whether it should be integrated.

---

### Breakdown of the 5 "Do Not Build" Features (Lines 147–151)

#### 1. Survival or Prognosis Prediction (e.g., predicting overall survival in months)
* **Why it is listed as "Do Not Build":**
  * Estimating life expectancy (e.g., *"Median survival: 14.2 months"*) makes a **direct clinical claim**. If an AI gives an inaccurate survival estimate, it can inappropriately influence whether a patient is offered aggressive therapy or palliative care.
  * In the BraTS benchmarks, image-only survival regression models notoriously perform poorly (often no better than predicting median survival based on patient age alone).
* **Can it be integrated?**
  * **UI/Website:** Trivial to add a card showing estimated survival.
  * **ML/Reality:** Would require acquiring a large, curated clinical dataset with complete follow-up data (overall survival, progression-free survival, surgical records, adjuvant chemo/radiation), training a survival model (e.g., Cox Proportional Hazards or DeepSurv), and undergoing prospective clinical validation.
* **Verdict:** **Do not build** unless you are running a formal clinical trial and have regulatory clearance.

---

#### 2. Per-Case "Confidence %" (e.g., *"Confidence: 94%"*)
* **Why it is listed as "Do Not Build":**
  * Deep neural networks produce raw softmax probabilities that are notoriously **overconfident and uncalibrated**. A model can be 98% "confident" while predicting the wrong tumor type or segmenting an artifact.
  * The project already provides a **reliability layer** (seed disagreement, test-time dropout variance, calibration curves, and cohort scope). Showing a naive single "94% confident" badge implies clinical certainty that does not exist.
* **Can it be integrated?**
  * **UI/Website:** Trivial (could just print `max(softmax_probs) * 100`).
  * **ML/Reality:** To do it honestly, you must conduct a rigorous **conformal prediction** or **temperature scaling** calibration study across multi-center external datasets so that "90% confidence" statistically guarantees 90% coverage.
* **Verdict:** Replace the idea of a simple single percentage with the platform’s existing **epistemic reliability panels** (seed agreement, calibration bins).

---

#### 3. Age/Sex Covariates in the Grade Model
* **Why it is listed as "Do Not Build":**
  * Patient age is one of the strongest epidemiological predictors of glioblastoma (high-grade tumors occur much more frequently in older adults).
  * However, the current model was trained and evaluated strictly on **3D MRI voxel intensities**. You cannot simply add two form fields (`Age`, `Sex`) in the web UI and expect the existing PyTorch model to use them.
* **Can it be integrated?**
  * **UI/Website:** Very easy (simple form inputs).
  * **ML/Reality:** Requires redesigning the model architecture to fuse 3D convolutional image embeddings with tabular demographic features, retraining from scratch on a pre-registered dataset split, and re-evaluating for demographic bias.
* **Verdict:** Feasible, but it is a **full ML re-training project**, not a frontend/website change.

---

#### 4. Advanced MRI: Perfusion (DSC/DCE), Spectroscopy (MRS), and Diffusion (DWI/ADC)
* **Why it is listed as "Do Not Build":**
  * The entire architecture and data pipeline are strictly bound to the **four standard structural sequences** (T1, T1ce, T2, FLAIR).
  * Advanced physiological MRI involves fundamentally different physics: Perfusion generates 4D hemodynamic time series (to calculate cerebral blood volume / rCBV); Diffusion uses gradient directions and b-values (to derive ADC maps); Spectroscopy produces chemical shift metabolic peaks (Cho/NAA ratios).
* **Can it be integrated?**
  * **UI/Website:** Complex. Requires time-intensity curve viewers, spectral graph viewers, and parametric colormap overlays.
  * **Backend/Pipeline:** Requires a major overhaul of the ingestion engine, 4D NIfTI processing, pharmacokinetic modeling, and completely new neural networks.
* **Verdict:** Valuable for neuro-oncology (distinguishing pseudoprogression from true tumor recurrence), but represents a **massive expansion in project scope**.

---

#### 5. Automatic Modality Classification
* **Why it is listed as "Do Not Build":**
  * Right now, the app expects users to assign files to slots (`t1`, `t1ce`, `t2`, `flair`) or name them with clear suffixes.
  * If the platform attempts to automatically guess which sequence is which and makes a mistake (e.g., swapping T1ce and T2), the segmentation model will fail silently and produce a convincing-looking but completely incorrect mask.
* **Can it be integrated?**
  * **UI/Website:** Seamless drag-and-drop into a single dropzone without slot matching.
  * **ML/Reality:** Requires training an auxiliary 3D CNN classifier or rule-based parser on intensity distributions and tissue contrast to reliably classify MRI series before feeding them to the segmentation pipeline. If DICOM were supported, this could largely be read from DICOM header metadata (`SeriesDescription`, `EchoTime`, `RepetitionTime`).
* **Verdict:** Highly practical if backed by DICOM metadata parsing or a validated sequence classifier.

---

### What About the Remaining Features in Sections 3–5 (Lines 164–181)?

If your question also refers to the **unfinished P0/P2 roadmap items** (such as server-side PDF exports, in-viewer contour editing, tumor board summaries, and 2×2 multi-sequence views):

| Roadmap Feature | Feasible to integrate in the website? | What is required? |
|---|---|---|
| **Server-Side PDF Export (P0-6)** | **Yes (High priority)** | Add `WeasyPrint` or a headless Chromium renderer in the FastAPI backend to generate formatted clinical PDF reports. |
| **2×2 Multi-Sequence View (P2)** | **Yes (High priority)** | Update `frontend/src/pages/ViewerPage.tsx` to display linked axial slices of T1, T1ce, T2, and FLAIR simultaneously in a 4-pane grid. |
| **Tumor Board Packet (P2)** | **Yes (High priority)** | Generate a unified summary bundling volume change curves, RANO status, treatment chronology, and key slice captures. |
| **In-Viewer Contour Editing (P2)** | **Yes (Moderate effort)** | Build a 2D canvas brush/eraser tool in the React viewer so clinicians can touch up segmentation contours directly on slice views. |
| **W/L Presets & Intensity (P2)** | **Yes (Low effort)** | Add Window/Level presets (Brain, Subdural, High-contrast) and a hover coordinate intensity inspector. |
| **DICOM / PACS Ingestion (P3)** | **Yes, but heavy backend effort** | Add Orthanc / DCM4CHEE integration or a client-side DICOM-to-NIfTI preprocessing pipeline (`dcm2niix` + HD-BET skull stripping). |

### Conclusion

* **The 5 items in Section 8 ("Do Not Build"):** Do **not** force these into the UI today. They require fresh ML architectures, clinical re-training, or external clinical validation to avoid making misleading medical claims.
* **The remaining items in P0, P1, and P2:** **Yes**, these should be completed, as they provide real clinical utility (multi-sequence comparison, PDF generation, tumor board packets, and interactive contour editing) while maintaining the integrity and safety of the system.

---

### 2.3 Exchange 3: Handoff Generation Request

#### User Prompt 3:
> Export the complete current conversation, including my prompts, your responses, feature evaluation tables, technical findings, decisions, and pending tasks, into `ANTIGRAVITY_HANDOFF.md` in the repository root.
> 
> Preserve the original wording and Markdown formatting as much as possible. Do not modify application code or any existing files. If you cannot access earlier parts of the conversation, explicitly state that limitation rather than inventing missing content. Verify that the handoff file was created successfully.

---

## 3. Consolidated Technical Findings & Architectural Audit

### 3.1 What is Implemented in the Codebase
1. **Core Diagnostic Metrics (P0):**
   * WT / TC / ET composite segmentation volumes in `backend/app/services/measurement_service.py`.
   * 3D maximal diameter, in-plane major/minor axes, and bidimensional products (Macdonald / RANO lineage).
   * Enhancing tumor (ET) presence flag, necrotic core fraction, edema-to-core ratio.
   * True scanner-space centroid coordinate calculations.
   * Composite Dice & HD95 ground-truth evaluation metrics.
   * Case worklist API (`/api/v1/cases`) and dedicated frontend page.
   * Per-case input-similarity domain shift check against BraTS reference statistics.

2. **Longitudinal Analysis (P1):**
   * Subject entity (`SubjectRecord`), study dates, and ordered timepoints.
   * Longitudinal tumor volume tracking panel comparing timepoint-over-timepoint trajectory.
   * Side-by-side prior vs. current comparison viewer (`frontend/src/pages/ComparePage.tsx`) with slice matching.
   * RANO 2010 high-grade glioma response assessment calculator (`docs/rano_response_assessment.md`).
   * Treatment timeline on the longitudinal axis (surgery, multi-course radiotherapy, chemotherapy, corticosteroids).

3. **Clinical Context (P2):**
   * Molecular markers and pathology record (IDH1/2, 1p/19q codeletion, MGMT methylation, ATRX, TERT, EGFR, CDKN2A/B, H3K27M, Ki-67).
   * Model vs. pathology reconciliation with cohort premise validation.
   * Extent of resection (EOR) calculator with 72-hour postoperative window verification.
   * 2D caliper distance and region (mean ± SD) measurement overlay tools saved per case.
   * Key-image view reference bookmarks.

### 3.2 What is Incomplete or Partially Implemented
1. **P0-6 Exports:** CSV measurements and NIfTI segmentation downloads are active, but **server-side PDF generation is deferred**.
2. **P0-8 Deletion & Retention:** Manual case deletion with storage cleanup is implemented, but **automated scheduled retention policy sweeps** are not implemented.
3. **P2 Manual Contour Correction:** Correction via file upload exists, but **in-viewer interactive brush/eraser editing is missing**.
4. **P2 Audit Trail:** An append-only activity history exists (`docs/activity_history.md`), but lacks role-based access control, read event tracking, and immutable tamper-resistant storage.
5. **P2 Multi-Sequence Viewing:** The viewer displays orthogonal planes (Axial, Coronal, Sagittal) for a single sequence; **simultaneous 2×2 synchronized viewing of T1ce, T2, FLAIR, and T1 is missing**.
6. **P2 Tumor Board Packet:** Aggregated one-click PDF digest combining key images, volume trajectory, and treatment history is not built.
7. **P2 Window/Level & Calibrated Intensity:** Radiological W/L presets and cursor intensity readouts remain arbitrary units.

### 3.3 Clinical Reality Gap (What Radiologists & Oncologists Need)
* **Hospital Integration:** Lack of DICOMweb / PACS connectivity (C-FIND, C-MOVE, WADO-RS) requires manual pre-processing outside clinical reading environments.
* **Clinical Raw DICOM Preprocessing:** Inability to handle raw, anisotropic, non-skull-stripped clinical acquisitions without degradation.
* **Radiation Oncology & Surgical Export:** Lack of DICOM RTSTRUCT or DICOM-SEG prevents direct export to Treatment Planning Systems (e.g., Eclipse, RayStation) or surgical neuronavigation.
* **Advanced MRI Modalities:** Inability to ingest or analyze Perfusion MRI (DSC/DCE rCBV), Diffusion (DWI/ADC), or MR Spectroscopy needed to differentiate pseudoprogression/radiation necrosis from true tumor progression.
* **Regulatory & Concurrency Constraints:** Not cleared as a medical device (FDA 510(k) / CE-MDR), and single-worker SQLite queue cannot support departmental concurrent reading.

---

## 4. Pending Tasks & Recommended Implementation Plan

### Phase A: High-Value, Clinically Honest Near-Term Tasks (P0 & P2 Completion)
1. **Server-Side PDF Reporting (P0-6):**
   * Integrate a headless renderer (e.g., WeasyPrint or Playwright) in FastAPI to generate standardized clinical and research PDF reports.
2. **2×2 Synchronized Multi-Sequence Grid (P2):**
   * Enhance `frontend/src/pages/ViewerPage.tsx` to provide a 4-up synchronized view (T1, T1ce, T2, FLAIR) for axial slice navigation.
3. **In-Viewer Interactive Contour Correction Brush (P2):**
   * Implement a HTML5 Canvas 2D brush and eraser overlay tool to allow clinicians to correct automated contours slice-by-slice without external software.
4. **Tumor Board Packet Generation (P2):**
   * Build a dedicated view/export combining anonymized key slice captures, longitudinal volume plots, and treatment timeline.
5. **Radiological W/L Presets & Coordinates (P2):**
   * Add preset window/level controls (Brain, Subdural, Stroke, Bone) and slice-coordinate hover readout.

### Phase B: Infrastructure & Interoperability (P3)
1. **DICOM Ingestion & Preprocessing Pipeline:**
   * Build an automated ingest service with `dcm2niix` and automated skull-stripping (e.g., HD-BET) to accept raw clinical DICOM series.
2. **DICOM RTSTRUCT / DICOM-SEG Export:**
   * Convert predicted multi-label masks into standard DICOM RT Structure Sets for radiation oncology treatment planning systems.
3. **PostgreSQL & Task Queue Upgrade:**
   * Migrate from SQLite and in-memory `asyncio.Queue` to PostgreSQL + Celery/Redis for multi-user departmental concurrency.
4. **Role-Based Access Control (RBAC):**
   * Introduce user roles (Attending Radiologist, Resident, Oncologist, Auditor) with full compliance read/write audit logging.

### Phase C: Safety Guardrails to Enforce (Section 8 "Do Not Build")
* **Do NOT output survival time or prognosis predictions.**
* **Do NOT output a naive single "confidence %" badge.** Rely on the epistemic reliability panel (calibration bins, seed disagreement).
* **Do NOT add Age/Sex form fields to the grade model without a formal multi-modal retraining and pre-registered study.**
* **Do NOT attempt automated modality classification without a validated sequence classifier or verified DICOM metadata.**
