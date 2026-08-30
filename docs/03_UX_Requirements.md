# Document 3 — UX Requirements

## Information Architecture & Routing Structure
* `/` — Landing/Home: project purpose, model summary (checkpoint version, dataset, headline Dice/HD95), persistent non-clinical-use disclaimer, "Upload a case" CTA.
* `/upload` — Case Selection: drag-and-drop + file picker for the 4 required modality files (or a pre-packaged case archive).
* `/cases/:id/analysis` — Analysis Workspace: 2D slice viewer + embedded 3D tumor-mesh viewer, live inference status polling.
* `/cases/:id/results` — Results Dashboard: per-class/mean Dice + HD95, tumor volume, checkpoint version, embedded 3D view for side-by-side review.
* `/history` — (V1) List of previously processed cases with timestamp, headline metrics, and model version used at the time.

## User Flows: Case Analysis Flow
```
[Read Disclaimer] ──► [Upload 4-Modality Case] ──► [Preprocess & Sliding-Window Inference]
                                                              │
[Review Results Dashboard] ◄── [2D Slice + 3D Mesh Review] ◄──┘
```

## Screen Requirements: Analysis Workspace Screen (`/cases/:id/analysis`)
* **Purpose:** Let the reviewer inspect AI-predicted tumor sub-regions against the underlying anatomy before trusting any metric.
* **Components:**
  * Modality selector + segmentation-overlay toggle for the 2D slice viewer (axial/coronal/sagittal), with a scroll/slider to move through slices.
  * Embedded 3D tumor-mesh viewer (react-three-fiber): rotate/zoom/pan, per-sub-region visibility toggle, consistent documented color legend.
  * Case metadata panel: case ID, modalities present, upload timestamp, live inference job status (queued/running/done/failed).
* **States:**
  * *Loading:* Skeleton loader during preprocessing & sliding-window inference.
  * *Error:* Specific message for invalid file, missing modality, or corrupted NIfTI — never a raw stack trace.
  * *Empty:* "No case selected yet" prompt back to `/upload`.

## Accessibility & Responsive Design
* Contrast ratio ≥ 4.5:1 for all text elements per WCAG 2.1 AA.
* Keyboard shortcut navigation for slice step-through (`KeyJ` / `KeyL`, mirroring standard radiology-tool conventions).
* Meshes decimated/vertex-capped so the 3D viewer maintains interactive frame rates on a typical laptop GPU.
* Every screen displaying a segmentation result carries a visible, persistently rendered non-clinical-use disclaimer.
