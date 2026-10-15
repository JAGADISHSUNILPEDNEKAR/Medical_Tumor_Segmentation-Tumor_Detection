# target.md — Sequential Build Order for Clinical Feature Completion

**Written:** 2026-10-12 against commit `f144174`
**Supersedes for sequencing purposes:** the unordered "Phase A / B / C" lists in
[`ANTIGRAVITY_HANDOFF.md`](ANTIGRAVITY_HANDOFF.md) §4
**Companion to:** [`roadmap.md`](roadmap.md) (which states *what* and *why*; this
states *in what order*, and *what must not be built*)

**Goal of this document.** Make the application as feature-rich and as
genuinely supportive to a radiologist and a neuro-oncologist as it can be
*without* the platform asserting a single thing it has not measured. Where a
feature is clinically valuable but cannot be fully evidenced, this document does
not drop it — it scopes it down to the honest version and writes the exact
disclaimer that must ship with it (§7).

---

## 1. How to read this

Items are numbered **1 → 22 in strict build order**. The order is not a
priority ranking. It is a dependency chain: each item is placed where it is
because building it earlier means building it twice, and building it later means
blocking something else.

| Column | Meaning |
|---|---|
| **Size** | S = under a day · M = a few days · L = a week or more · XL = a project with its own plan |
| **Blocks** | What cannot be built, or must be rebuilt, until this ships |
| **Disclaimer** | ✅ = a disclaimer in §7 must ship *with the feature*, in the same view, not in a footer |

Three tiers never move: **Tier 0 must precede every new surface** (it stops the
existing surfaces from being read wrongly), **Tier 5 must precede any
multi-user deployment**, and **Tier 6 item 21 cannot precede item 20** — see §4.

---

## 2. Standing constraints — unchanged by anything below

Every item in this document is scoped to add **no new clinical claim**.

1. **Research / decision-support prototype. Not a medical device. Not clinically
   validated.** No FDA 510(k), no De Novo, no CE-MDR. Must not be used for
   primary diagnosis or treatment decisions.
2. **The four-modality input contract holds.** T1, T1ce, T2, FLAIR are the only
   inputs to any model. Item 15 adds *viewing* of other sequences and explicitly
   excludes them from inference.
3. **Measurements are automated and mask-derived.** They are never labelled as a
   radiologist's measurement.
4. **An export can never disagree with the screen.** `export_service.py` does not
   recompute; it serialises. This rule decides item 6 — see §3.3.
5. **No prediction is recomputed from a human-edited mask.** Already enforced in
   `correction_service.py`; item 10 inherits it unchanged.
6. **A `null` is better than a `0.0`.** Empty denominators stay null.
7. **The reliability layer is a feature, not overhead.** Seed disagreement,
   calibration bins, operating point, cohort scope, withheld predictions,
   checkpoint SHA. A new panel that asserts more than it can support is a
   regression.

---

## 3. Corrections to the handoff, grounded in the tree

The handoff's feature inventory is accurate. Seven of its *effort and approach*
claims are not, and four of them change the build order. Read this section
before starting item 1.

### 3.1 The 2×2 grid is a layout change, not a data-loading project
The handoff implies four volumes must be loaded. They already are.
`ViewerPage.tsx:121` holds a `volumeCacheRef` — *"every modality of this case
parsed so far, shared by the 2D planes and the 3D view"* — and `ViewerPage.tsx:312`
already **prefetches all four** when the 3D view opens. The grid needs pane
layout, per-pane window state and linked slice indices. The parsing, the affine
handling and the caching exist. This is why item 4 sits early and is sized **M**,
not **L**.

### 3.2 "Brain / Subdural / Stroke / Bone" window presets are a CT concept and are wrong here
The handoff recommends them. They cannot be built as named, for three stacked
reasons:
* MRI has **no absolute intensity units** — unlike CT Hounsfield units, there is
  no physical scale for a preset to name.
* BraTS inputs are **per-volume foreground z-score normalised**
  (`backend/app/inference/brats/preprocessing.py:63`), so even relative values
  are not comparable between cases.
* The frontend NIfTI parser does **not** apply `scl_slope` / `scl_inter`
  (`frontend/src/lib/nifti.ts`), and the display path is 1–99th **percentile**
  windowing (`sliceExtraction.ts:399`).

So item 3 ships **per-sequence percentile window/level presets** (a different and
honest thing) and a readout labelled *stored voxel value, arbitrary units*. The
handoff's "**calibrated** cursor intensity readout" is **struck from scope** —
it cannot be delivered and is not deferred, it is withdrawn. See §7.3.

### 3.3 Server-side PDF rendering is the wrong resolution of the PDF question
`ARCHITECTURE.md:583` defers it on a dependency argument (weasyprint pulls cairo
and pango, awkward on the Hugging Face Space; reportlab means hand-duplicating
`ReportPage`). There is a second, stronger argument the handoff misses:

Key images are stored as **view references, not pixels**, by deliberate design
(`db/models.py:203`: *"A PNG would be a second copy of imaging data in a place
with no link back to the NIfTI it came from, and it would bake in a display
window that this application treats as presentation rather than data"*).
Rendering slices server-side therefore means re-implementing the affine-derived
orientation, the percentile normalisation and the overlay compositing in Python —
a second renderer that **can drift from the screen**, which constraint 4 forbids.

**The browser is already the renderer.** `PlaneView.tsx:245-266` draws through a
2D canvas, so `toBlob()` yields exactly what the reader saw. The correct shape is:
**client renders → server assembles and stores → server serves the PDF.** This
resolves the dependency question too: the server composes a document from
supplied PNGs and its own data, which needs no cairo and no pango.

That inversion is why **item 5 (slice capture) precedes item 6 (PDF)**.

### 3.4 The in-viewer brush already has its backend contract
`correction_service.py` validates geometry against the base mask, refuses to
resample, stores the base mask's SHA-256, recomputes measurements beside the
originals, and never overwrites model output. `PUT /results/{id}/correction`
exists (`api/v1/results.py:174`). The brush is therefore a **frontend** feature
that terminates in an existing, tested endpoint. The genuinely new primitive is
**writing** a NIfTI in the browser — the app currently only reads one. Hence
item 9 is split out ahead of item 10.

### 3.5 RANO 2.0 is additive, not a rewrite
`services/rano.py:46` already carries `CRITERIA_VERSION` and the module docstring
already names RANO 2.0 as unimplemented and version-tagged. A second rule set
lands beside the first. No migration, no reinterpretation of stored assessments.

### 3.6 RBAC is additive onto an isolation model that already exists
`auth/models.py` has **no `role` column** — that part is real work. But
`owner_id` scoping is already pervasive across every service and every route, so
roles layer onto an enforced boundary rather than inventing one.

### 3.7 The retention sweep has two constraints the handoff does not name
Celery/Redis is not available in the deployment target (single container, Hugging
Face Space, `DATABASE_URL=sqlite:////app/data/app.db`), so the sweep needs an
**in-process scheduler**. And it must respect the **same 409 guard** that
`delete_case` uses while a job is active, or a sweep will delete files from under
the single worker. Also note: the Space filesystem is **ephemeral** unless a
persistent disk is attached — a retention policy is a correctness feature there,
not just a compliance one.

---

## 4. The build order

### Tier 0 — Per-case honesty, before any new surface

Both items convert something the project has **already measured** into a warning
on the case in front of the reader. `roadmap.md` §6 makes this argument for the
domain-shift flag and it is the same argument here: highest value per unit of
work in the document. Neither adds a claim. Both must precede new surfaces,
because a new surface built on a silently-wrong read is worse than no surface.

---

#### 1. Grade calibration-support readout
**Size:** S · **Blocks:** nothing · **Disclaimer:** ✅ §7.1

The grade ensemble's probability is shown; the **evidential support for that
particular probability** is not. The project has measured that p(4) in the
0.5–0.6 calibration bin holds **zero** evaluation patients, and that on
BraTS-Africa input p(4) lands in roughly 0.52–0.78 — where the 0.5 decision
threshold hides the problem entirely.

Surface, per case: which calibration bin this case's p(4) falls into, **how many
evaluation patients populate that bin**, and the measured accuracy in it. When
the bin is empty or thinly populated, say so in words.

* **Where:** `inference/grade/reliability.py`, `schemas/grade.py`,
  `components/results/WhoGradePanel.tsx`
* **Acceptance:** an empty bin renders as *"no evaluation patients in this
  probability band"*, never as a percentage. Bin counts come from the stored
  calibration artifacts (`docs/grade_resnet18_roi_v2_calibration.json`), never
  recomputed at request time.
* **Why not a "confidence %":** this is the honest inverse of the banned feature.
  It reports the *support* for a number instead of dressing the number up. See §6.2.

---

#### 2. Modality slot plausibility warning
**Size:** S · **Blocks:** nothing · **Disclaimer:** ✅ §7.2

A reader who uploads T1ce into the T2 slot gets a convincing, completely wrong
segmentation and **no warning at all**. This is the highest-severity silent
failure mode in the application.

Check the four volumes' intensity relationships against the training fingerprint
and flag an implausible assignment. **This never reassigns a slot** — slot-based
identity stays the user's declaration, which is the honest design. It only says
*this does not look like what you said it is*.

* **Where:** extend `validation/input_domain.py`, reuse the input-similarity plumbing
  (`docs/input_domain_check.md`, `docs/input_domain_reference.json`), surface in
  `components/results/InputSimilarityPanel.tsx`
* **Acceptance:** a flag is a warning, never a block and never an auto-correction.
  Phrased as a question to the reader, not as a finding.
* **Relation to the banned item:** automatic modality *classification* stays
  banned (§6.5). A plausibility *warning* is its safe half and can ship now.

---

### Tier 1 — The reading view

Order within this tier matters more than anywhere else in the document.

---

#### 3. Per-sequence window/level state, presets and intensity readout
**Size:** S · **Blocks:** 4, 10 · **Disclaimer:** ✅ §7.3

Today `brightness` and `contrast` are **single global scalars** on `ViewerState`
(`types/viewer.ts:206-208`). Move them to **per-sequence** state, add
percentile-pair presets per sequence, and add a cursor readout of the stored
voxel value.

**Build this before item 4.** If the 2×2 grid lands first, four panes get
retrofitted with per-pane window state afterwards — the same work, done twice,
with a migration of the key-image `brightness`/`contrast` columns in the middle.

It also blocks item 10: a reader cannot responsibly contour what they cannot
window.

* **Where:** `types/viewer.ts`, `hooks/useViewerController.ts`,
  `components/viewer/ImageControls.tsx`, `lib/sliceExtraction.ts`,
  `services/key_image_store.py` (per-sequence window on restore)
* **Acceptance:** presets are named by **sequence and percentile pair** (e.g.
  *"FLAIR, 2–98%"*), never by CT tissue names. The readout is labelled
  *arbitrary units*. The source NIfTI is never modified — the existing
  display-only guarantee holds.

---

#### 4. 2×2 linked multi-sequence grid
**Size:** M · **Blocks:** 7, 10 · **Disclaimer:** —

The single largest gap between this viewer and a reading view. A reader cannot
separate true enhancement from intrinsic T1 hyperintensity (subacute blood,
proteinaceous fluid, melanin) without T1 and T1ce **simultaneously**, nor
non-enhancing tumour from vasogenic oedema without T2 and FLAIR alongside.
Thumbnails are previews, not panes.

Four panes, one plane, linked slice index and linked crosshair, per-pane window
from item 3, overlay toggleable per pane. Keep the existing 3-plane single-
sequence layout as a switchable mode — both are real reading patterns.

* **Where:** `pages/ViewerPage.tsx`, `components/viewer/PlaneView.tsx` (already
  parameterised per plane; parameterise per sequence too), `types/viewer.ts`
* **Already present:** all four volumes are parsed and cached — see §3.1
* **Acceptance:** panes stay linked through zoom, pan and slice change. A
  sequence that failed to load renders an explicit empty pane, never a blank one.
  No measurement is implied across panes.

---

#### 5. Slice capture substrate
**Size:** M · **Blocks:** 6, 7 · **Disclaimer:** ✅ §7.4

The primitive that two deliverables need and neither can fake: capture the
rendered canvas to a PNG and store it **beside** the key-image view reference,
never replacing it.

The view reference stays the source of truth; the PNG is a **rendering of a
stated view at a stated window**, carrying its sequence, plane, slice index,
window settings, overlay state, and the result and model identifiers that were
on screen. This keeps `db/models.py:203`'s reasoning intact — the picture can
always be traced back to the volume and the window that produced it.

* **Where:** new `frontend/src/lib/sliceCapture.ts` (canvas `toBlob`), new
  attachment store beside `services/key_image_store.py`, new route on
  `api/v1/results.py`
* **Acceptance:** a stored PNG without its full view reference is rejected. A
  capture records the provenance verdict that was displayed, so an image cannot
  be separated from whether its segmentation was verified.

---

### Tier 2 — Deliverables a clinician takes out of the application

---

#### 6. Server-assembled PDF report — completes P0-6
**Size:** M · **Blocks:** 7, 22 · **Disclaimer:** ✅ §7.5

`window.print()` (`ReportPage.tsx:83`) is the current path and it is fragile:
the output depends on the reader's browser, printer margins and headers.

Assemble server-side from the **nine sections `ReportPage` already renders** plus
client-captured PNGs from item 5. Composing a document from supplied images and
owned data needs **no cairo and no pango** — it resolves the dependency objection
in `ARCHITECTURE.md:583` rather than accepting it. See §3.3 for why the images
must come from the client.

* **Where:** `services/export_service.py`, `api/v1/results.py`,
  `pages/ReportPage.tsx` (capture + request), `requirements.txt`
* **Acceptance:** the PDF contains no number the result does not contain — it
  serialises, it does not recompute (constraint 4). Every page carries the
  non-clinical-use disclaimer and the provenance verdict in the footer. A
  `synthetic` or `invalid` provenance verdict is stamped on **every page**, not
  just the first.
* **Also log:** `export.report_pdf` added to `ACTIONS` in `services/activity.py:41`.

---

#### 7. Tumor board packet
**Size:** M · **Blocks:** nothing · **Disclaimer:** ✅ §7.6

One artifact for the weekly MDT meeting: de-identified key slices, the
longitudinal volume table, RANO status and its entered criteria, the treatment
chronology, the pathology and molecular record, and the extent-of-resection
figures. Low effort once items 5 and 6 exist; high visibility; **no new medical
claim** — every component already exists on screen.

**Keep it a table, not a chart.** `TimelinePanel.tsx:4` states the existing
decision: *"Deliberately a table and not a line chart. A series here is two or
three points."* That reasoning does not change because the destination is a
projector. Add a sparkline only when a subject actually has enough timepoints to
support one, and never a trend line through two points.

* **Where:** new `services/tumor_board_service.py`, new route, new frontend page
  reusing the longitudinal components
* **Acceptance:** de-identification is **on by default and cannot be silently
  switched off** — the packet states which de-identification profile produced it.
  Entered criteria are shown as entered, attributed to the person who entered
  them. Nothing in the packet is attributed to a model that did not produce it.

---

#### 8. Structured findings block
**Size:** S · **Blocks:** nothing · **Disclaimer:** ✅ §7.7

A copy-pasteable block of the automated numbers — WT/TC/ET volumes, maximal
diameters, bidimensional product, ET presence, necrotic fraction, edema-to-core
ratio, centroid — formatted for pasting into a dictation system.

**Scope limit, which is the whole point of the item:** numbers and their units
only. **No impression, no assessment, no comparison sentence.** Generating prose
that reads like a radiologist's impression is the failure mode this item is
deliberately shaped to avoid.

* **Where:** `components/results/TumorBurdenPanel.tsx`, `lib/measurement.ts`
* **Acceptance:** no sentence in the block contains an interpretive verb. The
  block opens with its provenance line.

---

### Tier 3 — The reader owns the contour

---

#### 9. Client-side NIfTI writer
**Size:** S–M · **Blocks:** 10 · **Disclaimer:** —

The real new primitive behind in-viewer editing. The app reads NIfTI
(`nifti-reader-js`, `lib/nifti.ts`) and writes none. Item 10 cannot start without
this, and it is worth landing and testing **alone**, because
`correction_service.py` rejects any byte-level mismatch in shape, affine or
spacing — it resamples nothing, by design.

* **Where:** new `frontend/src/lib/niftiWriter.ts`, with `pako` for gzip (already
  a dependency)
* **Acceptance:** round-trip a parsed volume unchanged, then round-trip it
  through `PUT /results/{id}/correction` and have the server's geometry
  validation accept it. Label values are BraTS-convention on the wire
  (`lib/labelConvention.ts`, `tests/test_label_conventions.py`).

---

#### 10. In-viewer brush and eraser
**Size:** L · **Blocks:** nothing · **Disclaimer:** ✅ §7.8

A radiologist will not sign off on a contour they cannot edit — already the
stated reasoning in `correction_service.py`. The upload path satisfies the letter
of that; nobody edits a mask in external software between cases.

2D brush and eraser on the slice canvas, per-label, adjustable radius, undo/redo,
slice-to-slice copy. On save, compose the full volume and send it through the
**existing** correction endpoint, which already does everything that matters.

* **Where:** `components/viewer/` (new overlay tool), `pages/ViewerPage.tsx`,
  `lib/niftiWriter.ts` from item 9. **Backend: no change.**
* **Acceptance:** inherits all four correction rules unchanged — model output
  preserved beside the edit, geometry identical, base mask SHA recorded, **grade
  and tumour type not recomputed**. Measurements recompute and display beside the
  originals. Interpolation between edited slices, if offered at all, is **off by
  default** and labelled as interpolated — it is a guess about slices the reader
  did not look at.

---

### Tier 4 — Oncology criteria depth

Independent of Tiers 1–3; may run in parallel by a second pair of hands.

---

#### 11. Non-enhancing / T2-FLAIR measurable disease
**Size:** S · **Blocks:** 12, 13 · **Disclaimer:** —

RANO 2.0 and RANO-LGG both assess **non-enhancing** disease, and the current
measurement set is organised around enhancement. The oedema label already exists;
what is missing is the non-enhancing composite and its diameters as a
first-class, separately reported measurement.

* **Where:** `services/measurement_service.py`, `schemas/measurements.py`,
  `components/results/TumorBurdenPanel.tsx`
* **Acceptance:** reported alongside WT/TC/ET, never folded into them. The label
  states that T2/FLAIR signal is not specific to tumour — it includes treatment
  effect and gliosis.

---

#### 12. RANO 2.0 as a second criteria set
**Size:** M · **Blocks:** nothing · **Disclaimer:** ✅ §7.9

Current trials and practice are moving to RANO 2.0 (2023): a revised baseline for
newly diagnosed glioblastoma, explicit handling of non-enhancing tumour, and
updated post-radiotherapy timing. The module already names it as unimplemented
and already version-tags its output (§3.5).

* **Where:** `services/rano.py` (second rule set beside RANO 2010),
  `schemas/subjects.py`, `components/longitudinal/RanoPanel.tsx`
* **Acceptance:** stored assessments keep the `criteria_version` they were
  computed under and are **never retro-recategorised**. Both versions can be
  displayed for one timepoint, and when they disagree the panel shows **both**
  rather than choosing. The two-comparison-point discipline that `rano.py` exists
  to enforce — progression from nadir, partial response from baseline — carries
  over. Still a calculator over entered criteria, never a prediction.
* **iRANO** (immunotherapy) is **deferred** — see §5.

---

#### 13. RANO-LGG for low-grade glioma
**Size:** M · **Blocks:** nothing · **Disclaimer:** ✅ §7.9

The grade model's own task is **4 vs 2–3**, so low-grade cases are squarely in
scope — yet the only response calculator is for high-grade glioma. For these
patients response is assessed on **T2/FLAIR** disease, which item 11 provides.
Without this, half the model's declared cohort has no response pathway at all.

* **Where:** `services/rano.py`, same panel
* **Acceptance:** the criteria set is selected by the reader, never inferred from
  the model's grade output — the model's group is not a diagnosis and must not
  silently pick a response standard.

---

#### 14. Demographic and clinical context record
**Size:** S · **Blocks:** nothing · **Disclaimer:** ✅ §7.10

Age, sex, KPS, presenting symptoms, baseline steroid dose — **recorded and
displayed, never fed to any model.** This is the integrable half of a banned item
(§6.3): age is among the strongest epidemiological predictors of glioblastoma,
and an oncologist reading the report needs it on the page. Feeding it to the
image-only grade ensemble is a retraining project; **writing it down is not**.

* **Where:** follows the `PathologyRecord` pattern exactly
  (`db/models.py:295`, `services/pathology.py`) — entered by the user, never
  predicted, displayed beside the model output
* **Acceptance:** the record states explicitly that the grade and type models
  take no covariates and did not see these values. No reconciliation, no
  "adjusted" anything.

---

#### 15. Reference-sequence viewing — ADC, rCBV, MRS maps
**Size:** M · **Blocks:** nothing · **Disclaimer:** ✅ §7.11

The integrable half of a banned item (§6.4), and the single most clinically
requested thing in the handoff's gap list. Distinguishing **pseudoprogression and
radiation necrosis from true recurrence** is the question that dominates
follow-up, and it is largely answered by looking at diffusion and perfusion.

Accept an **additional, optional reference volume** — a derived parametric map
(ADC, rCBV) — for **viewing in the grid from item 4 only**. It is **not a model
input**, it is not measured, it is not in the four-modality contract, and no
number is derived from it. Full 4D DSC/DCE time series, pharmacokinetic
modelling and MRS spectral display stay **deferred** (§5) — this is the viewing
slice of that scope, and it is the part that needs no new science.

* **Where:** ingest (`api/v1/upload.py`, `lib/uploadValidation.ts`) accepts an
  optional extra slot; `ViewerPage` renders it as a pane; **inference untouched**
* **Acceptance:** the pane is visually distinct from the four contract sequences
  and labelled as a reference sequence. A reference volume of mismatched geometry
  is **refused, not resampled** — the project resamples nothing, anywhere. The
  volume is excluded from the input-similarity check, from measurements and from
  every model path. Prove the exclusion with a test, not a comment.

---

### Tier 5 — Multi-user safety. Nothing here is optional before a second reader.

`roadmap.md` §7 is explicit: *"roles and concurrency must land before any
departmental use."*

---

#### 16. Roles, read events, and the upgrade to an audit trail
**Size:** L · **Blocks:** 22, and any real deployment · **Disclaimer:** ✅ §7.12

The activity history is append-only and covers writes and exports
(`docs/activity_history.md`), and the project deliberately **does not call it an
audit trail** because roles, read-event logging and tamper resistance are absent.
Fix all three and the name becomes earned.

Roles: attending radiologist · resident · oncologist · researcher · auditor.
Read events: viewing a case is an access to patient-derived data and is the event
an audit asks about first. Tamper resistance: hash-chain the event rows.

* **Where:** `auth/models.py` (new `role` column — none exists, §3.6),
  `auth/dependencies.py`, `api/deps.py`, every route, `services/activity.py`
  (`ACTIONS` grows read events), `db/session.py:20` `_ADDITIVE_COLUMNS` for the
  migration
* **Acceptance:** roles layer onto the existing `owner_id` scoping and never
  widen it — a role grants no cross-owner access that did not exist. Read-event
  logging cannot be disabled by a user. Renaming the feature to "audit trail" in
  the UI is part of **this** item's acceptance and nothing earlier.

---

#### 17. De-identification on ingest
**Size:** M · **Blocks:** 20 · **Disclaimer:** ✅ §7.13

Named in `roadmap.md` §5 and still unbuilt. Required before item 20, where DICOM
arrives carrying patient name, MRN, birth date, accession number, institution,
device serials and private tags.

* **Where:** `api/v1/upload.py`, `services/case_service.py`, a new profile module
* **Acceptance:** the profile applied is **recorded on the case** and shown in the
  packet from item 7. A NIfTI's own limited header fields are handled in the same
  pass, so the policy is one policy and not two. De-identification is **never
  described as anonymisation** — these are different standards and conflating
  them is a compliance claim the project cannot make.

---

#### 18. Scheduled retention sweep — completes P0-8
**Size:** S · **Blocks:** nothing · **Disclaimer:** —

Owner-scoped delete with file removal and a 409 guard ships. The automated sweep
does not, so patient-derived volumes are retained until someone remembers.

* **Where:** `app/main.py:27` `lifespan` (an in-process scheduler alongside the
  job-queue worker — Celery is not available, §3.7), `services/case_service.py`,
  `storage/filesystem.py`
* **Acceptance:** the sweep reuses the **existing** delete path and its 409
  guard — it must not delete files from under the single worker. Every sweep
  action is an activity event. The policy window is configuration, not a
  constant. Rows first, then files, and a filesystem error does not undo the
  rows — the existing `case_files_orphaned` reasoning carries over verbatim.

---

### Tier 6 — Infrastructure and hospital interoperability

Each is a project with its own plan. They are listed in the only order that works.

---

#### 19. PostgreSQL and an external task queue
**Size:** L · **Blocks:** 20, 21, 22 · **Disclaimer:** —

`sqlite:////app/data/app.db` and a process-local `asyncio.Queue` with one worker
(`services/job_queue.py:1-14`: jobs lost on crash, no retry, no coordination
between replicas). This **cannot serve concurrent readers** regardless of what is
built on top, and the Space filesystem is ephemeral besides.

* **Acceptance:** job recovery on restart — the documented gap where a crashed
  process leaves rows `QUEUED`/`RUNNING` forever — is **in scope for this item**,
  not deferred again. Queue concurrency stays bounded by available accelerator
  memory; parallel inference OOMs a single GPU, which is why the single worker
  exists in the first place.

---

#### 20. DICOM / DICOMweb ingest and clinical preprocessing
**Size:** XL · **Blocks:** 21 · **Disclaimer:** ✅ §7.14

The largest practical blocker to clinical use, and the one no UI work can
substitute for. Radiologists do not export, skull-strip, co-register and convert
four `.nii.gz` files by hand.

Two halves, and **the second is the hard one**: transport (C-FIND / C-MOVE /
WADO-RS, or Orthanc / dcm4chee in front) and **preprocessing to the contract the
models were trained on** — 1 mm isotropic, skull-stripped, co-registered to
SRI24, LPS. Clinical acquisitions are 5 mm anisotropic FLAIR with skull on.
Feeding them in unprocessed produces a confident, invalid segmentation.

* **Acceptance:** preprocessing is **auditable per case** — every step, its
  parameters and its software version, recorded on the case like checkpoint
  provenance is today. Any study the pipeline cannot bring to contract is
  **refused with the reason**, not processed anyway. The input-similarity check
  (`docs/input_domain_check.md`) runs **after** preprocessing and is load-bearing
  here, because this is precisely the path that introduces out-of-distribution
  input. This item also unlocks modality identification from `SeriesDescription`
  / `EchoTime` / `RepetitionTime` — header reading, not guessing, which is the
  only form of it this project should ever ship (§6.5).

---

#### 21. DICOM-SEG and RTSTRUCT export
**Size:** L · **Blocks:** nothing · **Disclaimer:** ✅ §7.15

Radiation oncology (Eclipse, RayStation) and neuronavigation (Brainlab,
StealthStation) **cannot import NIfTI**. Without this, the segmentation cannot
reach the systems that would act on it.

**This strictly cannot precede item 20, and not for convenience.** A conformant
DICOM-SEG or RTSTRUCT references the source series' SOP Instance UIDs and frame
of reference. Those UIDs exist only if the case arrived as DICOM. From a NIfTI
alone you cannot produce a conformant object — only a plausible-looking one that
a TPS will reject or, worse, silently misregister.

* **Acceptance:** exported structures carry the research-use designation in the
  structure set label itself, so the disclaimer survives import into a TPS where
  this application's UI is nowhere in sight. A corrected mask (item 10) exports
  as a **distinct, separately named** structure set from the model's own.

---

#### 22. HL7 ORU / FHIR DiagnosticReport
**Size:** L · **Blocks:** nothing · **Disclaimer:** ✅ §7.16

Push the report into Epic / Cerner and pull molecular pathology from the lab
system instead of typing it. Requires item 6 (the document), item 16 (roles and
an audit trail, because writing to an EHR is an attributable act) and item 19
(an EHR integration cannot hang off one in-process worker).

* **Acceptance:** the research-use designation is carried in the resource itself,
  not in an accompanying note. Inbound molecular data is reconciled against the
  existing pathology record, never silently overwriting a human entry.

---

## 5. Deferred — with the reason and the condition that unblocks it

Deferred is not rejected. Each has a stated unblock.

| Feature | Why deferred | What unblocks it |
|---|---|---|
| **Full perfusion (DSC/DCE), MRS, 4D DWI analysis** | Different physics, different dimensionality: 4D hemodynamic time series, pharmacokinetic modelling, chemical-shift spectra. New ingest, new viewers, new networks. Item 15 ships the *viewing* slice that needs no new science | A scoped sub-project with its own validation plan, after item 20 |
| **iRANO (immunotherapy response)** | Depends on RANO 2.0 landing first (item 12) and adds a distinct confirmation-scan rule for immunotherapy pseudoprogression | Item 12 in production, plus a clinical reviewer to validate the rule encoding |
| **Hanging protocols, multi-monitor layouts** | A browser tab cannot own a reading room's display topology. Partially approximated by item 4 | A desktop deployment target, or a deliberate decision to live inside one window |
| **DICOM Part 14 GSDF / calibrated grayscale** | **Not achievable in a browser at all.** Requires a calibrated diagnostic display and a calibrated display path. Must be *disclaimed* rather than deferred-with-a-plan — see §7.17 | Nothing in this repository. It is a hardware and platform property |
| **Automatic modality classification** | A wrong auto-assignment fails silently and convincingly. Item 2 ships the warning half | Item 20 — then it is DICOM header reading, not inference (§6.5) |
| **Calibrated uncertainty / voxel-wise uncertainty map** | Needs a calibration study the project has not run; `roadmap.md` §7 already defers it on exactly this ground | A multi-centre calibration study. Until then, item 1 |
| **Age/sex as grade-model covariates** | Architecture change, pre-registration, retrain, and a demographic bias re-evaluation. Item 14 ships the recording half | A pre-registered retraining study, not a UI change (§6.3) |
| **Survival / prognosis prediction** | Not deferred. **Do not build** — §6.1 | — |
| **Per-case "confidence %"** | Not deferred. **Do not build** — §6.2 | — |

---

## 6. Do not build — reaffirmed, with the buildable substitute

`roadmap.md` §8 holds. These are not UI limitations; they are the boundary
between a research prototype and an unapproved medical claim. **A disclaimer does
not cure an unvalidated number** — it only documents that you knew.

For four of the five, there is an honest adjacent feature that *is* in the build
order above. That substitute is the deliverable. The banned form is not.

### 6.1 Survival or prognosis prediction — **do not build, no substitute in scope**
*"Median survival: 14.2 months"* is a direct clinical claim with nothing behind
it. Image-only survival regression performs close to age-alone baselines in the
BraTS benchmarks. An inaccurate estimate can shift a patient between aggressive
therapy and palliative care. **No disclaimer makes this shippable.** If
population-level reference figures are ever wanted, they belong as **cited
literature by grade and molecular status, attributed to the publication**, never
as a per-patient output of this system — and even that is a documentation
decision, not a feature.

### 6.2 Per-case "confidence %" — **do not build → build item 1 instead**
Softmax is uncalibrated and overconfident; a model is routinely 98% "confident"
and wrong. The honest inverse is **item 1**: report the *evidential support* for
the probability — which calibration bin, how many evaluation patients in it. A
single badge would need conformal prediction or temperature scaling validated
across multi-centre external data, i.e. a study, not a component.

### 6.3 Age/sex covariates in the grade model — **do not build → build item 14 instead**
Two form fields do not make an image-only PyTorch model use them. Real integration
means fusing tabular features with 3D convolutional embeddings, pre-registering,
retraining and re-evaluating for demographic bias. **Item 14** records and
displays the same data without pretending the model saw it.

### 6.4 Perfusion / MRS / ADC-DWI features — **do not build as model features → build item 15 instead**
Outside the four-modality input contract. **Item 15** lets a reader *look* at a
derived map in the grid while the model contract stays exactly where it is, and
nothing numeric is derived from it.

### 6.5 Automatic modality classification — **do not build → build item 2 instead; revisit at item 20**
Slot-based identity is honest. A silent T1ce/T2 swap produces a convincing,
invalid mask. **Item 2** warns without reassigning. After **item 20**, identity
comes from DICOM headers — which is reading metadata, not classifying images, and
is the only form of this that should ever ship.

---

## 7. Disclaimer and note library

Exact copy to ship **with the feature, in the same view** — not in a footer and
not in a tooltip. Voice matches `CORRECTION_NOTE` (`correction_service.py:43`)
and `ViewerDisclaimer.tsx`: declarative, states what is claimed and what is not,
no hedging. Render these from a shared constant so the server and client cannot
drift, exactly as `CORRECTION_NOTE` already does.

**§7.1 — Grade calibration support (item 1)**
> The probability above is the ensemble's output. The bar beside it is how much
> evaluation data supports a probability in that band. Where the count is zero,
> this model's accuracy at this probability has not been measured — the number is
> not wrong, it is unverified. A probability near the decision threshold with no
> evaluation support behind it should not be read as a borderline case; it should
> be read as a case this model was not evaluated on.

**§7.2 — Modality slot plausibility (item 2)**
> The intensity relationships in these four volumes do not match what this slot
> assignment usually looks like. Nothing has been changed and nothing has been
> reassigned — the slots are as you declared them. Please confirm each file is in
> the correct slot before relying on this segmentation. A swapped sequence
> produces a confident and completely invalid result.

**§7.3 — Window/level and intensity readout (item 3)**
> Window presets here are percentile ranges of this volume, named by sequence.
> They are not calibrated tissue windows: MRI carries no absolute intensity
> units, and these volumes are normalised per case, so values are not comparable
> between studies or between patients. The readout is the stored voxel value in
> arbitrary units. Window and level are display-only; the source image is never
> modified.

**§7.4 — Captured slice images (item 5)**
> This image is a rendering of a stated view — sequence, plane, slice and window
> are recorded with it. It is a picture of the data at one display setting, not
> the data. Measure on the volume in the viewer, never on this image.

**§7.5 — PDF report (item 6) — on every page footer**
> Research prototype. Not a medical device, not clinically validated, and not for
> diagnosis, treatment, or other clinical decisions. Segmentation and
> measurements are automated and mask-derived; they are not a radiologist's
> measurements. Model output: {provenance verdict} · {model version} ·
> {checkpoint id}. Generated {timestamp} from result {result id}.

**§7.6 — Tumor board packet (item 7) — cover page**
> Prepared for discussion from a research prototype. Not a medical device and not
> clinically validated. De-identification profile: {profile}. This packet
> combines automated, mask-derived measurements with criteria entered by
> {entered-by} on {date}. Response category is calculated from those entered
> criteria under {criteria version} — it is not a model prediction. No item in
> this packet is a diagnosis.

**§7.7 — Structured findings block (item 8)**
> Automated, mask-derived measurements from {model version}, checkpoint
> {checkpoint id}. Values only — this block contains no interpretation,
> impression or comparison. Verify against the images before use.

**§7.8 — In-viewer contour editing (item 10)**
> You are editing a copy. The model's own segmentation is preserved unchanged
> beside your edit and both remain available — they are different claims.
> Volumes are recomputed from your edited mask; the WHO grade and the tumour type
> on this result are **not** recomputed, because they were produced from the
> model's mask and running them on an edited mask would be a different inference
> that has not been performed. Slices you did not edit are unchanged. Any
> interpolated slice is marked as interpolated.

**§7.9 — RANO 2.0 and RANO-LGG (items 12, 13)**
> Calculated from criteria you entered, under {criteria version}. This is
> threshold arithmetic over your inputs, not a model prediction and not a
> response assessment. Criteria version is recorded with every assessment;
> assessments stored under an earlier version are never recategorised. Where two
> criteria versions disagree for this timepoint, both are shown — the choice
> between them is clinical and is yours.

**§7.10 — Demographic and clinical context (item 14)**
> Entered by a user and displayed for reference. The grade and tumour-type models
> are image-only: they take no demographic or clinical covariates and did not see
> any of these values. Nothing on this result is adjusted for them.

**§7.11 — Reference sequence viewing (item 15)**
> This is a reference sequence, shown for viewing only. It is not part of the
> four-modality input contract: no model received it, no measurement is derived
> from it, and the input-similarity check does not cover it. Parametric maps such
> as ADC and rCBV depend on acquisition and post-processing parameters that this
> application does not read and cannot verify. Interpret it in your own reading
> environment.

**§7.12 — Audit trail (item 16)**
> Access and modification events are recorded and hash-chained. This record is
> for operational traceability in a research prototype. It has not been assessed
> against 21 CFR Part 11, HIPAA technical safeguards, or any other regulatory
> audit standard.

**§7.13 — De-identification (item 17)**
> De-identification profile {profile} removed the identifiers listed in {link}.
> This is de-identification, not anonymisation: residual identifiability from
> image content — facial structure in particular — and from dates and rare
> findings is not eliminated. Do not treat the output as anonymous data.

**§7.14 — DICOM ingest and preprocessing (item 20)**
> This study was converted and preprocessed to the input contract the models were
> trained on: {steps, parameters, software versions}. Preprocessing changes the
> images the models see. Segmentation accuracy on studies converted this way has
> not been separately validated against the published evaluation, which used
> already-preprocessed research data. Check the input-similarity result for this
> case before relying on anything below.

**§7.15 — DICOM-SEG / RTSTRUCT export (item 21) — in the structure set label itself**
> RESEARCH USE ONLY — NOT FOR TREATMENT PLANNING. Generated by an unvalidated
> research prototype, {model version}, checkpoint {checkpoint id}. Review and
> edit every contour before any clinical use.

**§7.16 — FHIR / HL7 report (item 22)**
> Research prototype output. Not a medical device, not clinically validated, not
> for diagnosis or treatment. Produced by {model version}, checkpoint
> {checkpoint id}, from automated segmentation. Requires review by a qualified
> reader before any clinical use.

**§7.17 — Display calibration (standing, in the viewer)**
> This viewer runs on an uncalibrated display. It does not implement the DICOM
> Part 14 Grayscale Standard Display Function, and the browser's own colour
> management is outside its control. Subtle low-contrast findings may not be
> visible here that are visible on a calibrated diagnostic display. This is a
> limitation of the platform, not a setting.

---

## 8. Progress

Maintain alongside `roadmap.md` §9.

| # | Item | Tier | Size | Disclaimer | State |
|---|---|---|---|---|---|
| 1 | Grade calibration-support readout | 0 | S | §7.1 | not started |
| 2 | Modality slot plausibility warning | 0 | S | §7.2 | not started |
| 3 | Per-sequence W/L + presets + readout | 1 | S | §7.3 | not started |
| 4 | 2×2 linked multi-sequence grid | 1 | M | — | not started |
| 5 | Slice capture substrate | 1 | M | §7.4 | not started |
| 6 | Server-assembled PDF (P0-6) | 2 | M | §7.5 | not started |
| 7 | Tumor board packet | 2 | M | §7.6 | not started |
| 8 | Structured findings block | 2 | S | §7.7 | not started |
| 9 | Client-side NIfTI writer | 3 | S–M | — | not started |
| 10 | In-viewer brush and eraser | 3 | L | §7.8 | not started |
| 11 | Non-enhancing / T2-FLAIR disease | 4 | S | — | not started |
| 12 | RANO 2.0 | 4 | M | §7.9 | not started |
| 13 | RANO-LGG | 4 | M | §7.9 | not started |
| 14 | Demographic / clinical context record | 4 | S | §7.10 | not started |
| 15 | Reference-sequence viewing | 4 | M | §7.11 | not started |
| 16 | Roles + read events → audit trail | 5 | L | §7.12 | not started |
| 17 | De-identification on ingest | 5 | M | §7.13 | not started |
| 18 | Retention sweep (P0-8) | 5 | S | — | not started |
| 19 | PostgreSQL + external queue | 6 | L | — | not started |
| 20 | DICOM / DICOMweb ingest + preprocessing | 6 | XL | §7.14 | not started |
| 21 | DICOM-SEG + RTSTRUCT export | 6 | L | §7.15 | not started |
| 22 | HL7 ORU / FHIR DiagnosticReport | 6 | L | §7.16 | not started |

**Withdrawn from scope** (not deferred — not deliverable as specified):
calibrated absolute-unit intensity readout (§3.2) · DICOM Part 14 GSDF
compliance (§5, disclaimed in §7.17).

**Do not build** (§6): survival/prognosis · per-case confidence % ·
age/sex as grade-model covariates · perfusion/MRS/DWI as model features ·
automatic modality classification.

---

## 9. What this buys a clinician, in order

* **After Tier 1** (items 1–5): a viewer a neuroradiologist can actually read
  from — four sequences at once, windowed per sequence, with the model's
  reliability stated per case rather than per model card.
* **After Tier 2** (6–8): something they can take to a meeting, put in a chart,
  or paste into a dictation system.
* **After Tier 3** (9–10): a contour they can sign off on, because they can fix it.
* **After Tier 4** (11–15): a neuro-oncologist's response assessment for
  **both** halves of the model's declared cohort, under current criteria, with
  the clinical context on the page and the sequences that answer the
  pseudoprogression question in view.
* **After Tier 5** (16–18): a second reader can log in without the first one's
  data governance becoming a problem.
* **After Tier 6** (19–22): the application sits inside a hospital's imaging and
  record systems rather than beside them.

Tier 6 is where "prototype" stops being the right word — and it is also where
regulatory clearance stops being optional. Nothing in Tiers 0–5 changes the
standing constraint in §2.
