# Architecture — Phase 4

Research / decision-support prototype. **Not a diagnostic device. Not clinically validated.**

## Boundary

Training stays in a separate notebook. This application validates and stores BraTS
NIfTI cases, runs them through a replaceable inference service, and visualizes the
result. Through Phase 4 the only implementation behind that boundary is
`MockInferenceService`, which synthesizes a geometric mask. **No trained model is
loaded and no real prediction is produced.**

```mermaid
flowchart LR
    User[User] --> FE[React SPA]
    FE -->|REST /api/v1| BE[FastAPI]
    BE --> Cases[CaseService]
    Cases --> Val[NIfTI validation]
    Cases --> Store[Isolated filesystem]
    Cases --> DB[(SQLite metadata)]
    BE --> Queue[JobQueue single worker]
    Queue --> Inf[InferenceService protocol]
    Inf --> Mock[MockInferenceService]
    Inf -.-> Real[Phase 6 RealBraTSInferenceService]
    Queue --> Res[ResultService]
    Res --> Artifacts[segmentation.nii.gz + result.json]
    FE -->|artifact stream| Artifacts
```

The frontend never learns which implementation ran. It sees `inference_source`
(`mock` today, `pytorch` later) and an artifact URL.

## Public API

Ingest endpoints create a case, validate it, enqueue a job, and return **202**:

- `POST /api/v1/predict` — four modalities
- `POST /api/v1/evaluate` — four modalities plus `seg` (Dice/HD95 are **not** computed yet)

Upload UX needs per-file progress, so these case endpoints are **additive**
(not a competing product API):

- `POST /api/v1/cases`
- `POST /api/v1/cases/{case_id}/files/{modality}`
- `POST /api/v1/cases/{case_id}/complete`
- `GET  /api/v1/cases/{case_id}`
- `POST /api/v1/cases/{case_id}/predict` — 202, or 409 if the case is not READY
- `GET  /api/v1/cases/{case_id}/artifacts/{artifact}` — streams an allowlisted NIfTI

Job and result polling:

- `GET /api/v1/jobs/{job_id}` — status, progress, `result_id` when COMPLETED
- `GET /api/v1/results/{result_id}` — measurements, segmentation availability

## Inference boundary

`backend/app/inference/base.py` defines the `InferenceService` protocol and the
`InferenceResult` dataclass. Swapping in a real model means adding one class that
satisfies the protocol and changing the construction site in `main.py`'s lifespan.
No route, schema, or frontend change is required.

Execution runs through `JobQueue`: an in-process asyncio FIFO with a **single**
worker, so two cases can never contend for the same GPU. Documented limits: the
queue is process-local (it does not coordinate across replicas), queued items are
lost on crash while the DB row stays QUEUED/RUNNING, and there is no retry.

## Frontend (`frontend/`)

- **React + TypeScript + Vite**, Tailwind CSS, strict mode.
- **API client**: every request goes through `src/lib/api.ts`, which resolves its
  origin from `VITE_API_URL` (empty = same-origin / Vite proxy). No component
  hardcodes a host.
- **State**: React local state for UI transitions; URL parameters drive case/job lookup.

**Phase 4 visualization** — entirely in-browser, with no PACS/Orthanc dependency:

- **Parsing**: `nifti-reader-js` + `pako` decode `.nii.gz` artifacts into Float32
  arrays, preserving affine and spacing.
- **2D multiplanar slicing**: `src/lib/sliceExtraction.ts` extracts arbitrary
  axial/coronal/sagittal slices from the flat array, applies percentile window
  normalization, and composites the overlay into an `ImageData` for canvas.
- **Coordinate convention**: NIfTI **RAS+** assumed across all axes. No reorientation,
  no resampling.
- **3D mesh generation**: dependency-free marching cubes in `src/lib/meshGeneration.ts`,
  scaling vertices by voxel spacing, with a per-region triangle budget.
- **3D rendering**: `Three.js` via `react-three-fiber`, with `OrbitControls` for
  rotate / zoom / pan.
- **Keyboard**: `KeyJ` / `KeyL` step the active plane, `KeyI` / `KeyK` cycle planes
  (`src/hooks/useSliceKeyboard.ts`).

## Case states

`CREATED` → `UPLOADING` → `VALIDATING` → `READY` or `FAILED`

## Job states

`QUEUED` → `RUNNING` → `COMPLETED` or `FAILED`

Transitions are explicit in `JobService`; there are no ad-hoc booleans.

## Storage

```text
{UPLOAD_DIR}/cases/{case-uuid}/input/{modality}.nii[.gz]
{UPLOAD_DIR}/cases/{case-uuid}/output/segmentation.nii.gz
{UPLOAD_DIR}/cases/{case-uuid}/output/result.json
{UPLOAD_DIR}/cases/{case-uuid}/metadata/case.json
```

Original filenames are never used as filesystem paths. Files are not served by the
frontend origin. Artifact URLs accept only the keys in `VIEWER_ARTIFACTS`; anything
else is a 400 before any path is constructed, so the route cannot be walked.

## Validation

**Hard failure:** unreadable NIfTI, not 3D, non-finite affine/spacing/voxels, missing
required modality, duplicate modality, shape/affine/spacing mismatch, `seg` labels
outside `{0,1,2,4}`, case over `MAX_UPLOAD_SIZE_MB`.

**Warning only:** all-zero volume; qform/sform disagreement.

No resampling. Label `3` is not rewritten to `4` on upload.

## BraTS label semantics

Uploads carry raw BraTS labels `{0, 1, 2, 4}`. The mock generates internally with
`3` for ET and converts once, in `to_brats_labels()` in `app/core/constants.py`,
before anything is written to disk. Every exported artifact and every API response
uses BraTS labels. A real model that emits class `3` must call the same function —
internal class `3` is never displayed as "BraTS label 3".

The frontend keeps the matching display config in `src/types/viewer.ts`
(`REGION_CONFIG`), keyed by BraTS label. Results and viewer read the same table,
so colours cannot drift apart.

## Measurements

Volumes are computed as `voxel_count × sx × sy × sz` from the NIfTI header zooms,
reported in mm³ and cm³ alongside the raw voxel count. Voxel counts are never
presented as physical volume.
