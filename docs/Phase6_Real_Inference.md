# Phase 6 — Real PyTorch Model Integration

**Status:** implemented, and verified against the trained `best_model.pth`.
The checkpoint loads strictly into the reconstructed architecture and its
recorded configuration fingerprint matches this service's — see
[§10 Trained checkpoint](#10-trained-checkpoint-verification).

Research / decision-support prototype. **Not a diagnostic device. Not clinically validated.**
Nothing in this document reports segmentation quality.

---

## 1. What Phase 6 changes

Phases 1–5 ran every case through `MockInferenceService`, which synthesizes a
geometric ellipsoid. Phase 6 adds `RealBraTSInferenceService`, which runs the 3D
U-Net trained in `notebook/notebookd7ed0558bb (2).ipynb`.

The mock is **not** removed. `INFERENCE_BACKEND` selects between them, and the
downstream contract is identical either way:

```
POST /predict → 202 → JobQueue (single FIFO worker)
                        → InferenceService.predict()
                            → segmentation.nii.gz
                        → MeasurementService
                        → MetricsService (when ground truth exists)
                        → ResultService
                    → GET /results, /measurements, /report, /artifacts
```

No route, schema, database column, or frontend file changed shape. The frontend
already typed `InferenceSource` as `"unavailable" | "mock" | "pytorch"`.

---

## 2. Notebook → production parity table

Every value is transcribed from the notebook. Nothing is inferred from generic
3D U-Net practice.

| Component | Notebook (source cell) | Production | Match |
|---|---|---|---|
| Modalities | T1, T1ce, T2, FLAIR (§8.3 `modality_order`) | same | ✅ |
| Channel order | `["t1","t1ce","t2","flair"]` — literal list, never sorted (§8.3) | `MODALITY_ORDER` constant, stacking driven by it | ✅ |
| Input dtype | `np.asanyarray(img.dataobj, dtype=np.float32)` (§8.3) | same | ✅ |
| Normalization | Per-case, per-modality z-score over `volume != 0`; background left at exactly 0 (§8.3 `foreground_zscore_normalize`) | same function, transcribed | ✅ |
| Foreground definition | `volume != background_value`, `background_value = 0.0` (§8.2) | same | ✅ |
| Global/dataset statistics | Never used — normalization is per case (§8 Assumption 2) | same | ✅ |
| Cropping | Bounding box of union of the 4 **raw** modalities' non-zero voxels, margin 4 voxels, clipped (§8.3 `compute_foreground_bbox`) | same | ✅ |
| Resampling | **None.** §5.1 measured 1251/1251 cases at 1.0 mm → `resampling_required = False` | none | ✅ |
| Reorientation | **None.** §8.1 measured 1251/1251 cases at `('L','P','S')` | none | ✅ |
| Intensity clipping / percentile norm | Not performed | not performed | ✅ |
| Skull stripping | Assumed already done (BraTS ships skull-stripped) | assumed, never applied | ✅ |
| Patch size | `(128, 128, 128)` (§9.1 selected, §9.2 locked for cycle 2) | `(128, 128, 128)` | ✅ |
| Model in_channels | 4 (§2.2 `ModelConfig`) | 4 | ✅ |
| Model classes | 4 (§2.2) | 4 | ✅ |
| Encoder stages | 5 (§2.2 `num_stages`) | 5 | ✅ |
| Base / max channels | 32 / 320 (§2.2, §12.1 — the CPU derate to 16 did not fire on the GPU run) | 32 / 320 | ✅ |
| Channel schedule | `[32, 64, 128, 256, 320]` (§12.3 printed output) | same | ✅ |
| Parameter count | `18,774,756` (§12.3 printed output) | `18,774,756` (asserted by test) | ✅ |
| Conv block | 2 × `Conv3d(k=3, p=1, bias=False)` → `InstanceNorm3d(affine=True)` → `LeakyReLU(0.01)` (§12.2) | same | ✅ |
| Downsampling | `Conv3d(ch, ch, k=2, s=2)` — strided conv, not pooling (§12.2 `Down3D`) | same | ✅ |
| Upsampling | `ConvTranspose3d(in, in, k=2, s=2)` then skip concat then ConvBlock3D (§12.2 `Up3D`) | same | ✅ |
| Skip connections | Concatenation, encoder stages 0..3 (§12.3) | same | ✅ |
| Output head | `Conv3d(32, 4, k=1)` (§12.3) | same | ✅ |
| Output activation | **None in the module** — raw logits; softmax applied in sliding-window (§16.1) | same | ✅ |
| Label mapping (train) | raw `{0,1,2,4}` → internal `{0,1,2,3}` (§2.1) | same constants | ✅ |
| Label mapping (export) | internal `3 → 4`, asserted (§19.2) | `internal_to_brats_labels`, raises on violation | ✅ |
| Sliding window | End-only zero padding to patch size; stride `round(p × (1−overlap))`; final tile flushed to far edge (§16.1) | same | ✅ |
| Overlap | `0.5` → stride `64` (§2.2 `PatchConfig`) | `0.5` → stride `64` | ✅ |
| Gaussian blending | Separable Gaussian, `sigma = 0.125 × dim`, centre `(dim−1)/2`, clipped at `1e-4` (§19.1) | same | ✅ |
| Accumulation | `prob_map += softmax(logits) × kernel`; `weight_map += kernel`; divide by `clip(weight_map, 1e-8)` (§16.1) | same | ✅ |
| Inference mode | `model.eval()` + `torch.no_grad()` (§16.1) | `model.eval()` + `torch.inference_mode()` | ✅ equivalent |
| Argmax | `np.argmax(prob_map, axis=0).astype(np.uint8)` (§19.2) | same | ✅ |
| Post-processing | Connected components per class `{1,2,3}`, `min_voxels=50`, 6-connectivity (`scipy.ndimage.label` default) (§20.1) | same | ✅ |
| Inverse crop | Zero-filled volume, cropped prediction written into `crop_bbox` (§21.1) | same | ✅ |
| Output labels | `{0, 1, 2, 4}`, asserted (§21.1, §22.4) | same, asserted | ✅ |
| Output affine | Reference (T1) affine from the preprocessed case (§21.1) | same, plus qform/sform codes copied | ✅ superset |
| Checkpoint format | `{"model_state_dict", "optimizer_state_dict", "scaler_state_dict", "epoch", "global_step", "best_val_metric", "compat_fingerprint", "torch_rng_state", "numpy_rng_state", "saved_at"}` (§15.4) | read; bare `state_dict` and `{"state_dict": …}` also accepted | ✅ |
| Compatibility fingerprint | SHA-256 of `{model, patch_selected_size, preprocessing}`, `sort_keys=True`, 12-char prefix (§15.4) | reproduced byte-for-byte | ✅ |
| Checkpoint strictness | `model.load_state_dict(payload["model_state_dict"])` — strict by default (§15.4) | explicit `strict=True` | ✅ |

### Two documented divergences

**1. CC filtering is applied before export.**
The notebook's `validate_one_case` (§21.1) computes its reported Dice/HD95 from
the **CC-filtered** mask but saves the **unfiltered** `pred_raw` to disk:

```python
pred_internal_clean = connected_component_filter(pred_internal, ...)   # metrics use this
...
_pred_raw_full = restore_prediction_to_original_space(pred_raw, ...)   # export uses this
```

Production applies the filter **before** export, so the stored
`segmentation.nii.gz` is the same mask the notebook's reported metrics were
computed from. This is the notebook's declared Section 20 post-processing step
applied consistently; the notebook's export path appears to have skipped it.
`metadata["postprocessing"]["removed_voxels_by_class"]` records exactly how many
voxels the filter removed, so the difference is always visible.

**2. Production squeezes singleton axes on load.**
`np.squeeze` is applied to each loaded volume so a `(X, Y, Z, 1)` upload behaves
the way the Phase 2 validator and `MockInferenceService` already treat it. It is
a no-op for the 3D volumes the notebook trained on and changes no voxel value.

---

## 3. Module layout

```
backend/app/inference/
├── base.py                  InferenceService protocol, InferenceResult, provenance accessors
├── factory.py               INFERENCE_BACKEND → implementation (torch imported lazily)
├── mock.py                  MockInferenceService (unchanged behaviour)
├── unavailable.py           UnavailableInferenceService — configured but not loadable
├── pytorch_service.py       RealBraTSInferenceService
└── brats/
    ├── config.py            Frozen training config + compat fingerprint + label maps
    ├── model.py             ConvBlock3D, Down3D, Up3D, UNet3D
    ├── checkpoint.py        Strict loading with explicit failure modes
    ├── preprocessing.py     Foreground z-score + foreground-bbox crop
    ├── sliding_window.py    Tiling, Gaussian kernel, weighted stitching
    └── postprocess.py       CC filter, label restoration, inverse crop

backend/scripts/
├── notebook_reference.py       Verbatim notebook transcription (parity only)
├── notebook_parity_check.py    Notebook vs production comparison harness
└── make_synthetic_checkpoint.py  Random-weight fixture generator
```

`app/inference/brats/config.py` is the single source of truth for everything the
checkpoint depends on. **None of it is environment-tunable** — a value that
drifts from training silently breaks inference, so it lives in source control.

---

## 4. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `INFERENCE_BACKEND` | `mock` | `mock` or `pytorch` |
| `MODEL_PATH` | *(empty)* | Path to `best_model.pth`. Required for `pytorch` |
| `MODEL_VERSION` | *(empty)* | Optional label; otherwise derived from the checkpoint |
| `INFERENCE_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:0` |
| `INFERENCE_STRICT_FINGERPRINT` | `true` | Refuse a checkpoint whose recorded config fingerprint differs |

`INFERENCE_STRICT_FINGERPRINT=false` relaxes **only** the metadata fingerprint
check. `load_state_dict(strict=True)` is unconditional — a checkpoint that does
not match the architecture key-for-key is always refused.

Run with the trained model:

```bash
export INFERENCE_BACKEND=pytorch
export MODEL_PATH=/abs/path/to/best_model.pth
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 5. Startup and health semantics

The model is built and loaded **once**, in the FastAPI lifespan. Every job
reuses the same in-memory model; no job reconstructs it.

A GPU is never required to start the API. If `INFERENCE_BACKEND=pytorch` but the
checkpoint is missing, corrupt, or incompatible, startup still succeeds and
installs `UnavailableInferenceService`:

| Situation | `model_loaded` | `inference_source` | `POST /predict` |
|---|---|---|---|
| `INFERENCE_BACKEND=mock` | `false` | `mock` | 202, synthetic result |
| `pytorch`, checkpoint loaded | `true` | `pytorch` | 202, real inference |
| `pytorch`, checkpoint unusable | `false` | `unavailable` | **503 `MODEL_UNAVAILABLE`** |

`/health` never reports `model_loaded: true` unless a checkpoint was actually
read from disk and loaded into the network. An unavailable backend refuses work
up front rather than accepting jobs that are guaranteed to fail — and it never
falls back to mock output, which would present a synthetic mask as a prediction.

`GET /model/info` returns `headline_metrics: null` always. The checkpoint's
`best_val_metric` is a **training-time proxy selection metric**, not a validated
headline Dice; it is surfaced only under `details.checkpoint.proxy_val_mean_soft_dice`
so it cannot be mistaken for one.

---

## 6. Memory and concurrency

- One model instance per process, loaded at startup.
- `torch.inference_mode()`; no autograd graph is built or retained.
- Per-tile device tensors are released immediately; peak device memory is one
  patch, not one patch per tile.
- The accumulator is the only full-volume float array: `(4, D, H, W)` float32
  plus a `(D, H, W)` weight map, both over the **cropped** volume.
- `torch.cuda.empty_cache()` after each case when running on CUDA.
- The existing single-worker FIFO `JobQueue` is unchanged, so two cases can
  never contend for the same GPU. No parallel inference was added.
- Inference runs in a thread executor, never inside the HTTP handler.

---

## 7. Provenance recorded on every result

`inference_source="pytorch"`, `synthetic=false`, plus `checkpoint_id` (SHA-256
prefix of the checkpoint file), `model_version`, `architecture`,
`model_parameters`, `input_shape`/`output_shape`, `input_spacing`,
`crop_bbox`, `normalization_stats`, `patch_size`, `sliding_window_overlap`,
`sliding_window_stride`, `num_patches`, `gaussian_sigma_scale`,
`postprocessing.removed_voxels_by_class`, `label_mapping`, `output_labels`,
`device`, `preprocess_seconds`, `inference_seconds`, `peak_gpu_memory_bytes`
(CUDA only), and the checkpoint's own `epoch` / `saved_at`.

A checkpoint written by `make_synthetic_checkpoint.py` carries
`synthetic_fixture: true`; the service propagates that into
`checkpoint_is_synthetic_fixture` and rewrites the result description, so a
random-weight fixture can never be presented as trained-model output.

---

## 8. Validating parity

```bash
# Reduced width, seconds — also runs as tests/test_notebook_parity.py
python backend/scripts/notebook_parity_check.py --synthetic

# Real trained geometry: 240x240x155 volume, 128^3 patches, stride 64
python backend/scripts/notebook_parity_check.py --synthetic --full

# Real trained WEIGHTS at a patch size that fits in 8 GB
python backend/scripts/notebook_parity_check.py \
    --checkpoint models/best_model.pth --base-channels 32 \
    --patch-size 64 64 64 --shape 100 100 90

# Against a real BraTS case and the trained checkpoint (needs a GPU / large host)
python backend/scripts/notebook_parity_check.py \
    --checkpoint models/best_model.pth \
    --case-dir /path/to/BraTS2021_00002 --full
```

### Results recorded so far

| Run | Geometry | Weights | Checks | Result |
|---|---|---|---|---|
| Reduced | 56×60×48, patch 32³, base 8 | random | 16/16 | bit-exact |
| **Trained geometry** | **240×240×155, patch 128³, stride 64, 4 tiles**, base 8 | random | 16/16 | bit-exact |
| **Trained weights** | 100×100×90, patch 64³, stride 32, 8 tiles, base 32 | **`best_model.pth`** | 16/16 | bit-exact |

In every run the probability maps agreed to `max |Δ| = 0.000e+00` — well inside
the 1e-6 float32 tolerance — and the predicted labels, post-processed labels,
exported BraTS mask, foreground and per-region voxel counts, output shape,
affine and spacing were all identical. The trained-weights run also exercised
the connected-component filter non-trivially (27 / 2 / 4 voxels removed for
NCR / ED / ET), and both implementations agreed after filtering.

The harness compares preprocessed voxels, normalization statistics, crop box,
probability map, predicted labels, post-processed labels, exported mask,
foreground and per-region voxel counts, output shape, affine and spacing.

---

## 9. Known limitations

1. **No real BraTS case has been segmented end-to-end.** The trained checkpoint
   loads and is verified, but no BraTS study was available on the development
   machine — only 8³ synthetic fixtures. The pipeline has not produced a
   segmentation of real anatomy, so **no Dice, HD95, tumor volume, or
   trained-model inference time is reported anywhere in this repository.**

2. **Full-scale inference does not fit on the development machine.** The host
   has 8 GB RAM and 8 cores. With the trained width (base_channels=32), a
   measured `64³` forward takes **4.40 s at 3.25 GB peak RSS**; an `80³`
   forward did not complete and drove the process into heavy swapping
   (1.19M pageouts, resident set collapsing to 0.08 GB at 8% CPU). The trained
   patch size is `128³` — roughly 8× the activation memory of `64³` — so real
   cases need a GPU or a substantially larger host. This is a hardware limit,
   not a pipeline defect.

3. **Parity with the real weights was verified at a reduced patch size.**
   Bit-exact notebook-vs-production parity is established at the real trained
   geometry (240×240×155 volume, 128³ patches, stride 64) with reduced width,
   and separately with the **real trained weights** at 64³ patches. The two
   together cover every code path; a single run combining real weights *and*
   128³ patches awaits a machine that can hold it.

4. **No test-time augmentation, no ensembling.** The notebook used neither.

5. **The queue is still process-local.** Phase 3's documented limits are
   unchanged: no cross-replica coordination, no crash recovery, no retry.

6. **MPS (Apple GPU) is untested.** `INFERENCE_DEVICE` will pass `mps` through
   to torch, but the notebook never used it and its numerics are unverified here.

---

## 10. Trained checkpoint verification

Verified against the supplied `best_model.pth` (copied to `models/best_model.pth`,
which is gitignored):

| Property | Value | How established |
|---|---|---|
| File size | 225,398,066 bytes | `stat` |
| `checkpoint_id` | `d76e9365aea3` | SHA-256 prefix of the file |
| Format | `model_state_dict` payload | notebook `save_checkpoint` (§15.4) |
| Top-level keys | `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `epoch`, `global_step`, `best_val_metric`, `compat_fingerprint`, `torch_rng_state`, `numpy_rng_state`, `saved_at` | read from the file |
| `epoch` | 19 | read from the file |
| `global_step` | 20,020 | read from the file |
| `saved_at` | `2026-09-15T19:29:39.635585` | read from the file |
| State-dict entries | 72 | read from the file |
| Parameters | **18,774,756** | matches notebook §12.3 output exactly |
| `compat_fingerprint` | **`af663316e8bd`** | **matches this service's derived fingerprint exactly** |
| Strict load | **succeeds** with `strict=True` | `load_brats_checkpoint` |
| Model load time | 0.385 s (CPU) | measured |
| `model_version` | `unet3d-ep19-d76e9365aea3` | derived from digest + epoch |

The fingerprint match is the strongest single piece of evidence in Phase 6. The
notebook hashed its live `ModelConfig`, selected patch size, and
`PreprocessingConfig` into the checkpoint at save time (§15.4). Reproducing that
digest from `app/inference/brats/config.py` means **every field name and value
in the transcription matches what training actually used** — including fields
that do not affect tensor shapes, such as `crop_margin_voxels`,
`normalize_mode`, `canonical_orientation`, and `sagittal_axis_index`, which a
successful `load_state_dict` alone would not have caught.

`best_val_metric = 0.6777643101863605` is the **training-time proxy selection
metric** the notebook used to choose the best epoch. It is not a validated
headline Dice, it is not reported as one, and `/model/info` returns
`headline_metrics: null`. It appears only under
`details.checkpoint.proxy_val_mean_soft_dice`.

### This is the checkpoint the notebook itself evaluated

The notebook's Section 19.3 cell printed:

```
Loaded best_model.pth (epoch=19, proxy_val_dice=0.6777643101863605) for evaluation.
```

Both values match the supplied file exactly (`epoch = 19`,
`best_val_metric = 0.6777643101863605`), so the checkpoint wired into production
is the same one the notebook's Stage G validation ran against — not a different
epoch or a different run.

`tests/test_real_checkpoint.py` re-asserts all of the above; it skips when the
checkpoint is not present.

---

## 11. End-to-end run with the trained checkpoint

The complete application path was exercised with `best_model.pth` loaded:

```
upload → Phase 2 validation → POST /predict → 202 → JobQueue
      → RealBraTSInferenceService → segmentation.nii.gz
      → MeasurementService → ResultService
      → GET /results, /measurements, /report, /artifacts/segmentation
```

**Two deviations, both stated rather than hidden:**

1. **Patch size 64³ instead of the trained 128³.** Required by the 8 GB host
   (§9.2). This changes the configuration fingerprint, so the loader refused
   the checkpoint until `strict_fingerprint=False` was passed explicitly —
   which is precisely what that guard is for. `load_state_dict(strict=True)`
   would *not* have caught it: UNet3D is fully convolutional, so the weights
   fit any patch size without complaint.
2. **The input is a generated phantom, not a brain.** No BraTS study was
   available on this machine.

Observed (pipeline facts — **not** quality metrics):

| | |
|---|---|
| Job status | `COMPLETED`, `inference_source = pytorch` |
| `model_version` | `unet3d-ep19-d76e9365aea3` |
| Input shape | `(96, 96, 80)`, spacing `(1.0, 1.0, 1.0)` |
| Crop bbox → cropped | `[[11,86],[11,86],[8,73]]` → `(75, 75, 65)` |
| Tiles / stride | 8 patches, stride `(32, 32, 32)` |
| Output artifact | `segmentation.nii.gz`, 3,334 bytes, `uint8` |
| Output shape | `(96, 96, 80)` — identical to input |
| Output affine | identical to input affine |
| Output spacing | `(1.0, 1.0, 1.0)` |
| Labels present | `[0]` |
| CC filter removed | NCR 0, ED 16, ET 0 |
| Preprocess / inference | 0.054 s / 58.6 s (CPU, under memory pressure) |
| `/measurements` | 200, `synthetic: false`, foreground 0 voxels |
| `/report` | 200, `inference_source: pytorch` |
| `/artifacts/segmentation` | 200, 3,334 bytes streamed |

The model predicted background everywhere. **That is the expected and
uninformative outcome for random noise shaped like an ellipsoid** — it is not
evidence about the model's accuracy in either direction, and it is not reported
as such. What it does demonstrate is that real weights drive the full
production path and that every downstream stage consumes the result correctly.
