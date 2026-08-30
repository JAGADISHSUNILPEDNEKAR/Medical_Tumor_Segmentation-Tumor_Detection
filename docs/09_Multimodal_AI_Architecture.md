# Document 9 — Track-Specific Architecture (Multimodal AI)

## 1. Modalities Handled
* **Volumetric Image Modality:** Four co-registered, skull-stripped MRI NIfTI volumes per case — native T1, post-contrast T1 (T1ce/T1Gd), T2, and T2-FLAIR.
* **Label Modality:** A voxel-wise segmentation label map (background + tumor sub-region classes) when ground truth is available.
* **Structured Data Modality:** Case/job/result metadata, checkpoint provenance, and per-class evaluation metrics.

## 2. Multimodal Processing Pipeline

```
       ┌────────────────────────┐             ┌────────────────────────┐
       │ 4-Modality NIfTI Case  │             │ Ground-Truth Seg Mask  │
       │ (T1, T1ce, T2, FLAIR)  │             │  (if available)        │
       └───────────┬────────────┘             └───────────┬────────────┘
                   │                                      │
        ┌──────────▼───────────┐               ┌──────────▼───────────┐
        │ Preprocessing        │               │ Evaluation Engine    │
        │ - Orientation/Resample│               │ - Dice (per class)   │
        │ - Foreground Z-Score  │               │ - Hausdorff95 (mm)   │
        └──────────┬───────────┘               └──────────┬───────────┘
                   │                                      │
                   │    ┌────────────────────────────┐    │
                   └───►│ 3D U-Net Inference Engine  │◄───┘
                        │ - Sliding-Window + Gaussian│
                        │ - Argmax + CC Filtering    │
                        └─────────────┬──────────────┘
                                      │
                        ┌─────────────▼──────────────┐
                        │ Mask + Mesh + Metrics       │
                        └────────────────────────────┘
```

## 3. Model Architecture & Training Design
* **Base Architecture:** Custom 3D U-Net — symmetric encoder/decoder with channel-wise skip concatenation.
  * Encoder: 4–5 stages, each two 3×3×3 conv blocks followed by strided-conv downsampling by 2×.
  * Decoder: mirrors encoder depth; transposed-conv (or trilinear upsample + conv) upsampling, concatenated with the matching encoder skip.
  * Normalization: **InstanceNorm3d** — batch-independent, standard for 3D medical segmentation given the very small batch sizes (1–4 patches) that 3D volumetric training requires.
  * Activation: LeakyReLU (negative slope ≈ 0.01).
  * Channels: starting width 16–32, doubling per downsampling stage, capped (e.g., 320) to bound GPU memory.
  * Input: `(4, D, H, W)` per-case tensor; Output: `(C, D, H, W)`, `C` = background + tumor sub-regions.
* **Transfer of Ideas from nnU-Net (not the framework itself):** dataset fingerprinting, resampling to a fingerprinted target spacing, foreground-only z-score normalization, patch-based training with foreground oversampling, compound Dice+CE loss, sliding-window inference with Gaussian-weighted stitching, and connected-component post-processing are all adopted conceptually; the official nnU-Net codebase, its automatic architecture search, and full 5-fold ensembling are **not** adopted for MVP.
* **Loss Function:** Compound Dice + Cross-Entropy —
$$
\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|C_{fg}|}\sum_{c \in C_{fg}} \mathcal{L}_{Dice}^{c}
$$
  combining CE's stable per-voxel gradient with Dice's overlap-focused, imbalance-robust signal.
* **Optimizer/Schedule:** SGD with Nesterov momentum (≈0.99) or AdamW as an easier-to-tune fallback; polynomial LR decay.
* **Data Augmentation:** Sagittal-axis flips, small-angle rotation, light scaling/intensity shift/gamma/Gaussian noise — all chosen to respect brain anatomy (no anterior-posterior or superior-inferior flips) and to avoid over-aggressive elastic deformation that would corrupt HD95-sensitive boundary shape.

## 4. Evaluation Metrics
* **Dice coefficient**, per class: $Dice_c = \dfrac{2|P_c \cap G_c|}{|P_c| + |G_c|}$ — computed per case, then macro-averaged across cases (never a single pooled voxel count).
* **Hausdorff95 (HD95)**: the 95th percentile of the surface-distance distribution between predicted and ground-truth boundaries, computed in physical millimeters (via voxel spacing, not voxel-index distance). Chosen over raw Hausdorff distance because HD is dominated by a single worst-case outlier voxel, making it unstable for model comparison — HD95 discards the top 5% most extreme distances while still penalizing genuine boundary disagreement, which is why it is the field-standard reported in BraTS leaderboards.
* **Edge cases:** both-empty → Dice = 1.0 / HD95 = 0 (documented, flagged in logs); one-sided-empty for HD95 → explicitly reported as "undefined," never substituted with a numeric placeholder.
