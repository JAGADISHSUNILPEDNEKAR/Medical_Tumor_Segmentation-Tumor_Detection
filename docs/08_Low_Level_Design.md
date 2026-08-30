# Document 8 — Low-Level Design (LLD)

## Module Architecture & Design Patterns

### Repository Pattern (Data Access Layer)
Abstracts metadata-DB access away from the API route handlers.
* Class: `CaseRepository`
  * `get_by_id(case_id: UUID) -> Case`
  * `create(modalities_present: list[str], has_ground_truth: bool) -> Case`
* Class: `JobRepository`
  * `create(case_id: UUID, job_type: str) -> Job`
  * `update_status(job_id: UUID, status: str) -> None`

### Pipeline Pattern (Preprocessing → Inference → Post-processing)
Each stage is an independently testable, independently executable unit, callable both from the FastAPI job runner and as a standalone CLI script.

```
                    ┌─────────────────────────────────┐
                    │      PipelineStage              │
                    │      <<Interface>>               │
                    │  +run(volume) -> volume          │
                    └────────────────┬────────────────┘
                                     │
   ┌─────────────────────┬───────────────────────┬─────────────────────┐
   │                      │                       │                     │
┌──┴──────────────┐  ┌────┴────────────┐  ┌───────┴───────────┐  ┌──────┴──────────┐
│ PreprocessStage  │  │ InferenceStage  │  │ PostProcessStage  │  │ VizExportStage  │
│ (resample+norm)  │  │ (sliding-window)│  │ (CC filtering)    │  │ (marching cubes)│
└──────────────────┘  └─────────────────┘  └────────────────────┘  └─────────────────┘
```

## Key Algorithms

### 1. Sliding-Window Inference with Gaussian-Weighted Blending
```python
import numpy as np
import torch

def sliding_window_infer(volume: np.ndarray, model, patch_size=(128, 128, 128), overlap=0.5) -> np.ndarray:
    """Tile the volume into overlapping patches, run the model on each,
    and blend overlapping predictions with a Gaussian weight (heavier at patch
    centers, lighter at patch edges) to avoid seam artifacts at patch borders."""
    gaussian_weight = _make_gaussian_kernel(patch_size)
    prob_map = np.zeros((model.num_classes, *volume.shape), dtype=np.float32)
    weight_map = np.zeros(volume.shape, dtype=np.float32)

    for patch, coords in _iter_patches(volume, patch_size, overlap):
        with torch.no_grad():
            logits = model(torch.from_numpy(patch).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        prob_map[:, *coords] += probs * gaussian_weight
        weight_map[coords] += gaussian_weight

    prob_map /= np.clip(weight_map, a_min=1e-6, a_max=None)
    return prob_map
```

### 2. Dice Coefficient (per class, with documented edge-case handling)
```python
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int, eps: float = 1e-6) -> float | None:
    pred_c = (pred_mask == class_id)
    gt_c = (gt_mask == class_id)

    if not gt_c.any() and not pred_c.any():
        return 1.0  # perfect agreement on absence — documented convention

    intersection = np.logical_and(pred_c, gt_c).sum()
    denom = pred_c.sum() + gt_c.sum()
    return float((2.0 * intersection + eps) / (denom + eps))
```

### 3. Connected-Component Post-Processing
```python
from scipy import ndimage

def filter_small_components(mask: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    cleaned = np.zeros_like(mask)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        labeled, num_features = ndimage.label(mask == class_id)
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() >= min_voxels:
                cleaned[component] = class_id
    return cleaned
```
