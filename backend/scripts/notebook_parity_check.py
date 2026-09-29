#!/usr/bin/env python3
"""Notebook vs production inference parity check.

Runs the SAME case through two independent implementations:

    A. scripts/notebook_reference.py  — verbatim transcription of the notebook
    B. app/inference/...              — the production RealBraTSInferenceService
                                        components

and compares every stage. Both sides load the same checkpoint, so any
difference is a pipeline difference, not a weights difference.

Comparisons and tolerances:

    preprocessed image        exact       (same arithmetic, same order)
    normalization mean/std    exact
    crop bbox                 exact
    probability map           atol 1e-6   (float32 accumulation order)
    predicted internal labels exact       (argmax of the above)
    post-CC-filter labels     exact
    exported BraTS labels     exact
    foreground voxel count    exact
    per-region voxel counts   exact
    output shape/affine/spacing exact

Usage:
    python scripts/notebook_parity_check.py --checkpoint ckpt.pth --case-dir DIR
    python scripts/notebook_parity_check.py --synthetic          # generated case
    python scripts/notebook_parity_check.py --synthetic --full   # 128^3, base 32
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import notebook_reference as ref  # noqa: E402
from app.inference.brats.config import TRAINING_CONFIG  # noqa: E402
from app.inference.brats.model import build_model  # noqa: E402
from app.inference.brats.postprocess import (  # noqa: E402
    connected_component_filter,
    internal_to_brats_labels,
    restore_prediction_to_original_space,
)
from app.inference.brats.preprocessing import load_and_preprocess_case  # noqa: E402
from app.inference.brats.sliding_window import (  # noqa: E402
    make_gaussian_weight_kernel,
    sliding_window_infer,
)

MODALITIES = ("t1", "t1ce", "t2", "flair")


# ── Synthetic case generation (clearly labeled) ──────────────────────────────
def write_synthetic_case(
    directory: Path, shape: tuple[int, int, int], seed: int = 7
) -> dict[str, Path]:
    """Write a SYNTHETIC 4-modality case: geometric blobs, not real anatomy.

    Structured to exercise the real code paths — a zero background rim so the
    foreground-bbox crop actually crops, and per-modality intensity scales so
    normalization has something to do.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-120.0, -120.0, -77.0]

    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float32)
    center = np.array([s / 2.0 for s in shape], dtype=np.float32)
    radius = np.array([s * 0.35 for s in shape], dtype=np.float32)
    inside = sum(
        ((grid[i] - center[i]) / radius[i]) ** 2 for i in range(3)
    ) <= 1.0

    paths: dict[str, Path] = {}
    for index, modality in enumerate(MODALITIES):
        base = 300.0 + 250.0 * index
        volume = np.zeros(shape, dtype=np.float32)
        noise = rng.normal(loc=base, scale=60.0 + 20.0 * index, size=shape)
        volume[inside] = np.abs(noise[inside]) + 1.0
        image = nib.Nifti1Image(volume, affine)
        image.header.set_zooms((1.0, 1.0, 1.0))
        path = directory / f"{modality}.nii.gz"
        nib.save(image, str(path))
        paths[modality] = path
    return paths


def resolve_case(case_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for modality in MODALITIES:
        for suffix in (".nii.gz", ".nii"):
            candidate = case_dir / f"{modality}{suffix}"
            if candidate.is_file():
                paths[modality] = candidate
                break
        else:
            raise SystemExit(f"Missing modality '{modality}' in {case_dir}")
    return paths


# ── Comparison helpers ───────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append((name, bool(passed), detail))

    def exact(self, name: str, a, b, detail: str = "") -> None:
        equal = np.array_equal(np.asarray(a), np.asarray(b))
        if not equal and not detail:
            diff = np.asarray(a) != np.asarray(b)
            detail = f"{int(np.count_nonzero(diff))} differing element(s)"
        self.check(name, equal, detail)

    def close(self, name: str, a, b, atol: float) -> None:
        a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a_arr.shape != b_arr.shape:
            self.check(name, False, f"shape {a_arr.shape} vs {b_arr.shape}")
            return
        max_abs = float(np.max(np.abs(a_arr - b_arr))) if a_arr.size else 0.0
        self.check(name, max_abs <= atol, f"max |Δ| = {max_abs:.3e} (atol {atol:.1e})")

    def render(self) -> bool:
        width = max(len(name) for name, _, _ in self.rows)
        print("=" * 78)
        print("NOTEBOOK → PRODUCTION PARITY")
        print("=" * 78)
        for name, passed, detail in self.rows:
            status = "PASS" if passed else "FAIL"
            suffix = f"   {detail}" if detail else ""
            print(f"[{status}] {name.ljust(width)}{suffix}")
        print("=" * 78)
        ok = all(passed for _, passed, _ in self.rows)
        print("RESULT:", "PARITY CONFIRMED" if ok else "PARITY BROKEN")
        print("=" * 78)
        return ok


def region_counts(mask: np.ndarray) -> dict[int, int]:
    return {label: int(np.count_nonzero(mask == label)) for label in (1, 2, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the real trained geometry (patch 128^3, base_channels 32).",
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--shape", type=int, nargs=3, default=None)
    parser.add_argument("--cc-min-voxels", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.full:
        patch_size = tuple(args.patch_size or TRAINING_CONFIG.patch.selected_size)
        base_channels = args.base_channels or TRAINING_CONFIG.model.base_channels
        shape = tuple(args.shape or (240, 240, 155))
    else:
        patch_size = tuple(args.patch_size or (32, 32, 32))
        base_channels = args.base_channels or 8
        shape = tuple(args.shape or (56, 60, 48))

    cc_min_voxels = (
        args.cc_min_voxels
        if args.cc_min_voxels is not None
        else TRAINING_CONFIG.postprocessing.connected_component_min_voxels
    )

    device = torch.device("cpu")
    prod_config = replace(
        TRAINING_CONFIG,
        model=replace(TRAINING_CONFIG.model, base_channels=base_channels),
        patch=replace(TRAINING_CONFIG.patch, selected_size=patch_size),
    )
    ref_config = ref.make_config(patch_size=patch_size, base_channels=base_channels)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if args.case_dir is not None:
            modality_paths = resolve_case(args.case_dir)
            case_label = f"real case at {args.case_dir}"
            synthetic_case = False
        else:
            modality_paths = write_synthetic_case(tmp_path / "case", shape)
            case_label = f"SYNTHETIC generated case, shape {shape}"
            synthetic_case = True

        # One set of weights, loaded into both implementations.
        if args.checkpoint is not None:
            payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
            state_dict = payload.get("model_state_dict", payload)
            checkpoint_label = str(args.checkpoint)
            synthetic_weights = bool(
                isinstance(payload, dict) and payload.get("synthetic_fixture")
            )
        else:
            torch.manual_seed(1234)
            state_dict = build_model(prod_config.model).state_dict()
            checkpoint_label = "SYNTHETIC in-memory random weights (no checkpoint given)"
            synthetic_weights = True

        prod_model = build_model(prod_config.model)
        prod_model.load_state_dict(state_dict, strict=True)
        prod_model.to(device).eval()

        ref_model = ref.UNet3D(ref_config.model)
        ref_model.load_state_dict(state_dict, strict=True)
        ref_model.to(device).eval()

        print("-" * 78)
        print(f"case        : {case_label}")
        print(f"weights     : {checkpoint_label}")
        print(f"patch size  : {patch_size}   base_channels: {base_channels}")
        print(f"device      : {device}   cc_min_voxels: {cc_min_voxels}")
        if synthetic_case or synthetic_weights:
            caveats = []
            if synthetic_case:
                caveats.append("the input is a generated phantom, not real anatomy")
            if synthetic_weights:
                caveats.append("the weights are untrained")
            print(
                "NOTE        : " + "; ".join(caveats) + ".\n"
                "              This run proves PIPELINE parity only — that both "
                "implementations compute the same thing.\n"
                "              It says nothing about segmentation quality, and no "
                "number from it is a metric."
            )
        print("-" * 78)

        report = Report()

        # ── A. notebook reference ────────────────────────────────────────────
        ref_row = {f"{m}_path": str(p) for m, p in modality_paths.items()}
        ref_row["seg_path"] = None
        t0 = time.perf_counter()
        ref_out = ref.infer_case_full(ref_row, ref_model, ref_config, device)
        ref_seconds = time.perf_counter() - t0
        ref_clean = ref.connected_component_filter(
            ref_out["pred_internal"], min_voxels=cc_min_voxels
        )
        ref_clean_raw = ref_clean.copy()
        ref_clean_raw[ref_clean == 3] = 4
        ref_full = ref.restore_prediction_to_original_space(
            ref_clean_raw, ref_out["original_shape"], ref_out["crop_bbox"]
        )

        # ── B. production implementation ─────────────────────────────────────
        t0 = time.perf_counter()
        prod_case = load_and_preprocess_case(modality_paths, prod_config.preprocessing)
        prod_prob, window_stats = sliding_window_infer(
            prod_case.image,
            prod_model,
            prod_config.model,
            prod_config.patch,
            device,
            weight_kernel_fn=lambda size: make_gaussian_weight_kernel(
                size, prod_config.patch.gaussian_sigma_scale
            ),
        )
        prod_internal = np.argmax(prod_prob, axis=0).astype(np.uint8)
        prod_seconds = time.perf_counter() - t0
        prod_clean, removed = connected_component_filter(
            prod_internal, min_voxels=cc_min_voxels
        )
        prod_raw = internal_to_brats_labels(prod_clean)
        prod_full = restore_prediction_to_original_space(
            prod_raw, prod_case.original_shape, prod_case.crop_bbox
        )

        # ── Comparisons ──────────────────────────────────────────────────────
        report.exact(
            "input shape",
            ref_out["original_shape"],
            prod_case.original_shape,
            f"{tuple(prod_case.original_shape)}",
        )
        report.exact(
            "crop bbox",
            [[s.start, s.stop] for s in ref_out["crop_bbox"]],
            [[s.start, s.stop] for s in prod_case.crop_bbox],
            str([[s.start, s.stop] for s in prod_case.crop_bbox]),
        )
        report.exact(
            "preprocessed shape",
            ref_out["image"].shape,
            prod_case.image.shape,
            str(tuple(prod_case.image.shape)),
        )
        report.close(
            "preprocessed voxels", ref_out["image"], prod_case.image, atol=0.0
        )
        ref_stats = [
            (ref_out["norm_stats"][m]["mean"], ref_out["norm_stats"][m]["std"])
            for m in MODALITIES
        ]
        prod_stats = [
            (prod_case.norm_stats[m]["mean"], prod_case.norm_stats[m]["std"])
            for m in MODALITIES
        ]
        report.close("normalization mean/std", ref_stats, prod_stats, atol=0.0)
        report.close("probability map", ref_out["prob_map"], prod_prob, atol=1e-6)
        report.exact(
            "model output shape",
            ref_out["prob_map"].shape,
            prod_prob.shape,
            str(tuple(prod_prob.shape)),
        )
        report.exact("predicted internal labels", ref_out["pred_internal"], prod_internal)
        report.exact("post-CC-filter internal labels", ref_clean, prod_clean)
        report.exact("exported BraTS mask", ref_full, prod_full)
        report.check(
            "exported labels ⊆ {0,1,2,4}",
            set(np.unique(prod_full).tolist()).issubset({0, 1, 2, 4}),
            f"observed {sorted(int(v) for v in np.unique(prod_full))}",
        )
        ref_fg = int(np.count_nonzero(ref_full))
        prod_fg = int(np.count_nonzero(prod_full))
        report.check(
            "foreground voxel count", ref_fg == prod_fg, f"{prod_fg} voxels"
        )
        report.exact(
            "per-region voxel counts",
            list(region_counts(ref_full).values()),
            list(region_counts(prod_full).values()),
            str(region_counts(prod_full)),
        )
        reference_img = nib.load(str(modality_paths["t1"]))
        report.close("output affine", reference_img.affine, prod_case.affine, atol=0.0)
        report.close(
            "output spacing",
            tuple(float(z) for z in reference_img.header.get_zooms()[:3]),
            prod_case.spacing,
            atol=0.0,
        )
        report.exact(
            "output shape == input shape",
            prod_full.shape,
            ref_out["original_shape"],
            str(tuple(prod_full.shape)),
        )

        ok = report.render()
        print(f"notebook reference wall time : {ref_seconds:.2f}s")
        print(f"production wall time         : {prod_seconds:.2f}s")
        print(f"patches evaluated            : {window_stats.num_patches}")
        print(f"sliding-window stride        : {window_stats.stride}")
        print(f"CC-filter voxels removed     : {removed}")

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "parity_passed": ok,
                        "case": case_label,
                        "weights": checkpoint_label,
                        "synthetic_case": synthetic_case,
                        "synthetic_weights": synthetic_weights,
                        "patch_size": list(patch_size),
                        "base_channels": base_channels,
                        "device": str(device),
                        "checks": [
                            {"name": n, "passed": p, "detail": d}
                            for n, p, d in report.rows
                        ],
                        "notebook_seconds": round(ref_seconds, 3),
                        "production_seconds": round(prod_seconds, 3),
                        "num_patches": window_stats.num_patches,
                        "stride": list(window_stats.stride),
                        "foreground_voxels": prod_fg,
                        "region_voxels": {
                            str(k): v for k, v in region_counts(prod_full).items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"JSON report written to {args.json_out}")

        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
