#!/usr/bin/env python3
"""Build a SYNTHETIC checkpoint in the notebook's exact save format.

The weights are randomly initialized. Nothing produced by this checkpoint is a
prediction, and no metric computed from it means anything. It exists so that
checkpoint loading, architecture reconstruction, the inference pipeline, and
the API integration can be exercised end-to-end on a machine that does not
have the trained `best_model.pth`.

Every payload written here carries `synthetic_fixture: True`, which the loader
propagates into result metadata so a fixture can never be mistaken for the
trained model.

Usage:
    python scripts/make_synthetic_checkpoint.py out.pth
    python scripts/make_synthetic_checkpoint.py out.pth --base-channels 8 \
        --patch-size 32 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.inference.brats.config import (  # noqa: E402
    TRAINING_CONFIG,
    checkpoint_compat_fingerprint,
)
from app.inference.brats.model import build_model, count_parameters  # noqa: E402


def build_synthetic_payload(
    *,
    base_channels: int,
    patch_size: tuple[int, int, int],
    seed: int,
    epoch: int,
) -> tuple[dict, int]:
    torch.manual_seed(seed)
    config = replace(
        TRAINING_CONFIG,
        model=replace(TRAINING_CONFIG.model, base_channels=base_channels),
        patch=replace(TRAINING_CONFIG.patch, selected_size=tuple(patch_size)),
    )
    model = build_model(config.model)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "epoch": epoch,
        "global_step": 0,
        # Deliberately None: a fixture must not carry a number that could be
        # read as a validation score.
        "best_val_metric": None,
        "compat_fingerprint": checkpoint_compat_fingerprint(config),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_fixture": True,
        "note": (
            "SYNTHETIC FIXTURE — randomly initialized weights. Not trained. "
            "Any segmentation or metric derived from this file is meaningless."
        ),
    }
    return payload, count_parameters(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-channels", type=int, default=TRAINING_CONFIG.model.base_channels)
    parser.add_argument("--patch-size", type=int, nargs=3, default=list(TRAINING_CONFIG.patch.selected_size))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()

    payload, params = build_synthetic_payload(
        base_channels=args.base_channels,
        patch_size=tuple(args.patch_size),
        seed=args.seed,
        epoch=args.epoch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print("=" * 70)
    print("SYNTHETIC CHECKPOINT WRITTEN — RANDOM WEIGHTS, NOT TRAINED")
    print("=" * 70)
    print(f"Path              : {args.output}")
    print(f"base_channels     : {args.base_channels}")
    print(f"patch_size        : {tuple(args.patch_size)}")
    print(f"parameters        : {params:,}")
    print(f"compat_fingerprint: {payload['compat_fingerprint']}")
    print(f"size              : {args.output.stat().st_size / 1e6:.1f} MB")
    print("Do not report any metric derived from this file.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
