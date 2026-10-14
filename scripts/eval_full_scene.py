"""
scripts/eval_full_scene.py
---------------------------
Full-scene inference + evaluation, for reviewer 1's request: "I strongly
recommend reporting metrics on full-scene inference to properly assess FP
rate." Table 2 in the manuscript only scores kelp-CONTAINING test tiles;
this script scores every pixel of a whole scene against ground truth, so
it directly measures the false-positive behaviour on water/land/eelgrass/
sunglint/etc. that the tile-only evaluation cannot see.

This only works on a scene whose kelp is FULLY labeled end-to-end (no
`_TileBound.tif`, i.e. no partially-annotated region was excluded) — per
your confirmation, that's the scene at LOO folder index 1 ("uxr", the
second data folder, reported as "Scene 2" in Table 2).

CRITICAL: the checkpoint you pass in must come from the LOO fold that held
THIS scene out as the test scene (test_indices: [1]) — a checkpoint trained
with this scene in train/val has seen its labels and the FP rate would be
meaningless. If you don't already have that checkpoint on disk, train it
first:

    python src/train.py --config configs/loo_fold1_holdout.yaml

(configs/loo_fold1_holdout.yaml should set data.loo.test_indices: [1] and
otherwise match whatever config produced Table 2's "satellite-only" row —
band_names = [blue, green, red, nir, rededge], vi_names = all 5,
substrate_onehot: false, no bathymetry/substrate/slope.)

Usage
-----
    python scripts/eval_full_scene.py \\
        --scene-dir   /path/to/data_root/<folder_at_index_1> \\
        --checkpoint  checkpoints/loo_fold1/best_xxx.ckpt \\
        --mean '[<12 or 10 comma-separated floats from that run's MLflow>]' \\
        --std  '[<...>]' \\
        --output-dir  outputs/full_scene_eval

The scene folder must contain the ground-truth `*_Mask.tif` used to build
the LOO training tiles for this scene (same rasterized kelp=1/else=0 mask,
full scene extent — this is what makes full-scene scoring possible here and
NOT for the other 9 scenes, which have unlabeled kelp outside their
`_TileBound`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import load_scene, run_inference, save_prediction, _find_file
from src.dataset import resolve_num_channels
from src.model import KelpSegmentationModel


DEFAULT_BAND_NAMES = ["blue", "green", "red", "nir", "rededge"]
DEFAULT_VI_NAMES   = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)
    # The number R1 actually wants: how often does the model cry "kelp" on
    # water/land/etc. that truly has none, as a fraction of all true-negative
    # opportunity — i.e. the classical false-positive rate, not to be
    # confused with (1 - precision).
    fpr = fp / (fp + tn + eps)
    return {
        "precision": precision, "recall": recall, "iou": iou, "f1": f1,
        "accuracy": accuracy, "false_positive_rate": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", required=True,
                         help="Folder for the fully-labeled scene (LOO index 1 / 'uxr').")
    parser.add_argument("--checkpoint", required=True,
                         help="Best checkpoint from the LOO fold with test_indices: [1].")
    parser.add_argument("--mask-pattern", default="*_Mask.tif",
                         help="Glob (within --scene-dir) for the full-scene ground-truth mask.")
    parser.add_argument("--band-names", nargs="+", default=DEFAULT_BAND_NAMES)
    parser.add_argument("--vi-names",   nargs="+", default=DEFAULT_VI_NAMES)
    parser.add_argument("--substrate-onehot", action="store_true")
    parser.add_argument("--mean", type=json.loads, required=True,
                         help='JSON list, e.g. \'[123.4, 98.7, ...]\' — must be the '
                              'mean/std logged by the matching LOO-fold training run, '
                              'NOT the mean/std from a different fold or from Table 2\'s '
                              '10-fold-averaged model (there isn\'t one — each LOO fold '
                              'has its own mean/std computed from its own 9 training scenes).')
    parser.add_argument("--std", type=json.loads, required=True)
    parser.add_argument("--architecture", default="Unet")
    parser.add_argument("--encoder", default="tu-maxvit_tiny_tf_512")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.70)
    parser.add_argument("--halo-size", type=int, default=64)
    parser.add_argument("--method", default="weighted", choices=["weighted", "halo", "max"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/full_scene_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = resolve_num_channels(args.band_names, args.vi_names,
                                        substrate_onehot=args.substrate_onehot)
    mean = np.array(args.mean, dtype=np.float32)
    std  = np.array(args.std,  dtype=np.float32)
    if len(mean) != in_channels:
        raise ValueError(
            f"--mean has {len(mean)} values but in_channels resolves to "
            f"{in_channels} for band_names={args.band_names} + "
            f"vi_names={args.vi_names} (substrate_onehot={args.substrate_onehot}). "
            "Pass the mean/std actually logged for this checkpoint's training run."
        )

    print(f"Loading scene: {args.scene_dir}")
    image_hwc, metadata = load_scene(
        args.scene_dir, args.band_names, args.vi_names,
        substrate_onehot=args.substrate_onehot,
    )
    print(f"  scene shape (H, W, C): {image_hwc.shape}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = KelpSegmentationModel.load_from_checkpoint(
        args.checkpoint,
        arch=args.architecture,
        encoder_name=args.encoder,
        in_channels=in_channels,
        out_classes=1,
        strict=False,
    ).to(device).eval()

    print("Running tiled inference over the full scene …")
    pred = run_inference(
        image_hwc, model, device, mean, std,
        tile_size=args.tile_size, overlap=args.overlap, halo_size=args.halo_size,
        method=args.method, threshold=args.threshold,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    scene_name = os.path.basename(os.path.normpath(args.scene_dir))
    pred_path = os.path.join(args.output_dir, f"{scene_name}_full_scene_pred.tif")
    save_prediction(pred, metadata, pred_path)
    print(f"  Prediction saved -> {pred_path}")

    mask_path = _find_file(args.scene_dir, args.mask_pattern)
    print(f"Scoring against ground truth: {mask_path}")
    with rasterio.open(mask_path) as src:
        y_true = (src.read(1) == 1)

    if y_true.shape != pred.shape:
        raise ValueError(
            f"Ground-truth mask shape {y_true.shape} != prediction shape "
            f"{pred.shape}. Check --mask-pattern and that the scene grids align."
        )

    tp, fp, fn, tn = confusion_counts(y_true, pred.astype(bool))
    m = metrics_from_counts(tp, fp, fn, tn)

    print("\n── Full-scene metrics ──────────────────────────────")
    for k in ("iou", "precision", "recall", "f1", "accuracy", "false_positive_rate"):
        print(f"  {k:<22}: {m[k]:.4f}")
    print(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}")
    print("──────────────────────────────────────────────────────")

    metrics_path = os.path.join(args.output_dir, f"{scene_name}_full_scene_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
