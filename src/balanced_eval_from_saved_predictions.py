"""
scripts/balanced_eval_from_saved_predictions.py
--------------------------------------------------
Compute balanced-subsample IoU / Precision / Recall / F1 / OA / KelpAcc /
NoKAcc, PER SCENE, from predictions ALREADY SAVED to disk (e.g. written
earlier by run_ndvi_baseline.py / run_rf_baseline.py via
--save-geotiff-masks), without re-running the classifier or re-predicting
anything. Also (optionally) writes the balanced-subsample selection masks,
so the exact same sampled pixels can later be reused to evaluate another
method (e.g. SKeMa) on an identical population.

For each scene subfolder found under --predictions-dir, the matching
original scene folder is located under --data_root, and for every predicted
tile the ground-truth mask and (if land masking is on) the bathymetry band
are read to build a class-balanced per-tile subsample: all kelp pixels +
--balanced-eval-ratio times as many randomly sampled no-kelp pixels, same
rule used everywhere else in this codebase (load_balanced_training_pixels /
evaluate_test_tiles_balanced).

Usage
-----
    python scripts/balanced_eval_from_saved_predictions.py \\
        --predictions-dir "C:\\Users\\mohsenghanbari\\KelpSegmentation\\outputs\\rf_predictions\\RF with landmask" \\
        --data_root "C:\\Annotated Dataset\\L2A Data\\G0512Stride256RevisedSlopeAddedAsCh13_4FullModelOnly10Scenes\\*" \\
        --apply-land-mask \\
        --save-selection-mask
"""

from __future__ import annotations

import argparse

import numpy as np

from src.baseline_common import evaluate_saved_predictions_balanced, metrics_from_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc balanced-subsample evaluation from already-saved predictions."
    )
    parser.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing <scene_folder_name>/<tile_filename>.tif "
             "predictions, as written by --save-geotiff-masks.",
    )
    parser.add_argument(
        "--data_root", required=True,
        help='Glob pattern for the original scene folders, e.g. '
             '"C:/Annotated Dataset/L2A Data/.../*" (same pattern used '
             'when the predictions were generated).',
    )
    parser.add_argument(
        "--apply-land-mask", action="store_true",
        help="Exclude land pixels before subsampling. Uses the tile's own "
             "bathymetry band (bathymetry > 0 = land) unless "
             "--land-mask-dir is also given.",
    )
    parser.add_argument(
        "--land-mask-dir", default=None,
        help="Path to an external land mask (raster or vector) covering "
             "the study area. Implies --apply-land-mask.",
    )
    parser.add_argument(
        "--land-value", type=float, default=None,
        help="For a raster external land mask: the value that means "
             "'land'. Defaults to 'any nonzero value'.",
    )
    parser.add_argument(
        "--balanced-eval-ratio", type=float, default=1.0,
        help="No-kelp:kelp ratio in the balanced subsample (default 1.0 = 1:1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed controlling which no-kelp pixels are sampled.",
    )
    parser.add_argument(
        "--save-selection-mask", action="store_true",
        help="Write the balanced-subsample selection mask for every "
             "matched tile as a GeoTIFF under --selection-mask-dir. "
             "Reload with src.baseline_common.load_selection_mask to reuse "
             "the exact same pixels for another method's evaluation.",
    )
    parser.add_argument(
        "--selection-mask-dir", default="outputs/selection_masks",
        help="Where to write selection masks when --save-selection-mask is set.",
    )
    args = parser.parse_args()

    apply_land_mask = args.apply_land_mask or (args.land_mask_dir is not None)
    save_selection_mask_dir = args.selection_mask_dir if args.save_selection_mask else None

    results = evaluate_saved_predictions_balanced(
        predictions_dir=args.predictions_dir,
        data_root=args.data_root,
        apply_land_mask=apply_land_mask,
        land_mask_dir=args.land_mask_dir,
        land_value=args.land_value,
        imbalance_ratio=args.balanced_eval_ratio,
        seed=args.seed,
        save_selection_mask_dir=save_selection_mask_dir,
    )

    if not results:
        print("\nNo matched scenes/tiles found -- nothing to report. Check "
              "--predictions-dir and --data_root.")
        return

    print(f"\nBalanced test-set (subsample ratio {args.balanced_eval_ratio}:1 "
          f"no-kelp:kelp) metrics, per scene:")
    print(f"{'Scene':<45}{'IoU':>10}{'Precision':>11}{'Recall':>9}"
          f"{'F1':>9}{'OA':>9}{'KelpAcc':>10}{'NoKAcc':>9}")

    metrics_list = []
    for scene_name in sorted(results.keys()):
        tp, fp, fn, tn = results[scene_name]
        m = metrics_from_counts(tp, fp, fn, tn)
        metrics_list.append(m)
        print(f"{scene_name:<45}{m['iou']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['accuracy']:>9.4f}"
              f"{m['kelp_accuracy']:>10.4f}{m['nokelp_accuracy']:>9.4f}")

    mean_iou      = float(np.mean([m["iou"] for m in metrics_list]))
    mean_p        = float(np.mean([m["precision"] for m in metrics_list]))
    mean_r        = float(np.mean([m["recall"] for m in metrics_list]))
    mean_f1       = float(np.mean([m["f1"] for m in metrics_list]))
    mean_oa       = float(np.mean([m["accuracy"] for m in metrics_list]))
    mean_kelp_a   = float(np.mean([m["kelp_accuracy"] for m in metrics_list]))
    mean_nokelp_a = float(np.mean([m["nokelp_accuracy"] for m in metrics_list]))
    print("-" * 112)
    print(f"{'Mean':<45}{mean_iou:>10.4f}{mean_p:>11.4f}{mean_r:>9.4f}"
          f"{mean_f1:>9.4f}{mean_oa:>9.4f}{mean_kelp_a:>10.4f}{mean_nokelp_a:>9.4f}")

    if save_selection_mask_dir is not None:
        print(f"\nSelection masks written under: {save_selection_mask_dir}")


if __name__ == "__main__":
    main()
