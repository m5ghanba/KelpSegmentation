"""
scripts/run_rf_baseline.py
---------------------------
Random Forest baseline for the 10-scene leave-one-out (LOO) experiment.

Same overall protocol as run_ndvi_baseline.py (see that file's docstring
for the shared pieces: land masking, balanced evaluation, GeoTIFF export).
This file additionally supports:
  - --imbalance-ratio / --class-weight (training-sample class ratio)
  - --tune-hyperparams (small RF hyperparameter grid search)

Segmentation-based post-processing (optional): SLIC superpixel segmentation
of each 512x512 tile, followed by a majority vote within each segment over
the raw per-pixel predictions -- every pixel in a segment is relabeled to
whichever class (kelp / no-kelp) was in the majority among that segment's
raw predictions. SLIC (Achanta et al., 2012) is a standard, widely-used
superpixel algorithm for exactly this kind of object-based post-processing
of pixel-wise land-cover classifications in remote sensing, including
specifically as a refinement step for Random Forest classifications. This
step is applied identically to the full-tile evaluation and (if
--balanced-eval is also set) to the balanced-subsample evaluation.
    --apply-segmentation
    --segmentation-n-segments 300
    --segmentation-compactness 10.0

Usage
-----
    python scripts/run_rf_baseline.py --data_root "/path/to/Annotated/*" \\
        --apply-land-mask --apply-segmentation
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baseline_common import (
    DEFAULT_BAND_NAMES,
    DEFAULT_VI_NAMES,
    confusion_counts,
    evaluate_test_tiles,
    evaluate_test_tiles_balanced,
    find_best_threshold,
    load_all_test_pixels,
    load_balanced_training_pixels,
    metrics_from_counts,
)
from src.dataset import split_loo

N_SCENES = 10
PROB_THRESHOLDS = np.linspace(0.01, 0.99, 198)

HYPERPARAM_GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20},
    {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 20,   "min_samples_leaf": 5},
]


def _fit_and_tune(x_bal, y_bal, x_val, y_val, cfg, seed, n_jobs):
    clf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **cfg)
    clf.fit(x_bal, y_bal)
    val_probs = clf.predict_proba(x_val)[:, 1]
    thr = find_best_threshold(val_probs, y_val, PROB_THRESHOLDS)
    tp, fp, fn, tn = confusion_counts(y_val, (val_probs > thr).astype(np.int64))
    val_f1 = metrics_from_counts(tp, fp, fn, tn)["f1"]
    return clf, thr, val_f1


def run_fold(
    data_root: str,
    test_idx: int,
    seed: int = 42,
    val_frac: float = 0.20,
    n_estimators: int = 200,
    max_depth=None,
    class_weight=None,
    imbalance_ratio: float = 1.0,
    n_jobs: int = -1,
    tune_hyperparams: bool = False,
    verbose_tuning: bool = True,
    apply_land_mask: bool = False,
    land_mask_dir: str = None,
    land_value: float = None,
    save_dir: str = None,
    balanced_eval: bool = False,
    balanced_eval_ratio: float = 1.0,
    balanced_eval_seed: int = None,
    balanced_save_dir: str = None,
    selection_mask_dir: str = None,
    apply_segmentation: bool = False,
    segmentation_n_segments: int = 300,
    segmentation_compactness: float = 10.0,
):
    x_train_paths, y_train_paths, x_val_paths, y_val_paths, x_test_paths, y_test_paths = split_loo(
        data_root, test_indices=[test_idx], val_frac=val_frac, seed=seed,
    )

    x_bal, y_bal = load_balanced_training_pixels(
        x_train_paths, y_train_paths,
        band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES, seed=seed,
        apply_land_mask=apply_land_mask, land_mask_dir=land_mask_dir, land_value=land_value,
        imbalance_ratio=imbalance_ratio,
    )

    has_val = val_frac > 0 and len(x_val_paths) > 0
    if has_val:
        x_val, y_val = load_all_test_pixels(
            x_val_paths, y_val_paths,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=apply_land_mask, land_mask_dir=land_mask_dir, land_value=land_value,
        )

    if tune_hyperparams:
        if not has_val:
            raise ValueError("--tune-hyperparams requires --val_frac > 0.")
        best_clf, best_thr, best_val_f1, best_cfg = None, 0.5, -1.0, None
        if verbose_tuning:
            print(f"    [scene {test_idx}] hyperparameter search "
                  f"({len(HYPERPARAM_GRID)} configs, imbalance_ratio={imbalance_ratio}, "
                  f"validation F1 for each):")
        for cfg in HYPERPARAM_GRID:
            clf, thr, val_f1 = _fit_and_tune(x_bal, y_bal, x_val, y_val, cfg, seed, n_jobs)
            if verbose_tuning:
                print(f"      val_f1={val_f1:.4f}  thr={thr:.2f}  {cfg}")
            if val_f1 > best_val_f1:
                best_clf, best_thr, best_val_f1, best_cfg = clf, thr, val_f1, cfg
        if verbose_tuning:
            print(f"    -> selected: val_f1={best_val_f1:.4f}  thr={best_thr:.2f}  {best_cfg}")
        clf, thr = best_clf, best_thr
        chosen_cfg = best_cfg
    else:
        cfg = {"n_estimators": n_estimators, "max_depth": max_depth,
                "min_samples_leaf": 1, "class_weight": class_weight}
        if has_val:
            clf, thr, _ = _fit_and_tune(x_bal, y_bal, x_val, y_val, cfg, seed, n_jobs)
        else:
            clf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **cfg)
            clf.fit(x_bal, y_bal)
            thr = 0.5
        chosen_cfg = cfg

    def predict_fn(x_kept: np.ndarray) -> np.ndarray:
        return (clf.predict_proba(x_kept)[:, 1] > thr).astype(np.int64)

    tp, fp, fn, tn = evaluate_test_tiles(
        x_test_paths, y_test_paths, predict_fn,
        band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        apply_land_mask=apply_land_mask, land_mask_dir=land_mask_dir, land_value=land_value,
        save_dir=save_dir,
        apply_segmentation=apply_segmentation,
        segmentation_n_segments=segmentation_n_segments,
        segmentation_compactness=segmentation_compactness,
    )

    balanced_counts = None
    if balanced_eval:
        b_seed = balanced_eval_seed if balanced_eval_seed is not None else seed
        balanced_counts = evaluate_test_tiles_balanced(
            x_test_paths, y_test_paths, predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=apply_land_mask, land_mask_dir=land_mask_dir, land_value=land_value,
            imbalance_ratio=balanced_eval_ratio, seed=b_seed,
            save_dir=balanced_save_dir, save_selection_mask_dir=selection_mask_dir,
            apply_segmentation=apply_segmentation,
            segmentation_n_segments=segmentation_n_segments,
            segmentation_compactness=segmentation_compactness,
        )

    return thr, chosen_cfg, tp, fp, fn, tn, balanced_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest LOO baseline.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--imbalance-ratio", type=float, default=1.0)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced", "balanced_subsample"])
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--tune-hyperparams", action="store_true")
    parser.add_argument("--quiet-tuning", action="store_true")
    parser.add_argument("--apply-land-mask", action="store_true")
    parser.add_argument("--land-mask-dir", default=None)
    parser.add_argument("--land-value", type=float, default=None)
    parser.add_argument("--save-geotiff-masks", action="store_true")
    parser.add_argument("--output-dir", default="outputs/rf_predictions")
    parser.add_argument("--balanced-eval", action="store_true")
    parser.add_argument("--balanced-eval-ratio", type=float, default=1.0)
    parser.add_argument("--balanced-eval-seed", type=int, default=None)
    parser.add_argument("--save-selection-mask", action="store_true")
    parser.add_argument("--selection-mask-dir", default="outputs/rf_selection_masks")
    parser.add_argument(
        "--apply-segmentation", action="store_true",
        help="Refine raw predictions via SLIC superpixel segmentation + "
             "majority voting within each segment before computing "
             "metrics/exports. See module docstring.",
    )
    parser.add_argument(
        "--segmentation-n-segments", type=int, default=300,
        help="Approximate number of SLIC superpixels per 512x512 tile.",
    )
    parser.add_argument(
        "--segmentation-compactness", type=float, default=10.0,
        help="SLIC compactness (higher = more spatially regular segments).",
    )
    args = parser.parse_args()
    apply_land_mask = args.apply_land_mask or (args.land_mask_dir is not None)
    save_dir = args.output_dir if args.save_geotiff_masks else None
    class_weight = None if args.class_weight == "none" else args.class_weight
    balanced_eval = args.balanced_eval or args.save_selection_mask
    selection_mask_dir = args.selection_mask_dir if args.save_selection_mask else None

    print(f"{'Scene':<6}{'Thresh':>8}{'IoU':>10}{'Precision':>11}{'Recall':>9}"
          f"{'F1':>9}{'OA':>9}{'KelpAcc':>10}{'NoKAcc':>9}")
    metrics_list = []
    balanced_metrics_list = []
    for scene_idx in range(N_SCENES):
        thr, cfg, tp, fp, fn, tn, balanced_counts = run_fold(
            args.data_root, scene_idx, seed=args.seed, val_frac=args.val_frac,
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            class_weight=class_weight, imbalance_ratio=args.imbalance_ratio,
            tune_hyperparams=args.tune_hyperparams,
            verbose_tuning=not args.quiet_tuning,
            apply_land_mask=apply_land_mask, land_mask_dir=args.land_mask_dir,
            land_value=args.land_value, save_dir=save_dir,
            balanced_eval=balanced_eval, balanced_eval_ratio=args.balanced_eval_ratio,
            balanced_eval_seed=args.balanced_eval_seed,
            selection_mask_dir=selection_mask_dir,
            apply_segmentation=args.apply_segmentation,
            segmentation_n_segments=args.segmentation_n_segments,
            segmentation_compactness=args.segmentation_compactness,
        )
        m = metrics_from_counts(tp, fp, fn, tn)
        metrics_list.append(m)
        cfg_note = f"  [{cfg}]" if args.tune_hyperparams else ""
        print(f"{scene_idx:<6}{thr:>8.2f}{m['iou']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['accuracy']:>9.4f}"
              f"{m['kelp_accuracy']:>10.4f}{m['nokelp_accuracy']:>9.4f}{cfg_note}")

        if balanced_counts is not None:
            bm = metrics_from_counts(*balanced_counts)
            balanced_metrics_list.append(bm)

    mean_iou      = float(np.mean([m["iou"] for m in metrics_list]))
    mean_p        = float(np.mean([m["precision"] for m in metrics_list]))
    mean_r        = float(np.mean([m["recall"] for m in metrics_list]))
    mean_f1       = float(np.mean([m["f1"] for m in metrics_list]))
    mean_oa       = float(np.mean([m["accuracy"] for m in metrics_list]))
    mean_kelp_a   = float(np.mean([m["kelp_accuracy"] for m in metrics_list]))
    mean_nokelp_a = float(np.mean([m["nokelp_accuracy"] for m in metrics_list]))
    print("-" * 80)
    print(f"{'Mean':<6}{'':>8}{mean_iou:>10.4f}{mean_p:>11.4f}{mean_r:>9.4f}"
          f"{mean_f1:>9.4f}{mean_oa:>9.4f}{mean_kelp_a:>10.4f}{mean_nokelp_a:>9.4f}")

    if save_dir is not None:
        print(f"\nPredicted GeoTIFFs written under: {save_dir}")

    if balanced_metrics_list:
        print(f"\nBalanced test-set (subsample ratio "
              f"{args.balanced_eval_ratio}:1 no-kelp:kelp) metrics:")
        print(f"{'Scene':<6}{'IoU':>10}{'Precision':>11}{'Recall':>9}"
              f"{'F1':>9}{'OA':>9}{'KelpAcc':>10}{'NoKAcc':>9}")
        for scene_idx, bm in enumerate(balanced_metrics_list):
            print(f"{scene_idx:<6}{bm['iou']:>10.4f}{bm['precision']:>11.4f}"
                  f"{bm['recall']:>9.4f}{bm['f1']:>9.4f}{bm['accuracy']:>9.4f}"
                  f"{bm['kelp_accuracy']:>10.4f}{bm['nokelp_accuracy']:>9.4f}")
        b_mean_iou      = float(np.mean([m["iou"] for m in balanced_metrics_list]))
        b_mean_p        = float(np.mean([m["precision"] for m in balanced_metrics_list]))
        b_mean_r        = float(np.mean([m["recall"] for m in balanced_metrics_list]))
        b_mean_f1       = float(np.mean([m["f1"] for m in balanced_metrics_list]))
        b_mean_oa       = float(np.mean([m["accuracy"] for m in balanced_metrics_list]))
        b_mean_kelp_a   = float(np.mean([m["kelp_accuracy"] for m in balanced_metrics_list]))
        b_mean_nokelp_a = float(np.mean([m["nokelp_accuracy"] for m in balanced_metrics_list]))
        print("-" * 72)
        print(f"{'Mean':<6}{b_mean_iou:>10.4f}{b_mean_p:>11.4f}{b_mean_r:>9.4f}"
              f"{b_mean_f1:>9.4f}{b_mean_oa:>9.4f}{b_mean_kelp_a:>10.4f}{b_mean_nokelp_a:>9.4f}")

        if selection_mask_dir is not None:
            print(f"\nBalanced-subsample selection masks written under: {selection_mask_dir}")


if __name__ == "__main__":
    main()
