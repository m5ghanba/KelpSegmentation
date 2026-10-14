"""
tests/test_baseline_common.py
------------------------------
Unit tests for src/baseline_common.py — pixel extraction, balanced training
sampling, dataset-level metrics, masking-first land masking (applied before
training and prediction), and GeoTIFF export of LOO predictions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import rasterio

from tests.conftest import TILE_H, TILE_W, _write_tif

from src.baseline_common import (
    DEFAULT_BAND_NAMES,
    DEFAULT_VI_NAMES,
    channel_index,
    confusion_counts,
    evaluate_test_tiles,
    evaluate_test_tiles_balanced,
    find_best_threshold,
    load_all_test_pixels,
    load_balanced_training_pixels,
    load_bathymetry_land_mask,
    load_selection_mask,
    metrics_from_counts,
)


@pytest.fixture()
def sparse_kelp_mask_path(tmp_path) -> str:
    """
    Binary mask with kelp as a minority class (~5%), matching the real
    within-tile class distribution -- unlike the shared `tile_mask_path`
    fixture (a ~50/50 random mask), which is unrepresentative for the
    balance test below.
    """
    rng = np.random.default_rng(2)
    data = (rng.random((1, TILE_H, TILE_W)) < 0.05).astype(np.uint8)
    path = str(tmp_path / "sparse_kelp_mask.tif")
    _write_tif(path, data)
    return path


@pytest.fixture()
def bathy_scene_tile_paths(tmp_path):
    """
    A small (64x64) 13-band tile living under <scene>/images/ and
    <scene>/masks/ (matching the real folder layout, needed by
    evaluate_test_tiles's scene-name recovery). Left half = land
    (bathy > 0), right half = water (bathy < 0). Kelp only in the water
    half. Returns (image_path, mask_path, n_land_pixels, n_water_pixels).
    """
    h = w = 64
    rng = np.random.default_rng(3)
    data = rng.integers(0, 500, size=(13, h, w), dtype=np.uint16).astype(np.float32)

    bathy = np.empty((h, w), dtype=np.float32)
    bathy[:, : w // 2] = 10.0    # land
    bathy[:, w // 2:] = -10.0    # water
    data[11] = bathy             # array index 11 == rasterio band 12 == bathymetry

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, w // 2:] = (rng.random((h, w // 2)) < 0.1).astype(np.uint8)  # kelp only in water half

    scene_dir = tmp_path / "scene_00"
    (scene_dir / "images").mkdir(parents=True)
    (scene_dir / "masks").mkdir(parents=True)
    img_path = str(scene_dir / "images" / "tile_0.tif")
    mask_path = str(scene_dir / "masks" / "tile_0.tif")
    _write_tif(img_path, data)
    _write_tif(mask_path, mask[np.newaxis])

    n_land = h * (w // 2)
    n_water = h * (w // 2)
    return img_path, mask_path, n_land, n_water


class TestLandMasking:
    """Land masking is applied *before* both training and prediction."""

    def test_bathymetry_mask_identifies_land(self, bathy_scene_tile_paths):
        img_path, _, n_land, n_water = bathy_scene_tile_paths
        land_mask = load_bathymetry_land_mask(img_path)
        assert land_mask.dtype == bool
        assert int(land_mask.sum()) == n_land
        assert int((~land_mask).sum()) == n_water

    def test_bathymetry_mask_respects_threshold(self, bathy_scene_tile_paths):
        img_path, _, n_land, n_water = bathy_scene_tile_paths
        # A very negative threshold should classify everything as "land"
        # (bathy > threshold is true everywhere).
        land_mask = load_bathymetry_land_mask(img_path, land_threshold=-1000.0)
        assert land_mask.all()

    def test_training_pixels_exclude_land_when_enabled(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        x_bal, y_bal = load_balanced_training_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        # Balanced set should still be exactly 50/50, and total size must
        # be bounded by the water-half pixel count (land pixels excluded).
        assert int(np.sum(y_bal == 1)) == int(np.sum(y_bal == 0))
        assert x_bal.shape[0] <= 2 * n_water

    def test_training_pixels_include_land_when_disabled(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        x_bal, y_bal = load_balanced_training_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert int(np.sum(y_bal == 1)) == int(np.sum(y_bal == 0))
        assert x_bal.shape[0] > 0

    def test_all_test_pixels_excludes_land_when_enabled(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        x_all, y_all = load_all_test_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        x_masked, y_masked = load_all_test_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        assert x_all.shape[0] == n_land + n_water
        assert x_masked.shape[0] == n_water
        assert int(y_masked.sum()) == int(y_all.sum())  # kelp only lives in water


class TestEvaluateTestTiles:
    def test_metrics_match_masked_manual_computation(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths

        def predict_fn(x_kept):
            # Predicts kelp everywhere it's asked to predict on.
            return np.ones(x_kept.shape[0], dtype=np.int64)

        tp, fp, fn, tn = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        # Land excluded entirely -> only water pixels contribute; predicting
        # "kelp everywhere" over water means fp = every no-kelp water pixel,
        # and tn = 0 (nothing was ever predicted no-kelp).
        x_masked, y_masked = load_all_test_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        expected_tp, expected_fp, expected_fn, expected_tn = confusion_counts(
            y_masked, np.ones_like(y_masked)
        )
        assert (tp, fp, fn, tn) == (expected_tp, expected_fp, expected_fn, expected_tn)

    def test_no_masking_covers_full_tile(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        tp, fp, fn, tn = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        # Predicting no-kelp everywhere over the full (unmasked) tile: every
        # true kelp pixel becomes a false negative, every true no-kelp pixel
        # (land + water) becomes a true negative.
        x_all, y_all = load_all_test_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert fn == int(y_all.sum())
        assert tp == 0
        assert tn == int((y_all == 0).sum())

    def test_geotiff_export_creates_expected_file(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_dir = str(tmp_path / "predictions")

        def predict_fn(x_kept):
            return np.ones(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_dir=save_dir,
        )

        expected_path = os.path.join(save_dir, "scene_00", "tile_0.tif")
        assert os.path.isfile(expected_path)

    def test_geotiff_export_matches_georeferencing(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_dir = str(tmp_path / "predictions")

        def predict_fn(x_kept):
            return np.ones(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_dir=save_dir,
        )
        out_path = os.path.join(save_dir, "scene_00", "tile_0.tif")
        with rasterio.open(mask_path) as ref, rasterio.open(out_path) as out:
            assert out.transform == ref.transform
            assert out.crs == ref.crs
            assert out.height == ref.height
            assert out.width == ref.width
            assert out.count == 1

    def test_geotiff_export_land_pixels_are_zero(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_dir = str(tmp_path / "predictions")

        def predict_fn(x_kept):
            return np.ones(x_kept.shape[0], dtype=np.int64)  # kelp everywhere it predicts

        evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_dir=save_dir,
        )
        out_path = os.path.join(save_dir, "scene_00", "tile_0.tif")
        with rasterio.open(out_path) as out:
            pred = out.read(1)
        land_mask = load_bathymetry_land_mask(img_path)
        assert np.all(pred[land_mask] == 0)      # land excluded from prediction -> written as 0
        assert np.all(pred[~land_mask] == 1)     # water side: predict_fn said "kelp everywhere"

    def test_no_save_dir_writes_nothing(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths

        files_before = set(str(p) for p in tmp_path.rglob("*") if p.is_file())

        def predict_fn(x_kept):
            return np.ones(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_dir=None,
        )

        files_after = set(str(p) for p in tmp_path.rglob("*") if p.is_file())
        assert files_after == files_before  # no new files written when save_dir is None


class TestEvaluateTestTilesBalanced:
    """
    evaluate_test_tiles_balanced: metrics computed on a class-balanced
    per-tile subsample of test pixels (matching the training-time sampling
    rule), plus an exportable selection mask recording exactly which pixels
    were sampled -- so the same pixels can be reused to evaluate another
    method (e.g. SKeMa) on an identical population.
    """

    def test_selection_is_balanced(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_selection_mask_dir = str(tmp_path / "selection_masks")

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, imbalance_ratio=1.0, seed=42,
            save_selection_mask_dir=save_selection_mask_dir,
        )

        sel_mask = load_selection_mask(os.path.join(save_selection_mask_dir, "scene_00", "tile_0.tif"))
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1).astype(bool)

        n_kelp_selected = int(np.sum(sel_mask & ground_truth))
        n_nokelp_selected = int(np.sum(sel_mask & ~ground_truth))
        n_kelp_total = int(np.sum(ground_truth))

        assert n_kelp_selected == n_kelp_total       # every kelp pixel is always kept
        assert n_nokelp_selected == n_kelp_total     # ratio=1.0 -> equal no-kelp count

    def test_selection_excludes_land(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_selection_mask_dir = str(tmp_path / "selection_masks")

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_selection_mask_dir=save_selection_mask_dir,
        )

        sel_mask = load_selection_mask(os.path.join(save_selection_mask_dir, "scene_00", "tile_0.tif"))
        land_mask = load_bathymetry_land_mask(img_path)
        assert not np.any(sel_mask & land_mask)  # nothing selected on land

    def test_imbalance_ratio_scales_nokelp_count(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_selection_mask_dir = str(tmp_path / "selection_masks")

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, imbalance_ratio=2.0, seed=42,
            save_selection_mask_dir=save_selection_mask_dir,
        )

        sel_mask = load_selection_mask(os.path.join(save_selection_mask_dir, "scene_00", "tile_0.tif"))
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1).astype(bool)

        n_kelp_total = int(np.sum(ground_truth))
        n_nokelp_selected = int(np.sum(sel_mask & ~ground_truth))
        assert n_nokelp_selected == 2 * n_kelp_total

    def test_same_seed_reproducible(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        for d in (dir_a, dir_b):
            evaluate_test_tiles_balanced(
                [img_path], [mask_path], predict_fn,
                band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
                apply_land_mask=True, seed=123, save_selection_mask_dir=d,
            )

        mask_a = load_selection_mask(os.path.join(dir_a, "scene_00", "tile_0.tif"))
        mask_b = load_selection_mask(os.path.join(dir_b, "scene_00", "tile_0.tif"))
        assert np.array_equal(mask_a, mask_b)  # same seed -> identical selection

    def test_metrics_reflect_only_selected_pixels(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_selection_mask_dir = str(tmp_path / "selection_masks")

        def predict_fn(x_kept):
            # Always predicts kelp.
            return np.ones(x_kept.shape[0], dtype=np.int64)

        tp, fp, fn, tn = evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, imbalance_ratio=1.0, seed=42,
            save_selection_mask_dir=save_selection_mask_dir,
        )

        sel_mask = load_selection_mask(os.path.join(save_selection_mask_dir, "scene_00", "tile_0.tif"))
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1).astype(bool)

        expected_tp = int(np.sum(sel_mask & ground_truth))
        expected_fp = int(np.sum(sel_mask & ~ground_truth))
        assert tp == expected_tp
        assert fp == expected_fp
        assert fn == 0   # predicted kelp everywhere selected -> no false negatives
        assert tn == 0   # predicted kelp everywhere selected -> no true negatives

    def test_predicted_map_export_matches_selection(self, bathy_scene_tile_paths, tmp_path):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        save_dir = str(tmp_path / "predictions")
        save_selection_mask_dir = str(tmp_path / "selection_masks")

        def predict_fn(x_kept):
            return np.ones(x_kept.shape[0], dtype=np.int64)

        evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, save_dir=save_dir,
            save_selection_mask_dir=save_selection_mask_dir,
        )

        sel_mask = load_selection_mask(os.path.join(save_selection_mask_dir, "scene_00", "tile_0.tif"))
        with rasterio.open(os.path.join(save_dir, "scene_00", "tile_0.tif")) as out:
            pred = out.read(1).astype(bool)

        assert np.array_equal(pred, sel_mask)  # predicted kelp everywhere it was asked to predict

    def test_balanced_eval_is_independent_of_full_eval(self, bathy_scene_tile_paths):
        """
        Sanity check: the balanced-subsample metrics should differ from the
        full-tile metrics for the same predict_fn, since they're computed
        over different (much smaller, class-balanced) pixel populations.
        """
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths

        def predict_fn(x_kept):
            return np.zeros(x_kept.shape[0], dtype=np.int64)

        tp_f, fp_f, fn_f, tn_f = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        tp_b, fp_b, fn_b, tn_b = evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, seed=42,
        )
        # Balanced tn should be much smaller than full tn (most no-kelp
        # water pixels are excluded from the balanced subsample).
        assert tn_b < tn_f


class TestFindBestThreshold:
    def test_perfectly_separable_scores(self):
        """Scores cleanly separated by class -> threshold should sit between them."""
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        y_true = np.array([1,   1,   1,   0,   0,   0])
        thr = find_best_threshold(scores, y_true, np.linspace(0.0, 1.0, 101))
        pred = (scores > thr).astype(np.int64)
        assert np.array_equal(pred, y_true)  # perfect separation achievable

    def test_tuning_on_imbalanced_set_avoids_precision_collapse(self):
        """
        Regression test for the prior-mismatch bug: a threshold picked on a
        balanced sample, applied naively at the scikit-learn default of 0.5,
        can massively over-predict the positive class on an imbalanced test
        set. Tuning against the natural (imbalanced) distribution should not
        produce a threshold worse than the naive default in this scenario.
        """
        rng = np.random.default_rng(0)
        # Imbalanced "validation" set: 2% kelp, scores are noisy but kelp
        # skews higher than no-kelp.
        n = 5000
        y_val = (rng.random(n) < 0.02).astype(np.int64)
        scores = np.where(
            y_val == 1,
            rng.normal(0.6, 0.15, n),
            rng.normal(0.3, 0.15, n),
        )
        tuned_thr = find_best_threshold(scores, y_val, np.linspace(0.0, 1.0, 101))

        tp_default, fp_default, fn_default, tn_default = confusion_counts(
            y_val, (scores > 0.5).astype(np.int64)
        )
        tp_tuned, fp_tuned, fn_tuned, tn_tuned = confusion_counts(
            y_val, (scores > tuned_thr).astype(np.int64)
        )

        f1_default = metrics_from_counts(tp_default, fp_default, fn_default, tn_default)["f1"]
        f1_tuned = metrics_from_counts(tp_tuned, fp_tuned, fn_tuned, tn_tuned)["f1"]
        assert f1_tuned >= f1_default


class TestImbalanceRatio:
    def test_default_ratio_is_still_1to1(self, tile_image_path, sparse_kelp_mask_path):
        """imbalance_ratio defaults to 1.0 -> identical to prior 1:1 behavior."""
        x, y = load_balanced_training_pixels(
            [tile_image_path], [sparse_kelp_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert int(np.sum(y == 1)) == int(np.sum(y == 0))

    def test_ratio_5_gives_five_times_as_many_nokelp(self, tile_image_path, sparse_kelp_mask_path):
        x, y = load_balanced_training_pixels(
            [tile_image_path], [sparse_kelp_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            imbalance_ratio=5.0,
        )
        n_kelp = int(np.sum(y == 1))
        n_nokelp = int(np.sum(y == 0))
        # sparse_kelp_mask_path has ~5% kelp in a 512x512 tile, so there's
        # plenty of no-kelp pixels available to hit the exact 5x target.
        assert n_nokelp == 5 * n_kelp

    def test_ratio_capped_by_available_nokelp_pixels(self, tmp_path):
        """
        If the requested ratio would need more no-kelp pixels than the tile
        actually has, every available no-kelp pixel is kept (no error, no
        oversampling/duplication).
        """
        from tests.conftest import _write_tif

        h = w = 32
        rng = np.random.default_rng(9)
        img = rng.integers(0, 500, size=(13, h, w), dtype=np.uint16).astype(np.float32)
        # 90% kelp, 10% no-kelp -- far too few no-kelp pixels for a 10x ask.
        mask = (rng.random((h, w)) < 0.9).astype(np.uint8)

        img_path = str(tmp_path / "img.tif")
        mask_path = str(tmp_path / "mask.tif")
        _write_tif(img_path, img)
        _write_tif(mask_path, mask[np.newaxis])

        x, y = load_balanced_training_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            imbalance_ratio=10.0,
        )
        n_kelp = int(np.sum(y == 1))
        n_nokelp_available = int(np.sum(mask == 0))
        n_nokelp_kept = int(np.sum(y == 0))
        assert n_nokelp_kept == n_nokelp_available  # capped, not padded
        assert n_kelp == int(np.sum(mask == 1))      # all kelp pixels always kept

    def test_higher_ratio_still_respects_land_mask(self, bathy_scene_tile_paths):
        img_path, mask_path, n_land, n_water = bathy_scene_tile_paths
        x, y = load_balanced_training_pixels(
            [img_path], [mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, imbalance_ratio=5.0,
        )
        assert x.shape[0] <= n_water  # land pixels never enter the sample


class TestChannelIndex:
    def test_band_index(self):
        assert channel_index("blue") == DEFAULT_BAND_NAMES.index("blue")

    def test_vi_index(self):
        expected = len(DEFAULT_BAND_NAMES) + DEFAULT_VI_NAMES.index("ndvi")
        assert channel_index("ndvi") == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            channel_index("not_a_channel")


class TestLoadBalancedTrainingPixels:
    def test_output_shapes_match(self, tile_image_path, tile_mask_path):
        x, y = load_balanced_training_pixels(
            [tile_image_path], [tile_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == len(DEFAULT_BAND_NAMES) + len(DEFAULT_VI_NAMES)

    def test_labels_are_balanced(self, tile_image_path, sparse_kelp_mask_path):
        """
        Kelp and no-kelp counts must match exactly (per-tile balancing).
        Uses a sparse-kelp mask (kelp as the minority class), matching the
        real class distribution within kelp-containing tiles -- the shared
        `tile_mask_path` fixture is an unrepresentative ~50/50 random mask.
        """
        x, y = load_balanced_training_pixels(
            [tile_image_path], [sparse_kelp_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        n_kelp = int(np.sum(y == 1))
        n_nokelp = int(np.sum(y == 0))
        assert n_kelp == n_nokelp

    def test_multiple_tiles_concatenated(self, tile_image_path, tile_mask_path):
        x, y = load_balanced_training_pixels(
            [tile_image_path, tile_image_path], [tile_mask_path, tile_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        x1, y1 = load_balanced_training_pixels(
            [tile_image_path], [tile_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert x.shape[0] == 2 * x1.shape[0]


class TestLoadAllTestPixels:
    def test_covers_every_pixel(self, tile_image_path, tile_mask_path):
        """Unlike the balanced loader, this must return H*W pixels per tile."""
        x, y = load_all_test_pixels(
            [tile_image_path], [tile_mask_path],
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
        )
        assert x.shape[0] == 512 * 512
        assert y.shape[0] == 512 * 512


class TestConfusionCounts:
    def test_perfect_prediction(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        assert (tp, fp, fn, tn) == (2, 0, 0, 2)

    def test_known_counts(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        assert (tp, fp, fn, tn) == (2, 1, 1, 1)


class TestMetricsFromCounts:
    def test_known_iou(self):
        """tp=60, fp=20, fn=20 -> IoU = 60/100 = 0.60 (matches paper Eq. 3)."""
        m = metrics_from_counts(tp=60, fp=20, fn=20)
        assert abs(m["iou"] - 0.60) < 1e-9

    def test_zero_denominator_safe(self):
        m = metrics_from_counts(tp=0, fp=0, fn=0)
        assert m["iou"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_overall_accuracy(self):
        """tp=60, fp=20, fn=20, tn=900 -> OA = (60+900)/1000 = 0.96."""
        m = metrics_from_counts(tp=60, fp=20, fn=20, tn=900)
        assert abs(m["accuracy"] - 0.96) < 1e-9

    def test_kelp_accuracy_equals_recall(self):
        m = metrics_from_counts(tp=60, fp=20, fn=20, tn=900)
        assert abs(m["kelp_accuracy"] - m["recall"]) < 1e-12

    def test_nokelp_accuracy(self):
        """tn=900, fp=20 -> no-kelp accuracy = 900/920."""
        m = metrics_from_counts(tp=60, fp=20, fn=20, tn=900)
        assert abs(m["nokelp_accuracy"] - (900 / 920)) < 1e-9

    def test_accuracy_fields_present_with_default_tn(self):
        """tn defaults to 0 -- keys must still exist (backward compatible call site)."""
        m = metrics_from_counts(tp=60, fp=20, fn=20)
        for key in ("accuracy", "kelp_accuracy", "nokelp_accuracy"):
            assert key in m

    def test_perfect_classifier_all_metrics_one(self):
        m = metrics_from_counts(tp=50, fp=0, fn=0, tn=950)
        assert abs(m["accuracy"] - 1.0) < 1e-9
        assert abs(m["kelp_accuracy"] - 1.0) < 1e-9
        assert abs(m["nokelp_accuracy"] - 1.0) < 1e-9
