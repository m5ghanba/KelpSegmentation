"""
tests/test_segmentation.py
----------------------------
Unit tests for src/baseline_common.py's segmentation-based post-processing:
segment_majority_vote_refine, and its integration into evaluate_test_tiles
and evaluate_test_tiles_balanced.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baseline_common import (
    DEFAULT_BAND_NAMES,
    DEFAULT_VI_NAMES,
    confusion_counts,
    evaluate_test_tiles,
    evaluate_test_tiles_balanced,
    metrics_from_counts,
    segment_majority_vote_refine,
)


def _write_tif(path, data, crs="EPSG:3005"):
    import rasterio
    from rasterio.transform import from_bounds

    bands, h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                        count=bands, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data)


@pytest.fixture()
def noisy_scene_tile_paths(tmp_path):
    """
    A (64x64) 13-band tile living under <scene>/images/ and <scene>/masks/.
    Left quarter = land (bathy > 0). Ground truth kelp is a compact circular
    patch in the water region. The tile's NIR band is set up so that a
    naive threshold predictor gets the kelp patch mostly right but with
    ~20% random speckle flips elsewhere in the water -- exactly the kind of
    noise segmentation-based majority voting should clean up.
    Returns (image_path, mask_path).
    """
    h = w = 64
    rng = np.random.default_rng(11)
    data = rng.integers(0, 500, size=(13, h, w), dtype=np.uint16).astype(np.float32)

    land_region = np.zeros((h, w), dtype=bool)
    land_region[:, : w // 4] = True
    bathy = np.where(land_region, 10.0, -10.0)
    data[11] = bathy  # bathymetry band (rasterio band 12)

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w * 3 // 4
    kelp_patch = ((yy - cy) ** 2 + (xx - cx) ** 2) < (h // 5) ** 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[kelp_patch & ~land_region] = 1

    nir = data[3]
    nir[land_region] = 2000.0
    nir[kelp_patch & ~land_region] = 2000.0
    water_nokelp = ~land_region & (mask == 0)
    nir[water_nokelp] = 200.0
    speckle_idx = np.flatnonzero(water_nokelp.reshape(-1))
    n_speckle = int(0.2 * len(speckle_idx))
    speckle_pick = rng.choice(speckle_idx, size=n_speckle, replace=False)
    nir_flat = nir.reshape(-1)
    nir_flat[speckle_pick] = 2000.0
    data[3] = nir_flat.reshape(h, w)

    scene_dir = tmp_path / "scene_00"
    (scene_dir / "images").mkdir(parents=True)
    (scene_dir / "masks").mkdir(parents=True)
    img_path = str(scene_dir / "images" / "tile_0.tif")
    mask_path = str(scene_dir / "masks" / "tile_0.tif")
    _write_tif(img_path, data)
    _write_tif(mask_path, mask[np.newaxis])
    return img_path, mask_path


class TestSegmentMajorityVoteRefine:
    def test_output_is_binary(self):
        h, w, c = 32, 32, 3
        rng = np.random.default_rng(0)
        image_hwc = rng.random((h, w, c))
        raw_pred = rng.integers(0, 2, h * w).astype(np.int64)
        valid = np.ones(h * w, dtype=bool)
        refined = segment_majority_vote_refine(image_hwc, raw_pred, valid, n_segments=20)
        assert set(np.unique(refined).tolist()) <= {0, 1}
        assert refined.shape == raw_pred.shape

    def test_invalid_pixels_untouched(self):
        h, w, c = 32, 32, 3
        rng = np.random.default_rng(1)
        image_hwc = rng.random((h, w, c))
        raw_pred = np.ones(h * w, dtype=np.int64)  # predicts kelp everywhere
        valid = np.zeros(h * w, dtype=bool)
        valid[: h * w // 2] = True  # only first half is "valid" (e.g. water)
        refined = segment_majority_vote_refine(image_hwc, raw_pred, valid, n_segments=20)
        # Invalid half must be untouched (still 1, since raw_pred was all 1s there too)
        assert np.array_equal(refined[~valid], raw_pred[~valid])

    def test_cleans_up_speckle_noise(self):
        """
        A compact true region with scattered raw-prediction noise should
        become spatially cleaner (higher accuracy) after refinement.
        """
        h, w, c = 64, 64, 3
        rng = np.random.default_rng(2)
        image_hwc = rng.random((h, w, c))

        y_true = np.zeros((h, w), dtype=np.int64)
        y_true[10:30, 10:30] = 1

        raw_pred = y_true.copy()
        flip_mask = rng.random((h, w)) < 0.2
        raw_pred[flip_mask] = 1 - raw_pred[flip_mask]

        valid = np.ones(h * w, dtype=bool)
        refined = segment_majority_vote_refine(
            image_hwc, raw_pred.reshape(-1), valid, n_segments=50, compactness=10.0,
        )

        raw_acc = (raw_pred.reshape(-1) == y_true.reshape(-1)).mean()
        refined_acc = (refined == y_true.reshape(-1)).mean()
        assert refined_acc > raw_acc

    def test_all_same_prediction_is_a_noop(self):
        """If every valid pixel already agrees, refinement changes nothing."""
        h, w, c = 32, 32, 2
        rng = np.random.default_rng(3)
        image_hwc = rng.random((h, w, c))
        raw_pred = np.zeros(h * w, dtype=np.int64)  # all no-kelp
        valid = np.ones(h * w, dtype=bool)
        refined = segment_majority_vote_refine(image_hwc, raw_pred, valid, n_segments=20)
        assert np.array_equal(refined, raw_pred)

    def test_no_valid_pixels_returns_unchanged_copy(self):
        h, w, c = 16, 16, 2
        rng = np.random.default_rng(4)
        image_hwc = rng.random((h, w, c))
        raw_pred = rng.integers(0, 2, h * w).astype(np.int64)
        valid = np.zeros(h * w, dtype=bool)
        refined = segment_majority_vote_refine(image_hwc, raw_pred, valid, n_segments=10)
        assert np.array_equal(refined, raw_pred)

    def test_ties_go_to_nokelp(self):
        """A segment with exactly 50/50 kelp/no-kelp raw predictions should end up 0."""
        h, w, c = 4, 4, 1
        # Force a single segment covering the whole tiny tile via high compactness
        # and n_segments=1, then give exactly balanced raw predictions.
        image_hwc = np.zeros((h, w, c))
        raw_pred = np.array([1, 0, 1, 0,
                              1, 0, 1, 0,
                              1, 0, 1, 0,
                              1, 0, 1, 0], dtype=np.int64)
        valid = np.ones(h * w, dtype=bool)
        refined = segment_majority_vote_refine(image_hwc, raw_pred, valid, n_segments=1, compactness=1.0)
        # Whole tile is one segment with 8 kelp / 8 no-kelp -> tie -> no-kelp (0)
        assert np.all(refined == 0)


class TestEvaluateTestTilesSegmentation:
    def test_segmentation_improves_noisy_predictor(self, noisy_scene_tile_paths):
        img_path, mask_path = noisy_scene_tile_paths

        def predict_fn(x_kept):
            nir_idx = DEFAULT_BAND_NAMES.index("nir")
            return (x_kept[:, nir_idx] > 1000).astype(np.int64)

        tp1, fp1, fn1, tn1 = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        tp2, fp2, fn2, tn2 = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, apply_segmentation=True, segmentation_n_segments=80,
        )
        m1 = metrics_from_counts(tp1, fp1, fn1, tn1)
        m2 = metrics_from_counts(tp2, fp2, fn2, tn2)
        assert m2["iou"] > m1["iou"]
        assert m2["f1"] > m1["f1"]

    def test_segmentation_output_still_binary_via_confusion_counts(self, noisy_scene_tile_paths):
        """Sanity check that refined predictions never break confusion_counts."""
        img_path, mask_path = noisy_scene_tile_paths

        def predict_fn(x_kept):
            nir_idx = DEFAULT_BAND_NAMES.index("nir")
            return (x_kept[:, nir_idx] > 1000).astype(np.int64)

        tp, fp, fn, tn = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, apply_segmentation=True,
        )
        assert tp + fp + fn + tn > 0

    def test_disabled_by_default(self, noisy_scene_tile_paths):
        """apply_segmentation defaults to False -- must not change existing behavior."""
        img_path, mask_path = noisy_scene_tile_paths

        def predict_fn(x_kept):
            nir_idx = DEFAULT_BAND_NAMES.index("nir")
            return (x_kept[:, nir_idx] > 1000).astype(np.int64)

        tp1, fp1, fn1, tn1 = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True,
        )
        tp2, fp2, fn2, tn2 = evaluate_test_tiles(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, apply_segmentation=False,
        )
        assert (tp1, fp1, fn1, tn1) == (tp2, fp2, fn2, tn2)


class TestEvaluateTestTilesBalancedSegmentation:
    def test_segmentation_runs_end_to_end(self, noisy_scene_tile_paths):
        img_path, mask_path = noisy_scene_tile_paths

        def predict_fn(x_kept):
            nir_idx = DEFAULT_BAND_NAMES.index("nir")
            return (x_kept[:, nir_idx] > 1000).astype(np.int64)

        tp, fp, fn, tn = evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, apply_segmentation=True, segmentation_n_segments=80,
            seed=42,
        )
        assert tp + fp + fn + tn > 0

    def test_disabled_by_default(self, noisy_scene_tile_paths):
        img_path, mask_path = noisy_scene_tile_paths

        def predict_fn(x_kept):
            nir_idx = DEFAULT_BAND_NAMES.index("nir")
            return (x_kept[:, nir_idx] > 1000).astype(np.int64)

        counts1 = evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, seed=42,
        )
        counts2 = evaluate_test_tiles_balanced(
            [img_path], [mask_path], predict_fn,
            band_names=DEFAULT_BAND_NAMES, vi_names=DEFAULT_VI_NAMES,
            apply_land_mask=True, seed=42, apply_segmentation=False,
        )
        assert counts1 == counts2
