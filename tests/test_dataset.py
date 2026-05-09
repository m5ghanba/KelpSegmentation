"""
tests/test_dataset.py
---------------------
Unit tests for src/dataset.py — SatelliteDataset and split utilities.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import pytest

from src.dataset import (
    SatelliteDataset,
    compute_dataset_mean_std,
    get_files_from_folders,
    normalize_input_mean_std,
    split_geographic,
    split_random,
    split_loo,
)
from tests.conftest import (
    DEFAULT_BAND_NAMES,
    DEFAULT_VI_NAMES,
    N_CHANNELS,
    TILE_H,
    TILE_W,
    _write_tif,
)


# ---------------------------------------------------------------------------
# normalize_input_mean_std
# ---------------------------------------------------------------------------

class TestNormalizeInputMeanStd:
    def test_output_shape_preserved(self):
        img  = np.ones((64, 64, 13), dtype=np.float32) * 100
        mean = np.full(13, 50.0, dtype=np.float32)
        std  = np.full(13, 25.0, dtype=np.float32)
        out  = normalize_input_mean_std(img, mean, std)
        assert out.shape == img.shape

    def test_known_values(self):
        """pixel = 100, mean = 50, std = 25 → normalized = 2.0"""
        img  = np.full((1, 1, 1), 100.0, dtype=np.float32)
        mean = [50.0]
        std  = [25.0]
        out  = normalize_input_mean_std(img, mean, std)
        assert abs(out[0, 0, 0] - 2.0) < 1e-5

    def test_nan_handled(self):
        """NaN inputs must not propagate into the output."""
        img  = np.full((4, 4, 3), np.nan, dtype=np.float32)
        mean = [0.0, 0.0, 0.0]
        std  = [1.0, 1.0, 1.0]
        out  = normalize_input_mean_std(img, mean, std)
        assert not np.any(np.isnan(out))

    def test_zero_std_no_division_error(self):
        """epsilon guard prevents division by zero."""
        img  = np.ones((4, 4, 2), dtype=np.float32)
        mean = [1.0, 1.0]
        std  = [0.0, 0.0]
        out  = normalize_input_mean_std(img, mean, std)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# SatelliteDataset
# ---------------------------------------------------------------------------

class TestSatelliteDataset:
    def test_len(self, tile_image_path, tile_mask_path):
        ds = SatelliteDataset(
            [tile_image_path] * 3, [tile_mask_path] * 3,
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        assert len(ds) == 3

    def test_output_shapes(self, tile_image_path, tile_mask_path):
        ds = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        img, mask = ds[0]
        # image: (C, H, W)
        assert img.shape == (N_CHANNELS, TILE_H, TILE_W)
        # mask: (1, H, W) — binary kelp class
        assert mask.shape == (1, TILE_H, TILE_W)

    def test_output_dtype(self, tile_image_path, tile_mask_path):
        ds = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        img, mask = ds[0]
        assert img.dtype == np.float32
        assert mask.dtype == np.float32

    def test_mask_binary(self, tile_image_path, tile_mask_path):
        """Mask values must be 0.0 or 1.0 only."""
        ds = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        _, mask = ds[0]
        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_substrate_values_valid(self, tile_image_path, tile_mask_path):
        """
        Substrate values on disk are {1,2,3} after preprocessing remaps 4→3.
        The dataset must preserve them unchanged — value 4 should not appear.
        """
        ds = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        img, _ = ds[0]
        substrate_ch = img[5]   # substrate is raw band index 5
        unique = set(np.unique(substrate_ch).tolist())
        assert 4.0 not in unique, (
            "Value 4 found in substrate channel — remapping should happen "
            "at preprocessing time, not at read time."
        )
        # All values must be within the valid set {0, 1, 2, 3}
        assert unique.issubset({0.0, 1.0, 2.0, 3.0})

    def test_num_channels_property(self):
        ds = SatelliteDataset.__new__(SatelliteDataset)
        ds.band_names = DEFAULT_BAND_NAMES
        ds.vi_names   = DEFAULT_VI_NAMES
        assert ds.num_channels == N_CHANNELS

    def test_normalization_applied(self, tile_image_path, tile_mask_path):
        """With non-trivial mean/std, normalized values should differ from raw."""
        mean = np.full(N_CHANNELS, 200.0, dtype=np.float32)
        std  = np.full(N_CHANNELS,  50.0, dtype=np.float32)
        ds_raw  = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
        )
        ds_norm = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=DEFAULT_BAND_NAMES,
            vi_names=DEFAULT_VI_NAMES,
            mean=mean, std=std,
        )
        img_raw,  _ = ds_raw[0]
        img_norm, _ = ds_norm[0]
        assert not np.allclose(img_raw, img_norm)

    def test_subset_channels(self, tile_image_path, tile_mask_path):
        """Dataset with only spectral bands (no substrate/bathy/slope/VIs)."""
        spectral = ["blue", "green", "red", "nir"]
        ds = SatelliteDataset(
            [tile_image_path], [tile_mask_path],
            classes=["kelp"],
            band_names=spectral,
            vi_names=[],
        )
        img, _ = ds[0]
        assert img.shape[0] == 4


# ---------------------------------------------------------------------------
# get_files_from_folders
# ---------------------------------------------------------------------------

class TestGetFilesFromFolders:
    def test_basic(self, tmp_path):
        scene = tmp_path / "S1"
        (scene / "images").mkdir(parents=True)
        (scene / "masks").mkdir(parents=True)
        (scene / "images" / "a.tif").touch()
        (scene / "images" / "b.tif").touch()
        (scene / "masks"  / "a.tif").touch()
        (scene / "masks"  / "b.tif").touch()

        imgs, masks = get_files_from_folders([str(scene)])
        assert len(imgs) == 2
        assert len(masks) == 2

    def test_missing_subfolder_warns(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        imgs, masks = get_files_from_folders([str(empty)])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert imgs == [] and masks == []


# ---------------------------------------------------------------------------
# Split strategies (smoke tests — verify counts add up and no overlap)
# ---------------------------------------------------------------------------

def _make_scene_tree(root, n_scenes: int = 10, tiles_per_scene: int = 5):
    """Create a minimal directory tree for split-strategy tests."""
    for s in range(n_scenes):
        scene = root / f"scene_{s:02d}"
        (scene / "images").mkdir(parents=True)
        (scene / "masks").mkdir(parents=True)
        for t in range(tiles_per_scene):
            (scene / "images" / f"tile_{t}.tif").touch()
            (scene / "masks"  / f"tile_{t}.tif").touch()
    return str(root / "*")


class TestSplitStrategies:
    def test_geographic_counts(self, tmp_path):
        pattern = _make_scene_tree(tmp_path, n_scenes=10, tiles_per_scene=5)
        X_tr, y_tr, X_v, y_v, X_te, y_te = split_geographic(
            pattern, train_frac=0.70, val_frac=0.15, seed=42
        )
        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == 50
        assert len(X_tr) == len(y_tr)
        assert len(X_v)  == len(y_v)
        assert len(X_te) == len(y_te)

    def test_geographic_no_overlap(self, tmp_path):
        pattern = _make_scene_tree(tmp_path, n_scenes=10, tiles_per_scene=5)
        X_tr, _, X_v, _, X_te, _ = split_geographic(pattern, seed=0)
        assert set(X_tr).isdisjoint(set(X_v))
        assert set(X_tr).isdisjoint(set(X_te))

    def test_random_counts(self, tmp_path):
        pattern = _make_scene_tree(tmp_path, n_scenes=10, tiles_per_scene=5)
        X_tr, y_tr, X_v, y_v, X_te, y_te = split_random(
            pattern, val_frac=0.15, test_frac=0.15, seed=42
        )
        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == 50

    def test_loo_test_folder_isolated(self, tmp_path):
        pattern = _make_scene_tree(tmp_path, n_scenes=10, tiles_per_scene=5)
        X_tr, _, X_v, _, X_te, _ = split_loo(
            pattern, test_indices=[0, 1], val_frac=0.20, seed=42
        )
        # test set should come only from scenes 0 and 1
        assert len(X_te) == 10
        # no test tile should appear in train or val
        assert set(X_te).isdisjoint(set(X_tr))
        assert set(X_te).isdisjoint(set(X_v))

    def test_loo_out_of_range_raises(self, tmp_path):
        pattern = _make_scene_tree(tmp_path, n_scenes=5, tiles_per_scene=3)
        with pytest.raises(ValueError, match="out of range"):
            split_loo(pattern, test_indices=[99])

    def test_empty_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            split_geographic(str(tmp_path / "nonexistent" / "*"))
