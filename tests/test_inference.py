"""
tests/test_inference.py
-----------------------
Unit tests for inference.py — scene loading, tiling, and end-to-end inference.

All tests use synthetic data (no real GeoTIFFs required) and a tiny dummy
model so CI runs in seconds on CPU.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from inference import (
    _find_scene_files,
    _tile_grid,
    _extract_tile,
    create_weight_map,
    load_scene,
    run_inference,
    save_prediction,
)
from tests.conftest import _write_tif, N_CHANNELS


# ---------------------------------------------------------------------------
# Tiny dummy model (returns zero logits → sigmoid ≈ 0.5 everywhere)
# ---------------------------------------------------------------------------

class DummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return torch.zeros(B, 1, H, W)


# ---------------------------------------------------------------------------
# create_weight_map
# ---------------------------------------------------------------------------

class TestCreateWeightMap:
    def test_shape(self):
        wmap = create_weight_map(tile_size=64, halo_size=8)
        assert wmap.shape == (64, 64)

    def test_centre_equals_one(self):
        wmap = create_weight_map(tile_size=64, halo_size=8)
        centre = wmap[32, 32]
        assert abs(centre - 1.0) < 1e-6

    def test_corner_less_than_centre(self):
        wmap = create_weight_map(tile_size=64, halo_size=8)
        assert wmap[0, 0] < wmap[32, 32]

    def test_all_positive(self):
        wmap = create_weight_map(tile_size=64, halo_size=8)
        assert np.all(wmap > 0)

    def test_no_halo_all_ones(self):
        """halo_size=0 → weight map is all ones."""
        wmap = create_weight_map(tile_size=32, halo_size=0)
        assert np.allclose(wmap, 1.0)


# ---------------------------------------------------------------------------
# _find_scene_files
# ---------------------------------------------------------------------------

class TestFindSceneFiles:
    def test_finds_all_five(self, scene_dir):
        paths = _find_scene_files(scene_dir)
        assert len(paths) == 5
        for p in paths:
            assert os.path.isfile(p)

    def test_raises_if_pattern_missing(self, tmp_path):
        """Missing file for one pattern → ValueError."""
        with pytest.raises(ValueError, match="Expected exactly 1"):
            _find_scene_files(str(tmp_path))

    def test_raises_if_ambiguous(self, scene_dir, tmp_path):
        """Two files matching the same pattern → ValueError."""
        import shutil
        dup = glob.glob(os.path.join(scene_dir, "*_B2B3B4B8.tif"))[0]
        shutil.copy(dup, dup.replace("B2B3B4B8", "B2B3B4B8_copy_B2B3B4B8"))
        with pytest.raises(ValueError, match="Expected exactly 1"):
            _find_scene_files(scene_dir)


# ---------------------------------------------------------------------------
# load_scene
# ---------------------------------------------------------------------------

class TestLoadScene:
    ALL_BANDS = ["blue", "green", "red", "nir", "rededge",
                 "substrate", "bathymetry", "slope"]
    ALL_VIS   = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]

    def test_output_shape_all_channels(self, scene_dir):
        """All 8 raw bands + 5 VIs = 13 channels."""
        image, _ = load_scene(scene_dir, self.ALL_BANDS, self.ALL_VIS)
        assert image.shape == (64, 64, N_CHANNELS)

    def test_spectral_only_no_aux(self, scene_dir):
        """When only spectral bands are requested, aux files are not opened."""
        bands = ["blue", "green", "red", "nir", "rededge"]
        image, _ = load_scene(scene_dir, bands, vi_names=[])
        assert image.shape == (64, 64, 5)

    def test_subset_no_rededge(self, scene_dir):
        """Request only 4 spectral bands — no 20 m file needed."""
        bands = ["blue", "green", "red", "nir"]
        image, _ = load_scene(scene_dir, bands, vi_names=[])
        assert image.shape == (64, 64, 4)

    def test_channel_order_matches_band_names(self, scene_dir):
        """Output channel count must equal len(band_names) + len(vi_names)."""
        bands = ["blue", "nir", "slope"]
        vis   = ["ndvi"]
        image, _ = load_scene(scene_dir, bands, vis)
        assert image.shape[2] == len(bands) + len(vis)

    def test_substrate_values_valid(self, scene_dir):
        """
        Substrate values on disk are {1,2,3} after preprocessing remaps 4→3.
        load_scene must preserve them unchanged — value 4 should not appear.
        """
        image, _ = load_scene(scene_dir, self.ALL_BANDS, vi_names=[])
        sub_idx  = self.ALL_BANDS.index("substrate")
        unique   = set(np.unique(image[:, :, sub_idx]).tolist())
        assert 4.0 not in unique, (
            "Value 4 found in substrate channel — remapping should happen "
            "at preprocessing time, not at read time."
        )
        assert unique.issubset({0.0, 1.0, 2.0, 3.0})

    def test_metadata_has_crs(self, scene_dir):
        _, meta = load_scene(scene_dir, ["blue", "green", "red", "nir"],
                              vi_names=[])
        assert "crs" in meta or meta.get("crs") is not None

    def test_no_vi_names(self, scene_dir):
        """When vi_names=[], output has exactly len(band_names) channels."""
        image, _ = load_scene(scene_dir, self.ALL_BANDS, vi_names=[])
        assert image.shape[2] == len(self.ALL_BANDS)

    def test_unknown_vi_raises(self, scene_dir):
        with pytest.raises(ValueError, match="Unknown VI name"):
            load_scene(scene_dir, ["blue", "green", "red", "nir"],
                       vi_names=["totally_unknown_vi"])


# ---------------------------------------------------------------------------
# _tile_grid
# ---------------------------------------------------------------------------

class TestTileGrid:
    def test_small_image_one_tile(self):
        """Image smaller than tile_size → exactly one tile at (0,0)."""
        coords = _tile_grid(h=100, w=100, tile_size=512, overlap=0.5)
        assert coords == [(0, 0)]

    def test_covers_full_image(self):
        """Every pixel must be covered by at least one tile."""
        H, W, tile_size, overlap = 400, 600, 128, 0.5
        coords = _tile_grid(H, W, tile_size, overlap)
        covered = np.zeros((H, W), dtype=bool)
        for ri, rj in coords:
            covered[ri:ri+tile_size, rj:rj+tile_size] = True
        assert covered.all(), "Some pixels are not covered by any tile."

    def test_overlap_increases_tile_count(self):
        coords_lo = _tile_grid(512, 512, 128, overlap=0.0)
        coords_hi = _tile_grid(512, 512, 128, overlap=0.5)
        assert len(coords_hi) > len(coords_lo)


# ---------------------------------------------------------------------------
# _extract_tile
# ---------------------------------------------------------------------------

class TestExtractTile:
    def _img(self, H=100, W=100, C=13):
        return np.random.default_rng(0).random((H, W, C)).astype(np.float32)

    def test_interior_tile_exact_size(self):
        img  = self._img()
        tile = _extract_tile(img, i=0, j=0, tile_size=64, padding_mode="reflect")
        assert tile.shape == (64, 64, 13)

    def test_border_tile_padded(self):
        """Tile at the bottom-right corner must be padded to tile_size."""
        img  = self._img(H=70, W=70)
        tile = _extract_tile(img, i=50, j=50, tile_size=64, padding_mode="reflect")
        assert tile.shape == (64, 64, 13)

    def test_constant_padding_fills_zeros(self):
        img  = self._img(H=70, W=70)
        tile = _extract_tile(img, i=50, j=50, tile_size=64, padding_mode="constant")
        # The padded region (rows 20+ and cols 20+) should contain zeros
        assert np.any(tile[25:, 25:] == 0.0)


# ---------------------------------------------------------------------------
# run_inference — end-to-end smoke tests
# ---------------------------------------------------------------------------

class TestRunInference:
    """
    Uses a 64×64 synthetic scene and a DummyModel that outputs zero logits.
    At threshold=0.5, sigmoid(0) = 0.5 which is NOT > 0.5, so all predictions
    should be 0.  We just verify shapes, dtypes, and that nothing crashes.
    """
    ALL_BANDS = ["blue", "green", "red", "nir", "rededge",
                 "substrate", "bathymetry", "slope"]
    ALL_VIS   = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]

    def _run(self, method: str, scene_dir, norm_stats):
        image_hwc, _ = load_scene(scene_dir, self.ALL_BANDS, self.ALL_VIS)
        mean, std    = norm_stats
        model        = DummyModel()
        device       = torch.device("cpu")

        preds = run_inference(
            image_hwc, model, device, mean, std,
            tile_size=32, overlap=0.5, halo_size=4,
            method=method, padding_mode="reflect",
            batch_size=4, threshold=0.5,
        )
        return preds

    def test_weighted_shape(self, scene_dir, norm_stats):
        preds = self._run("weighted", scene_dir, norm_stats)
        assert preds.shape == (64, 64)

    def test_halo_shape(self, scene_dir, norm_stats):
        preds = self._run("halo", scene_dir, norm_stats)
        assert preds.shape == (64, 64)

    def test_max_shape(self, scene_dir, norm_stats):
        preds = self._run("max", scene_dir, norm_stats)
        assert preds.shape == (64, 64)

    def test_output_dtype_uint8(self, scene_dir, norm_stats):
        preds = self._run("weighted", scene_dir, norm_stats)
        assert preds.dtype == np.uint8

    def test_output_binary(self, scene_dir, norm_stats):
        preds = self._run("weighted", scene_dir, norm_stats)
        assert set(np.unique(preds).tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# save_prediction
# ---------------------------------------------------------------------------

class TestSavePrediction:
    def test_file_created(self, scene_dir, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        preds = np.zeros((64, 64), dtype=np.uint8)
        meta  = {
            "driver": "GTiff",
            "dtype":  "uint8",
            "width":  64,
            "height": 64,
            "count":  1,
            "crs":    "EPSG:3005",
            "transform": from_bounds(0, 0, 1, 1, 64, 64),
        }
        out_path = str(tmp_path / "pred.tif")
        save_prediction(preds, meta, out_path)
        assert os.path.isfile(out_path)

    def test_saved_values_match(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        rng   = np.random.default_rng(99)
        preds = rng.integers(0, 2, (64, 64), dtype=np.uint8)
        meta  = {
            "driver": "GTiff", "dtype": "uint8",
            "width": 64, "height": 64, "count": 1,
            "crs": "EPSG:3005",
            "transform": from_bounds(0, 0, 1, 1, 64, 64),
        }
        out_path = str(tmp_path / "pred_check.tif")
        save_prediction(preds, meta, out_path)

        with rasterio.open(out_path) as src:
            loaded = src.read(1)
        assert np.array_equal(preds, loaded)
