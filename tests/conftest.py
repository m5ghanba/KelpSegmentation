"""
tests/conftest.py
-----------------
Shared pytest fixtures for KelpSegmentation tests.

All fixtures that create files use tmp_path (pytest's built-in per-test
temporary directory) so nothing leaks between tests.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Constants mirrored from the project
# ---------------------------------------------------------------------------

N_RAW_BANDS = 8          # Blue, Green, Red, NIR, Red-edge, Substrate, Bathy, Slope
N_VI        = 5          # ndvi, ndwi, gndvi, clgreen, ndvire
N_CHANNELS  = N_RAW_BANDS + N_VI   # 13

TILE_H = 512
TILE_W = 512

DEFAULT_BAND_NAMES = [
    "blue", "green", "red", "nir", "rededge",
    "substrate", "bathymetry", "slope",
]
DEFAULT_VI_NAMES = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]


# ---------------------------------------------------------------------------
# Rasterio helper
# ---------------------------------------------------------------------------

def _write_tif(
    path: str,
    data: np.ndarray,   # (bands, H, W)
    crs:  str = "EPSG:3005",
) -> None:
    """Write a minimal GeoTIFF for testing."""
    bands, H, W = data.shape
    transform = from_bounds(0, 0, 1, 1, W, H)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=H, width=W,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Single-tile fixtures  (used by test_dataset.py)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tile_image_path(tmp_path) -> str:
    """
    A GeoTIFF with 13 bands (the full channel stack expected by SatelliteDataset)
    written as a single 512×512 tile.

    The substrate channel (band index 5) only contains values {1, 2, 3}.
    Value 4 is absent because the 4→3 remapping is now done at preprocessing
    time (notebook Cell 4) before tiles are written to disk — not at read time.
    """
    rng  = np.random.default_rng(0)
    data = rng.integers(0, 500, size=(13, TILE_H, TILE_W), dtype=np.uint16).astype(np.float32)

    # Substrate channel: values {1, 2, 3} only — 4 no longer exists on disk
    data[5] = rng.integers(1, 4, size=(TILE_H, TILE_W), dtype=np.uint16).astype(np.float32)

    path = str(tmp_path / "tile_image.tif")
    _write_tif(path, data)
    return path


@pytest.fixture()
def tile_mask_path(tmp_path) -> str:
    """Binary mask (0 or 1) for a single 512×512 tile."""
    rng  = np.random.default_rng(1)
    data = rng.integers(0, 2, size=(1, TILE_H, TILE_W), dtype=np.uint8)
    path = str(tmp_path / "tile_mask.tif")
    _write_tif(path, data)
    return path


# ---------------------------------------------------------------------------
# Scene-folder fixture  (used by test_inference.py)
# ---------------------------------------------------------------------------

@pytest.fixture()
def scene_dir(tmp_path) -> str:
    """
    A synthetic scene folder containing the five raw GeoTIFFs expected by
    DatasetInference / inference.load_scene.

    Spatial size is kept small (64×64) for fast tests.
    """
    H, W = 64, 64
    rng  = np.random.default_rng(42)
    scene = tmp_path / "SCENE_UXQ"
    scene.mkdir()

    # *_B2B3B4B8.tif  — 4 bands, 10 m
    _write_tif(str(scene / "SCENE_UXQ_B2B3B4B8.tif"),
               rng.integers(0, 500, (4, H, W), dtype=np.uint16).astype(np.float32))

    # *_B5B6B7B8A_B11B12.tif  — 6 bands (only first = red-edge is used)
    _write_tif(str(scene / "SCENE_UXQ_B5B6B7B8A_B11B12.tif"),
               rng.integers(0, 500, (6, H, W), dtype=np.uint16).astype(np.float32))

    # *_Substrate.tif  — values {1, 2, 3} only.
    # Value 4 is absent because the 4→3 remapping is done at preprocessing
    # time (notebook Cell 4) before the file is written to disk.
    sub = rng.integers(1, 4, (1, H, W), dtype=np.uint8).astype(np.float32)
    _write_tif(str(scene / "SCENE_UXQ_Substrate.tif"), sub)

    # *_Bathymetry.tif  — negative depths
    bathy = rng.uniform(-200, 0, (1, H, W)).astype(np.float32)
    _write_tif(str(scene / "SCENE_UXQ_Bathymetry.tif"), bathy)

    # *_Slope.tif  — 0–45 degree values
    slope = rng.uniform(0, 45, (1, H, W)).astype(np.float32)
    _write_tif(str(scene / "SCENE_UXQ_Slope.tif"), slope)

    return str(scene)


# ---------------------------------------------------------------------------
# Normalization stats fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def norm_stats():
    """Plausible mean/std arrays of length N_CHANNELS (13)."""
    rng  = np.random.default_rng(7)
    mean = rng.uniform(50, 200, N_CHANNELS).astype(np.float32)
    std  = rng.uniform(10,  50, N_CHANNELS).astype(np.float32)
    return mean, std
