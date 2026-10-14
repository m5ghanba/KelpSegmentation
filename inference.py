"""
inference.py
------------
Inference CLI entry point for kelp / no-kelp segmentation.

Usage
-----
    python inference.py --config configs/infer_config.yaml

Scene folder layout
-------------------
Each sub-folder inside ``inference.input_dir`` must contain the GeoTIFF files
that correspond to the ``band_names`` listed in the config.  Only the files
actually needed are opened — if ``substrate``, ``bathymetry``, and ``slope``
are all absent from ``band_names``, the corresponding ``*_Substrate.tif`` /
``*_Bathymetry.tif`` / ``*_Slope.tif`` files do not need to exist.

Required file patterns per band group
--------------------------------------
  blue / green / red / nir          → *_B2B3B4B8.tif         (always required)
  rededge                           → *_B5B6B7B8A_B11B12.tif  (band 1)
  substrate                         → *_Substrate.tif
  bathymetry                        → *_Bathymetry.tif
  slope                             → *_Slope.tif

Tiling merge methods
--------------------
  weighted : Linear-fade weight map averaging of overlapping tile logits.
  halo     : Uses only the centre of each tile, discards border halo.
  max      : Element-wise maximum of overlapping binary predictions.

CHANGES vs. the original (for the manuscript revision):
  - Removed the unconditional substrate value 4→3 remap. That remap matched
    the BoPs substrate scheme (3 classes, {1,2,3}), but this project reports
    results using the RF substrate scheme exclusively (4 classes, {1,2,3,4}
    — rock/mixed/sand/mud), which src/dataset.py's SatelliteDataset never
    remapped either. Applying the remap here regardless of scheme silently
    folded real "mud" pixels into "sand" at inference time only — training
    tiles kept class 4 intact. If you ever reintroduce BoPs-substrate scenes,
    handle the scheme difference explicitly (e.g. a `substrate_scheme` config
    key), don't hardcode a remap for everyone.
  - in_channels / channel assembly now delegates to
    `src.dataset.resolve_num_channels` so this file and SatelliteDataset can't
    drift out of sync on the channel count again.
  - `load_scene()` and `resolve_in_channels()` gained `substrate_onehot`
    support, mirroring SatelliteDataset, for running full-scene inference
    with a model trained on the one-hot substrate ablation.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import yaml
from tqdm import tqdm

# ---- local imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import (
    normalize_input_mean_std,
    SatelliteDataset,
    resolve_num_channels,
    DEFAULT_SUBSTRATE_CLASSES,
)
from src.model import KelpSegmentationModel


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_in_channels(cfg: Dict[str, Any]) -> int:
    """Derive in_channels from band_names + vi_names, with optional override."""
    data_cfg = cfg["data"]
    n = resolve_num_channels(
        data_cfg.get("band_names", []),
        data_cfg.get("vi_names", []),
        substrate_onehot=data_cfg.get("substrate_onehot", False),
    )
    return cfg.get("model", {}).get("in_channels", n)


# ---------------------------------------------------------------------------
# Weight map  (matches notebook's create_weight_map exactly)
# ---------------------------------------------------------------------------

def create_weight_map(tile_size: int, halo_size: int) -> np.ndarray:
    """
    Return a (tile_size, tile_size) float32 weight map where edge pixels
    fade linearly to zero over ``halo_size`` pixels.  Centre pixels = 1.0.
    """
    wmap = np.ones((tile_size, tile_size), dtype=np.float32)
    for i in range(halo_size):
        fade = (i + 1) / halo_size
        wmap[i, :]                 = fade
        wmap[tile_size - 1 - i, :] = fade
        wmap[:, i]                 = np.minimum(wmap[:, i],                 fade)
        wmap[:, tile_size - 1 - i] = np.minimum(wmap[:, tile_size - 1 - i], fade)
    return wmap


# ---------------------------------------------------------------------------
# Scene loading — driven entirely by band_names from the config
# ---------------------------------------------------------------------------

# Maps each optional auxiliary band name to the glob pattern of its file.
# The 10 m spectral bands are always loaded from *_B2B3B4B8.tif.
# The 20 m file is only opened when "rededge" is requested.
_AUX_PATTERNS: Dict[str, str] = {
    "rededge":    "*_B5B6B7B8A_B11B12.tif",
    "substrate":  "*_Substrate.tif",
    "bathymetry": "*_Bathymetry.tif",
    "slope":      "*_Slope.tif",
}

# Which bands live inside *_B2B3B4B8.tif and their 1-based rasterio indices
_SPECTRAL_10M: Dict[str, int] = {
    "blue": 1, "green": 2, "red": 3, "nir": 4,
}


def _find_file(scene_dir: str, pattern: str) -> str:
    """Return the single file matching *pattern* in scene_dir, or raise."""
    matches = glob.glob(os.path.join(scene_dir, pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching '{pattern}' in {scene_dir!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_scene(
    scene_dir:  str,
    band_names: List[str],
    vi_names:   List[str],
    substrate_onehot:  bool = False,
    substrate_classes: Tuple[int, ...] = DEFAULT_SUBSTRATE_CLASSES,
) -> Tuple[np.ndarray, Dict]:
    """
    Load exactly the bands listed in ``band_names``, compute the VIs in
    ``vi_names``, and return the (H, W, C) array ready for inference.

    Channel order in the output array mirrors ``SatelliteDataset.__getitem__``:
    raw bands in the order given by ``band_names``, then VIs in the order
    given by ``vi_names``, then (if substrate_onehot) one-hot substrate
    channels appended last.

    Parameters
    ----------
    scene_dir  : path to one scene sub-folder
    band_names : list of band names, e.g. ["blue","green","red","nir","rededge"]
                 Must match training config exactly. Must NOT contain
                 "substrate" when substrate_onehot=True (same contract as
                 SatelliteDataset).
    vi_names   : list of VI names, e.g. ["ndvi","ndwi"]
    substrate_onehot : if True, read the substrate band and append a
                 one-hot encoding instead of expecting a scalar "substrate"
                 entry in band_names.
    substrate_classes : which raw substrate values become one-hot channels.

    Returns
    -------
    image_hwc : np.ndarray  shape (H, W, len(band_names) + len(vi_names) [+ K])
                raw (un-normalized) float32 array
    metadata  : dict  rasterio profile from the 10 m file (for georeferenced output)
    """
    if substrate_onehot and "substrate" in band_names:
        raise ValueError(
            "substrate_onehot=True expects 'substrate' to be excluded from "
            "band_names — see src/dataset.py SatelliteDataset docstring."
        )

    # ---- Determine which files are actually needed -------------------------
    need_10m     = any(b in _SPECTRAL_10M for b in band_names)
    need_20m     = "rededge"    in band_names
    need_sub     = substrate_onehot or ("substrate" in band_names)
    need_bathy   = "bathymetry" in band_names
    need_slope   = "slope"      in band_names

    # ---- Always load the 10 m file (at minimum for scene dimensions) --------
    path_10m = _find_file(scene_dir, "*_B2B3B4B8.tif")
    with rasterio.open(path_10m) as src:
        metadata = src.meta.copy()
        H_full   = src.height
        W_full   = src.width
        # Read only the spectral bands that are actually requested
        requested_10m_indices = [
            _SPECTRAL_10M[b] for b in band_names if b in _SPECTRAL_10M
        ]
        if requested_10m_indices:
            raw_10m = src.read(requested_10m_indices)   # (n, H, W)
            img_10m = np.transpose(raw_10m, (1, 2, 0)).astype(np.float32)  # (H,W,n)
        else:
            img_10m = None

    H, W = H_full, W_full

    # ---- Red-edge: band 1 of *_B5B6B7B8A_B11B12.tif, upsampled to 10 m ----
    if need_20m:
        path_20m = _find_file(scene_dir, "*_B5B6B7B8A_B11B12.tif")
        with rasterio.open(path_20m) as src:
            raw_re = src.read(
                indexes=[1],
                out_shape=(1, H, W),
                resampling=Resampling.nearest,
            )                                                    # (1, H, W)
        img_re = np.transpose(raw_re, (1, 2, 0)).astype(np.float32)  # (H,W,1)
    else:
        img_re = None

    # ---- Substrate ------------------------------------------------------------
    # RF substrate scheme only: valid classes {1,2,3,4} (rock, mixed sediment,
    # sand, mud); 0 = no coastal-data coverage (sentinel fill, not a class).
    # No remap — see module docstring.
    substrate_raw = None
    if need_sub:
        path_sub = _find_file(scene_dir, "*_Substrate.tif")
        with rasterio.open(path_sub) as src:
            substrate_raw = src.read(1).astype(np.float32)   # (H, W)

    substrate = None
    if substrate_raw is not None and not substrate_onehot:
        substrate = substrate_raw[:, :, np.newaxis]

    # ---- Bathymetry ---------------------------------------------------------
    if need_bathy:
        path_bathy = _find_file(scene_dir, "*_Bathymetry.tif")
        with rasterio.open(path_bathy) as src:
            bathymetry = src.read(1).astype(np.float32)[:, :, np.newaxis]
    else:
        bathymetry = None

    # ---- Slope --------------------------------------------------------------
    if need_slope:
        path_slope = _find_file(scene_dir, "*_Slope.tif")
        with rasterio.open(path_slope) as src:
            slope = src.read(1).astype(np.float32)[:, :, np.newaxis]
    else:
        slope = None

    # ---- Assemble channels in band_names order ------------------------------
    # This mirrors exactly how SatelliteDataset stacks its channels,
    # so the channel index seen by the model is identical to training.
    band_arrays: Dict[str, np.ndarray] = {}
    if img_10m is not None:
        spectral_names_in_order = [b for b in band_names if b in _SPECTRAL_10M]
        for idx, name in enumerate(spectral_names_in_order):
            band_arrays[name] = img_10m[:, :, idx : idx + 1]
    if img_re      is not None: band_arrays["rededge"]    = img_re
    if substrate   is not None: band_arrays["substrate"]  = substrate
    if bathymetry  is not None: band_arrays["bathymetry"] = bathymetry
    if slope       is not None: band_arrays["slope"]      = slope

    # Stack in exactly the order declared in band_names
    raw_channels = [band_arrays[b] for b in band_names]
    image_hwc    = np.concatenate(raw_channels, axis=-1)   # (H, W, len(band_names))

    # ---- Band position map for VI calculators — follows stacking order ------
    band_pos = {name: idx for idx, name in enumerate(band_names)}

    # ---- Compute vegetation indices using SatelliteDataset calculators ------
    dummy             = SatelliteDataset.__new__(SatelliteDataset)
    dummy._band_pos   = band_pos
    dummy._vi_calculators = {
        "ndvi":    dummy._calc_ndvi,
        "ndwi":    dummy._calc_ndwi,
        "gndvi":   dummy._calc_gndvi,
        "clgreen": dummy._calc_clgreen,
        "ndvire":  dummy._calc_ndvire,
    }

    vi_arrays = []
    for vi in vi_names:
        calc = dummy._vi_calculators.get(vi)
        if calc is None:
            raise ValueError(
                f"Unknown VI name: {vi!r}. "
                f"Available: {list(dummy._vi_calculators.keys())}"
            )
        vi_arrays.append(calc(image_hwc)[..., np.newaxis])

    if vi_arrays:
        image_hwc = np.concatenate([image_hwc] + vi_arrays, axis=-1)

    # ---- One-hot substrate, appended last (mirrors SatelliteDataset) --------
    if substrate_onehot:
        onehot = np.stack(
            [(substrate_raw == c).astype(np.float32) for c in substrate_classes],
            axis=-1,
        )
        image_hwc = np.concatenate([image_hwc, onehot], axis=-1)

    return image_hwc, metadata


# ---------------------------------------------------------------------------
# Tile grid  (mirrors notebook's generate_tiles ceiling logic exactly)
# ---------------------------------------------------------------------------

def _tile_grid(h: int, w: int, tile_size: int, overlap: float) -> List[Tuple[int, int]]:
    """Return list of (row, col) top-left corners for a covering tile grid."""
    step = max(int(tile_size * (1 - overlap)), 1)
    tiles_y = 1 if h <= tile_size else ((h - tile_size) + step - 1) // step + 1
    tiles_x = 1 if w <= tile_size else ((w - tile_size) + step - 1) // step + 1
    return [(y * step, x * step) for y in range(tiles_y) for x in range(tiles_x)]


def _extract_tile(
    image_hwc:    np.ndarray,
    i:            int,
    j:            int,
    tile_size:    int,
    padding_mode: str,
) -> np.ndarray:
    """Extract one tile from a scene image and pad to tile_size if needed."""
    H, W  = image_hwc.shape[:2]
    tile  = image_hwc[i : min(i + tile_size, H), j : min(j + tile_size, W)]
    ah, aw = tile.shape[:2]

    if ah < tile_size or aw < tile_size:
        pad_b = tile_size - ah
        pad_r = tile_size - aw
        mode  = padding_mode
        if mode == "reflect" and (ah <= 1 or aw <= 1):
            mode = "edge"
        if mode == "constant":
            tile = np.pad(tile, ((0, pad_b), (0, pad_r), (0, 0)),
                          mode=mode, constant_values=0)
        else:
            tile = np.pad(tile, ((0, pad_b), (0, pad_r), (0, 0)), mode=mode)
    return tile


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(
    image_hwc:    np.ndarray,
    model:        torch.nn.Module,
    device:       torch.device,
    mean:         np.ndarray,
    std:          np.ndarray,
    tile_size:    int   = 512,
    overlap:      float = 0.70,
    halo_size:    int   = 64,
    method:       str   = "weighted",
    padding_mode: str   = "reflect",
    batch_size:   int   = 8,
    threshold:    float = 0.5,
) -> np.ndarray:
    """
    Run tiled inference on a full (H, W, C) scene image.

    The global training mean/std is applied tile-by-tile (not recomputed per
    tile), matching the notebook's DatasetInference.generate_tiles →
    normalize_tile_mean_std flow.

    Returns
    -------
    predictions : np.ndarray  shape (H, W), dtype uint8  (0 = background, 1 = kelp)
    """
    model.eval()
    H, W = image_hwc.shape[:2]

    weight_map = create_weight_map(tile_size, halo_size)
    coords     = _tile_grid(H, W, tile_size, overlap)

    if method == "weighted":
        prob_accum   = np.zeros((H, W), dtype=np.float64)
        weight_accum = np.zeros((H, W), dtype=np.float64)
    else:
        predictions = np.zeros((H, W), dtype=np.uint8)

    batch_tiles:  List[torch.Tensor]    = []
    batch_coords: List[Tuple[int, int]] = []

    def _flush() -> None:
        tensor = torch.cat(batch_tiles, dim=0).to(device)   # (B, C, tile, tile)
        with torch.no_grad():
            logits = model(tensor)                           # (B, 1, tile, tile)

        if method == "weighted":
            probs = logits.squeeze(1).sigmoid().cpu().numpy()
            for prob, (ri, rj) in zip(probs, batch_coords):
                eh     = min(tile_size, H - ri)
                ew     = min(tile_size, W - rj)
                wslice = weight_map[:eh, :ew]
                prob_accum[ri:ri+eh, rj:rj+ew]   += prob[:eh, :ew] * wslice
                weight_accum[ri:ri+eh, rj:rj+ew] += wslice

        elif method == "halo":
            preds = (logits.squeeze(1).sigmoid() > threshold).cpu().numpy().astype(np.uint8)
            for pred, (ri, rj) in zip(preds, batch_coords):
                si = halo_size if ri > 0               else 0
                sj = halo_size if rj > 0               else 0
                ei = (tile_size - halo_size) if (ri + tile_size) < H else tile_size
                ej = (tile_size - halo_size) if (rj + tile_size) < W else tile_size
                fi, fj = ri + si, rj + sj
                predictions[fi:fi+(ei-si), fj:fj+(ej-sj)] = pred[si:ei, sj:ej]

        else:  # max
            preds = (logits.squeeze(1).sigmoid() > threshold).cpu().numpy().astype(np.uint8)
            for pred, (ri, rj) in zip(preds, batch_coords):
                eh = min(tile_size, H - ri)
                ew = min(tile_size, W - rj)
                predictions[ri:ri+eh, rj:rj+ew] = np.maximum(
                    predictions[ri:ri+eh, rj:rj+ew], pred[:eh, :ew]
                )

        batch_tiles.clear()
        batch_coords.clear()

    for (ri, rj) in tqdm(coords, total=len(coords), desc="Predicting tiles"):
        tile = _extract_tile(image_hwc, ri, rj, tile_size, padding_mode)
        # Per-tile application of the (global, training-set) mean/std
        tile = normalize_input_mean_std(tile, mean, std)
        t    = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float()
        batch_tiles.append(t)
        batch_coords.append((ri, rj))
        if len(batch_tiles) == batch_size:
            _flush()

    if batch_tiles:
        _flush()

    if method == "weighted":
        weight_accum = np.where(weight_accum == 0, 1.0, weight_accum)
        predictions  = (prob_accum / weight_accum > threshold).astype(np.uint8)

    return predictions


# ---------------------------------------------------------------------------
# Save output GeoTIFF
# ---------------------------------------------------------------------------

def save_prediction(
    predictions: np.ndarray,
    metadata:    Dict,
    output_path: str,
) -> None:
    """Write a binary (0/1) prediction as a single-band uint8 GeoTIFF."""
    meta = metadata.copy()
    meta.update({"driver": "GTiff", "dtype": "uint8", "count": 1, "compress": "lzw"})
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(predictions.astype(np.uint8), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tiled kelp segmentation inference on a directory of scenes."
    )
    parser.add_argument("--config", required=True, help="Path to infer_config.yaml")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    infer_cfg = cfg["inference"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    tile_cfg  = cfg.get("tiling", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Channel configuration (must match training exactly) ---------------
    band_names  = data_cfg.get("band_names", [
        "blue", "green", "red", "nir", "rededge",
        "substrate", "bathymetry", "slope",
    ])
    vi_names    = data_cfg.get("vi_names", [
        "ndvi", "ndwi", "gndvi", "clgreen", "ndvire",
    ])
    substrate_onehot = data_cfg.get("substrate_onehot", False)
    in_channels = resolve_in_channels(cfg)

    print(f"Band names       : {band_names}")
    print(f"VI names         : {vi_names}")
    print(f"substrate_onehot : {substrate_onehot}")
    print(f"in_channels      : {in_channels}")

    # ---- Normalization stats -----------------------------------------------
    mean = np.array(data_cfg["mean"], dtype=np.float32)
    std  = np.array(data_cfg["std"],  dtype=np.float32)
    if len(mean) == 0 or len(std) == 0:
        raise ValueError(
            "data.mean and data.std must be set in infer_config.yaml.\n"
            "Copy the values printed by train.py during mean/std computation."
        )
    if len(mean) != in_channels:
        raise ValueError(
            f"data.mean has {len(mean)} values but in_channels = {in_channels}. "
            "Ensure mean/std match the band_names + vi_names (+ substrate_onehot) "
            "configuration used during training."
        )

    # ---- Load model --------------------------------------------------------
    ckpt_path = model_cfg["checkpoint"]
    model = KelpSegmentationModel.load_from_checkpoint(
        ckpt_path,
        arch=model_cfg.get("architecture",    "Unet"),
        encoder_name=model_cfg.get("encoder", "tu-maxvit_tiny_tf_512"),
        in_channels=in_channels,
        out_classes=model_cfg.get("out_classes", 1),
        encoder_weights=model_cfg.get("encoder_weights", None),
        strict=False,
    )
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}\n")

    # ---- Discover scene folders --------------------------------------------
    input_dir     = infer_cfg["input_dir"]
    output_suffix = infer_cfg.get("output_suffix", "_kelp_pred.tif")
    subfolders    = sorted([f.path for f in os.scandir(input_dir) if f.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"No sub-folders found in {input_dir!r}")
    print(f"Found {len(subfolders)} scene folder(s) to process.\n")

    # ---- Tiling parameters -------------------------------------------------
    tile_size    = tile_cfg.get("tile_size",    512)
    overlap      = tile_cfg.get("overlap",      0.70)
    halo_size    = tile_cfg.get("halo_size",    64)
    method       = tile_cfg.get("method",       "weighted")
    padding_mode = tile_cfg.get("padding_mode", "reflect")
    batch_size   = tile_cfg.get("batch_size",   8)
    threshold    = cfg.get("threshold",         0.5)

    # ---- Process each scene ------------------------------------------------
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        output_path = os.path.join(folder, folder_name + output_suffix)

        print(f"Processing : {folder_name}")
        print(f"  Output   : {output_path}")

        try:
            image_hwc, metadata = load_scene(
                folder, band_names, vi_names,
                substrate_onehot=substrate_onehot,
            )
            print(f"  Scene shape (H, W, C): {image_hwc.shape}")

            predictions = run_inference(
                image_hwc, model, device, mean, std,
                tile_size=tile_size,
                overlap=overlap,
                halo_size=halo_size,
                method=method,
                padding_mode=padding_mode,
                batch_size=batch_size,
                threshold=threshold,
            )
            save_prediction(predictions, metadata, output_path)
            print(f"  Saved    → {output_path}\n")

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            print("  Continuing with next folder …\n")
            continue

    print("All scenes processed.")


if __name__ == "__main__":
    main()
