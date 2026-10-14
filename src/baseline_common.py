"""
src/baseline_common.py
-----------------------
Shared utilities for the traditional (non-deep-learning) baselines used to
answer Reviewer #1 Comment 1 and Reviewer #2 Comment 2: a quantitative
comparison against classic kelp-mapping methods (NDVI thresholding, Random
Forest).

Design notes
------------
- Baselines are evaluated on the *exact same* LOO folds (``split_loo``) and
  the *exact same* kelp-containing 512x512 tiles used to produce the DL
  model's Table 2 results, so the resulting metrics are directly comparable
  and can be dropped into the same table.
- Training pixels are class-ratio-controlled *per tile*: all kelp pixels
  plus ``imbalance_ratio`` times as many randomly sampled no-kelp pixels
  from that same tile (default ratio 1.0 = full 1:1 balancing). This mirrors
  how traditional pixel-based classifiers (RF, decision trees,
  maximum-likelihood) are typically trained in the kelp-mapping literature
  on hand-picked ROIs, and keeps peak memory bounded (we never materialise
  the full imbalanced pixel pool for a fold at once).
- Test pixels are NOT balanced by default: every (remaining, i.e. non-land
  when land masking is enabled) pixel in every held-out test tile is used,
  so IoU / Precision / Recall / F1 are computed under the natural, highly
  imbalanced class distribution. A separate, optional balanced-subsample
  evaluation path exists for OA / kelp-class accuracy / no-kelp-class
  accuracy -- see ``evaluate_test_tiles_balanced``.
- Metrics use the dataset-level TP/FP/FN/TN aggregation defined in the paper
  (Eqs. 1-4 for IoU/Precision/Recall/F1), aggregated per held-out scene; the
  "Mean" row is the simple arithmetic mean of the 10 per-scene values,
  matching Table 2's convention exactly (verified against the reported
  0.5648 mean IoU).

Land masking
------------
Traditional spectral-index / pixel classifiers (unlike the DL model, which
sees full scene context) can confuse terrestrial vegetation for kelp, since
NDVI is high for both. Land masking here is applied *before* both training
and prediction: land pixels are excluded from the balanced training set and
from the test pixels entirely, so the classifier never sees them and they
never enter the confusion-matrix counts. Two land-mask sources are
supported:

1. Default (``apply_land_mask=True``, ``land_mask_dir=None``): use the
   scene's own bathymetry band. Per the paper (Section 2.2.1), bathymetry
   values represent depth (negative) or above-water elevation (positive)
   relative to mean sea level; pixels with bathymetry > 0 are treated as
   land.
2. External mask (``land_mask_dir=<path>``): supply your own land mask
   covering the study area, as either:
     - a single-band raster (any GDAL format, e.g. GeoTIFF) -- reprojected
       and resampled (nearest) onto each tile's exact grid; a pixel is
       land if its value is nonzero (or equals ``land_value``, if given).
     - a vector polygon layer (.shp / .gpkg / .geojson) of land area --
       rasterized onto each tile's grid; a pixel is land if it falls
       inside any polygon.

GeoTIFF export
---------------
``evaluate_test_tiles`` / ``evaluate_test_tiles_balanced`` optionally write
the predicted map for every test tile as a single-band GeoTIFF (same
georeferencing as the original mask file), organised as
``<save_dir>/<scene_folder_name>/<tile_filename>.tif`` -- mirroring the
original ``images``/``masks`` folder layout.

Segmentation-based post-processing
-----------------------------------
Both ``evaluate_test_tiles`` and ``evaluate_test_tiles_balanced`` accept an
optional post-processing step: SLIC superpixel segmentation of the tile,
followed by a majority vote within each segment over the raw per-pixel
predictions -- every pixel in a segment is relabeled to whichever class
(kelp / no-kelp) was in the majority among that segment's raw predictions.
SLIC (Simple Linear Iterative Clustering; Achanta et al., 2012) is a
standard, widely used superpixel algorithm for exactly this kind of
object-based post-processing of pixel-wise land-cover classifications in
remote sensing, including specifically as a refinement step for Random
Forest classifications. See ``segment_majority_vote_refine``.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.dataset import ALL_BAND_INDICES, SatelliteDataset

# Satellite-only channel configuration: the best-performing configuration
# identified by the ablation study (Table 1) -- no bathymetry, no substrate.
DEFAULT_BAND_NAMES: List[str] = ["blue", "green", "red", "nir", "rededge"]
DEFAULT_VI_NAMES:   List[str] = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]

VECTOR_EXTENSIONS = (".shp", ".gpkg", ".geojson", ".json")


def channel_index(
    name: str,
    band_names: List[str] = DEFAULT_BAND_NAMES,
    vi_names:   List[str] = DEFAULT_VI_NAMES,
) -> int:
    """
    Return the 0-based channel index of ``name`` in the stacked
    (raw bands, then vegetation indices) tensor produced by SatelliteDataset.
    """
    if name in band_names:
        return band_names.index(name)
    if name in vi_names:
        return len(band_names) + vi_names.index(name)
    raise ValueError(f"Unknown channel: {name!r}")


def _tile_pixels(dataset: SatelliteDataset, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a flattened (n_pixels, n_channels) feature array and a
    (n_pixels,) binary label array for a single tile.
    """
    image, mask = dataset[idx]                   # image: (C,H,W); mask: (1,H,W)
    image = np.asarray(image)
    mask = np.asarray(mask)
    c, h, w = image.shape
    x = image.reshape(c, h * w).T                 # (H*W, C)
    y = mask.reshape(h * w).astype(np.int64)       # (H*W,)
    return x, y


# ---------------------------------------------------------------------------
# Land masking (applied before both training and prediction)
# ---------------------------------------------------------------------------

def load_bathymetry_land_mask(image_path: str, land_threshold: float = 0.0) -> np.ndarray:
    """
    Return a (H, W) boolean land mask (True = land, to be excluded) derived
    directly from the tile's own bathymetry band: land = bathymetry >
    land_threshold (above mean sea level).
    """
    with rasterio.open(image_path) as src:
        bathy = src.read(ALL_BAND_INDICES["bathymetry"]).astype(np.float32)
    return bathy > land_threshold


def load_external_land_mask(
    image_path: str,
    land_mask_path: str,
    land_value: Optional[float] = None,
) -> np.ndarray:
    """
    Return a (H, W) boolean land mask (True = land) built by reprojecting
    (raster) or rasterizing (vector) an external land-mask source onto a
    single tile's exact grid (transform, CRS, shape).

    See the module docstring for supported formats.
    """
    with rasterio.open(image_path) as src:
        tile_transform = src.transform
        tile_crs = src.crs
        h, w = src.height, src.width

    ext = os.path.splitext(land_mask_path)[1].lower()

    if ext in VECTOR_EXTENSIONS:
        try:
            import geopandas as gpd
        except ImportError as e:
            raise ImportError(
                "Reading a vector land mask (.shp/.gpkg/.geojson) requires "
                "geopandas: pip install geopandas"
            ) from e
        from rasterio.features import rasterize

        gdf = gpd.read_file(land_mask_path)
        if gdf.crs is not None and tile_crs is not None and gdf.crs != tile_crs:
            gdf = gdf.to_crs(tile_crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            return np.zeros((h, w), dtype=bool)
        land = rasterize(shapes, out_shape=(h, w), transform=tile_transform,
                          fill=0, dtype="uint8")
        return land.astype(bool)

    # Raster land mask: reproject/resample onto the tile's grid.
    from rasterio.warp import Resampling, reproject

    with rasterio.open(land_mask_path) as mask_src:
        dst = np.zeros((h, w), dtype=mask_src.dtypes[0])
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=dst,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=tile_transform,
            dst_crs=tile_crs,
            resampling=Resampling.nearest,
        )
    if land_value is not None:
        return dst == land_value
    return dst != 0


def get_land_mask_for_tile(
    image_path: str,
    apply_land_mask: bool,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Return the (H, W) boolean land mask for a tile, or None if
    ``apply_land_mask`` is False. Uses ``land_mask_dir`` (external raster or
    vector source) if given, otherwise falls back to the tile's own
    bathymetry band (bathymetry > 0 => land).
    """
    if not apply_land_mask:
        return None
    if land_mask_dir is not None:
        return load_external_land_mask(image_path, land_mask_dir, land_value=land_value)
    return load_bathymetry_land_mask(image_path)


# ---------------------------------------------------------------------------
# Shared balanced-subsampling logic (used by training AND, separately, by
# evaluate_test_tiles_balanced for a matched test-time subsample)
# ---------------------------------------------------------------------------

def _balanced_subsample_indices(
    y: np.ndarray,
    imbalance_ratio: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Return indices into ``y`` selecting all kelp (y==1) pixels plus
    ``imbalance_ratio`` times as many randomly sampled no-kelp (y==0)
    pixels, capped at however many no-kelp pixels are actually available.
    Returns ``None`` if there are no kelp pixels at all (nothing to anchor
    the sample to).
    """
    kelp_idx = np.flatnonzero(y == 1)
    if len(kelp_idx) == 0:
        return None

    nokelp_idx = np.flatnonzero(y == 0)
    n_nokelp_target = int(round(imbalance_ratio * len(kelp_idx)))
    if len(nokelp_idx) > n_nokelp_target:
        nokelp_idx = rng.choice(nokelp_idx, size=n_nokelp_target, replace=False)
    # else: keep every available no-kelp pixel (can't reach the target ratio)

    return np.concatenate([kelp_idx, nokelp_idx])


def load_balanced_training_pixels(
    image_paths: List[str],
    mask_paths:  List[str],
    band_names:  List[str] = DEFAULT_BAND_NAMES,
    vi_names:    List[str] = DEFAULT_VI_NAMES,
    seed: int = 42,
    apply_land_mask: bool = False,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
    imbalance_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a class-ratio-controlled pixel training set.

    For every training tile, land pixels are excluded first (if
    ``apply_land_mask``), then all remaining kelp pixels are kept and
    ``imbalance_ratio`` times as many no-kelp pixels are randomly sampled
    *from that same tile* (capped at however many no-kelp pixels the tile
    actually has). Concatenated across all tiles this yields a globally
    ~(1 : imbalance_ratio) kelp:no-kelp dataset.

    Returns
    -------
    X : (N, C) float32 array
    y : (N,) int64 array of {0, 1}
    """
    rng = np.random.default_rng(seed)
    ds = SatelliteDataset(
        image_paths, mask_paths,
        classes=["kelp"],
        band_names=band_names,
        vi_names=vi_names,
    )

    x_parts, y_parts = [], []
    for i in range(len(ds)):
        x, y = _tile_pixels(ds, i)

        land_mask = get_land_mask_for_tile(
            ds.image_paths[i], apply_land_mask, land_mask_dir, land_value,
        )
        if land_mask is not None:
            keep_water = ~land_mask.reshape(-1)
            x, y = x[keep_water], y[keep_water]

        keep = _balanced_subsample_indices(y, imbalance_ratio, rng)
        if keep is None:
            continue  # no kelp pixels left in this tile (after masking); skip

        x_parts.append(x[keep])
        y_parts.append(y[keep])

    if not x_parts:
        raise RuntimeError("No kelp pixels found in any training tile (after land masking).")

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def load_all_test_pixels(
    image_paths: List[str],
    mask_paths:  List[str],
    band_names:  List[str] = DEFAULT_BAND_NAMES,
    vi_names:    List[str] = DEFAULT_VI_NAMES,
    apply_land_mask: bool = False,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load every pixel from a list of test tiles, with land pixels excluded
    first if ``apply_land_mask`` is True (naturally imbalanced otherwise).
    """
    ds = SatelliteDataset(
        image_paths, mask_paths,
        classes=["kelp"],
        band_names=band_names,
        vi_names=vi_names,
    )
    x_parts, y_parts = [], []
    for i in range(len(ds)):
        x, y = _tile_pixels(ds, i)

        land_mask = get_land_mask_for_tile(
            ds.image_paths[i], apply_land_mask, land_mask_dir, land_value,
        )
        if land_mask is not None:
            keep_water = ~land_mask.reshape(-1)
            x, y = x[keep_water], y[keep_water]

        x_parts.append(x)
        y_parts.append(y)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Threshold tuning (shared by NDVI thresholding and RF's decision threshold)
# ---------------------------------------------------------------------------

def find_best_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """
    Return the threshold in ``thresholds`` that maximises kelp-class F1 for
    the rule ``scores > threshold``. ``scores``/``y_true`` should come from
    a held-out set with the *natural* (imbalanced) class distribution.
    """
    best_thr, best_f1 = float(thresholds[0]), -1.0
    for thr in thresholds:
        pred = (scores > thr).astype(np.int64)
        tp, fp, fn, tn = confusion_counts(y_true, pred)
        f1 = metrics_from_counts(tp, fp, fn, tn)["f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# Segmentation-based post-processing (SLIC + majority vote)
# ---------------------------------------------------------------------------

def segment_majority_vote_refine(
    image_hwc: np.ndarray,
    raw_pred_flat: np.ndarray,
    valid_flat: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Refine a raw per-pixel binary prediction map using SLIC superpixel
    segmentation followed by majority voting within each segment.

    The tile is segmented into ``n_segments`` superpixels using SLIC
    (Achanta et al., 2012) over all channels of ``image_hwc`` (each channel
    min-max normalised first, so bands and spectral indices -- which have
    very different native numeric ranges -- contribute comparably to the
    segmentation distance metric, rather than the segmentation being
    dominated by whichever channel happens to have the largest raw values).
    Segment membership is independent of predicted class: every segment,
    regardless of its own internal spectral characteristics, then has every
    one of its ``valid_flat`` pixels relabeled to whichever class (kelp /
    no-kelp) was in the majority among ``raw_pred_flat`` within that segment
    (ties go to no-kelp). Pixels outside ``valid_flat`` (e.g. land pixels
    that were never predicted on) are left untouched.

    Parameters
    ----------
    image_hwc : (H, W, C) array
        The tile's full multi-channel image (same channel set used for
        classification), used only to drive the segmentation -- not
        related to the predicted class.
    raw_pred_flat : (H*W,) array of {0, 1}
        The classifier's raw, pre-refinement prediction for every pixel
        (row-major flattened, matching ``image_hwc.reshape(-1, C)`` order).
        Values at positions where ``valid_flat`` is False are ignored for
        voting purposes and left unchanged in the output.
    valid_flat : (H*W,) boolean array
        True at pixels that have a real prediction to vote with (e.g. water
        pixels, if land masking is enabled) and that should be relabeled.

    Returns
    -------
    (H*W,) array of {0, 1}: the refined predictions.
    """
    try:
        from skimage.segmentation import slic
    except ImportError as e:
        raise ImportError(
            "Segmentation post-processing requires scikit-image: "
            "pip install scikit-image"
        ) from e

    h, w, c = image_hwc.shape

    # Per-channel min-max normalization so no single band/index dominates
    # the SLIC distance metric (reflectance bands and spectral indices have
    # very different native scales).
    img_norm = np.empty_like(image_hwc, dtype=np.float64)
    for ch in range(c):
        band = image_hwc[:, :, ch].astype(np.float64)
        band_min = float(np.nanmin(band))
        band_max = float(np.nanmax(band))
        if band_max > band_min:
            img_norm[:, :, ch] = (band - band_min) / (band_max - band_min)
        else:
            img_norm[:, :, ch] = 0.0

    segments = slic(
        img_norm,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=-1,
        start_label=0,
    )
    seg_ids = segments.reshape(-1)

    seg_ids_valid = seg_ids[valid_flat]
    preds_valid = raw_pred_flat[valid_flat]

    refined = raw_pred_flat.copy()
    if len(seg_ids_valid) == 0:
        return refined  # nothing valid to vote on in this tile

    n_seg = int(seg_ids.max()) + 1
    kelp_counts = np.bincount(
        seg_ids_valid, weights=(preds_valid == 1).astype(np.float64), minlength=n_seg,
    )
    total_counts = np.bincount(seg_ids_valid, minlength=n_seg)
    nokelp_counts = total_counts - kelp_counts
    winning_label = (kelp_counts > nokelp_counts).astype(np.int64)  # ties -> no-kelp (0)

    refined[valid_flat] = winning_label[seg_ids_valid]
    return refined


# ---------------------------------------------------------------------------
# Per-tile evaluation with optional GeoTIFF export and segmentation
# post-processing of the predicted maps
# ---------------------------------------------------------------------------

def _scene_name_from_path(path: str) -> str:
    """
    Recover the original scene folder name from a tile path of the form
    '.../<scene_folder>/masks/<tile>.tif' or '.../<scene_folder>/images/<tile>.tif'.
    """
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def evaluate_test_tiles(
    image_paths: List[str],
    mask_paths:  List[str],
    predict_fn:  Callable[[np.ndarray], np.ndarray],
    band_names:  List[str] = DEFAULT_BAND_NAMES,
    vi_names:    List[str] = DEFAULT_VI_NAMES,
    apply_land_mask: bool = False,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
    save_dir: Optional[str] = None,
    apply_segmentation: bool = False,
    segmentation_n_segments: int = 300,
    segmentation_compactness: float = 10.0,
) -> Tuple[int, int, int, int]:
    """
    Evaluate a trained classifier tile-by-tile on the held-out test tiles,
    matching the masking-first protocol used for training: land pixels (if
    ``apply_land_mask``) are excluded before prediction and therefore never
    contribute to the confusion-matrix counts.

    ``predict_fn`` takes a (n_pixels, n_channels) feature array (already
    land-filtered) and returns a (n_pixels,) array of {0, 1} predictions.

    If ``apply_segmentation`` is True, the raw per-pixel predictions for
    every (non-land) pixel in the tile are refined via SLIC superpixel
    segmentation + majority voting (``segment_majority_vote_refine``)
    before metrics/export -- see the module docstring.

    If ``save_dir`` is given, the (possibly segmentation-refined) predicted
    map for every test tile is written as a single-band GeoTIFF to
    ``save_dir/<scene_folder_name>/<mask_filename>``. Land pixels are
    written as 0.

    Returns
    -------
    (tp, fp, fn, tn) aggregated across all given tiles.
    """
    ds = SatelliteDataset(
        image_paths, mask_paths,
        classes=["kelp"],
        band_names=band_names,
        vi_names=vi_names,
    )

    tp_total = fp_total = fn_total = tn_total = 0
    for i in range(len(ds)):
        x, y = _tile_pixels(ds, i)
        image_path = ds.image_paths[i]
        mask_path = mask_paths[i]

        with rasterio.open(mask_path) as mask_src:
            h, w = mask_src.height, mask_src.width
            out_profile = mask_src.profile.copy()

        land_mask = get_land_mask_for_tile(image_path, apply_land_mask, land_mask_dir, land_value)
        if land_mask is not None:
            keep_water = ~land_mask.reshape(-1)
        else:
            keep_water = np.ones(h * w, dtype=bool)

        x_kept = x[keep_water]
        if len(x_kept):
            y_pred_kept_raw = np.asarray(predict_fn(x_kept)).astype(np.int64)
        else:
            y_pred_kept_raw = np.array([], dtype=np.int64)

        if apply_segmentation:
            full_raw_pred = np.zeros(h * w, dtype=np.int64)
            full_raw_pred[keep_water] = y_pred_kept_raw
            img_hwc = x.reshape(h, w, x.shape[1])
            refined_full_pred = segment_majority_vote_refine(
                img_hwc, full_raw_pred, keep_water,
                n_segments=segmentation_n_segments, compactness=segmentation_compactness,
            )
            y_pred_kept = refined_full_pred[keep_water]
        else:
            y_pred_kept = y_pred_kept_raw

        y_kept = y[keep_water]
        tp, fp, fn, tn = confusion_counts(y_kept, y_pred_kept)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        if save_dir is not None:
            full_pred = np.zeros(h * w, dtype=np.uint8)
            full_pred[keep_water] = y_pred_kept.astype(np.uint8)
            full_pred = full_pred.reshape(h, w)

            scene_name = _scene_name_from_path(mask_path)
            out_dir = os.path.join(save_dir, scene_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(mask_path))

            out_profile.update(dtype="uint8", count=1)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(full_pred[np.newaxis, ...])

    return tp_total, fp_total, fn_total, tn_total


def evaluate_test_tiles_balanced(
    image_paths: List[str],
    mask_paths:  List[str],
    predict_fn:  Callable[[np.ndarray], np.ndarray],
    band_names:  List[str] = DEFAULT_BAND_NAMES,
    vi_names:    List[str] = DEFAULT_VI_NAMES,
    apply_land_mask: bool = False,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
    imbalance_ratio: float = 1.0,
    seed: int = 42,
    save_dir: Optional[str] = None,
    save_selection_mask_dir: Optional[str] = None,
    apply_segmentation: bool = False,
    segmentation_n_segments: int = 300,
    segmentation_compactness: float = 10.0,
) -> Tuple[int, int, int, int]:
    """
    Like ``evaluate_test_tiles``, but computes metrics on a class-balanced
    SUBSAMPLE of each test tile's (land-masked) pixels, rather than on every
    pixel: all kelp pixels are kept, and ``imbalance_ratio`` times as many
    no-kelp pixels are randomly sampled from the same tile.

    If ``apply_segmentation`` is True, predictions are computed for EVERY
    water pixel in the tile (not just the balanced subsample), refined via
    SLIC + majority voting over the full tile, and only then subsampled
    down to the balanced set for metric computation -- segmentation needs
    the full spatial neighbourhood to be meaningful, so this costs more
    compute than the unsegmented path (which only predicts on the
    subsampled pixels).

    If ``save_selection_mask_dir`` is given, a boolean selection mask (1 =
    pixel included in the balanced subsample, 0 = not) is written per test
    tile as a single-band GeoTIFF to
    ``save_selection_mask_dir/<scene_folder_name>/<mask_filename>``.

    Returns
    -------
    (tp, fp, fn, tn) aggregated across the balanced subsample of all given
    tiles.
    """
    ds = SatelliteDataset(
        image_paths, mask_paths,
        classes=["kelp"],
        band_names=band_names,
        vi_names=vi_names,
    )
    rng = np.random.default_rng(seed)

    tp_total = fp_total = fn_total = tn_total = 0
    for i in range(len(ds)):
        x, y = _tile_pixels(ds, i)
        image_path = ds.image_paths[i]
        mask_path = mask_paths[i]

        with rasterio.open(mask_path) as mask_src:
            h, w = mask_src.height, mask_src.width
            out_profile = mask_src.profile.copy()

        land_mask = get_land_mask_for_tile(image_path, apply_land_mask, land_mask_dir, land_value)
        if land_mask is not None:
            keep_water = ~land_mask.reshape(-1)
        else:
            keep_water = np.ones(h * w, dtype=bool)

        water_positions = np.flatnonzero(keep_water)
        x_water, y_water = x[keep_water], y[keep_water]

        if apply_segmentation:
            y_pred_water_raw = (
                np.asarray(predict_fn(x_water)).astype(np.int64) if len(x_water)
                else np.array([], dtype=np.int64)
            )
            full_raw_pred = np.zeros(h * w, dtype=np.int64)
            full_raw_pred[keep_water] = y_pred_water_raw
            img_hwc = x.reshape(h, w, x.shape[1])
            refined_full_pred = segment_majority_vote_refine(
                img_hwc, full_raw_pred, keep_water,
                n_segments=segmentation_n_segments, compactness=segmentation_compactness,
            )
            y_pred_water = refined_full_pred[keep_water]
        else:
            y_pred_water = None  # computed lazily below, only for the sampled subset

        sel_idx = _balanced_subsample_indices(y_water, imbalance_ratio, rng)
        if sel_idx is None:
            continue  # no kelp pixels in this tile after masking; nothing to sample

        y_sel = y_water[sel_idx]
        if apply_segmentation:
            y_pred_sel = y_pred_water[sel_idx]
        else:
            x_sel = x_water[sel_idx]
            y_pred_sel = np.asarray(predict_fn(x_sel)).astype(np.int64)

        tp, fp, fn, tn = confusion_counts(y_sel, y_pred_sel)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        selected_full_positions = water_positions[sel_idx]
        scene_name = _scene_name_from_path(mask_path)

        if save_selection_mask_dir is not None:
            sel_mask = np.zeros(h * w, dtype=np.uint8)
            sel_mask[selected_full_positions] = 1
            sel_mask = sel_mask.reshape(h, w)

            out_dir = os.path.join(save_selection_mask_dir, scene_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(mask_path))

            sel_profile = out_profile.copy()
            sel_profile.update(dtype="uint8", count=1)
            with rasterio.open(out_path, "w", **sel_profile) as dst:
                dst.write(sel_mask[np.newaxis, ...])

        if save_dir is not None:
            full_pred = np.zeros(h * w, dtype=np.uint8)
            full_pred[selected_full_positions] = y_pred_sel.astype(np.uint8)
            full_pred = full_pred.reshape(h, w)

            out_dir = os.path.join(save_dir, scene_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(mask_path))

            pred_profile = out_profile.copy()
            pred_profile.update(dtype="uint8", count=1)
            with rasterio.open(out_path, "w", **pred_profile) as dst:
                dst.write(full_pred[np.newaxis, ...])

    return tp_total, fp_total, fn_total, tn_total


def load_selection_mask(path: str) -> np.ndarray:
    """
    Read a selection-mask GeoTIFF written by ``evaluate_test_tiles_balanced``
    and return it as a (H, W) boolean array (True = pixel was included in
    the balanced subsample).
    """
    with rasterio.open(path) as src:
        return src.read(1).astype(bool)


def evaluate_saved_predictions_balanced(
    predictions_dir: str,
    data_root: str,
    apply_land_mask: bool = False,
    land_mask_dir: Optional[str] = None,
    land_value: Optional[float] = None,
    imbalance_ratio: float = 1.0,
    seed: int = 42,
    save_selection_mask_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Compute balanced-subsample TP/FP/FN/TN, per scene, from predictions
    ALREADY SAVED to disk (e.g. written earlier by ``evaluate_test_tiles``
    via ``--save-geotiff-masks``), without re-running the classifier or
    re-predicting anything.

    Returns
    -------
    Dict mapping scene folder name -> (tp, fp, fn, tn) for that scene's
    balanced subsample.
    """
    rng = np.random.default_rng(seed)

    scene_dirs = {
        os.path.basename(p.rstrip("/\\")): p
        for p in glob.glob(data_root) if os.path.isdir(p)
    }

    pred_scene_dirs = sorted(
        d for d in glob.glob(os.path.join(predictions_dir, "*")) if os.path.isdir(d)
    )

    results: Dict[str, Tuple[int, int, int, int]] = {}

    for pred_scene_dir in pred_scene_dirs:
        scene_name = os.path.basename(pred_scene_dir.rstrip("/\\"))
        if scene_name not in scene_dirs:
            if verbose:
                print(f"[WARNING] No matching original scene folder found for "
                      f"'{scene_name}' under data_root={data_root!r}; skipping.")
            continue

        orig_scene_dir = scene_dirs[scene_name]
        masks_dir = os.path.join(orig_scene_dir, "masks")
        images_dir = os.path.join(orig_scene_dir, "images")

        tp_scene = fp_scene = fn_scene = tn_scene = 0
        n_tiles_evaluated = 0

        pred_tile_paths = sorted(glob.glob(os.path.join(pred_scene_dir, "*.tif")))
        for pred_path in pred_tile_paths:
            tile_name = os.path.basename(pred_path)
            mask_path = os.path.join(masks_dir, tile_name)
            image_path = os.path.join(images_dir, tile_name)

            if not os.path.isfile(mask_path):
                if verbose:
                    print(f"[WARNING] No matching ground-truth mask for "
                          f"'{scene_name}/{tile_name}'; skipping.")
                continue
            if apply_land_mask and land_mask_dir is None and not os.path.isfile(image_path):
                if verbose:
                    print(f"[WARNING] No matching image tile (needed for "
                          f"bathymetry-based land masking) for "
                          f"'{scene_name}/{tile_name}'; skipping.")
                continue

            with rasterio.open(mask_path) as mask_src:
                h, w = mask_src.height, mask_src.width
                y = mask_src.read(1).reshape(-1).astype(np.int64)
                y = (y > 0).astype(np.int64)

            with rasterio.open(pred_path) as pred_src:
                y_pred = pred_src.read(1).reshape(-1).astype(np.int64)

            if y_pred.shape[0] != y.shape[0]:
                if verbose:
                    print(f"[WARNING] Shape mismatch between prediction and "
                          f"mask for '{scene_name}/{tile_name}'; skipping.")
                continue

            land_mask = get_land_mask_for_tile(image_path, apply_land_mask, land_mask_dir, land_value)
            if land_mask is not None:
                keep_water = ~land_mask.reshape(-1)
            else:
                keep_water = np.ones(h * w, dtype=bool)

            water_positions = np.flatnonzero(keep_water)
            y_water = y[keep_water]
            y_pred_water = y_pred[keep_water]

            sel_idx = _balanced_subsample_indices(y_water, imbalance_ratio, rng)
            if sel_idx is None:
                continue

            y_sel = y_water[sel_idx]
            y_pred_sel = y_pred_water[sel_idx]

            tp, fp, fn, tn = confusion_counts(y_sel, y_pred_sel)
            tp_scene += tp
            fp_scene += fp
            fn_scene += fn
            tn_scene += tn
            n_tiles_evaluated += 1

            if save_selection_mask_dir is not None:
                selected_full_positions = water_positions[sel_idx]
                sel_mask = np.zeros(h * w, dtype=np.uint8)
                sel_mask[selected_full_positions] = 1
                sel_mask = sel_mask.reshape(h, w)

                out_dir = os.path.join(save_selection_mask_dir, scene_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, tile_name)

                with rasterio.open(mask_path) as mask_src:
                    sel_profile = mask_src.profile.copy()
                sel_profile.update(dtype="uint8", count=1)
                with rasterio.open(out_path, "w", **sel_profile) as dst:
                    dst.write(sel_mask[np.newaxis, ...])

        if verbose:
            print(f"  [{scene_name}] evaluated {n_tiles_evaluated} tiles")

        if n_tiles_evaluated > 0:
            results[scene_name] = (tp_scene, fp_scene, fn_scene, tn_scene)
        elif verbose:
            print(f"[WARNING] No usable tiles found for scene '{scene_name}'; "
                  f"omitted from results.")

    return results


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for the kelp (positive) class."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float]:
    """
    Precision / Recall / IoU / F1 exactly as defined in Eqs. 1-4 of the
    paper (kelp class only), plus overall accuracy and per-class accuracy.

    - ``accuracy``: overall pixel accuracy, (tp+tn)/(tp+fp+fn+tn).
    - ``kelp_accuracy``: kelp-class accuracy, tp/(tp+fn) -- identical to
      recall (the producer's accuracy for the kelp class).
    - ``nokelp_accuracy``: no-kelp-class accuracy, tn/(tn+fp) (specificity).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy        = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    kelp_accuracy   = recall
    nokelp_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "kelp_accuracy": kelp_accuracy,
        "nokelp_accuracy": nokelp_accuracy,
    }
