"""
src/dataset.py
--------------
PyTorch Dataset for kelp / no-kelp semantic segmentation on Sentinel-2 imagery,
plus all data-split helpers (geographic folder-based, random tile-based, and LOO).

CHANGES vs. the original (for the manuscript revision):
  - `resolve_num_channels()` — single source of truth for in_channels, used by
    both SatelliteDataset and the train.py / inference.py CLIs, so the
    substrate-one-hot channel count can't drift out of sync between them.
  - `SatelliteDataset(..., substrate_onehot=True)` — replaces the scalar
    substrate channel with a K-class one-hot encoding (see class docstring).
  - `split_loo(..., val_mode="scene", val_indices=[...])` — optional
    scene-disjoint (leak-free) train/val split, alongside the original
    tile-level shuffle ("val_mode='tile'", still the default, unchanged
    behaviour / unchanged results for anything not opting in).
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as BaseDataset


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def normalize_input_mean_std(
    image_hwc: np.ndarray,
    mean_per_channel: Sequence[float],
    std_per_channel: Sequence[float],
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Apply per-channel mean/std normalization to an (H, W, C) array."""
    image_hwc = np.nan_to_num(image_hwc).astype(np.float32)
    mean = np.array(mean_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    std  = np.array(std_per_channel,  dtype=np.float32)[np.newaxis, np.newaxis, :]
    return (image_hwc - mean) / (std + epsilon)


# ---------------------------------------------------------------------------
# Channel / index definitions
# ---------------------------------------------------------------------------

# Maps a human-readable channel name to its 1-based band index in the TIFF.
# Bands 1-4: Blue, Green, Red, NIR (10 m)
# Band  5:   Red-edge
# Band 11:   Substrate
# Band 12:   Bathymetry
# Band 13:   Slope
ALL_BAND_INDICES: Dict[str, int] = {
    "blue":       1,
    "green":      2,
    "red":        3,
    "nir":        4,
    "rededge":    5,
    "substrate":  11,
    "bathymetry": 12,
    "slope":      13,
}

# Indices of computed vegetation indices inside image_hwc *after* raw bands are
# stacked.  Position depends on which raw bands were requested; these are
# re-computed at runtime based on the loaded band order.
VI_NAMES = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]

# Substrate raster is categorical: RF source uses classes {1,2,3,4} (rock,
# mixed sediment, sand, mud); 0 = no coastal-data coverage (sentinel fill
# value, not a fifth substrate class). One-hot expansion covers only the
# four valid classes — a no-data pixel is the all-zero row, distinguishable
# from every real class by construction.
DEFAULT_SUBSTRATE_CLASSES: Tuple[int, ...] = (1, 2, 3, 4)


def resolve_num_channels(
    band_names: Sequence[str],
    vi_names: Sequence[str],
    substrate_onehot: bool = False,
    n_substrate_classes: int = len(DEFAULT_SUBSTRATE_CLASSES),
) -> int:
    """
    Single source of truth for model in_channels, so train.py / inference.py
    and SatelliteDataset can never disagree on the channel count.

    `band_names` must NOT include "substrate" when substrate_onehot=True
    (the one-hot channels replace it) — see SatelliteDataset docstring.
    """
    n = len(band_names) + len(vi_names)
    if substrate_onehot:
        n += n_substrate_classes
    return n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SatelliteDataset(BaseDataset):
    """
    PyTorch Dataset for kelp segmentation on Sentinel-2 / auxiliary imagery.

    Parameters
    ----------
    image_paths : list of str
        Paths to multi-band GeoTIFF images.
    mask_paths : list of str
        Paths to single-band GeoTIFF masks (0 = background, 1 = kelp).
    classes : list of str
        Subset of ["water", "kelp", "land"] to predict.
    augmentation : callable, optional
        An albumentations Compose pipeline that accepts image= and mask= kwargs.
    mean : array-like, optional
        Per-channel mean for normalization (computed from training set).
    std : array-like, optional
        Per-channel standard deviation for normalization.
    band_names : list of str, optional
        Which raw bands to load from the TIFF, in order.  Defaults to
        ["blue","green","red","nir","rededge","substrate","bathymetry","slope"].
    vi_names : list of str, optional
        Which vegetation indices to append.  Defaults to all five.
    substrate_onehot : bool
        If True, read the substrate band separately from `band_names` (which
        must therefore NOT contain "substrate") and append a K-channel
        one-hot encoding (see `substrate_classes`) at the END of the channel
        stack, after raw bands and after vegetation indices. Default False
        (original scalar-substrate-channel behaviour, unchanged).
    substrate_classes : sequence of int
        Which raw substrate values become one-hot channels, in order. Default
        (1, 2, 3, 4) — the four RF substrate classes. Value 0 (no coastal
        coverage) is intentionally excluded: it becomes the implicit
        all-zero row.

    Channel order in the returned (C, H, W) image, when substrate_onehot is
    used: [raw bands in band_names order] + [VIs in vi_names order] +
    [one-hot substrate channels in substrate_classes order].
    Use `resolve_num_channels()` (module-level) to get the matching in_channels.
    """

    CLASSES = ["water", "kelp", "land"]

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        classes: Optional[List[str]] = None,
        augmentation: Optional[Callable] = None,
        mean: Optional[Sequence[float]] = None,
        std:  Optional[Sequence[float]] = None,
        band_names: Optional[List[str]] = None,
        vi_names:   Optional[List[str]] = None,
        substrate_onehot: bool = False,
        substrate_classes: Sequence[int] = DEFAULT_SUBSTRATE_CLASSES,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augmentation = augmentation
        self.mean = mean
        self.std  = std

        if classes is None:
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(c.lower()) for c in classes]

        # Resolve which raw bands to read (1-based rasterio indices)
        if band_names is None:
            band_names = list(ALL_BAND_INDICES.keys())
        self.band_names = band_names

        self.substrate_onehot  = substrate_onehot
        self.substrate_classes = list(substrate_classes)
        if self.substrate_onehot and "substrate" in self.band_names:
            raise ValueError(
                "substrate_onehot=True expects 'substrate' to be excluded "
                "from band_names — the one-hot channels replace the scalar "
                "substrate band and are appended separately. Remove "
                "'substrate' from band_names in the config."
            )

        self.rasterio_bands = [ALL_BAND_INDICES[b] for b in band_names]
        self._substrate_band_idx = ALL_BAND_INDICES["substrate"]

        # Which VIs to compute
        if vi_names is None:
            vi_names = list(VI_NAMES)
        self.vi_names = vi_names

        # Build look-up: raw band position (0-based) by name
        self._band_pos: Dict[str, int] = {name: i for i, name in enumerate(band_names)}

        # Map VI name → method
        self._vi_calculators: Dict[str, Callable] = {
            "ndvi":    self._calc_ndvi,
            "ndwi":    self._calc_ndwi,
            "gndvi":   self._calc_gndvi,
            "clgreen": self._calc_clgreen,
            "ndvire":  self._calc_ndvire,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # ---- Load raw bands (+ substrate band too, in one read, if onehot)---
        combined_bands = list(self.rasterio_bands)
        if self.substrate_onehot:
            combined_bands = combined_bands + [self._substrate_band_idx]

        with rasterio.open(self.image_paths[index]) as src:
            raw = src.read(combined_bands)               # (C[+1], H, W)
        raw_hwc = np.transpose(raw, (1, 2, 0)).astype(np.float32)  # (H, W, C[+1])

        if self.substrate_onehot:
            image_hwc     = raw_hwc[..., :-1]
            substrate_raw = raw_hwc[..., -1]
        else:
            image_hwc     = raw_hwc
            substrate_raw = None

        # ---- Compute vegetation indices ------------------------------------
        # (self._band_pos indexes only into band_names / image_hwc as built
        #  above, so this is unaffected by whether substrate_onehot is set.)
        indices = []
        for vi_name in self.vi_names:
            calc = self._vi_calculators.get(vi_name)
            if calc is not None:
                indices.append(calc(image_hwc)[..., np.newaxis])
        if indices:
            image_hwc = np.concatenate([image_hwc] + indices, axis=-1)

        # ---- One-hot substrate, appended last ------------------------------
        if substrate_raw is not None:
            onehot = np.stack(
                [(substrate_raw == c).astype(np.float32) for c in self.substrate_classes],
                axis=-1,
            )
            image_hwc = np.concatenate([image_hwc, onehot], axis=-1)

        # ---- Load & process mask -------------------------------------------
        mask = self._read_mask(self.mask_paths[index])

        # ---- Normalize -------------------------------------------------------
        if self.mean is not None and self.std is not None:
            image_hwc = normalize_input_mean_std(image_hwc, self.mean, self.std)

        # ---- Augment --------------------------------------------------------
        if self.augmentation is not None:
            sample   = self.augmentation(image=image_hwc, mask=mask)
            image_hwc = sample["image"]
            mask      = sample["mask"]

        # ---- Final transpose ------------------------------------------------
        image_final = np.transpose(image_hwc, (2, 0, 1))   # (C, H, W)
        mask_final  = np.transpose(mask,      (2, 0, 1))   # (num_classes, H, W)

        return image_final.astype(np.float32), mask_final.astype(np.float32)

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------

    def _read_mask(self, mask_path: str) -> np.ndarray:
        """Return (H, W, num_classes) float mask."""
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(int)
        masks = [(mask == v) for v in self.class_values]
        return np.stack(masks, axis=-1).astype("float")

    # ------------------------------------------------------------------
    # Vegetation index calculators
    # ------------------------------------------------------------------

    def _get(self, image_hwc: np.ndarray, band: str) -> np.ndarray:
        """Return a band slice by name; zeros if band not loaded."""
        pos = self._band_pos.get(band)
        if pos is None:
            return np.zeros(image_hwc.shape[:2], dtype=np.float32)
        return image_hwc[..., pos]

    def _calc_ndvi(self, img: np.ndarray) -> np.ndarray:
        nir, red = self._get(img, "nir"), self._get(img, "red")
        return (nir - red) / (nir + red + 1e-10)

    def _calc_ndwi(self, img: np.ndarray) -> np.ndarray:
        green, nir = self._get(img, "green"), self._get(img, "nir")
        return (green - nir) / (green + nir + 1e-10)

    def _calc_gndvi(self, img: np.ndarray) -> np.ndarray:
        nir, green = self._get(img, "nir"), self._get(img, "green")
        return (nir - green) / (nir + green + 1e-10)

    def _calc_clgreen(self, img: np.ndarray) -> np.ndarray:
        nir, green = self._get(img, "nir"), self._get(img, "green")
        # Clip to avoid exploding values when green ≈ 0 (e.g. scene UYQ)
        return np.where(green < 1e-4, 20.0, nir / (green + 1e-10) - 1)

    def _calc_ndvire(self, img: np.ndarray) -> np.ndarray:
        re, red = self._get(img, "rededge"), self._get(img, "red")
        return (re - red) / (re + red + 1e-10)

    # ------------------------------------------------------------------
    # Channel count helper
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        return resolve_num_channels(
            self.band_names, self.vi_names,
            substrate_onehot=self.substrate_onehot,
            n_substrate_classes=len(self.substrate_classes),
        )


# ---------------------------------------------------------------------------
# Mean / std computation
# ---------------------------------------------------------------------------

def compute_dataset_mean_std(
    dataset: SatelliteDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std over an entire dataset using Welford-style
    online accumulation.  Returns (mean, std) arrays of shape (C,).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    sum_ch   = None
    sum_sq   = None
    n_pixels = 0
    n_ch     = None

    for images, _ in loader:
        imgs = images.numpy()                        # (B, C, H, W)
        B, C, H, W = imgs.shape
        if n_ch is None:
            n_ch   = C
            sum_ch = np.zeros(C, dtype=np.float64)
            sum_sq = np.zeros(C, dtype=np.float64)
        sum_ch += imgs.sum(axis=(0, 2, 3))
        sum_sq += (imgs ** 2).sum(axis=(0, 2, 3))
        n_pixels += B * H * W

    mean = (sum_ch / n_pixels).astype(np.float32)
    std  = np.sqrt(sum_sq / n_pixels - (sum_ch / n_pixels) ** 2).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Data-split strategies
# ---------------------------------------------------------------------------

def get_files_from_folders(folder_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    Collect image and mask file paths from a list of scene folders.

    Each folder must contain sub-directories named ``images/`` and ``masks/``.
    Files are matched positionally (both globs are sorted so they align).
    """
    image_paths: List[str] = []
    mask_paths:  List[str] = []
    for folder in folder_list:
        img_dir  = os.path.join(folder, "images")
        mask_dir = os.path.join(folder, "masks")
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            image_paths.extend(sorted(glob.glob(os.path.join(img_dir,  "*"))))
            mask_paths.extend( sorted(glob.glob(os.path.join(mask_dir, "*"))))
        else:
            print(f"[WARNING] Folder {folder!r} is missing 'images/' or 'masks/' sub-dir.")
    return image_paths, mask_paths


def _check_counts(splits: Dict[str, Tuple[List, List]]) -> None:
    for name, (imgs, masks) in splits.items():
        if len(imgs) != len(masks):
            raise ValueError(
                f"Image/mask count mismatch in '{name}' split: "
                f"{len(imgs)} images vs {len(masks)} masks."
            )


# ---- Strategy 1: geographic folder split (70 / 15 / 15) ------------------

def split_geographic(
    data_root: str,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Shuffle scene folders and split them into train / val / test by folder.

    This preserves geographic independence between splits.

    Parameters
    ----------
    data_root : str
        Glob pattern that expands to the scene folders, e.g.
        ``"/data/Annotated/*"``.
    train_frac, val_frac : float
        Fractions; test receives whatever is left.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Six lists: X_train, y_train, X_val, y_val, X_test, y_test
    """
    folders = sorted([f for f in glob.glob(data_root) if os.path.isdir(f)])
    if not folders:
        raise FileNotFoundError(f"No folders found at {data_root!r}")

    random.seed(seed)
    random.shuffle(folders)

    n = len(folders)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)

    train_folders = folders[:n_train]
    val_folders   = folders[n_train : n_train + n_val]
    test_folders  = folders[n_train + n_val :]

    X_train, y_train = get_files_from_folders(train_folders)
    X_val,   y_val   = get_files_from_folders(val_folders)
    X_test,  y_test  = get_files_from_folders(test_folders)

    _check_counts({"train": (X_train, y_train),
                   "val":   (X_val,   y_val),
                   "test":  (X_test,  y_test)})

    print(f"Geographic split — train: {len(X_train)} "
          f"({len(train_folders)} folders), "
          f"val: {len(X_val)} ({len(val_folders)} folders), "
          f"test: {len(X_test)} ({len(test_folders)} folders)")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---- Strategy 2: random tile-level split ----------------------------------

def split_random(
    data_root:   str,
    val_frac:    float = 0.15,
    test_frac:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Pool all tiles from all folders and split at the tile level using
    sklearn's train_test_split.

    Useful when you want to train on virtually all available data.
    """
    folders = sorted([f for f in glob.glob(data_root) if os.path.isdir(f)])
    if not folders:
        raise FileNotFoundError(f"No folders found at {data_root!r}")

    image_files: List[str] = []
    mask_files:  List[str] = []
    for folder in folders:
        img_dir = os.path.join(folder, "images")
        mask_dir = os.path.join(folder, "masks")

        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            image_files.extend(sorted(glob.glob(os.path.join(img_dir, "*"))))
            mask_files.extend(sorted(glob.glob(os.path.join(mask_dir, "*"))))
        else:
            print(f"[WARNING] Folder {folder!r} missing images/ or masks/; skipping.")

    if len(image_files) != len(mask_files):
        raise ValueError("Global image/mask count mismatch.")

    # First carve off test
    X_rest, X_test, y_rest, y_test = train_test_split(
        image_files, mask_files,
        test_size=test_frac, random_state=seed,
    )
    # Then carve off val from the remainder
    adjusted_val = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=adjusted_val, random_state=seed,
    )

    _check_counts({"train": (X_train, y_train),
                   "val":   (X_val,   y_val),
                   "test":  (X_test,  y_test)})

    print(f"Random split — train: {len(X_train)}, "
          f"val: {len(X_val)}, test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---- Strategy 3: leave-one-(or-few)-out ----------------------------------

def split_loo(
    data_root:    str,
    test_indices: List[int],
    val_frac:     float = 0.20,
    seed:         int   = 42,
    val_mode:     str   = "tile",
    val_indices:  Optional[List[int]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Hold out the scene folder(s) at ``test_indices`` as the test set.

    Parameters
    ----------
    data_root : str
        Glob pattern for scene folders.
    test_indices : list of int
        Zero-based indices into the sorted folder list to use as test scenes.
        E.g. ``[3]`` for the fourth folder, ``[9, 1, 3]`` for three scenes.
    val_frac : float
        Only used when val_mode="tile": fraction of remaining tiles used for
        validation (default 0.20). When val_mode="scene" and val_indices is
        not given, val_frac instead sets how many *whole scenes* (rounded,
        minimum 1) are held out for validation.
    seed : int
        Random seed. Governs both the tile-level shuffle (val_mode="tile")
        and, when val_indices is not given, which scene(s) are chosen for
        validation (val_mode="scene").
    val_mode : {"tile", "scene"}
        "tile" (default, ORIGINAL BEHAVIOUR — unchanged): pool all tiles
        from every non-test scene, shuffle at tile level, and cut the last
        val_frac fraction off as validation. Tiles from the same scene can
        land in both train and validation (they overlap spatially by
        design — 50% tile overlap within a scene), which is the leakage
        reviewers flagged: it does not touch the reported *test* metrics
        (test scenes are always fully disjoint), but it can make
        checkpoint/early-stopping selection slightly optimistic.
        "scene" (NEW, leak-free): hold out whole scene folder(s) for
        validation instead, so train/val/test are all scene-disjoint.
        Use `val_indices` to pick them explicitly (recommended, e.g. for
        the fixed 7-train/1-val/2-test comparison split), or let `val_frac`
        pick that many scenes deterministically via `seed`.
    val_indices : list of int, optional
        Only used when val_mode="scene". Zero-based folder indices (from
        the same sorted folder list as test_indices) to use as validation
        scenes. Must not overlap test_indices.

    Folder order reference (10-scene dataset):
        0:uws  1:uxr  2:uxq  3:uyq  4:ucu
        5:uxs  6:uca  7:uwt  8:udu  9:uuu
    """
    folders = sorted([f for f in glob.glob(data_root) if os.path.isdir(f)])
    if not folders:
        raise FileNotFoundError(f"No folders found at {data_root!r}")
    if max(test_indices) >= len(folders):
        raise ValueError(
            f"test_indices {test_indices} out of range for {len(folders)} folders."
        )

    test_folders = [folders[i] for i in test_indices]
    X_test, y_test = get_files_from_folders(test_folders)

    if val_mode == "tile":
        trainval_folders = [f for i, f in enumerate(folders) if i not in test_indices]
        X_all, y_all = get_files_from_folders(trainval_folders)

        # Shuffle at tile level
        combined = list(zip(X_all, y_all))
        random.seed(seed)
        random.shuffle(combined)
        X_all[:], y_all[:] = zip(*combined)  # type: ignore[assignment]

        split_idx = int((1.0 - val_frac) * len(X_all))
        X_train, y_train = X_all[:split_idx],  y_all[:split_idx]
        X_val,   y_val   = X_all[split_idx:],  y_all[split_idx:]

    elif val_mode == "scene":
        remaining_indices = [i for i in range(len(folders)) if i not in test_indices]

        if val_indices is not None:
            if set(val_indices) & set(test_indices):
                raise ValueError(
                    f"val_indices {val_indices} overlaps test_indices {test_indices}."
                )
            chosen_val_indices = list(val_indices)
        else:
            n_val_folders = max(1, round(val_frac * len(remaining_indices)))
            rng = random.Random(seed)
            shuffled = remaining_indices[:]
            rng.shuffle(shuffled)
            chosen_val_indices = shuffled[:n_val_folders]

        val_folders   = [folders[i] for i in chosen_val_indices]
        train_folders = [
            f for i, f in enumerate(folders)
            if i not in test_indices and i not in chosen_val_indices
        ]
        X_train, y_train = get_files_from_folders(train_folders)
        X_val,   y_val   = get_files_from_folders(val_folders)

    else:
        raise ValueError(f"Unknown val_mode: {val_mode!r}. Use 'tile' or 'scene'.")

    _check_counts({"train": (X_train, y_train),
                   "val":   (X_val,   y_val),
                   "test":  (X_test,  y_test)})

    print(f"LOO split (val_mode={val_mode!r}) — "
          f"test folders: {[os.path.basename(f) for f in test_folders]}")
    if val_mode == "scene":
        print(f"  val folders: {[os.path.basename(folders[i]) for i in chosen_val_indices]}")
    print(f"  train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test
