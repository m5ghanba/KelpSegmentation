"""
src/transforms.py
-----------------
Albumentations augmentation pipelines for kelp segmentation.

Notes on augmentation choices for satellite + auxiliary data:
  - Flips and 90° rotations are safe for all channels.
  - Affine (translate/scale/rotate) is safe.
  - Brightness / contrast / gamma / noise augmentations are deliberately
    excluded because they corrupt spectral ratios (NDVI, NDWI …) and
    pre-computed auxiliary channels (bathymetry, substrate, slope).
  - ElasticTransform is excluded because it can distort kelp-bed boundaries.
  - CoarseDropout / GridDistortion are available but off by default.

CHANGE vs. the original: `get_training_augmentation()` now takes a `mode`
argument so the manuscript's augmentation ablation table (R1) can be produced
by looping over modes with everything else held fixed. mode="flip" (default)
reproduces the ORIGINAL behaviour exactly (flip-only — rot90/affine were
present as commented-out code before but never actually applied, despite the
manuscript's augmentation-rationale paragraph reading as if they might have
been considered and excluded for a stated reason; that reason applies only to
rotations that AREN'T multiples of 90°, so flip+rot90 is a legitimate new
comparison, not something already ruled out).
"""

from __future__ import annotations
from typing import Optional
import albumentations as A

_VALID_MODES = ("flip", "flip_rot90", "flip_affine")


def get_training_augmentation(
    mode:         str   = "flip",
    hflip_p:      float = 0.5,
    vflip_p:      float = 0.5,
    rotate90_p:   float = 0.5,
    affine_p:     float = 0.3,
) -> A.Compose:
    """
    Return the training augmentation pipeline.

    Parameters
    ----------
    mode : {"flip", "flip_rot90", "flip_affine"}
        "flip"        — horizontal + vertical flip only (default; matches
                         the original / manuscript-reported behaviour).
        "flip_rot90"  — flip + random 90° rotation. Safe for all channels
                         (no interpolation), so this is a legitimate
                         additional axis-aligned augmentation, not the
                         "rotations that require interpolation" the
                         manuscript already argues against.
        "flip_affine" — flip + rot90 + affine (translate ±10%, scale
                         0.9-1.1, rotate ±15°). Involves interpolation —
                         included only as a comparison point per R1's
                         request, not because we currently believe it's
                         safe for the auxiliary channels' precision.
    hflip_p, vflip_p : float
        Probabilities for horizontal / vertical flips.
    rotate90_p : float
        Probability of a random 90° rotation (used when mode != "flip").
    affine_p : float
        Probability of the affine transform (used when mode == "flip_affine").
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown augmentation mode {mode!r}. Use one of {_VALID_MODES}.")

    transforms = [
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
    ]

    if mode in ("flip_rot90", "flip_affine"):
        transforms.append(A.RandomRotate90(p=rotate90_p))

    if mode == "flip_affine":
        transforms.append(
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=affine_p,
            )
        )

    # --- Deliberately excluded in every mode ---
    # A.RandomBrightnessContrast(...)  # breaks spectral indices
    # A.RandomGamma(...)               # same issue
    # A.GaussNoise(...)               # corrupts bathymetry precision
    # A.ElasticTransform(...)         # distorts kelp boundaries
    # A.CoarseDropout(...)            # optional: simulates cloud cover
    # A.GridDistortion(...)           # optional: water-surface ripple

    return A.Compose(transforms)


def get_validation_augmentation() -> Optional[A.Compose]:
    """Return None — no augmentation at validation / test time."""
    return None
