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
"""

from __future__ import annotations
from typing import Optional
import albumentations as A


def get_training_augmentation(
    hflip_p:      float = 0.5,
    vflip_p:      float = 0.5,
    rotate90_p:   float = 0.5,
    affine_p:     float = 0.3,
) -> A.Compose:
    """
    Return the training augmentation pipeline.

    Parameters
    ----------
    hflip_p, vflip_p : float
        Probabilities for horizontal / vertical flips.
    rotate90_p : float
        Probability of a random 90° rotation.
    affine_p : float
        Probability of affine (translate ±10 %, scale 0.9–1.1, rotate ±15°).
    """
    transforms = [
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        # A.RandomRotate90(p=rotate90_p),

        # # Geometric transformations – safe for all channels
        # A.Affine(
        #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        #     scale=(0.9, 1.1),
        #     rotate=(-15, 15),
        #     p=affine_p,
        # ),

        # --- Deliberately excluded ---
        # A.RandomBrightnessContrast(...)  # breaks spectral indices
        # A.RandomGamma(...)               # same issue
        # A.GaussNoise(...)               # corrupts bathymetry precision
        # A.ElasticTransform(...)         # distorts kelp boundaries
        # A.CoarseDropout(...)            # optional: simulates cloud cover
        # A.GridDistortion(...)           # optional: water-surface ripple
    ]
    return A.Compose(transforms)


def get_validation_augmentation() -> Optional[A.Compose]:
    """Return None — no augmentation at validation / test time."""
    return None
