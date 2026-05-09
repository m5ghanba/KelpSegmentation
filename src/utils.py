"""
src/utils.py
------------
Visualisation helpers for kelp segmentation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Image stretching
# ---------------------------------------------------------------------------

def stretch_image(
    image: np.ndarray,
    min_value: float = 0,
    max_value: float = 255,
) -> np.ndarray:
    """
    Stretch a 3-band (H, W, 3) float array to [min_value, max_value].

    Default clip ranges are tuned for Sentinel-2 L2A reflectance scaled
    to ~0–10 000.  If all values in a band fall outside the default range,
    a 2 / 98 percentile cut is used as fallback.
    """
    stretched = np.zeros_like(image, dtype=np.float32)
    default_ranges = {
        0: (0, 200),   # NIR
        1: (0, 400),   # Red
        2: (0, 600),   # Green
    }
    cumulative_cuts = {0: (1, 45), 1: (2, 95), 2: (2, 95)}

    for i in range(image.shape[2]):
        band = image[:, :, i]
        dmin, dmax = default_ranges.get(i, (0, 1000))

        if np.all(band >= dmax) or np.all(band <= dmin):
            lo, hi = cumulative_cuts.get(i, (2, 98))
            bmin, bmax = np.percentile(band, lo), np.percentile(band, hi)
        else:
            bmin, bmax = dmin, dmax

        clipped = np.clip(band, bmin, bmax)
        stretched[:, :, i] = (
            (clipped - bmin) / (bmax - bmin) * (max_value - min_value) + min_value
        )

    return stretched.astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualize a sample
# ---------------------------------------------------------------------------

def visualize(rgb_plot: bool = True, **images: np.ndarray) -> None:
    """
    Plot images side-by-side.

    Keys ``"image"`` and ``"mask"`` receive special treatment:
      - ``"image"``  (C, H, W): displayed as NIR-R-G false colour composite
                                (or greyscale first band if rgb_plot=False).
      - ``"mask"``   (num_classes, H, W): displayed as greyscale.
      - Everything else: displayed as greyscale.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        ax = plt.subplot(1, n, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(" ".join(name.split("_")).title())

        if name == "image":
            img_hwc = image.transpose(1, 2, 0)
            if rgb_plot and img_hwc.shape[2] >= 4:
                vis = np.stack(
                    [img_hwc[:, :, 3],   # NIR
                     img_hwc[:, :, 2],   # Red
                     img_hwc[:, :, 1]],  # Green
                    axis=-1,
                )
                ax.imshow(stretch_image(vis.astype(np.float32)))
            else:
                band = img_hwc[:, :, 0]
                lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                norm = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
                ax.imshow(norm, cmap="gray")

        elif name == "mask":
            squeezed = image.squeeze()
            ax.imshow(squeezed if squeezed.ndim == 2 else image[0], cmap="gray")

        else:
            ax.imshow(image, cmap="gray")

    plt.tight_layout()
    plt.show()
