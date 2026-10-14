"""
infer_single_tile.py

Run kelp / no-kelp segmentation inference on ONE 512x512 Sentinel-2 tile.

Input directory
---------------
The directory must contain exactly one:

    *_image.tiff

Optionally, it can also contain:

    *_mask.tiff

The image GeoTIFF created by the tile-generation script contains 13 bands:

    1  B02 / blue
    2  B03 / green
    3  B04 / red
    4  B08 / nir
    5  B05 / rededge
    6  B06
    7  B07
    8  B8A
    9  B11
    10 B12
    11 substrate
    12 bathymetry
    13 slope

However, the model does NOT use all 13 raw bands.

The model input is constructed as:

    B02
    B03
    B04
    B08
    B05
    NDVI
    NDWI
    GNDVI
    CLgreen
    NDVI_RE
    [substrate, if requested]
    [bathymetry, if requested]
    [slope, if requested]

Therefore:

    10 channels = B02/B03/B04/B08/B05 + 5 VIs
    11 channels = above + substrate
    12 channels = above + substrate + bathymetry
    13 channels = above + substrate + bathymetry + slope

B06, B07, B8A, B11, and B12 are NOT included as raw model channels.
B11 and B12 are used only for the corresponding auxiliary channels.

Normalization values are loaded from a separate JSON file containing
`mean` and `std` arrays. The number of values must match the final
model input channel count.

Examples
--------
Base 10-channel model:

    python src/infer_single_tile.py \
        --checkpoint path/to/model.ckpt \
        --normalization-json path/to/normalization.json \
        --input_dir path/to/tile

10 + substrate:

    python src/infer_single_tile.py \
        --checkpoint path/to/model.ckpt \
        --normalization-json path/to/normalization.json \
        --input_dir path/to/tile \
        --use-substrate

10 + substrate + bathymetry:

    python src/infer_single_tile.py \
        --checkpoint path/to/model.ckpt \
        --normalization-json path/to/normalization.json \
        --input_dir path/to/tile \
        --use-substrate \
        --use-bathymetry

10 + substrate + bathymetry + slope:

    python src/infer_single_tile.py \
        --checkpoint path/to/model.ckpt \
        --normalization-json path/to/normalization.json \
        --input_dir path/to/tile \
        --use-substrate \
        --use-bathymetry \
        --use-slope
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import (
    normalize_input_mean_std,
    SatelliteDataset,
    resolve_num_channels,
)
from src.model import KelpSegmentationModel


# ---------------------------------------------------------------------------
# Fixed assumptions
# ---------------------------------------------------------------------------

# These are the ONLY raw Sentinel-2 bands used by the model.
#
# The input TIFF contains 10 Sentinel-2 bands, but only the first five
# are passed to the model:
#
#   B02, B03, B04, B08, B05
#
# B06, B07, B8A, B11, B12 are not included as raw model channels.
BASE_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "rededge",
]

OPTIONAL_BANDS = {
    "substrate": 11,
    "bathymetry": 12,
    "slope": 13,
}

# Fixed vegetation indices. These are appended after the five raw bands.
VI_NAMES = [
    "ndvi",
    "ndwi",
    "gndvi",
    "clgreen",
    "ndvire",
]

MODEL_ARCHITECTURE = "Unet"
MODEL_ENCODER = "tu-maxvit_tiny_tf_512"
MODEL_OUT_CLASSES = 1
MODEL_ENCODER_WEIGHTS = None
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def load_normalization(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalization mean/std values from a JSON file."""

    with open(path, "r") as f:
        data = json.load(f)

    if "mean" not in data or "std" not in data:
        raise ValueError(
            f"Normalization JSON must contain both 'mean' and 'std': {path}"
        )

    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            "Normalization 'mean' and 'std' must be 1-D arrays."
        )

    if len(mean) != len(std):
        raise ValueError(
            f"Normalization mean/std length mismatch: "
            f"{len(mean)} vs {len(std)}."
        )

    if np.any(std == 0):
        raise ValueError("Normalization std contains zero values.")

    return mean, std


def resolve_in_channels(
    band_names: List[str],
    vi_names: List[str],
    substrate_onehot: bool = False,
) -> int:
    """Resolve the number of model input channels."""

    return resolve_num_channels(
        band_names,
        vi_names,
        substrate_onehot=substrate_onehot,
    )


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def find_single_file(input_dir: str, pattern: str) -> str:
    matches = sorted(
        glob.glob(os.path.join(input_dir, pattern))
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {input_dir!r}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one file matching '{pattern}' in "
            f"{input_dir!r}, found {len(matches)}:\n"
            + "\n".join(matches)
        )

    return matches[0]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_tile(
    input_dir: str,
    use_substrate: bool = False,
    use_bathymetry: bool = False,
    use_slope: bool = False,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Load one 512x512 tile.

    The input TIFF contains up to 13 raw bands, but only the following
    channels are selected for model input:

        B02, B03, B04, B08, B05

    followed later by the five vegetation indices.

    Optional auxiliary channels are selected directly from:

        band 11 = substrate
        band 12 = bathymetry
        band 13 = slope

    Returns
    -------
    image_hwc : raw model input bands in H,W,C order

        Contains:

            B02, B03, B04, B08, B05
            [+ substrate]
            [+ bathymetry]
            [+ slope]

    metadata : rasterio metadata

    raw_image : original complete raw TIFF contents in C,H,W order
    """

    image_path = find_single_file(
        input_dir,
        "*_image.tiff",
    )

    with rasterio.open(image_path) as src:

        if src.width != 512 or src.height != 512:
            raise ValueError(
                f"Input image must be 512x512, but is "
                f"{src.width}x{src.height}."
            )

        n_bands = src.count

        # The base model always requires bands 1-5.
        required_bands = 5

        if use_substrate:
            required_bands = max(required_bands, 11)

        if use_bathymetry:
            required_bands = max(required_bands, 12)

        if use_slope:
            required_bands = max(required_bands, 13)

        if n_bands < required_bands:
            raise ValueError(
                f"Input image has {n_bands} bands, but the requested "
                f"configuration requires at least {required_bands} bands."
            )

        # Keep the complete raw image for visualization and diagnostics.
        raw = src.read().astype(np.float32)

        metadata = src.meta.copy()

    # ------------------------------------------------------------------
    # Select ONLY the raw channels used by the model.
    #
    # TIFF band order:
    #
    # 1  B02
    # 2  B03
    # 3  B04
    # 4  B08
    # 5  B05
    # 6  B06       <- NOT USED
    # 7  B07       <- NOT USED
    # 8  B8A       <- NOT USED
    # 9  B11       <- NOT USED as raw band
    # 10 B12       <- NOT USED as raw band
    # 11 substrate
    # 12 bathymetry
    # 13 slope
    # ------------------------------------------------------------------

    selected_indices = [
        0,  # B02 / blue
        1,  # B03 / green
        2,  # B04 / red
        3,  # B08 / nir
        4,  # B05 / rededge
    ]

    if use_substrate:
        selected_indices.append(10)  # band 11

    if use_bathymetry:
        selected_indices.append(11)  # band 12

    if use_slope:
        selected_indices.append(12)  # band 13

    selected = raw[selected_indices]

    # Convert C,H,W -> H,W,C
    image_hwc = np.transpose(
        selected,
        (1, 2, 0),
    )

    return image_hwc, metadata, raw


# ---------------------------------------------------------------------------
# Compute vegetation indices
# ---------------------------------------------------------------------------

def add_vegetation_indices(
    image_hwc: np.ndarray,
    band_names: List[str],
    vi_names: List[str],
) -> np.ndarray:
    """
    Compute VIs using exactly the same calculators as SatelliteDataset.

    The VIs are appended after the raw model bands.

    For the standard configuration, the output order is:

        B02
        B03
        B04
        B08
        B05
        NDVI
        NDWI
        GNDVI
        CLgreen
        NDVI_RE
        [optional auxiliary channels]
    """

    if not vi_names:
        return image_hwc

    band_pos = {
        name: idx
        for idx, name in enumerate(band_names)
    }

    dummy = SatelliteDataset.__new__(SatelliteDataset)

    dummy._band_pos = band_pos

    dummy._vi_calculators = {
        "ndvi": dummy._calc_ndvi,
        "ndwi": dummy._calc_ndwi,
        "gndvi": dummy._calc_gndvi,
        "clgreen": dummy._calc_clgreen,
        "ndvire": dummy._calc_ndvire,
    }

    vi_arrays = []

    for vi in vi_names:

        calc = dummy._vi_calculators.get(vi)

        if calc is None:
            raise ValueError(
                f"Unknown VI name: {vi!r}. "
                f"Available: {list(dummy._vi_calculators.keys())}"
            )

        vi_array = calc(image_hwc)[..., np.newaxis]

        vi_arrays.append(vi_array)

    return np.concatenate(
        [image_hwc] + vi_arrays,
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Single-tile inference
# ---------------------------------------------------------------------------

def run_single_tile_inference(
    image_hwc: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the complete 512x512 tile.

    No tiling, overlap, halo, or batching is necessary because the
    input itself is exactly 512x512.
    """

    if image_hwc.shape[0] != 512 or image_hwc.shape[1] != 512:
        raise ValueError(
            f"Expected 512x512 input, got {image_hwc.shape[:2]}."
        )

    if image_hwc.shape[-1] != len(mean):
        raise ValueError(
            f"Input has {image_hwc.shape[-1]} channels, but normalization "
            f"expects {len(mean)} channels."
        )

    normalized = normalize_input_mean_std(
        image_hwc,
        mean,
        std,
    )

    tensor = (
        torch.from_numpy(normalized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    model.eval()

    with torch.no_grad():

        logits = model(tensor)

        probability = (
            logits.squeeze(1)
            .sigmoid()
            .cpu()
            .numpy()[0]
        )

    prediction = (
        probability > threshold
    ).astype(np.uint8)

    return prediction, probability


# ---------------------------------------------------------------------------
# Save prediction GeoTIFF
# ---------------------------------------------------------------------------

def save_prediction(
    prediction: np.ndarray,
    metadata: Dict,
    output_path: str,
) -> None:

    meta = metadata.copy()

    meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "compress": "lzw",
    })

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **meta,
    ) as dst:

        dst.write(
            prediction.astype(np.uint8),
            1,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[float, float, float, float]:

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    tp = intersection

    fp = np.logical_and(
        prediction,
        ~ground_truth,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        ground_truth,
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return iou, precision, recall, f1


# ---------------------------------------------------------------------------
# Visualization
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


def create_false_color(
    raw_image: np.ndarray,
) -> np.ndarray:
    """
    Create NIR-R-G false-colour image.

    Sentinel-2 bands:

        B02 = blue
        B03 = green
        B04 = red
        B08 = NIR

    Output RGB channels:

        Red   = NIR
        Green = Red
        Blue  = Green
    """

    false_color = np.stack(
        [
            raw_image[3],  # B08 / NIR
            raw_image[2],  # B04 / Red
            raw_image[1],  # B03 / Green
        ],
        axis=-1,
    )

    return stretch_image(
        false_color
    )

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
) -> np.ndarray:

    output = image.copy()
    mask = mask.astype(bool)

    output[mask] = color

    return output


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------

def get_arial_font(
    size: int,
):
    """
    Load Arial font.

    Tries the Windows Arial path first, then a local Arial font,
    then falls back to Pillow's default font.
    """

    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        "arial.ttf",
    ]

    for font_path in font_paths:

        try:

            return ImageFont.truetype(
                font_path,
                size,
            )

        except OSError:

            pass

    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

def add_scale_bar(
    image: np.ndarray,
    pixel_size_m: float = 10.0,
    scale_length_m: float = 1000.0,
) -> np.ndarray:
    """
    Add a 1000 m scale bar.

    Pixel size:
        10 m / pixel

    Scale:
        1000 m total
        Two 500 m alternating black/white segments

    Appearance:
        5 px bar thickness
        Dark-gray background
        White labels
        Arial 16 px

    Position:
        Bottom-left
    """

    image = image.copy()

    height, width = image.shape[:2]

    scale_length_px = int(
        scale_length_m / pixel_size_m
    )

    scale_margin = 10

    # Make sure the scale bar fits inside the image.
    if scale_length_px >= width - 2 * scale_margin:
        return image

    scale_height_px = 5

    scale_color = (0, 0, 0)
    scale_text_color = (255, 255, 255)
    background_color = (50, 50, 50)

    font = get_arial_font(16)

    # Bottom edge of the scale bar.
    y_start = height - scale_margin

    # Top edge of the scale bar.
    y_end = y_start - scale_height_px

    # Two equal 500 m segments.
    segment_length_px = scale_length_px // 2

    x_start = scale_margin

    # Height reserved for the text above the bar.
    text_area_height = 22

    # Background rectangle.
    bg_x0 = x_start - 7
    bg_y0 = y_end - text_area_height - 7
    bg_x1 = x_start + scale_length_px + 30
    bg_y1 = y_start + 7

    pil_image = Image.fromarray(
        image
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    draw.rectangle(
        [
            bg_x0,
            bg_y0,
            bg_x1,
            bg_y1,
        ],
        fill=background_color,
    )

    # Draw the two alternating 500 m segments.
    for i in range(2):

        x0 = (
            x_start
            + i * segment_length_px
        )

        x1 = (
            x0
            + segment_length_px
        )

        color = (
            scale_color
            if i % 2 == 0
            else (255, 255, 255)
        )

        draw.rectangle(
            [
                x0,
                y_end,
                x1,
                y_start,
            ],
            fill=color,
        )

    # Scale labels:
    #
    #     0       500       1000 m
    #
    text_positions = [
        (
            x_start,
            "0",
        ),
        (
            x_start + segment_length_px,
            f"{int(scale_length_m / 2)}",
        ),
        (
            x_start + scale_length_px,
            f"{int(scale_length_m)} m",
        ),
    ]

    for x, text in text_positions:

        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = (
            text_bbox[2]
            - text_bbox[0]
        )

        text_x = (
            x
            - text_width // 2
        )

        draw.text(
            (
                text_x,
                y_end - text_area_height,
            ),
            text,
            font=font,
            fill=scale_text_color,
        )

    return np.asarray(
        pil_image
    )


# ---------------------------------------------------------------------------
# Metrics annotation
# ---------------------------------------------------------------------------

def add_metrics_annotation(
    image: np.ndarray,
    iou: float,
    f1: float,
) -> np.ndarray:
    """
    Add IoU and F1-Score to the bottom-right corner.

    Appearance:
        Arial 25 px
        White text
        Black background rectangles
        35 px margin from bottom/right
        10 px gap between lines
    """

    pil_image = Image.fromarray(
        image
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    font = get_arial_font(25)

    text_line1 = f"IoU: {iou:.4f}"
    text_line2 = f"F1-Score: {f1:.4f}"

    img_width, img_height = (
        pil_image.size
    )

    # Use anchor="lt" so the bounding boxes correspond
    # directly to the text drawing positions.
    bbox1 = draw.textbbox(
        (0, 0),
        text_line1,
        font=font,
        anchor="lt",
    )

    bbox2 = draw.textbbox(
        (0, 0),
        text_line2,
        font=font,
        anchor="lt",
    )

    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]

    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    margin = 35
    gap_between_lines = 10

    # Bottom-right placement.
    x1 = (
        img_width
        - w1
        - margin
    )

    y1 = (
        img_height
        - h1
        - h2
        - margin
        - gap_between_lines
    )

    x2 = (
        img_width
        - w2
        - margin
    )

    y2 = (
        img_height
        - h2
        - margin
    )

    # Exact-size black background for IoU.
    draw.rectangle(
        [
            x1,
            y1,
            x1 + w1,
            y1 + h1,
        ],
        fill="black",
    )

    # Exact-size black background for F1.
    draw.rectangle(
        [
            x2,
            y2,
            x2 + w2,
            y2 + h2,
        ],
        fill="black",
    )

    # White text.
    draw.text(
        (x1, y1),
        text_line1,
        font=font,
        fill="white",
        anchor="lt",
    )

    draw.text(
        (x2, y2),
        text_line2,
        font=font,
        fill="white",
        anchor="lt",
    )

    return np.asarray(
        pil_image
    )


# ---------------------------------------------------------------------------
# Save visualization
# ---------------------------------------------------------------------------

def save_visualizations(
    raw_image: np.ndarray,
    prediction: np.ndarray,
    output_dir: str,
    stem: str,
    ground_truth: np.ndarray | None = None,
    iou: float | None = None,
    f1: float | None = None,
) -> None:

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Use the SAME kelp overlay colour for ground truth and prediction.
    # ------------------------------------------------------------------

    kelp_color = (0, 255, 0)

    # ------------------------------------------------------------------
    # False colour NIR-R-G
    # ------------------------------------------------------------------

    false_color = create_false_color(
        raw_image
    )

    # Add the scale bar to the false-colour image itself.
    false_color = add_scale_bar(
        false_color
    )

    false_color_path = os.path.join(
        output_dir,
        f"{stem}_false_color.png",
    )

    Image.fromarray(
        false_color
    ).save(
        false_color_path
    )

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    if ground_truth is not None:

        gt_overlay = overlay_mask(
            false_color,
            ground_truth,
            kelp_color,
        )

        gt_path = os.path.join(
            output_dir,
            f"{stem}_ground_truth.png",
        )

        Image.fromarray(
            gt_overlay
        ).save(
            gt_path
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    prediction_overlay = overlay_mask(
        false_color,
        prediction,
        (255,255,0),
    )

    # ------------------------------------------------------------------
    # Metrics annotation
    # ------------------------------------------------------------------

    if iou is not None and f1 is not None:

        prediction_overlay = (
            add_metrics_annotation(
                prediction_overlay,
                iou,
                f1,
            )
        )

    # The false_color image already contains the scale bar, so it is
    # preserved in the prediction image through the overlay operation.
    prediction_png_path = os.path.join(
        output_dir,
        f"{stem}_prediction.png",
    )

    Image.fromarray(
        prediction_overlay
    ).save(
        prediction_png_path
    )

    print(
        f"Saved false colour  -> {false_color_path}"
    )

    if ground_truth is not None:

        print(
            f"Saved ground truth  -> {gt_path}"
        )

    print(
        f"Saved prediction    -> {prediction_png_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run kelp segmentation inference on one 512x512 "
            "Sentinel-2 tile."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )

    parser.add_argument(
        "--normalization-json",
        required=True,
        help=(
            "Path to JSON file containing normalization "
            "'mean' and 'std'."
        ),
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing *_image.tiff and optionally "
            "*_mask.tiff."
        ),
    )

    parser.add_argument(
        "--use-substrate",
        action="store_true",
        help="Use band 11 as the substrate channel.",
    )

    parser.add_argument(
        "--use-bathymetry",
        action="store_true",
        help="Use band 12 as the bathymetry channel.",
    )

    parser.add_argument(
        "--use-slope",
        action="store_true",
        help="Use band 13 as the slope channel.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Fixed data/model assumptions
    # ------------------------------------------------------------------

    vi_names = VI_NAMES.copy()

    substrate_onehot = False

    # IMPORTANT:
    #
    # These are the only raw bands included in the model input.
    #
    # B02, B03, B04, B08, B05
    #
    # The remaining Sentinel-2 bands in the input TIFF are ignored.
    band_names = BASE_BANDS.copy()

    if args.use_substrate:
        band_names.append("substrate")

    if args.use_bathymetry:
        band_names.append("bathymetry")

    if args.use_slope:
        band_names.append("slope")

    in_channels = resolve_in_channels(
        band_names,
        vi_names,
        substrate_onehot=substrate_onehot,
    )

    print()
    print("Input configuration")
    print("-------------------")
    print(f"Raw model bands : {band_names}")
    print(f"VIs             : {vi_names}")
    print(f"Channels        : {in_channels}")
    print()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    mean, std = load_normalization(
        args.normalization_json
    )

    if len(mean) != in_channels:
        raise ValueError(
            f"Normalization JSON has {len(mean)} values, "
            f"but the selected configuration requires "
            f"{in_channels} channels."
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    model = KelpSegmentationModel.load_from_checkpoint(
        args.checkpoint,
        arch=MODEL_ARCHITECTURE,
        encoder_name=MODEL_ENCODER,
        in_channels=in_channels,
        out_classes=MODEL_OUT_CLASSES,
        encoder_weights=MODEL_ENCODER_WEIGHTS,
        strict=False,
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: {args.checkpoint}"
    )

    print()

    # ------------------------------------------------------------------
    # Load one tile
    # ------------------------------------------------------------------

    image_hwc, metadata, raw_image = load_tile(
        args.input_dir,
        use_substrate=args.use_substrate,
        use_bathymetry=args.use_bathymetry,
        use_slope=args.use_slope,
    )

    print(
        f"Raw image shape: "
        f"{raw_image.shape[0]} bands x "
        f"{raw_image.shape[1]} x "
        f"{raw_image.shape[2]}"
    )

    print(
        f"Selected raw model input: "
        f"{image_hwc.shape}"
    )

    # ------------------------------------------------------------------
    # Add VIs
    # ------------------------------------------------------------------

    image_hwc = add_vegetation_indices(
        image_hwc,
        band_names,
        vi_names,
    )

    print(
        f"Final model input shape: "
        f"{image_hwc.shape}"
    )

    if image_hwc.shape[-1] != in_channels:

        raise ValueError(
            f"Final image has "
            f"{image_hwc.shape[-1]} channels, "
            f"but model expects "
            f"{in_channels}."
        )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    threshold = DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    print(
        "Running inference..."
    )

    prediction, probability = (
        run_single_tile_inference(
            image_hwc=image_hwc,
            model=model,
            device=device,
            mean=mean,
            std=std,
            threshold=threshold,
        )
    )

    print(
        "Inference complete."
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    image_path = find_single_file(
        args.input_dir,
        "*_image.tiff",
    )

    image_stem = Path(
        image_path
    ).stem

    output_dir = os.path.join(
        args.input_dir,
        "inference_output",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save prediction GeoTIFF
    # ------------------------------------------------------------------

    prediction_path = os.path.join(
        output_dir,
        f"{image_stem}_prediction.tif",
    )

    save_prediction(
        prediction,
        metadata,
        prediction_path,
    )

    print(
        f"Saved prediction GeoTIFF -> "
        f"{prediction_path}"
    )

    # ------------------------------------------------------------------
    # Optional ground truth
    # ------------------------------------------------------------------

    mask_matches = sorted(
        glob.glob(
            os.path.join(
                args.input_dir,
                "*_mask.tiff",
            )
        )
    )

    ground_truth = None

    iou = None
    precision = None
    recall = None
    f1 = None

    if len(mask_matches) > 1:

        raise ValueError(
            f"Expected at most one *_mask.tiff, "
            f"found {len(mask_matches)}."
        )

    if len(mask_matches) == 1:

        mask_path = mask_matches[0]

        with rasterio.open(mask_path) as src:

            if src.width != 512 or src.height != 512:

                raise ValueError(
                    f"Ground-truth mask must be 512x512, "
                    f"but is "
                    f"{src.width}x{src.height}."
                )

            ground_truth = src.read(1)

        # Ensure binary
        ground_truth = (
            ground_truth > 0
        ).astype(np.uint8)

        iou, precision, recall, f1 = (
            calculate_metrics(
                prediction,
                ground_truth,
            )
        )

        print()
        print("Metrics")
        print("-------")
        print(f"IoU       : {iou:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1        : {f1:.4f}")

        metrics_path = os.path.join(
            output_dir,
            "metrics.txt",
        )

        with open(
            metrics_path,
            "w",
        ) as f:

            f.write(
                f"IoU: {iou:.6f}\n"
            )

            f.write(
                f"Precision: {precision:.6f}\n"
            )

            f.write(
                f"Recall: {recall:.6f}\n"
            )

            f.write(
                f"F1: {f1:.6f}\n"
            )

        print(
            f"Saved metrics -> {metrics_path}"
        )

    else:

        print()

        print(
            "No *_mask.tiff found. "
            "Skipping ground-truth evaluation."
        )

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    save_visualizations(
        raw_image=raw_image,
        prediction=prediction,
        output_dir=output_dir,
        stem=image_stem,
        ground_truth=ground_truth,
        iou=iou,
        f1=f1,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()