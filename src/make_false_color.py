
#!/usr/bin/env python3

"""
Create false-color PNGs from Sentinel-2 GeoTIFF images and overlay
the kelp mask boundary.

Input directory structure:
    input_dir/
        images/
            image_001.tiff
            image_002.tiff
            ...
        masks/
            image_001.tiff
            image_002.tiff
            ...

Image GeoTIFF:
    - Sentinel-2 bands
    - Band 4 = NIR
    - Band 3 = Red
    - Band 2 = Green
    - Expected size: 512 x 512

Mask GeoTIFF:
    - Scalar mask
    - 0 = no kelp
    - 1 = kelp

Output:
    input_dir/
        pngs/
            image_001.png
            image_002.png
            ...

False-color stretching:
    NIR: 0 - 300
    Red: 0 - 500
    Green: 0 - 700

The output is an RGB PNG using:
    R channel = NIR
    G channel = Red
    B channel = Green

Kelp boundary:
    Lime/yellow line
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NIR_MIN, NIR_MAX = 0, 300
RED_MIN, RED_MAX = 0, 500
GREEN_MIN, GREEN_MAX = 0, 700

# Lime/yellow boundary color
BOUNDARY_COLOR = np.array([0, 255, 0], dtype=np.uint8)

# Boundary thickness in pixels
BOUNDARY_WIDTH = 1


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def stretch_band(band, vmin, vmax):
    """
    Stretch a band from [vmin, vmax] to [0, 255].

    Values below vmin are clipped to 0.
    Values above vmax are clipped to 255.
    """
    band = band.astype(np.float32)

    stretched = (band - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)

    return (stretched * 255).astype(np.uint8)


def get_mask_boundary(mask):
    """
    Extract the boundary of the kelp region.

    mask:
        0 = no kelp
        1 = kelp
    """
    kelp = mask == 1

    # Find pixels along the boundary between kelp and non-kelp.
    boundary = find_boundaries(
        kelp,
        mode="inner"
    )

    return boundary


def make_boundary_thicker(boundary, width=1):
    """
    Increase the thickness of the boundary if requested.
    """
    if width <= 1:
        return boundary

    from scipy.ndimage import binary_dilation

    return binary_dilation(
        boundary,
        iterations=width - 1
    )


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_pair(image_path, mask_path, output_path):
    """
    Process one image/mask pair.
    """

    # -------------------------------------------------------------
    # Read Sentinel-2 image
    # -------------------------------------------------------------

    with rasterio.open(image_path) as src:

        if src.count < 4:
            raise ValueError(
                f"{image_path} has only {src.count} bands. "
                "At least 4 bands are required."
            )

        # Rasterio uses 1-based band indexing.
        #
        # Sentinel-2:
        # Band 2 = Green
        # Band 3 = Red
        # Band 4 = NIR
        green = src.read(2)
        red = src.read(3)
        nir = src.read(4)

        image_height = src.height
        image_width = src.width

    # -------------------------------------------------------------
    # Read mask
    # -------------------------------------------------------------

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    # -------------------------------------------------------------
    # Check dimensions
    # -------------------------------------------------------------

    if nir.shape != (512, 512):
        print(
            f"Warning: {image_path.name} has size "
            f"{nir.shape[1]} x {nir.shape[0]}, not 512 x 512."
        )

    if mask.shape != nir.shape:
        raise ValueError(
            f"Dimension mismatch:\n"
            f"Image: {image_path.name} -> {nir.shape}\n"
            f"Mask:  {mask_path.name} -> {mask.shape}"
        )

    # -------------------------------------------------------------
    # Create false-color image
    # -------------------------------------------------------------

    nir_8bit = stretch_band(
        nir,
        NIR_MIN,
        NIR_MAX
    )

    red_8bit = stretch_band(
        red,
        RED_MIN,
        RED_MAX
    )

    green_8bit = stretch_band(
        green,
        GREEN_MIN,
        GREEN_MAX
    )

    # False color:
    #
    # RGB = NIR, Red, Green
    #
    false_color = np.stack(
        [
            nir_8bit,
            red_8bit,
            green_8bit
        ],
        axis=-1
    )

    # -------------------------------------------------------------
    # Extract kelp boundary
    # -------------------------------------------------------------

    boundary = get_mask_boundary(mask)

    if BOUNDARY_WIDTH > 1:
        boundary = make_boundary_thicker(
            boundary,
            BOUNDARY_WIDTH
        )

    # -------------------------------------------------------------
    # Overlay boundary
    # -------------------------------------------------------------

    output = false_color.copy()

    output[boundary] = BOUNDARY_COLOR

    # -------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------

    Image.fromarray(output, mode="RGB").save(
        output_path
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create NIR-R-G false-color PNGs with kelp mask "
            "boundaries overlaid in lime."
        )
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help=(
            "Directory containing 'images' and 'masks' folders."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "'pngs' inside the input directory."
        )
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"

    if args.output is None:
        output_dir = input_dir / "pngs"
    else:
        output_dir = Path(args.output)

    # -------------------------------------------------------------
    # Validate directories
    # -------------------------------------------------------------

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}"
        )

    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Masks directory not found: {masks_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Find GeoTIFF files
    # -------------------------------------------------------------

    image_files = sorted(
        list(images_dir.glob("*.tiff"))
        + list(images_dir.glob("*.tif"))
    )

    mask_files = sorted(
        list(masks_dir.glob("*.tiff"))
        + list(masks_dir.glob("*.tif"))
    )

    if len(image_files) == 0:
        raise RuntimeError(
            f"No GeoTIFF files found in {images_dir}"
        )

    if len(mask_files) == 0:
        raise RuntimeError(
            f"No GeoTIFF files found in {masks_dir}"
        )

    print(f"Found {len(image_files)} image files.")
    print(f"Found {len(mask_files)} mask files.")

    # -------------------------------------------------------------
    # Match images and masks
    # -------------------------------------------------------------

    if len(image_files) != len(mask_files):
        raise ValueError(
            "The number of images and masks is different:\n"
            f"Images: {len(image_files)}\n"
            f"Masks:  {len(mask_files)}"
        )

    # Match:
    #   tile_0_image.tiff -> tile_0_mask.tiff
    #   tile_1_image.tiff -> tile_1_mask.tiff
    #   etc.
    #
    # The image and mask have the same tile ID, but use
    # "_image" and "_mask" respectively.

    mask_lookup = {
        p.stem.replace("_mask", ""): p
        for p in mask_files
    }

    pairs = []

    for image_path in image_files:

        image_key = image_path.stem.replace("_image", "")

        if image_key not in mask_lookup:
            raise ValueError(
                f"No matching mask found for image "
                f"'{image_path.name}'. "
                f"Expected a mask like "
                f"'{image_key}_mask.tiff'"
            )

        mask_path = mask_lookup[image_key]

        pairs.append(
            (image_path, mask_path)
        )

    # -------------------------------------------------------------
    # Process files
    # -------------------------------------------------------------

    for i, (image_path, mask_path) in enumerate(pairs, start=1):

        output_path = output_dir / f"{image_path.stem}.png"

        print(
            f"[{i}/{len(pairs)}] "
            f"{image_path.name} -> {output_path.name}"
        )

        process_pair(
            image_path,
            mask_path,
            output_path
        )

    print()
    print("Done!")
    print(f"Created {len(pairs)} PNG files.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()