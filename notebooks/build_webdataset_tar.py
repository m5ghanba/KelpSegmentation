"""
build_webdataset_tar.py
-----------------------
Builds a single WebDataset-compatible .tar archive from the existing
GeoTIFF tile folder structure.

Input layout expected (your current tile output):
    <data_root>/
        <SceneID>/
            images/
                tile_0_image.tiff
                tile_1_image.tiff
                ...
            masks/
                tile_0_mask.tiff
                tile_1_mask.tiff
                ...

Output layout inside kelp_tiles.tar:
    20200908T192939_20200908T193355_T09UWS_tile_0000_image.tif
    20200908T192939_20200908T193355_T09UWS_tile_0000_mask.tif
    20200908T192939_20200908T193355_T09UXR_tile_0000_image.tif
    ...

The image and mask for a given tile share the same key prefix
(<SceneID>_tile_NNNN), which is what WebDataset uses to group them
into a single sample.

Usage:
    python build_webdataset_tar.py \
        --data_root /path/to/geotiff_tiles/scenes \
        --output    /path/to/kelp_tiles.tar

    # Dry-run: print what would be added without writing the tar
    python build_webdataset_tar.py \
        --data_root /path/to/geotiff_tiles/scenes \
        --output    /path/to/kelp_tiles.tar \
        --dry_run
"""

import argparse
import glob
import os
import tarfile
from pathlib import Path


def collect_tile_pairs(data_root: str) -> list[tuple[str, str, str, int]]:
    """
    Walk scene folders and collect (scene_id, tile_index, image_path, mask_path)
    tuples, sorted first by scene_id then by tile index.

    Returns
    -------
    list of (scene_id, tile_idx, image_path, mask_path)
    """
    pairs = []
    scene_folders = sorted(
        f for f in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(f)
    )
    if not scene_folders:
        raise FileNotFoundError(f"No scene folders found under {data_root!r}")

    for scene_folder in scene_folders:
        scene_id = os.path.basename(scene_folder)
        img_dir  = os.path.join(scene_folder, "images")
        msk_dir  = os.path.join(scene_folder, "masks")

        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            print(f"[WARNING] Skipping {scene_id!r}: missing images/ or masks/ subfolder")
            continue

        image_files = sorted(glob.glob(os.path.join(img_dir, "*")))
        mask_files  = sorted(glob.glob(os.path.join(msk_dir, "*")))

        if len(image_files) != len(mask_files):
            raise ValueError(
                f"Scene {scene_id!r}: {len(image_files)} images vs "
                f"{len(mask_files)} masks — counts must match."
            )

        for idx, (img_path, msk_path) in enumerate(zip(image_files, mask_files)):
            pairs.append((scene_id, idx, img_path, msk_path))

    return pairs


def build_tar(data_root: str, output_path: str, dry_run: bool = False) -> None:
    pairs = collect_tile_pairs(data_root)
    total = len(pairs)
    print(f"Found {total} tile pairs across "
          f"{len(set(p[0] for p in pairs))} scenes.")

    if dry_run:
        print("\n[DRY RUN] First 10 entries that would be written:")
        for scene_id, idx, img, msk in pairs[:10]:
            key = f"{scene_id}_tile_{idx:04d}"
            print(f"  {key}_image.tif  ←  {img}")
            print(f"  {key}_mask.tif   ←  {msk}")
        print(f"\n  ... and {max(0, total - 10)} more pairs.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with tarfile.open(output_path, "w") as tar:
        for i, (scene_id, idx, img_path, msk_path) in enumerate(pairs):
            key = f"{scene_id}_tile_{idx:04d}"

            tar.add(img_path, arcname=f"{key}_image.tif")
            tar.add(msk_path, arcname=f"{key}_mask.tif")

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [{i + 1}/{total}] added {key}")

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\nDone. Written to: {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build a WebDataset .tar from GeoTIFF tile folders."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the scenes/ folder containing per-scene subfolders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the .tar file, e.g. /data/hf_upload/kelp_tiles.tar",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be added without writing the tar.",
    )
    args = parser.parse_args()
    build_tar(args.data_root, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
