The tiling notebook (02_preprocess_training_tiles.ipynb) and split_loo / split_geographic / split_random functions work against the GeoTIFF folder structure. The build_webdataset_tar.py script above is a one-time export tool that reads those same folders and repackages them into a tar. 

Step 1 — Dry-run to verify
python build_webdataset_tar.py \
    --data_root /path/to/geotiff_tiles/scenes \
    --output    /tmp/kelp_tiles.tar \
    --dry_run
Check that the printed filenames look correct (scene ID prefix, zero-padded tile index, _image.tif / _mask.tif). Confirm the total tile count matches what you expect.

Step 2 — Build the tar
bashpython build_webdataset_tar.py \
    --data_root /path/to/geotiff_tiles/scenes \
    --output    /path/to/hf_upload/kelp_tiles.tar
This may take a few minutes depending on disk speed. The script prints progress every 100 tiles and reports the final file size.