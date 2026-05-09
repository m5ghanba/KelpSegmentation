# KelpSegmentation

Binary semantic segmentation of kelp / no-kelp on Sentinel-2 satellite imagery
along the British Columbia coast.

- **Input:** Multi-band GeoTIFF tiles (spectral bands + substrate + bathymetry + slope)
- **Output:** Binary mask GeoTIFF (1 = kelp, 0 = background)
- **Architecture:** Any `segmentation-models-pytorch` model (default: U-Net with MaxViT encoder)
- **Experiment tracking:** MLflow (runs stored locally in `mlruns/`, view with `mlflow ui`)

---

## Repository structure

```
KelpSegmentation/
├── data/                        # raw tiles — NOT committed (gitignored)
│   └── <SceneName>/
│       ├── images/              # multi-band GeoTIFFs
│       └── masks/               # single-band binary masks
├── notebooks/
│   ├── 01_preprocess_scenes.ipynb                # First step preprocessing for either training or inference
│   └── 02_preprocess_training_tiles.ipynb               # Second step training (only) preprocessing
├── src/
│   ├── __init__.py
│   ├── dataset.py               # SatelliteDataset + split strategies
│   ├── transforms.py            # Albumentations pipelines
│   ├── model.py                 # PyTorch Lightning segmentation module
│   ├── train.py                 # Training CLI entry point
│   ├── evaluate.py              # Metric computation utilities
│   └── utils.py                 # Visualisation helpers
├── configs/
│   ├── train_config.yaml        # Training configuration
│   └── infer_config.yaml        # Inference configuration
├── checkpoints/                 # saved model weights (gitignored)
├── outputs/                     # inference results (gitignored)
├── inference.py                 # Inference CLI entry point
├── pyproject.toml
└── README.md
```

---

## Environment setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd KelpSegmentation

# 2. Create a virtual environment (Python 3.10, 3.11, or 3.12)
python -m venv .venv

# 3. Activate
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 4. Install setuptools (avoids deprecation warnings on Python 3.12+)
pip install "setuptools<81"

# 5. Install the package + all runtime dependencies
pip install -e .

# For GPU support (CUDA 12.1 example):
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Note: rasterio and GDAL are easiest to install via conda:
#   conda install -c conda-forge rasterio gdal
```

---

## Data layout

Each scene folder must contain two sub-directories:

```
<data_root>/
└── SceneName/
    ├── images/
    │   ├── tile_0_image.tiff
    │   └── ...
    └── masks/
        ├── tile_0_mask.tiff
        └── ...
```

Multi-band TIFF band layout (1-based):

| Band | Content      | Resolution |
|------|-------------|------------|
| 1    | Blue        | 10 m       |
| 2    | Green       | 10 m       |
| 3    | Red         | 10 m       |
| 4    | NIR         | 10 m       |
| 5    | Red-edge    | 20 m (resampled) |
| 11   | Substrate   | —          |
| 12   | Bathymetry  | —          |
| 13   | Slope       | —          |

Masks are single-band GeoTIFFs with values 0 (background / water / land) and 1 (kelp).

---

## Input channels

Both `band_names` and `vi_names` are configurable in the YAML files.
The model's `in_channels` is computed automatically from these lists.

**Available raw bands:** `blue green red nir rededge substrate bathymetry slope`

**Available vegetation indices:** `ndvi ndwi gndvi clgreen ndvire`

---

## Training

Edit `configs/train_config.yaml`, then run:

```bash
python src/train.py --config configs/train_config.yaml
```

The script will:
1. Build the data split (geographic / random / LOO).
2. Compute per-channel mean & std from training tiles.
3. Train with early stopping and checkpoint saving.
4. Evaluate on the held-out test set.
5. Log everything (params, metrics, mean/std JSON, checkpoint) to MLflow.

**View training metrics:**

```bash
mlflow ui
# Open http://localhost:5000
```

### Data split strategies

| Strategy    | When to use |
|-------------|-------------|
| `geographic` | Default. Splits by scene folder → geographically independent train / val / test. |
| `random`     | All tiles pooled and split at tile level. Use when you want to train on as much data as possible. |
| `loo`        | Specific scene folder(s) held out as test; remaining tiles split train / val. Used for LOO cross-validation experiments. |

Configure via `data.split_strategy` in `train_config.yaml`.

### Key training options

| Field | Description |
|-------|-------------|
| `model.architecture` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, … |
| `model.encoder` | Any smp encoder, e.g. `tu-maxvit_tiny_tf_512`, `resnet18` |
| `training.loss` | `dice` (default), `bce_dice`, `tversky`, `jaccard`, `focal_dice` |
| `training.epochs` | Max training epochs |
| `training.patience` | Early stopping patience (epochs without val IoU improvement) |
| `data.split_strategy` | `geographic`, `random`, or `loo` |
| `data.band_names` | Which raw bands to load |
| `data.vi_names` | Which vegetation indices to compute |

---

### Fine-tuning for new regions

When applying SKeMa to a new geographic region outside British Columbia, fine-tuning from the BC-trained checkpoint is strongly recommended over training from scratch.

**Why fine-tuning is almost always better:**

Training from scratch discards everything the model learned about spectral signatures, bathymetric gradients, and kelp appearance. Fine-tuning from the BC weights gives a warm start, converges 3–5× faster, and typically reaches a higher final IoU — especially when the new dataset is small. The model's encoder has already learned to distinguish water, substrate, and vegetation from multi-spectral imagery; only the regional calibration needs updating.

**When to train from scratch instead:**

- The input channel configuration has changed (`band_names` or `vi_names` differ from the original run), making the checkpoint weights dimensionally incompatible.
- You are combining the new regional data with the full BC dataset and retraining on the pooled set.
- The new region uses a different sensor or has spectral characteristics so different from the BC coast that the old weights are genuinely misleading (rare within Sentinel-2).

**How to fine-tune:**

1. Train the base model on the BC dataset as normal. Note the checkpoint path.
2. Create a new config file (e.g. `configs/train_config_newregion.yaml`) and set:

```yaml
finetune:
  enabled:    true
  checkpoint: "checkpoints/best_bc_model.ckpt"

  # Freeze encoder for the first 10 epochs when the new dataset is small
  # (typically < ~500 tiles). Lets the decoder adapt to regional statistics
  # before the full network is updated.
  freeze_encoder: true
  freeze_epochs:  10

training:
  epochs:  100          # fine-tuning typically needs fewer epochs
  lr:      0.00005      # 4–5× lower than scratch; avoids overwriting learned features
  patience: 15
```

3. Run training as usual:

```bash
python src/train.py --config configs/train_config_newregion.yaml
```

The script will print which weights were loaded and whether any keys were missing or unexpected. Fine-tuning params are logged to MLflow alongside all other hyperparameters for full experiment traceability.

---


## Inference

Edit `configs/infer_config.yaml` — in particular set `data.mean` and `data.std`
to the values printed by `train.py` — then run:

```bash
python inference.py --config configs/infer_config.yaml
```

For each scene folder inside `inference.input_dir`, the script reads the
multi-band TIFF, runs tiled inference, and writes a binary GeoTIFF prediction.

### Tiling methods

| Method | Description |
|--------|-------------|
| `weighted` | Gaussian-weighted averaging of overlapping tile probability maps. Best quality (default). |
| `halo` | Uses only the central region of each tile, discards border halo. |
| `max` | Element-wise maximum of overlapping binary predictions. Fast. |

---

## Augmentation rationale

The following transforms are applied during training:

| Transform | Rationale |
|-----------|-----------|
| `HorizontalFlip` / `VerticalFlip` | Safe for all channels |


The following transforms are **excluded**:

| Transform | Reason for exclusion |
|-----------|----------------------|
| `RandomBrightnessContrast` | Corrupts spectral ratios (NDVI, NDWI, …) |
| `RandomGamma` | Same issue |
| `RandomRotate90` | Safe for satellite imagery - not applied|
| `Affine` (translate ±10 %, scale 0.9–1.1, rotate ±15°) | Geometric robustness  - not applied|
| `GaussNoise` | Corrupts bathymetry precision |
| `ElasticTransform` | Distorts kelp-bed boundaries unnaturally |

---

## Known installation issues and fixes

### Build backend error
If you see `BackendUnavailable: Cannot import 'setuptools.backends.legacy'`, the `pyproject.toml` already uses the correct backend (`setuptools.build_meta`). Make sure you have a recent setuptools:
```bash
pip install "setuptools>=64"
```

### GDAL on Windows
GDAL cannot be compiled from source on Windows without a full C++ toolchain. Install it from a pre-built wheel **before** running `pip install -e .`:

```bash
# Find the correct wheel for your Python version at:
# https://github.com/cgohlke/geospatial-wheels/releases
pip install GDAL-3.9.2-cp312-cp312-win_amd64.whl   # adjust filename
pip install -e .
```

GDAL is intentionally excluded from `pyproject.toml` because the correct wheel varies by OS and Python version.

### NumPy 2.x compatibility
Several compiled extensions (rasterio, segmentation-models-pytorch) are built against NumPy 1.x and will crash at import under NumPy 2.x. The `pyproject.toml` already pins `numpy<2`. If you see the warning after installing, force-reinstall:

```bash
pip install -e . --force-reinstall
```

### poetry.lock (optional — reproducible installs)
A `poetry.lock` file pins every transitive dependency to the exact tested version. To generate one:

```bash
pip install poetry
# Make sure pyproject.toml is present, then:
poetry lock
```

Commit `poetry.lock` to the repo. Collaborators can then run `poetry install` to reproduce your exact environment. This is most useful for long-term reproducibility; for day-to-day use the `pip install -e .` approach is simpler.

### Mean/std JSON path on Windows
The stats are written to `artifacts/normalization/` in the working directory.

---

## Pre-processing pipelines

Pre-processing uses **Jupyter notebooks** rather than CLI scripts. This is the standard approach in the remote sensing community for data preparation: it lets you set paths once at the top, run cells step-by-step, inspect intermediate rasters, and safely skip steps that have already been completed. To use these notebooks, ensure your notebook IDE is using your existing venv (install `ipykernel`, run `python -m ipykernel install --user --name kelp-env --display-name "Python (kelp-env)"`, then select the **"Python (kelp-env)"** kernel in your IDE).

The two notebooks live in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `01_preprocess_scenes.ipynb` | Extract the five scene files needed for either training or inference from raw `.SAFE` folders |
| `02_preprocess_training_tiles.ipynb` | Convert CVAT XML annotations + scene files into training tiles |

---

### Inference pre-processing (`01_preprocess_scenes.ipynb`)

**Input:** A directory of Sentinel-2 `.SAFE` folders.  
**Output:** For each `.SAFE`, a sub-folder containing the five files required by `inference.py`:

```
<SAFE_DIR>/
└── S2A_MSIL2A_20230801_..._T09UXQ/
    ├── S2A_...T09UXQ_B2B3B4B8.tif
    ├── S2A_...T09UXQ_B5B6B7B8A_B11B12.tif
    ├── S2A_...T09UXQ_Substrate.tif
    ├── S2A_...T09UXQ_Bathymetry.tif
    └── S2A_...T09UXQ_Slope.tif
```

**Steps:**

1. **Open the notebook** and set all paths in the first (configuration) cell:
   - `SAFE_DIR` — directory containing your `.SAFE` folders
   - `COASTWIDE_BATHYMETRY` / `COASTWIDE_SLOPE_OUTPUT` — coastwide bathymetry and where to write the slope
   - `BOPS_SUBSTRATE_DIR` / `RF_SUBSTRATE_DIR` — directories for the two substrate datasets
   - `BOPS_SCENE_IDENTIFIERS` — scene name fragments that should use BoPs substrate (default: `["UXQ", "UXS", "UDU"]`)
   - Fill values for NoData pixels

2. **Cell 1** *(run once):* Computes the coastwide slope from the bathymetry using Horn's algorithm. Skips automatically if the output already exists.

3. **Cell 2:** Extracts the 10 m (`_B2B3B4B8.tif`) and 20 m (`_B5B6B7B8A_B11B12.tif`) spectral band stacks from each `.SAFE` folder. Applies the processing baseline ≥ 4.0 DN offset correction.

4. **Cell 3:** Warps bathymetry, slope, and per-region substrate files onto each scene's 10 m grid using `gdalwarp`. Scene identifiers determine whether BoPs or RF substrate files are used.

5. **Cell 4:** Merges the per-region substrate tiles into a single `_Subs.tif` and deletes the intermediate regional files.

6. **Cell 5:** Fills NoData/NaN pixels with the configured fill values and renames files to the final suffixes (`_Substrate.tif`, `_Bathymetry.tif`, `_Slope.tif`).

**Requirements:** `gdalwarp` must be on your PATH (installed with GDAL).

---

### Training pre-processing (`02_preprocess_training_tiles.ipynb`)

**Input:** Scene folders that already contain the five files from notebook 01, plus annotation files.  
**Output:** `images/` and `masks/` sub-folders of `TILE_SIZE × TILE_SIZE` GeoTIFF tiles, ready for `src/train.py`.

**Full workflow:**

#### Step 1 — Annotate in CVAT

1. Create a false-colour PNG (NIR–R–G) of your scene to highlight kelp.
2. Upload to [CVAT](https://app.cvat.ai) and annotate kelp beds as **polygons** with label `kelp` (or `Kelp`). Mark confirmed non-kelp areas with label `no-kelp`.
3. Export annotations as **CVAT XML 1.1** format and save `annotations.xml` in the scene folder.

#### Step 2 — Run Cell 1 (CVAT XML → binary GeoTIFF mask)

Set `SCENES_DIR`, `CVAT_XML_FILENAME`, and `S2_REFERENCE_SUFFIX` in the configuration cell, then run Cell 1. It reads the exported XML, burns all polygons labelled `kelp` or `Kelp` as **1** and leaves everything else as **0**, then writes a georeferenced binary mask named `*_Mask.tif` (using the same base name as the `*_B2B3B4B8.tif` file). No intermediate multiclass file is needed.

#### Step 3 — Create `_TileBound.tif` in QGIS (manual, per scene)

This step prevents tiles that partially overlap the annotation boundary from entering training — those tiles contain real kelp that was not annotated, so including them would teach the model to treat kelp as background.

1. Open `*_B2B3B4B8.tif` and `*_Mask.tif` in QGIS.
2. Create a new vector layer (*Layer → Create Layer → New Shapefile Layer*), geometry type **Polygon**, CRS matching `*_B2B3B4B8.tif`.
3. Toggle editing and draw polygons that tightly enclose the **fully annotated** region. Ensure all unannotated kelp falls outside.
4. Save layer edits.
5. Rasterize (*Raster → Conversion → Rasterize*): burn value = **1**, extent = `*_B2B3B4B8.tif`, resolution = 10 m. Save as `*_TileBound.tif` in the scene folder.

**Skip this step** if the scene is fully annotated — Cell 2 will include all kelp-containing tiles.

#### Step 4 — Run Cell 2 (tile extraction)

Run Cell 2 to slide a `TILE_SIZE × TILE_SIZE` window (stride `STRIDE`) over each scene. A tile is written only if it contains at least one kelp pixel and — when `_TileBound.tif` is present — all pixels in the tile fall inside the annotated boundary.

Output tile channel layout (13 channels, Float32):

| Channels | Content |
|----------|---------|
| 0–3 | Blue, Green, Red, NIR (10 m) |
| 4–9 | B5, B6, B7, B8A, B11, B12 (20 m, upsampled ×2 to match 10 m grid) |
| 10 | Substrate (from `*_Substrate.tif`) |
| 11 | Bathymetry |
| 12 | Slope |

