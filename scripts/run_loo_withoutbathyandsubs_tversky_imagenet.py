"""
Run 10 Leave-One-Out (LOO) training experiments for KelpSegmentation.

Purpose:
    Generate one YAML config for each LOO experiment, where each experiment
    uses a different scene/tile as the test set (indices 0-9).

For each experiment:
    - Only `data.loo.test_indices` is changed.
    - The generated config is saved under:
        configs/loo_withoutbathyandsubs_tversky_imagenet/
    - Config names are:
        loo_test0.yaml
        loo_test1.yaml
        ...
        loo_test9.yaml

Usage:
    1. Generate the 10 configs only:
        python scripts/run_loo_withoutbathyandsubs_tversky_imagenet.py

    2. Generate the configs and run all 10 training experiments:
        python scripts/run_loo_withoutbathyandsubs_tversky_imagenet.py --execute

Notes:
    - Experiments are run sequentially.
    - Each experiment uses one test index and the remaining data for
      training/validation according to the LOO split configuration.


    - Make sure the base config uses:
        split_strategy: "loo"
      otherwise the `data.loo.test_indices` setting will not be used.


    - Checkpoint and MLflow output locations should be configured so that
      results from different LOO experiments are not overwritten.


    -This assumes a train_config.yaml of:
# ──────────────────────────────────────────────────────────────────────────────
# KelpSegmentation — Training Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Run:  python src/train.py --config configs/train_config.yaml

# ── Data ──────────────────────────────────────────────────────────────────────
data:
  # Glob pattern that expands to scene folders (each folder contains images/ masks/).
  # Example: "/data/Annotated/G0512Stride256RevisedOnly10Scenes_BoPsSubstrate_combined/*"
  data_root: "C:/Annotated Dataset/L2A Data/Test_Tiles_afterRSPaperrevision/*"

  # ── Split strategy ──────────────────────────────────────────────────────────
  # Choose one of: "geographic" | "random" | "loo"
  split_strategy: "loo"

  # geographic: shuffle folders, split by folder (preserves geographic independence)
  geographic:
    train_frac: 0.70
    val_frac:   0.15
    # test receives the remainder

  # random: pool all tiles and split at tile level (use when you want max data for training to get the best model, in which case set 0.0015 for val_frac and test_frac)
  random:
    val_frac:  0.0015
    test_frac: 0.0015

  # loo (leave-one-out): hold out specific scene folders as test; This can be multiple folders so leave ONE out is perhaps not the best name
  # remaining folders' tiles are split train/val at tile level.
  # Folder order for 10-scene dataset:
  #   0:uws  1:uxr  2:uxq  3:uyq  4:ucu  5:uxs  6:uca  7:uwt  8:udu  9:uuu
  loo:
    test_indices: [0]     # e.g. [9,1,3] or [0,2,6]
    val_frac: 0.20        # fraction of remaining tiles held out for val
    # val_mode: "scene"   # "tile", default, or "scene"
    # val_indices: [0]

  random_seed: 42

  # ── Channels ────────────────────────────────────────────────────────────────
  # Which raw bands to load from the TIFF (in this order).
  # Available: blue, green, red, nir, rededge, substrate, bathymetry, slope
  band_names:
    - blue
    - green
    - red
    - nir
    - rededge
    #- substrate
    #- bathymetry
    #- slope

  # Which vegetation indices to compute and append after the raw bands.
  # Available: ndvi, ndwi, gndvi, clgreen, ndvire
  vi_names:
    - ndvi
    - ndwi
    - gndvi
    - clgreen
    - ndvire

  # If true, compute mean/std from training data before training (recommended).
  # If false, supply precomputed_mean / precomputed_std below.
  compute_mean_std: true

  # Precomputed normalization stats — only used when compute_mean_std: false.
  # Length must equal len(band_names) + len(vi_names).
  # precomputed_mean: [145.0, 120.0, 100.0, 180.0, 130.0, 1.5, -10.0, 5.0, 0.1, -0.1, 0.05, 5.0, 0.2]
  # precomputed_std:  [ 30.0,  25.0,  22.0,  40.0,  28.0, 0.8,  50.0, 2.0, 0.2,  0.3, 0.15, 8.0, 0.3]

  # DataLoader settings
  batch_size:  8
  num_workers: 8

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  # Any smp architecture: Unet, UnetPlusPlus, DeepLabV3Plus, FPN, PSPNet, …
  architecture:    "Unet"
  # Any smp encoder: tu-maxvit_tiny_tf_512, resnet18, resnet50, efficientnet-b0, …
  encoder:         "tu-maxvit_tiny_tf_512"
  # Pre-trained encoder weights. Use null for no pre-training (required for
  # non-standard in_channels that don't match imagenet's 3 channels).
  encoder_weights: imagenet
  # Binary segmentation → always 1.
  out_classes: 1
  # in_channels is computed automatically from band_names + vi_names.
  # Override here only if you need to hard-code a specific value.
  # in_channels: 13

# ── Training ──────────────────────────────────────────────────────────────────
training:
  epochs:        50
  lr:            0.0002
  eta_min:       0.00001      # CosineAnnealingLR minimum LR
  weight_decay:  0.0001
  optimizer:     "adam"       # "adam" or "adamw"
  patience:      10           # early stopping patience (monitored: valid_kelp_per_image_iou)

  # ── Loss ──────────────────────────────────────────────────────────────────
  # Options: dice | bce_dice | tversky | jaccard | focal_dice
  loss: "tversky"
  # Extra loss kwargs (only relevant for some losses)
  bce_weight:    0.5          # for bce_dice
  dice_weight:   0.5          # for bce_dice
  tversky_alpha: 0.3          # for tversky  (α > β → penalise FP more)
  tversky_beta:  0.7          # for tversky  (β > α → penalise FN more)

  # Sigmoid threshold for binary prediction during metric computation.
  threshold: 0.5

  # Use mixed precision ("medium" or "high") — helps on Ampere+ GPUs.
  float32_matmul_precision: "medium"

# ── Augmentation ──────────────────────────────────────────────────────────────
# See src/transforms.py for notes on which transforms are safe.
augmentation:
  hflip_p:    0.5
  vflip_p:    0.5
  rotate90_p: 0.5
  affine_p:   0.3

# ── Checkpoints ───────────────────────────────────────────────────────────────
checkpoints:
  dir:      "checkpoints"     # relative to repo root; gitignored
  monitor:  "valid_kelp_per_image_iou"
  mode:     "max"             # save when monitored metric improves
  # Enable checkpointing only in the last N epochs (set 0 to always checkpoint)
  save_last_n_epochs: 0

# ── Fine-tuning ───────────────────────────────────────────────────────────────
# Set enabled: true to initialise the model from a pre-trained checkpoint
# instead of random weights.  All other training settings (epochs, lr,
# augmentation, data split, …) apply as normal.
#
# When to use fine-tuning vs. training from scratch
# ──────────────────────────────────────────────────
# Fine-tune (enabled: true) when:
#   • You have a new geographic region with limited labelled data.
#   • The new sensor and spectral bands match the original training data.
#   • You want faster convergence and a better starting point.
#
# Train from scratch (enabled: false) when:
#   • The input channel configuration has changed (different band_names /
#     vi_names), making the pre-trained weights incompatible.
#   • You are combining the new data with the original dataset and retraining
#     on the full pool.
#   • The new region is spectrally so different that the old weights
#     are likely to be actively misleading.
#
# Recommended fine-tuning settings for a new region:
#   epochs:   50–100  (fewer than scratch; early stopping usually triggers)
#   lr:       0.00005 (5× lower than scratch; avoids overwriting learned features)
#   freeze_encoder: true  (if new dataset is very small, < ~500 tiles)
finetune:
  enabled:        false

  # Path to the .ckpt file saved by a previous training run.
  checkpoint:     "checkpoints/best_model.ckpt"

  # If true, freeze all encoder weights for the first `freeze_epochs` epochs,
  # then unfreeze and continue training the full network.
  # Useful when the new dataset is small and you want to avoid over-writing
  # the rich spectral features learned on the original BC dataset.
  freeze_encoder:  false
  freeze_epochs:   10    # only used when freeze_encoder: true
  
# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow:
  experiment_name: "KelpSegmentation"
  tracking_uri:    "mlruns"   # local folder; change to http://... for remote server
  run_name:        null       # auto-generated if null
  log_model:       false      # set true to log the full model artifact (large)
  tags:
    project: "kelp-monitoring"
    dataset: "Sentinel-2"


"""


from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run 10 LOO training experiments."
    )

    parser.add_argument(
        "--template",
        default="configs/train_config.yaml",
        help="Path to the base training config."
    )

    parser.add_argument(
        "--output-dir",
        default="configs/loo_withoutbathyandsubs_tversky_imagenet",
        help="Directory where the generated LOO configs will be saved."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run all 10 training experiments sequentially."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read the original training configuration
    # ------------------------------------------------------------------
    with open(args.template, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    generated_configs = []

    # ------------------------------------------------------------------
    # Generate one config for each LOO test scene
    # ------------------------------------------------------------------
    for test_index in range(10):

        # Make a fresh copy so each experiment starts from the
        # exact same original configuration.
        config = yaml.safe_load(
            yaml.safe_dump(base_config, sort_keys=False)
        )

        # Change ONLY the LOO test index.
        config["data"]["loo"]["test_indices"] = [test_index]

        # Output filename
        output_path = os.path.join(
            args.output_dir,
            f"loo_test{test_index}.yaml"
        )

        # Save config
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
                allow_unicode=True
            )

        generated_configs.append(output_path)

        print(f"Wrote: {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Generated {len(generated_configs)} LOO configurations")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    for path in generated_configs:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        test_index = cfg["data"]["loo"]["test_indices"]
        print(f"  {path}  -> test_indices: {test_index}")

    # ------------------------------------------------------------------
    # Run experiments sequentially if requested
    # ------------------------------------------------------------------
    if args.execute:

        print()
        print("=" * 70)
        print("Running all 10 LOO experiments sequentially")
        print("=" * 70)

        for i, config_path in enumerate(generated_configs, start=1):

            print()
            print("=" * 70)
            print(f"LOO EXPERIMENT {i}/10")
            print(f"Config: {config_path}")
            print("=" * 70)

            subprocess.run(
                [
                    sys.executable,
                    "src/train.py",
                    "--config",
                    config_path,
                ],
                check=True,
            )

        print()
        print("=" * 70)
        print("All 10 LOO experiments completed successfully.")
        print("=" * 70)

    else:
        print()
        print("Configs were generated but NOT executed.")
        print()
        print("To run all 10 experiments sequentially:")
        print(
            f"  python {os.path.basename(__file__)} --execute"
        )


if __name__ == "__main__":
    main()