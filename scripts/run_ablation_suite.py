"""
scripts/run_ablation_suite.py
-------------------------------
Generates one train_config.yaml per (experiment, seed) on a SINGLE fixed
scene split — the 7-train / 1-val / 2-test design you proposed, reused
across every reviewer-requested comparison so we don't have to rerun full
LOO/three-fold for each one:

  - indices:            bands only  vs.  bands + 5 indices
  - substrate encoding:  none  vs.  scalar  vs.  one-hot
  - augmentation:        flip  vs.  flip+rot90  vs.  flip+affine
  - loss:                dice / bce_dice / tversky / jaccard / focal_dice
  - pretrained weights:  encoder_weights=None  vs.  "imagenet"

Each experiment is run at `--seeds` (default 1 2 3) training seeds, with the
data split held IDENTICAL across every run (same test/val scene indices,
same data.random_seed) — only training.seed (model init / shuffling /
augmentation randomness) varies. See src/train.py's docstring for why
these two seeds are deliberately separate.

This script only WRITES configs (and, optionally, a shell script listing
the train.py commands) — it does not itself launch potentially 100+ GPU
training runs. Review the generated configs, then either:
  - loop over them yourself (your own scheduler / tmux / SLURM array), or
  - pass --execute to run them here, sequentially, one after another
    (only sensible for a quick local check — NOT recommended for the full
    suite; sequential single-GPU training of everything below is a lot
    of wall-clock time).

Fixed split scenes (EDIT THESE to whatever you'd rather use — they are
arbitrary, chosen only to be distinct from LOO folder index 1, which is
reserved for the full-scene evaluation fold):
  test_indices = [0, 6]   (folders "uws", "uca")
  val_indices  = [8]      (folder "udu")
  -> train on the remaining 7 folders.

Every generated run is MLflow-tagged with `ablation_group` and `variant`
(and train.py additionally tags `training_seed` / `data_random_seed`
automatically) so scripts/compile_ablation_tables.py can group results
without re-parsing every YAML.

Usage
-----
    python scripts/run_ablation_suite.py \\
        --data-root "/path/to/Annotated/*" \\
        --config-out-dir configs/ablation \\
        --seeds 1 2 3
    # then, per generated config:
    python src/train.py --config configs/ablation/<group>/<variant>_seed1.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from typing import Any, Dict, List

import yaml

ALL_BAND_NAMES = ["blue", "green", "red", "nir", "rededge"]
ALL_VI_NAMES   = ["ndvi", "ndwi", "gndvi", "clgreen", "ndvire"]

FIXED_TEST_INDICES = [0, 6]
FIXED_VAL_INDICES  = [8]


def base_config(data_root: str, tracking_uri: str, experiment_name: str,
                 epochs: int, batch_size: int, num_workers: int) -> Dict[str, Any]:
    return {
        "data": {
            "data_root": data_root,
            "split_strategy": "loo",
            "random_seed": 42,  # fixed — does not vary; irrelevant to the
                                # split itself once val_mode="scene" with
                                # explicit indices, kept only for logging.
            "loo": {
                "test_indices": FIXED_TEST_INDICES,
                "val_mode": "scene",
                "val_indices": FIXED_VAL_INDICES,
            },
            "band_names": list(ALL_BAND_NAMES),
            "vi_names": list(ALL_VI_NAMES),
            "substrate_onehot": False,
            "compute_mean_std": True,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model": {
            "architecture": "Unet",
            "encoder": "tu-maxvit_tiny_tf_512",
            "out_classes": 1,
            "encoder_weights": None,
        },
        "training": {
            "epochs": epochs,
            "lr": 2e-4,
            "eta_min": 1e-5,
            "loss": "dice",
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "threshold": 0.5,
            "patience": 10,
            "seed": 1,
        },
        "augmentation": {
            "mode": "flip",
            "hflip_p": 0.5, "vflip_p": 0.5, "rotate90_p": 0.5, "affine_p": 0.3,
        },
        "checkpoints": {
            "dir": "checkpoints/ablation",
            "monitor": "valid_kelp_per_image_iou",
            "mode": "max",
        },
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "log_model": True,
            "tags": {},
        },
    }


def build_experiments() -> List[Dict[str, Any]]:
    """
    Each entry: {"group": ..., "variant": ..., "overrides": {dotted-ish nested
    dict merged into base_config}}. Keep this list edited to taste — it's
    meant to be a starting point covering every reviewer comment, not a
    fixed prescription.
    """
    exps: List[Dict[str, Any]] = []

    # ---- 1. Indices: does the CNN need precomputed indices? (R1) -----------
    exps.append({"group": "indices", "variant": "bands_only",
                  "overrides": {"data": {"band_names": list(ALL_BAND_NAMES), "vi_names": []}}})
    exps.append({"group": "indices", "variant": "bands_plus_indices",
                  "overrides": {"data": {"band_names": list(ALL_BAND_NAMES), "vi_names": list(ALL_VI_NAMES)}}})

    # ---- 2. Substrate encoding (R1: one-hot vs scalar; also a "none" control)
    sat_bands = list(ALL_BAND_NAMES)
    exps.append({"group": "substrate_encoding", "variant": "none",
                  "overrides": {"data": {"band_names": sat_bands, "substrate_onehot": False}}})
    exps.append({"group": "substrate_encoding", "variant": "scalar",
                  "overrides": {"data": {"band_names": sat_bands + ["substrate"], "substrate_onehot": False}}})
    exps.append({"group": "substrate_encoding", "variant": "one_hot",
                  "overrides": {"data": {"band_names": sat_bands, "substrate_onehot": True}}})

    # ---- 3. Augmentation (R1 + code/manuscript mismatch found in review) ---
    for mode in ("flip", "flip_rot90", "flip_affine"):
        exps.append({"group": "augmentation", "variant": mode,
                      "overrides": {"augmentation": {"mode": mode}}})

    # ---- 4. Loss (R1 + R2) --------------------------------------------------
    loss_variants = [
        ("dice", {}),
        ("bce_dice", {"bce_weight": 0.5, "dice_weight": 0.5}),
        ("tversky_fn_heavy", {"tversky_alpha": 0.3, "tversky_beta": 0.7}),  # penalize FN more
        ("jaccard", {}),
        ("focal_dice", {"focal_weight": 0.5, "dice_weight": 0.5}),
    ]
    for name, kwargs in loss_variants:
        loss_name = "tversky" if name == "tversky_fn_heavy" else name
        exps.append({"group": "loss", "variant": name,
                      "overrides": {"training": {"loss": loss_name, **kwargs}}})

    # ---- 5. Pretrained vs. random init (R2) ---------------------------------
    exps.append({"group": "pretrained", "variant": "random_init",
                  "overrides": {"model": {"encoder_weights": None}}})
    exps.append({"group": "pretrained", "variant": "imagenet",
                  "overrides": {"model": {"encoder_weights": "imagenet"}}})

    return exps


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-out-dir", default="configs/ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--experiment-name", default="KelpSegmentation-Ablation")
    parser.add_argument("--execute", action="store_true",
                         help="Actually run `python src/train.py --config ...` for every "
                              "generated config, sequentially. Off by default — see "
                              "module docstring for why.")
    args = parser.parse_args()

    base = base_config(args.data_root, args.tracking_uri, args.experiment_name,
                        args.epochs, args.batch_size, args.num_workers)
    experiments = build_experiments()

    generated_paths: List[str] = []
    for exp in experiments:
        group, variant, overrides = exp["group"], exp["variant"], exp["overrides"]
        group_dir = os.path.join(args.config_out_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        for seed in args.seeds:
            cfg = deep_merge(base, overrides)
            cfg["training"]["seed"] = seed
            cfg["mlflow"]["run_name"] = f"{group}__{variant}__seed{seed}"
            cfg["mlflow"]["tags"] = {
                **cfg["mlflow"].get("tags", {}),
                "ablation_group": group,
                "variant": variant,
            }
            cfg["checkpoints"]["dir"] = os.path.join("checkpoints", "ablation", group, f"{variant}_seed{seed}")

            out_path = os.path.join(group_dir, f"{variant}_seed{seed}.yaml")
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            generated_paths.append(out_path)
            print(f"Wrote {out_path}")

    print(f"\n{len(generated_paths)} configs written under {args.config_out_dir}/")
    print(f"Fixed split for ALL of them — test_indices={FIXED_TEST_INDICES}, "
          f"val_indices={FIXED_VAL_INDICES} (edit FIXED_TEST_INDICES / "
          f"FIXED_VAL_INDICES at the top of this script if you want different scenes).")

    if args.execute:
        print("\n--execute set: running every config sequentially now …")
        for path in generated_paths:
            print(f"\n=== {path} ===")
            subprocess.run([sys.executable, "src/train.py", "--config", path], check=True)
    else:
        print("\nNot executing (pass --execute to run sequentially, or launch the "
              "configs yourself). Example single run:")
        print(f"  python src/train.py --config {generated_paths[0]}")


if __name__ == "__main__":
    main()
