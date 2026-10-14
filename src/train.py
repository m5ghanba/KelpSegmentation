"""
src/train.py
------------
Training CLI entry point for kelp / no-kelp segmentation.

Usage
-----
    python src/train.py --config configs/train_config.yaml

What this script does
---------------------
1. Loads config from YAML.
2. Selects the data-split strategy (geographic / random / loo).
3. Optionally computes per-channel mean & std from the training set.
4. Builds Dataset, DataLoaders, model, and Lightning Trainer.
5. Optionally loads pre-trained weights for fine-tuning a new region
   (set finetune.enabled: true in the config).
6. Trains with early stopping and model checkpointing.
7. Runs test-set evaluation after training.
8. Logs all hyperparameters, metrics, and the best checkpoint to MLflow.

CHANGES vs. the original (for the manuscript revision):
  - `training.seed` (default 42) is now applied via `pl.seed_everything()`,
    SEPARATELY from `data.random_seed`. This lets you hold the data split
    perfectly fixed (same data.random_seed / same loo.test_indices /
    loo.val_indices) while varying only model init + dataloader shuffling +
    augmentation randomness across `training.seed` — exactly what the
    multi-seed comparison tables (indices, substrate encoding, loss,
    augmentation, architecture, pretrained-vs-scratch) need. Before this
    change there was no explicit seeding at all outside of `random.seed()`
    inside the split functions, so repeated runs of the same config were not
    actually reproducible in model init / shuffling.
  - `data.loo.val_mode` ("tile", default, or "scene") and `data.loo.val_indices`
    are now passed through to `split_loo()` — see src/dataset.py for the
    leak-free scene-disjoint validation option.
  - `data.substrate_onehot` is passed through to SatelliteDataset; in_channels
    resolution now delegates to `src.dataset.resolve_num_channels` so it can
    never disagree with what SatelliteDataset actually produces.
  - `augmentation.mode` ("flip", "flip_rot90", "flip_affine") is passed
    through to `get_training_augmentation()`.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import MLFlowLogger
except ImportError:
    raise ImportError("pytorch_lightning is required: pip install pytorch-lightning")

# ---- local imports ---------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    SatelliteDataset,
    compute_dataset_mean_std,
    split_geographic,
    split_random,
    split_loo,
    resolve_num_channels,
)
from src.model import KelpSegmentationModel
from src.transforms import get_training_augmentation, get_validation_augmentation


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_in_channels(cfg: Dict[str, Any]) -> int:
    """Derive in_channels from band_names + vi_names (+ substrate_onehot) in data config."""
    data_cfg = cfg["data"]
    n = resolve_num_channels(
        data_cfg.get("band_names", []),
        data_cfg.get("vi_names",   []),
        substrate_onehot=data_cfg.get("substrate_onehot", False),
    )
    # Allow explicit override
    return cfg.get("model", {}).get("in_channels", n)


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def build_splits(cfg: Dict[str, Any]):
    data_cfg   = cfg["data"]
    strategy   = data_cfg.get("split_strategy", "geographic").lower()
    data_root  = data_cfg["data_root"]
    seed       = data_cfg.get("random_seed", 42)

    if strategy == "geographic":
        geo = data_cfg.get("geographic", {})
        return split_geographic(
            data_root,
            train_frac=geo.get("train_frac", 0.70),
            val_frac=geo.get("val_frac",     0.15),
            seed=seed,
        )
    elif strategy == "random":
        rand = data_cfg.get("random", {})
        return split_random(
            data_root,
            val_frac=rand.get("val_frac",   0.15),
            test_frac=rand.get("test_frac", 0.15),
            seed=seed,
        )
    elif strategy == "loo":
        loo = data_cfg.get("loo", {})
        return split_loo(
            data_root,
            test_indices=loo["test_indices"],
            val_frac=loo.get("val_frac", 0.20),
            seed=seed,
            val_mode=loo.get("val_mode", "tile"),
            val_indices=loo.get("val_indices", None),
        )
    else:
        raise ValueError(f"Unknown split_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> None:
    # ------------------------------------------------------------------ setup
    data_cfg   = cfg["data"]
    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    aug_cfg    = cfg.get("augmentation", {})
    ckpt_cfg   = cfg.get("checkpoints", {})
    mlflow_cfg = cfg.get("mlflow", {})

    # ---- Seeding ------------------------------------------------------------
    # Deliberately separate from data_cfg["random_seed"]: that seed controls
    # WHICH tiles/scenes go into train/val/test (must stay fixed across a
    # multi-seed comparison run); training.seed controls model init,
    # dataloader shuffling, and augmentation randomness (the thing you
    # actually want to vary across seeds 1/2/3 while holding the split fixed).
    training_seed = train_cfg.get("seed", 42)
    pl.seed_everything(training_seed, workers=True)

    torch.set_float32_matmul_precision(
        train_cfg.get("float32_matmul_precision", "high")
    )

    band_names       = data_cfg.get("band_names", None)
    vi_names         = data_cfg.get("vi_names",   None)
    substrate_onehot = data_cfg.get("substrate_onehot", False)
    in_channels      = resolve_in_channels(cfg)

    # ------------------------------------------------------------------ splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_splits(cfg)

    # ------------------------------------------------------------------ augmentation
    train_aug = get_training_augmentation(
        mode=aug_cfg.get("mode", "flip"),
        hflip_p=aug_cfg.get("hflip_p",    0.5),
        vflip_p=aug_cfg.get("vflip_p",    0.5),
        rotate90_p=aug_cfg.get("rotate90_p", 0.5),
        affine_p=aug_cfg.get("affine_p",   0.3),
    )
    val_aug = get_validation_augmentation()

    # ------------------------------------------------------------------ mean/std
    if data_cfg.get("compute_mean_std", True):
        print("Computing mean and std from training data …")
        temp_ds = SatelliteDataset(
            X_train, y_train,
            classes=["kelp"],
            band_names=band_names,
            vi_names=vi_names,
            substrate_onehot=substrate_onehot,
        )
        mean, std = compute_dataset_mean_std(
            temp_ds,
            batch_size=32,
            num_workers=data_cfg.get("num_workers", 4),
        )
        print(f"  mean = {mean.tolist()}")
        print(f"  std  = {std.tolist()}")
    else:
        mean = data_cfg.get("precomputed_mean")
        std  = data_cfg.get("precomputed_std")
        if mean is None or std is None:
            raise ValueError(
                "compute_mean_std is false but precomputed_mean / precomputed_std "
                "are not set in the config."
            )
        import numpy as np
        mean = np.array(mean, dtype="float32")
        std  = np.array(std,  dtype="float32")

    # ------------------------------------------------------------------ datasets
    train_ds = SatelliteDataset(
        X_train, y_train,
        classes=["kelp"],
        augmentation=train_aug,
        mean=mean, std=std,
        band_names=band_names,
        vi_names=vi_names,
        substrate_onehot=substrate_onehot,
    )
    val_ds = SatelliteDataset(
        X_val, y_val,
        classes=["kelp"],
        augmentation=val_aug,
        mean=mean, std=std,
        band_names=band_names,
        vi_names=vi_names,
        substrate_onehot=substrate_onehot,
    )
    test_ds = SatelliteDataset(
        X_test, y_test,
        classes=["kelp"],
        augmentation=val_aug,
        mean=mean, std=std,
        band_names=band_names,
        vi_names=vi_names,
        substrate_onehot=substrate_onehot,
    )

    bs = data_cfg.get("batch_size", 8)
    nw = data_cfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))

    # ------------------------------------------------------------------ model
    epochs = train_cfg.get("epochs", 50)
    t_max  = epochs * len(train_loader)

    model = KelpSegmentationModel(
        arch=model_cfg.get("architecture",    "Unet"),
        encoder_name=model_cfg.get("encoder", "tu-maxvit_tiny_tf_512"),
        in_channels=in_channels,
        out_classes=model_cfg.get("out_classes", 1),
        lr=train_cfg.get("lr",           2e-4),
        t_max=t_max,
        eta_min=train_cfg.get("eta_min", 1e-5),
        loss_name=train_cfg.get("loss",  "dice"),
        loss_kwargs={
            "bce_weight":    train_cfg.get("bce_weight",    0.5),
            "dice_weight":   train_cfg.get("dice_weight",   0.5),
            "tversky_alpha": train_cfg.get("tversky_alpha", 0.5),
            "tversky_beta":  train_cfg.get("tversky_beta",  0.5),
        },
        encoder_weights=model_cfg.get("encoder_weights", None),
        optimizer_name=train_cfg.get("optimizer",  "adam"),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        threshold=train_cfg.get("threshold",       0.5),
    )

    # ------------------------------------------------------------------ fine-tune
    ft_cfg = cfg.get("finetune", {}) or {}
    if ft_cfg.get("enabled", False):
        ft_ckpt = ft_cfg.get("checkpoint")
        if not ft_ckpt:
            raise ValueError(
                "finetune.enabled is true but finetune.checkpoint is not set."
            )
        if not os.path.exists(ft_ckpt):
            raise FileNotFoundError(
                f"Fine-tune checkpoint not found: {ft_ckpt}"
            )

        # Load only the model weights — not the optimizer/scheduler state.
        # This lets the new training run use fresh LR scheduling while
        # inheriting all learned feature representations.
        state = torch.load(ft_ckpt, map_location="cpu")
        # Lightning checkpoints store weights under "state_dict"
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"\nFine-tuning from: {ft_ckpt}")
        if missing:
            print(f"  Keys not loaded (will be randomly initialised): {missing}")
        if unexpected:
            print(f"  Keys in checkpoint not in model (ignored): {unexpected}")
        print("  Weights loaded successfully.\n")

        # Optional: freeze encoder for the first N epochs so the decoder
        # can adapt to the new region's statistics before the full network
        # is updated.  The callback below unfreezes after freeze_epochs.
        if ft_cfg.get("freeze_encoder", False):
            freeze_epochs = ft_cfg.get("freeze_epochs", 10)
            print(f"  Encoder frozen for first {freeze_epochs} epoch(s).")
            for name, param in model.model.encoder.named_parameters():
                param.requires_grad = False

            class _UnfreezeEncoder(pl.Callback):
                """Unfreeze encoder weights after ``freeze_epochs`` epochs."""
                def __init__(self, n: int) -> None:
                    super().__init__()
                    self._n = n
                    self._unfrozen = False

                def on_train_epoch_start(self,
                                          trainer: pl.Trainer,
                                          pl_module: pl.LightningModule) -> None:
                    if not self._unfrozen and trainer.current_epoch >= self._n:
                        for param in pl_module.model.encoder.parameters():
                            param.requires_grad = True
                        self._unfrozen = True
                        print(f"\n  Epoch {trainer.current_epoch}: "
                              f"encoder unfrozen — full network now training.\n")

            extra_callbacks: list = [_UnfreezeEncoder(freeze_epochs)]
        else:
            extra_callbacks = []

        # Collected here, logged inside the MLflow run below
        _ft_mlflow_params = {
            "finetune.checkpoint":     ft_ckpt,
            "finetune.freeze_encoder": ft_cfg.get("freeze_encoder", False),
            "finetune.freeze_epochs":  (
                ft_cfg.get("freeze_epochs", 10)
                if ft_cfg.get("freeze_encoder", False) else "n/a"
            ),
        }
    else:
        extra_callbacks = []
        _ft_mlflow_params = {"finetune.enabled": False}
        print("Training from scratch (finetune.enabled: false).\n")

    # ------------------------------------------------------------------ callbacks
    ckpt_dir = ckpt_cfg.get("dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor = ckpt_cfg.get("monitor", "valid_kelp_per_image_iou")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_{epoch:03d}_{valid_kelp_per_image_iou:.4f}",
        monitor=monitor,
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=monitor,
        patience=train_cfg.get("patience", 10),
        mode="max",
        verbose=True,
    )

    # ------------------------------------------------------------------ MLflow
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "KelpSegmentation"))

    run_name = mlflow_cfg.get("run_name", None)
    tags     = mlflow_cfg.get("tags", {}) or {}
    # Always tag the training seed and data-split seed, even if the config's
    # own `mlflow.tags` doesn't mention them — makes it trivial to group runs
    # by "same split, different seed" later without re-parsing every config.
    tags = {**tags, "training_seed": str(training_seed),
            "data_random_seed": str(data_cfg.get("random_seed", 42))}

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Log all config values as flat parameters
        _log_params_flat(mlflow, cfg)
        # Log fine-tuning params (resolved at runtime, not just from config)
        for k, v in _ft_mlflow_params.items():
            mlflow.log_param(k, v)
        # Log mean/std as JSON artifacts
        _log_mean_std(mean, std, run.info.run_id)

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_cfg.get("experiment_name", "KelpSegmentation"),
            tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
            run_id=run.info.run_id,
        )

        # ---------------------------------------------------------------- train
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint_cb, early_stop_cb] + extra_callbacks,
            logger=mlflow_logger,
            log_every_n_steps=1,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        # ---------------------------------------------------------------- test
        best_ckpt = checkpoint_cb.best_model_path
        print(f"\nBest checkpoint: {best_ckpt}")
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=best_ckpt)

        # Log test metrics explicitly to MLflow
        if test_results:
            for key, val in test_results[0].items():
                mlflow.log_metric(key, val)

        # Log the checkpoint as an artifact
        if best_ckpt and mlflow_cfg.get("log_model", False):
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
        
        # Save pth file as well.
        if best_ckpt and os.path.isfile(best_ckpt):
            checkpoint = torch.load(
                best_ckpt, map_location="cpu", weights_only=False
            )
            best_pth = os.path.splitext(best_ckpt)[0] + ".pth"
            torch.save(checkpoint["state_dict"], best_pth)
            print(f"Best model weights saved: {best_pth}")

        print("\n── Test Results ─────────────────────────────")
        for k, v in (test_results[0] if test_results else {}).items():
            print(f"  {k}: {v:.4f}")
        print("─────────────────────────────────────────────")
        print(f"MLflow run ID: {run.info.run_id}")
        print("View with:  mlflow ui")


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _log_params_flat(mlf, cfg: Dict, prefix: str = "") -> None:
    """Recursively log all scalar config values as MLflow params."""
    for key, val in cfg.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            _log_params_flat(mlf, val, prefix=f"{full_key}.")
        elif isinstance(val, (list, tuple)):
            mlf.log_param(full_key, json.dumps(val))
        elif val is not None:
            mlf.log_param(full_key, val)


def _log_mean_std(mean, std, run_id: str) -> None:
    """Save computed mean/std as a JSON file and log to MLflow."""
    import numpy as np
    stats = {
        "mean": mean.tolist() if hasattr(mean, "tolist") else list(mean),
        "std":  std.tolist()  if hasattr(std,  "tolist") else list(std),
    }
    # create a local artifacts folder (portable)
    base_dir = os.path.join("artifacts", "normalization")
    os.makedirs(base_dir, exist_ok=True)

    path = os.path.join(base_dir, f"mean_std_{run_id[:8]}.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    mlflow.log_artifact(path, artifact_path="normalization")
    print(f"  Mean/std saved → {path}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a kelp segmentation model."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to train_config.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
