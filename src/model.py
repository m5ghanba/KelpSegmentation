"""
src/model.py
------------
PyTorch Lightning module wrapping any segmentation-models-pytorch architecture
for binary kelp / no-kelp segmentation.

Supported losses
----------------
  dice       – DiceLoss (default)
  bce_dice   – 0.5 × BCE + 0.5 × Dice
  tversky    – TverskyLoss (set alpha/beta in config)
  jaccard    – JaccardLoss
  focal_dice – 0.5 × FocalLoss + 0.5 × DiceLoss

Metrics reported per epoch
--------------------------
  Kelp class    : per-image IoU, dataset IoU, precision, recall, F1
  Background    : same (derived by swapping tp/tn/fp/fn)
  Overall       : pixel accuracy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

try:
    import pytorch_lightning as pl
except ImportError:
    raise ImportError("pytorch_lightning is required: pip install pytorch-lightning")


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """Return a loss function by name."""
    loss_name = loss_name.lower()

    if loss_name == "dice":
        return smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    elif loss_name == "bce_dice":
        bce_w  = kwargs.get("bce_weight",  0.5)
        dice_w = kwargs.get("dice_weight", 0.5)
        bce  = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        def combined(pred, target):
            return bce_w * bce(pred, target) + dice_w * dice(pred, target)
        return combined  # type: ignore[return-value]

    elif loss_name == "tversky":
        alpha = kwargs.get("tversky_alpha", 0.5)
        beta  = kwargs.get("tversky_beta",  0.5)
        return smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE, from_logits=True, alpha=alpha, beta=beta
        )

    elif loss_name == "jaccard":
        return smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)

    elif loss_name == "focal_dice":
        focal = smp.losses.FocalLoss(mode="binary")
        dice  = smp.losses.DiceLoss(mode="binary", from_logits=True)
        def combined(pred, target):
            return 0.5 * dice(pred, target) + 0.5 * focal(torch.sigmoid(pred), target)
        return combined  # type: ignore[return-value]

    else:
        raise ValueError(f"Unknown loss: {loss_name!r}. "
                         "Choose from: dice, bce_dice, tversky, jaccard, focal_dice")


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class KelpSegmentationModel(pl.LightningModule):
    """
    Parameters
    ----------
    arch : str
        smp architecture name, e.g. "Unet", "UnetPlusPlus", "DeepLabV3Plus".
    encoder_name : str
        Any smp-supported encoder, e.g. "tu-maxvit_tiny_tf_512", "resnet18".
    in_channels : int
        Number of input channels (raw bands + vegetation indices).
    out_classes : int
        1 for binary segmentation.
    lr : float
        Initial learning rate.
    t_max : int
        T_max for CosineAnnealingLR (typically epochs × steps_per_epoch).
    eta_min : float
        Minimum LR for the cosine scheduler.
    loss_name : str
        See build_loss() for options.
    loss_kwargs : dict
        Extra keyword arguments forwarded to build_loss().
    encoder_weights : str or None
        Pre-trained weights for the encoder, e.g. "imagenet" or None.
    optimizer_name : str
        "adam" or "adamw".
    weight_decay : float
        Weight decay for AdamW.
    threshold : float
        Sigmoid threshold for converting logits to binary predictions.
    """

    def __init__(
        self,
        arch:             str   = "Unet",
        encoder_name:     str   = "tu-maxvit_tiny_tf_512",
        in_channels:      int   = 13,
        out_classes:      int   = 1,
        lr:               float = 2e-4,
        t_max:            int   = 1000,
        eta_min:          float = 1e-5,
        loss_name:        str   = "dice",
        loss_kwargs:      Optional[Dict[str, Any]] = None,
        encoder_weights:  Optional[str] = None,
        optimizer_name:   str   = "adam",
        weight_decay:     float = 1e-4,
        threshold:        float = 0.5,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=encoder_weights,
            **model_kwargs,
        )

        self.loss_fn   = build_loss(loss_name, **(loss_kwargs or {}))
        self.threshold = threshold

        self.training_step_outputs:   List[Dict] = []
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs:       List[Dict] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Shared step / epoch-end
    # ------------------------------------------------------------------

    def shared_step(self, batch, stage: str) -> Dict:
        image, mask = batch

        assert image.ndim == 4, "Expected (B, C, H, W)"
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, (
            f"Image spatial dims must be divisible by 32; got {h}×{w}."
        )
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0.0

        logits = self.forward(image)
        loss   = self.loss_fn(logits, mask)

        prob_mask = logits.sigmoid()
        pred_mask = (prob_mask > self.threshold).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs: List[Dict], stage: str) -> None:
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # ---- Kelp (target) class ----------------------------------------
        kelp_per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        kelp_dataset_iou   = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        kelp_precision      = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        kelp_recall         = smp.metrics.recall(   tp, fp, fn, tn, reduction="micro")
        kelp_f1             = smp.metrics.f1_score(  tp, fp, fn, tn, reduction="micro")

        # ---- Background class -------------------------------------------
        bg_tp, bg_fp, bg_fn, bg_tn = tn, fn, fp, tp
        bg_per_image_iou = smp.metrics.iou_score(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro-imagewise")
        bg_dataset_iou   = smp.metrics.iou_score(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro")
        bg_precision      = smp.metrics.precision(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro")
        bg_recall         = smp.metrics.recall(   bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro")
        bg_f1             = smp.metrics.f1_score(  bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro")

        # ---- Overall accuracy -------------------------------------------
        accuracy = (tp.sum() + tn.sum()).float() / (tp.sum() + fp.sum() + fn.sum() + tn.sum()).float()

        metrics = {
            f"{stage}_kelp_per_image_iou": kelp_per_image_iou,
            f"{stage}_kelp_dataset_iou":   kelp_dataset_iou,
            f"{stage}_kelp_precision":     kelp_precision,
            f"{stage}_kelp_recall":        kelp_recall,
            f"{stage}_kelp_f1":            kelp_f1,
            f"{stage}_bg_per_image_iou":   bg_per_image_iou,
            f"{stage}_bg_dataset_iou":     bg_dataset_iou,
            f"{stage}_bg_precision":       bg_precision,
            f"{stage}_bg_recall":          bg_recall,
            f"{stage}_bg_f1":              bg_f1,
            f"{stage}_accuracy":           accuracy,
        }
        self.log_dict(metrics, prog_bar=True)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
        if hp["optimizer_name"].lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=hp["lr"])

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp["t_max"], eta_min=hp["eta_min"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler":  scheduler,
                "interval":   "step",
                "frequency":  1,
            },
        }
