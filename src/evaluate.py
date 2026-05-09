"""
src/evaluate.py
---------------
Stand-alone metric computation for kelp segmentation predictions.

Useful both from within train.py (test-set evaluation) and from external
analysis scripts or notebooks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch


def compute_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute a full set of binary segmentation metrics from confusion-matrix
    components accumulated over a dataset.

    Parameters
    ----------
    tp, fp, fn, tn : torch.Tensor
        Per-image confusion-matrix components, shape (N, 1).
    prefix : str
        Optional string prepended to every key (e.g. "test_").

    Returns
    -------
    dict of {metric_name: float}
    """
    p = prefix

    # ---- Kelp (target) class --------------------------------------------
    kelp_iou_img     = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
    kelp_iou_dataset = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    kelp_precision   = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
    kelp_recall      = smp.metrics.recall(   tp, fp, fn, tn, reduction="micro").item()
    kelp_f1          = smp.metrics.f1_score(  tp, fp, fn, tn, reduction="micro").item()

    # ---- Background class -----------------------------------------------
    bg_tp, bg_fp, bg_fn, bg_tn = tn, fn, fp, tp
    bg_iou_img     = smp.metrics.iou_score(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro-imagewise").item()
    bg_iou_dataset = smp.metrics.iou_score(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro").item()
    bg_precision   = smp.metrics.precision(bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro").item()
    bg_recall      = smp.metrics.recall(   bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro").item()
    bg_f1          = smp.metrics.f1_score(  bg_tp, bg_fp, bg_fn, bg_tn, reduction="micro").item()

    # ---- Overall accuracy -----------------------------------------------
    accuracy = ((tp.sum() + tn.sum()).float() /
                (tp.sum() + fp.sum() + fn.sum() + tn.sum()).float()).item()

    return {
        f"{p}kelp_per_image_iou": kelp_iou_img,
        f"{p}kelp_dataset_iou":   kelp_iou_dataset,
        f"{p}kelp_precision":     kelp_precision,
        f"{p}kelp_recall":        kelp_recall,
        f"{p}kelp_f1":            kelp_f1,
        f"{p}bg_per_image_iou":   bg_iou_img,
        f"{p}bg_dataset_iou":     bg_iou_dataset,
        f"{p}bg_precision":       bg_precision,
        f"{p}bg_recall":          bg_recall,
        f"{p}bg_f1":              bg_f1,
        f"{p}accuracy":           accuracy,
    }


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
    prefix: str = "test_",
) -> Dict[str, float]:
    """
    Run a model over an entire DataLoader and return aggregated metrics.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode.
    loader : DataLoader
        Yields (image, mask) batches.
    device : torch.device
    threshold : float
        Sigmoid threshold for binarisation.
    prefix : str
        Prepended to metric keys.

    Returns
    -------
    dict of {metric_name: float}
    """
    model.eval()
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits    = model(images)
            prob      = logits.sigmoid()
            pred_mask = (prob > threshold).float()

            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), masks.long(), mode="binary"
            )
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

    return compute_metrics(
        torch.cat(all_tp),
        torch.cat(all_fp),
        torch.cat(all_fn),
        torch.cat(all_tn),
        prefix=prefix,
    )
