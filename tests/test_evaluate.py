"""
tests/test_evaluate.py
----------------------
Unit tests for src/evaluate.py — metric computation.

Tests use analytically-known confusion-matrix inputs so that expected values
can be calculated by hand.  Any future refactoring of the metric code will be
caught immediately if these values change.
"""

from __future__ import annotations

import pytest
import torch

from src.evaluate import compute_metrics, evaluate_loader


# ---------------------------------------------------------------------------
# Helper: build confusion-matrix tensors from scalar totals
# ---------------------------------------------------------------------------

def _cm(tp_n: int, fp_n: int, fn_n: int, tn_n: int, batch: int = 4):
    """
    Return per-image (N, 1) tensors that sum to the given totals.
    Distributes counts evenly across N=batch images.
    """
    def t(n):
        base = n // batch
        rem  = n % batch
        vals = [base] * batch
        for i in range(rem):
            vals[i] += 1
        return torch.tensor(vals, dtype=torch.long).unsqueeze(1)
    return t(tp_n), t(fp_n), t(fn_n), t(tn_n)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_prediction(self):
        """All pixels correct → IoU = 1, precision = 1, recall = 1, F1 = 1."""
        tp, fp, fn, tn = _cm(100, 0, 0, 100)
        m = compute_metrics(tp, fp, fn, tn)
        assert abs(m["kelp_dataset_iou"] - 1.0) < 1e-5
        assert abs(m["kelp_precision"]   - 1.0) < 1e-5
        assert abs(m["kelp_recall"]      - 1.0) < 1e-5
        assert abs(m["kelp_f1"]          - 1.0) < 1e-5
        assert abs(m["accuracy"]         - 1.0) < 1e-5

    def test_no_true_positives(self):
        """No TP → IoU = 0, precision = 0 (or undefined), recall = 0."""
        tp, fp, fn, tn = _cm(0, 50, 50, 100)
        m = compute_metrics(tp, fp, fn, tn)
        assert m["kelp_dataset_iou"] < 1e-5
        assert m["kelp_recall"]      < 1e-5

    def test_prefix_applied(self):
        """Prefix must be prepended to all returned keys."""
        tp, fp, fn, tn = _cm(10, 2, 3, 20)
        m = compute_metrics(tp, fp, fn, tn, prefix="test_")
        for key in m:
            assert key.startswith("test_"), f"Key missing prefix: {key}"

    def test_background_metrics_present(self):
        """Both kelp and background class metrics must be returned."""
        tp, fp, fn, tn = _cm(50, 10, 5, 80)
        m = compute_metrics(tp, fp, fn, tn)
        for key in [
            "kelp_dataset_iou", "kelp_precision", "kelp_recall", "kelp_f1",
            "bg_dataset_iou",   "bg_precision",   "bg_recall",   "bg_f1",
            "accuracy",
        ]:
            assert key in m, f"Missing metric: {key}"

    def test_all_values_finite(self):
        tp, fp, fn, tn = _cm(30, 5, 7, 60)
        m = compute_metrics(tp, fp, fn, tn)
        for key, val in m.items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"
            assert val == val, f"{key} is NaN"           # NaN check
            assert abs(val) < 1e9,         f"{key} is Inf: {val}"

    def test_known_iou(self):
        """
        tp=60, fp=20, fn=20, tn=100
        IoU = tp / (tp + fp + fn) = 60 / 100 = 0.60
        """
        tp, fp, fn, tn = _cm(60, 20, 20, 100)
        m = compute_metrics(tp, fp, fn, tn)
        assert abs(m["kelp_dataset_iou"] - 0.60) < 1e-4

    def test_known_accuracy(self):
        """
        tp=60, fp=20, fn=20, tn=100
        accuracy = (tp + tn) / total = 160 / 200 = 0.80
        """
        tp, fp, fn, tn = _cm(60, 20, 20, 100)
        m = compute_metrics(tp, fp, fn, tn)
        assert abs(m["accuracy"] - 0.80) < 1e-4


# ---------------------------------------------------------------------------
# evaluate_loader — smoke test with a minimal model
# ---------------------------------------------------------------------------

class TestEvaluateLoader:
    def test_runs_and_returns_metrics(self):
        """Verify evaluate_loader doesn't crash and returns a non-empty dict."""
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Tiny identity-ish model: outputs logits ≈ 0 → sigmoid ≈ 0.5
        class TinyModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])

        images = torch.randn(8, 13, 64, 64)
        masks  = torch.randint(0, 2, (8, 1, 64, 64)).float()
        loader = DataLoader(TensorDataset(images, masks), batch_size=4)

        model   = TinyModel()
        device  = torch.device("cpu")
        metrics = evaluate_loader(model, loader, device, prefix="test_")

        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert "test_kelp_dataset_iou" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
