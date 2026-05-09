"""
tests/test_model.py
-------------------
Unit tests for src/model.py — KelpSegmentationModel and loss factory.

Key things verified
-------------------
  - build_loss returns a callable for every supported loss name.
  - Model forward pass produces the expected output shape.
  - Spatial dimensions divisible by 32 are accepted; others raise an
    assertion (guarding against the classic "wrong input shape" runtime error).
  - shared_step runs end-to-end and returns all expected keys.
  - configure_optimizers returns a dict with the right keys.
"""

from __future__ import annotations

import pytest
import torch

from src.model import KelpSegmentationModel, build_loss


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------

class TestBuildLoss:
    @pytest.mark.parametrize("name", ["dice", "bce_dice", "tversky", "jaccard", "focal_dice"])
    def test_known_losses_callable(self, name):
        loss_fn = build_loss(name)
        assert callable(loss_fn)

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("unsupported_loss_xyz")

    def test_bce_dice_runs(self):
        loss_fn = build_loss("bce_dice", bce_weight=0.5, dice_weight=0.5)
        pred   = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss   = loss_fn(pred, target)
        assert loss.item() > 0

    def test_tversky_custom_alpha_beta(self):
        loss_fn = build_loss("tversky", tversky_alpha=0.3, tversky_beta=0.7)
        pred   = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss   = loss_fn(pred, target)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class TestModelConstruction:
    def test_default_construction(self):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            encoder_weights=None,
        )
        assert model is not None

    def test_forward_output_shape(self):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            encoder_weights=None,
        )
        model.eval()
        x   = torch.randn(1, 13, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_output_shape_batch(self):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            encoder_weights=None,
        )
        model.eval()
        x   = torch.randn(4, 13, 512, 512)
        out = model(x)
        assert out.shape == (4, 1, 512, 512)


# ---------------------------------------------------------------------------
# Dimension guard — the most important model test for CI
# ---------------------------------------------------------------------------

class TestDimensionGuard:
    """
    Spatial dims not divisible by 32 must raise an AssertionError in
    shared_step, catching the classic encoder/decoder shape mismatch early.
    """

    def _make_model(self):
        return KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            encoder_weights=None,
        )

    def _make_batch(self, H, W):
        image = torch.randn(1, 13, H, W)
        mask  = torch.randint(0, 2, (1, 1, H, W)).float()
        return image, mask

    @pytest.mark.parametrize("H,W", [(256, 256), (512, 512), (64, 128)])
    def test_divisible_by_32_accepted(self, H, W):
        model = self._make_model()
        batch = self._make_batch(H, W)
        # Should not raise
        result = model.shared_step(batch, "test")
        assert "loss" in result

    @pytest.mark.parametrize("H,W", [(65, 65), (513, 512), (100, 200)])
    def test_not_divisible_by_32_raises(self, H, W):
        model = self._make_model()
        batch = self._make_batch(H, W)
        with pytest.raises(AssertionError):
            model.shared_step(batch, "test")


# ---------------------------------------------------------------------------
# shared_step output keys
# ---------------------------------------------------------------------------

class TestSharedStep:
    def test_output_keys(self):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            encoder_weights=None,
        )
        image = torch.randn(2, 13, 256, 256)
        mask  = torch.randint(0, 2, (2, 1, 256, 256)).float()
        result = model.shared_step((image, mask), "train")
        for key in ("loss", "tp", "fp", "fn", "tn"):
            assert key in result, f"Missing key: {key}"

    def test_loss_is_finite(self):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            loss_name="dice",
            encoder_weights=None,
        )
        image = torch.randn(2, 13, 256, 256)
        mask  = torch.randint(0, 2, (2, 1, 256, 256)).float()
        result = model.shared_step((image, mask), "train")
        assert torch.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------

class TestConfigureOptimizers:
    @pytest.mark.parametrize("optim_name", ["adam", "adamw"])
    def test_returns_dict_with_optimizer(self, optim_name):
        model = KelpSegmentationModel(
            arch="Unet", encoder_name="resnet18",
            in_channels=13, out_classes=1,
            optimizer_name=optim_name,
            encoder_weights=None,
        )
        opt_cfg = model.configure_optimizers()
        assert "optimizer" in opt_cfg
        assert "lr_scheduler" in opt_cfg
