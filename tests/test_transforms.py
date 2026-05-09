"""
tests/test_transforms.py
------------------------
Unit tests for src/transforms.py — augmentation pipelines.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.transforms import get_training_augmentation, get_validation_augmentation


class TestGetTrainingAugmentation:
    def test_returns_compose(self):
        import albumentations as A
        aug = get_training_augmentation()
        assert isinstance(aug, A.Compose)

    def test_image_shape_preserved(self):
        aug = get_training_augmentation()
        rng   = np.random.default_rng(0)
        image = rng.integers(0, 255, (512, 512, 13), dtype=np.uint8).astype(np.float32)
        mask  = rng.integers(0, 2,   (512, 512, 1),  dtype=np.uint8).astype(np.float32)
        result = aug(image=image, mask=mask)
        assert result["image"].shape == image.shape
        assert result["mask"].shape  == mask.shape

    def test_mask_values_stay_binary(self):
        """Augmentation must never introduce non-binary values in the mask."""
        aug = get_training_augmentation()
        rng   = np.random.default_rng(1)
        image = rng.integers(0, 200, (512, 512, 13), dtype=np.uint8).astype(np.float32)
        mask  = rng.integers(0, 2,   (512, 512, 1),  dtype=np.uint8).astype(np.float32)
        for _ in range(10):
            result = aug(image=image, mask=mask)
            unique = np.unique(result["mask"])
            assert set(unique.tolist()).issubset({0.0, 1.0}), (
                f"Non-binary mask values after augmentation: {unique}"
            )

    def test_custom_probabilities(self):
        """Verify the factory accepts custom probability arguments without error."""
        aug = get_training_augmentation(
            hflip_p=1.0,
            vflip_p=1.0,
            rotate90_p=1.0,
            affine_p=0.0,
        )
        rng   = np.random.default_rng(2)
        image = rng.integers(0, 255, (512, 512, 13), dtype=np.uint8).astype(np.float32)
        mask  = rng.integers(0, 2,   (512, 512, 1),  dtype=np.uint8).astype(np.float32)
        result = aug(image=image, mask=mask)
        assert result["image"].shape == image.shape

    def test_non_square_image(self):
        """Pipeline must handle non-square spatial dimensions."""
        aug   = get_training_augmentation()
        rng   = np.random.default_rng(3)
        image = rng.integers(0, 200, (256, 512, 13), dtype=np.uint8).astype(np.float32)
        mask  = rng.integers(0, 2,   (256, 512, 1),  dtype=np.uint8).astype(np.float32)
        result = aug(image=image, mask=mask)
        assert result["image"].shape == image.shape
        assert result["mask"].shape  == mask.shape


class TestGetValidationAugmentation:
    def test_returns_none(self):
        """Validation pipeline should be None (no augmentation)."""
        assert get_validation_augmentation() is None
