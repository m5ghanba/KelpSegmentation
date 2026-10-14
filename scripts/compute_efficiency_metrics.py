"""
compute_efficiency_metrics.py
-----------------------------
Parameter count + computational complexity (MACs) + inference throughput
(FPS) for the architectures used in the efficiency-comparison table.

FPS is measured as model forward-pass throughput on the GPU available to
PyTorch. For the manuscript, this should be run on the same NVIDIA RTX 4070
used for model training/inference.

The FPS measurement:
    - uses a dummy 512 x 512 x C tensor
    - runs several warm-up iterations
    - synchronizes CUDA before and after timing
    - excludes model loading, raster I/O, preprocessing, postprocessing,
      and visualization
    - reports images/tiles per second for batch_size=1

Usage
-----
    pip install thop

    python scripts/compute_efficiency_metrics.py --in-channels 10 --tile-size 512

Optional:
    python scripts/compute_efficiency_metrics.py \
        --in-channels 10 \
        --tile-size 512 \
        --fps-warmup 20 \
        --fps-iters 100

The script automatically uses CUDA if available. If CUDA is not available,
MACs and parameter counts are still reported, but FPS is skipped.

Known naming caveats
--------------------
- "Segformer" is a true SegFormer architecture with an MLP decode head,
  not a U-Net decoder variant, when supported by the installed
  segmentation-models-pytorch version.
- ConvNeXt / ResNet101 encoder names may need a "tu-" prefix depending on
  the installed segmentation-models-pytorch version. Both spellings are
  tried below.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class ArchSpec:
    label: str
    arch: str
    encoder_name: str
    encoder_weights: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


def default_architectures() -> List[ArchSpec]:
    """
    Mirrors the manuscript's architecture-comparison table plus the
    additional architectures requested for the efficiency comparison.
    """
    return [
        # ---- Main model + existing comparisons -----------------------------
        ArchSpec(
            "U-Net / MaxViT-Tiny (main model)",
            "Unet",
            "tu-maxvit_tiny_tf_512",
        ),
        ArchSpec(
            "U-Net / ResNet18",
            "Unet",
            "resnet18",
        ),
        ArchSpec(
            "U-Net / MiT-B1 (U-Net decoder, NOT true SegFormer)",
            "Unet",
            "mit_b1",
        ),
        ArchSpec(
            "U-Net / ConvNeXt-Tiny",
            "Unet",
            "convnext_tiny",
        ),
        ArchSpec(
            "DeepLabV3+ / MobileNetV2",
            "DeepLabV3Plus",
            "mobilenet_v2",
        ),

        # ---- Additional architectures ---------------------------------------
        ArchSpec(
            "UNet++ / ResNet34",
            "UnetPlusPlus",
            "resnet34",
        ),
        ArchSpec(
            "DeepLabV3+ / ResNet101",
            "DeepLabV3Plus",
            "resnet101",
        ),
        ArchSpec(
            "DeepLabV3+ / ConvNeXt-Tiny",
            "DeepLabV3Plus",
            "convnext_tiny",
        ),
        ArchSpec(
            "SegFormer-B0 (true, if supported)",
            "Segformer",
            "mit_b0",
        ),
        ArchSpec(
            "SegFormer-B2 (true, if supported)",
            "Segformer",
            "mit_b2",
        ),
    ]


def _try_create_model(
    spec: ArchSpec,
    in_channels: int,
    out_classes: int = 1,
):
    import segmentation_models_pytorch as smp

    attempts = [spec.encoder_name]

    if not spec.encoder_name.startswith("tu-"):
        attempts.append("tu-" + spec.encoder_name)

    last_exc = None

    for encoder_name in attempts:
        try:
            model = smp.create_model(
                spec.arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                encoder_weights=spec.encoder_weights,
                **spec.extra_kwargs,
            )
            return model, encoder_name

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    raise last_exc


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def compute_macs(
    model: torch.nn.Module,
    in_channels: int,
    tile_size: int,
) -> Optional[float]:
    """
    Returns MACs for one (1, C, H, W) forward pass.

    Note:
        FLOPs are approximately 2 x MACs.
    """
    try:
        from thop import profile
    except ImportError:
        print(
            "  [thop not installed - run `pip install thop` "
            "for MACs; skipping]"
        )
        return None

    dummy = torch.randn(
        1,
        in_channels,
        tile_size,
        tile_size,
    )

    model.eval()

    with torch.no_grad():
        macs, _params = profile(
            model,
            inputs=(dummy,),
            verbose=False,
        )

    return macs


def measure_fps(
    model: torch.nn.Module,
    in_channels: int,
    device: torch.device,
    tile_size: int = 512,
    batch_size: int = 1,
    n_warmup: int = 20,
    n_iters: int = 100,
) -> float:
    """
    Measure model forward-pass throughput in images/tiles per second.

    The timing excludes model construction and dummy-input creation.

    For CUDA:
        - warm-up iterations allow kernels/caches to settle
        - torch.cuda.synchronize() is called before and after timing
          because CUDA operations are asynchronous
    """

    model = model.to(device).eval()

    dummy = torch.randn(
        batch_size,
        in_channels,
        tile_size,
        tile_size,
        device=device,
    )

    with torch.no_grad():

        # --------------------------------------------------------------
        # Warm-up
        # --------------------------------------------------------------
        for _ in range(n_warmup):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # --------------------------------------------------------------
        # Timed forward passes
        # --------------------------------------------------------------
        start = time.perf_counter()

        for _ in range(n_iters):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    total_images = n_iters * batch_size

    return total_images / elapsed


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--in-channels",
        type=int,
        default=10,
        help=(
            "Number of input channels. For the main satellite "
            "configuration this is 10: 5 Sentinel-2 bands + "
            "5 spectral indices."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Spatial tile size.",
    )

    parser.add_argument(
        "--out-classes",
        type=int,
        default=1,
        help="Number of output classes.",
    )

    parser.add_argument(
        "--fps-warmup",
        type=int,
        default=20,
        help="Number of warm-up forward passes before FPS measurement.",
    )

    parser.add_argument(
        "--fps-iters",
        type=int,
        default=100,
        help="Number of timed forward passes for FPS measurement.",
    )

    parser.add_argument(
        "--fps-batch-size",
        type=int,
        default=1,
        help="Batch size used for FPS measurement.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print()
        print("=" * 100)
        print("GPU FPS measurement enabled")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device: {torch.cuda.current_device()}")
        print("=" * 100)
        print()

    else:
        device = torch.device("cpu")

        print()
        print("=" * 100)
        print("WARNING: CUDA is not available.")
        print("Parameter counts and MACs will be computed, but FPS will be")
        print("skipped because the manuscript FPS should be measured on the GPU.")
        print("=" * 100)
        print()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print(
        f"{'Architecture':<45}"
        f"{'Encoder used':<22}"
        f"{'Params (M)':>12}"
        f"{'MACs (G)':>12}"
        f"{'FPS':>12}"
    )

    print("-" * 103)

    rows = []

    # ------------------------------------------------------------------
    # Evaluate architectures
    # ------------------------------------------------------------------

    for spec in default_architectures():

        try:
            model, encoder_used = _try_create_model(
                spec,
                args.in_channels,
                args.out_classes,
            )

        except Exception as exc:  # noqa: BLE001
            print(
                f"{spec.label:<45}"
                f"[SKIPPED - {type(exc).__name__}: {exc}]"
            )
            continue

        # --------------------------------------------------------------
        # Parameter count
        # --------------------------------------------------------------

        total_params, _ = count_params(model)

        # --------------------------------------------------------------
        # MACs
        # --------------------------------------------------------------

        macs = compute_macs(
            model,
            args.in_channels,
            args.tile_size,
        )

        if macs is not None:
            macs_g = f"{macs / 1e9:.2f}"
        else:
            macs_g = "n/a"

        # --------------------------------------------------------------
        # FPS
        # --------------------------------------------------------------

        if device.type == "cuda":

            try:
                fps = measure_fps(
                    model=model,
                    in_channels=args.in_channels,
                    device=device,
                    tile_size=args.tile_size,
                    batch_size=args.fps_batch_size,
                    n_warmup=args.fps_warmup,
                    n_iters=args.fps_iters,
                )

                fps_display = f"{fps:.2f}"

            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [FPS measurement failed for {spec.label}: "
                    f"{type(exc).__name__}: {exc}]"
                )

                fps = None
                fps_display = "n/a"

        else:
            fps = None
            fps_display = "n/a"

        # --------------------------------------------------------------
        # Print row
        # --------------------------------------------------------------

        print(
            f"{spec.label:<45}"
            f"{encoder_used:<22}"
            f"{total_params / 1e6:>12.2f}"
            f"{macs_g:>12}"
            f"{fps_display:>12}"
        )

        rows.append(
            {
                "label": spec.label,
                "encoder_used": encoder_used,
                "params": total_params,
                "macs": macs,
                "fps": fps,
            }
        )

        # Free GPU memory before moving to the next architecture.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 103)
    print("FPS measurement details")
    print("=" * 103)
    print(f"Device:       {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Input:        {args.fps_batch_size} x {args.in_channels} x "
          f"{args.tile_size} x {args.tile_size}")
    print(f"Warm-up:      {args.fps_warmup} iterations")
    print(f"Timed passes: {args.fps_iters} iterations")
    print()
    print(
        "FPS represents model forward-pass throughput only and excludes "
        "raster I/O, preprocessing, postprocessing, and visualization."
    )
    print("=" * 103)


if __name__ == "__main__":
    main()