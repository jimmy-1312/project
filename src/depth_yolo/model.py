"""
RGB-D first-conv surgery for YOLOv8.

YOLOv8's stem is `Conv(3 → 64, k=3, s=2)` (or 32/16 for n/s sizes). To accept
an RGB-D input we replace it with `Conv(4 → C, k=3, s=2)`. The 4th channel is
initialized as the per-output-channel mean of the RGB weights — a documented
"average-init" trick that lets the new model start with sensible activations
on the new channel without zeroing existing learned features.

Public API:
    convert_first_conv_to_rgbd(model, init="rgb_mean") → modified model (in place)
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def _find_first_conv(yolo_model):
    """
    Walk an Ultralytics YOLO wrapper and return its very first Conv2d.

    Layout (as of ultralytics 8.x):
        YOLO (wrapper) -> .model (DetectionModel)
                              -> .model[0] (Conv module)
                                  -> .conv (nn.Conv2d)

    We're conservative: also handle the case where `.model` is already
    a Sequential. Raises if no nn.Conv2d is found in the first stem.
    """
    import torch.nn as nn  # lazy

    inner = yolo_model.model if hasattr(yolo_model, "model") else yolo_model
    if hasattr(inner, "model"):  # DetectionModel.model is a Sequential
        inner = inner.model

    first_block = inner[0]
    # Search any nn.Conv2d submodule (handles wrapper Conv class)
    for m in first_block.modules():
        if isinstance(m, nn.Conv2d):
            return first_block, m
    raise RuntimeError("No nn.Conv2d found in the first block of the model.")


def convert_first_conv_to_rgbd(
    yolo_model,
    init: Literal["rgb_mean", "zero", "random"] = "rgb_mean",
):
    """
    Replace the first Conv2d's input channels from 3 to 4. Returns the same
    YOLO wrapper so callers can chain.

    init: how to initialize the 4th input channel of the new conv weight:
        "rgb_mean" — mean of the existing R/G/B kernels per output channel
                     (preserves response magnitude; recommended)
        "zero"     — zero — depth contributes nothing initially, must learn
                     from scratch (slower but cleanest ablation)
        "random"   — same Kaiming init as a fresh layer (loses the stem
                     entirely; not recommended)

    The new conv inherits stride/padding/bias/dilation/groups from the old
    one. The replacement is in-place: `yolo_model.model.model[0].conv` is
    overwritten.
    """
    import torch
    import torch.nn as nn

    block, old_conv = _find_first_conv(yolo_model)

    if old_conv.in_channels == 4:
        logger.info("First conv already has in_channels=4; skipping surgery.")
        return yolo_model
    if old_conv.in_channels != 3:
        raise ValueError(
            f"Expected in_channels=3, got {old_conv.in_channels}. "
            "This conversion only handles standard RGB stems."
        )

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        # First three input channels: copy as-is.
        new_conv.weight[:, :3] = old_conv.weight

        # Fourth input channel.
        if init == "rgb_mean":
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        elif init == "zero":
            new_conv.weight[:, 3:4].zero_()
        elif init == "random":
            nn.init.kaiming_normal_(new_conv.weight[:, 3:4])
        else:
            raise ValueError(f"Unknown init mode: {init!r}")

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    # Replace on the parent block. We try several attribute names that
    # ultralytics has used historically.
    for attr in ("conv", "0", "_modules"):
        if hasattr(block, attr) and getattr(block, attr) is old_conv:
            setattr(block, attr, new_conv)
            break
    else:
        # Fallback: walk the parent's modules dict and swap by identity.
        for name, child in block.named_children():
            if child is old_conv:
                setattr(block, name, new_conv)
                break
        else:
            raise RuntimeError(
                "Found the old Conv2d but could not locate its parent attribute "
                "for replacement. Open an issue with the YOLO model layout."
            )

    logger.info(
        f"Converted first conv: {old_conv.in_channels}→4, init={init!r}, "
        f"out_channels={old_conv.out_channels}, kernel={old_conv.kernel_size}"
    )
    return yolo_model
