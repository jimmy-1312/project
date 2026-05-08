"""Depth Anything V2 wrapper. Outputs metric depth (m) for the *Metric-* variants,
otherwise unitless [0,1] depth (higher = closer)."""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

import config


_AGG_MODES = {"mean", "median", "max", "min", "top_k", "top_p"}


class DepthEstimator:
    def __init__(self, model_name: str = None, device: str = None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.model_name = model_name or config.DEPTH_MODEL_NAME
        self.device = device or config.DEVICE
        self.is_metric = "Metric" in self.model_name

        print(f"DepthEstimator: {self.model_name} on {self.device} "
              f"({'metric' if self.is_metric else 'relative'})")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Image → depth map (H,W) float32. Metric model: meters. Relative: [0,1] (higher=closer)."""
        if isinstance(image, np.ndarray):
            if image.size == 0:
                return np.zeros((0, 0), dtype=np.float32)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise TypeError(f"expected ndarray or PIL.Image, got {type(image)}")

        H, W = pil.size[1], pil.size[0]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        if self.is_metric:
            return np.where(np.isfinite(pred), pred, 0.0).astype(np.float32)

        # Relative: min-max normalize → [0, 1], higher = closer
        d_min, d_max = pred.min(), pred.max()
        if d_max > d_min:
            return (pred - d_min) / (d_max - d_min)
        return np.zeros_like(pred, dtype=np.float32)

    def scale_depth_to_meters(
        self,
        depth_map: np.ndarray,
        gt_depth: Optional[np.ndarray] = None,
        max_depth: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Convert relative depth → metric. With GT: least-squares fit. Without: linear×max_depth."""
        max_d = max_depth or getattr(config, "MAX_DEPTH_M", 10.0)
        min_d = getattr(config, "MIN_DEPTH_M", 0.1)

        if gt_depth is not None and np.isfinite(gt_depth).sum() > 10:
            valid = ((gt_depth > min_d) & (gt_depth < max_d)
                     & np.isfinite(depth_map) & np.isfinite(gt_depth))
            if np.sum(valid) > 10:
                y, x = gt_depth[valid], depth_map[valid]
                A = np.vstack([x, np.ones(len(x))]).T
                scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
                return np.clip(scale * depth_map + shift, 0, max_d), float(scale), float(shift)

        return np.clip(depth_map * max_d, 0, max_d), float(max_d), 0.0


def _aggregate_values(values: np.ndarray, mode: str, k: Optional[int],
                      p: Optional[float], closest_side: str) -> float:
    """Reduce 1-D depth samples to a scalar.
    closest_side: 'high' (relative depth) or 'low' (metric)."""
    if mode not in _AGG_MODES:
        raise ValueError(f"mode must be one of {_AGG_MODES}")
    if closest_side not in ("high", "low"):
        raise ValueError("closest_side must be 'high' or 'low'")

    values = np.asarray(values, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    if mode == "mean":   return float(values.mean())
    if mode == "median": return float(np.median(values))
    if mode == "max":    return float(values.max())
    if mode == "min":    return float(values.min())

    if mode == "top_k":
        if not (k and k > 0):
            raise ValueError("top_k requires k > 0")
        n = int(min(k, values.size))
    else:  # top_p
        if not (p and 0.0 < p <= 1.0):
            raise ValueError("top_p requires 0 < p <= 1")
        n = max(1, int(np.ceil(values.size * p)))

    if closest_side == "high":
        kept = np.partition(values, -n)[-n:]
    else:
        kept = np.partition(values, n - 1)[:n]
    return float(kept.mean())
