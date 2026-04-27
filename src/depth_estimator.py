"""
Depth Anything V2 Depth Estimation Module

This module provides monocular depth estimation using the Depth Anything V2 model.
It estimates relative depth maps and can convert them to metric depth.
"""

import numpy as np
import torch
from typing import List, Optional, Sequence, Tuple, Union
from PIL import Image
import cv2

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    AutoImageProcessor, AutoModelForDepthEstimation = None, None

import config


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
    ) -> None:
        self.model_name = model_name or getattr(config, 'DEPTH_MODEL_NAME', 'depth-anything/Depth-Anything-V2-Small-hf')
        self.device = device or getattr(config, 'DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Metric variants (Metric-Indoor / Metric-Outdoor) output absolute depth
        # in meters. Relative variants output unitless inverse-depth-like values
        # that we min-max normalize to [0, 1]. Detect from the model name so we
        # don't squash real meters into a relative range.
        self.is_metric = "Metric" in self.model_name

        if AutoImageProcessor is None:
            raise ImportError("Please install transformers: pip install transformers")

        print(f"Loading Depth Estimator ({self.model_name}) on {self.device}...")
        print(f"  → Output mode: {'metric (meters)' if self.is_metric else 'relative ([0,1])'}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("Depth Estimator loaded successfully.")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate a depth map from the input image.

        Output convention depends on the loaded model (set by self.is_metric):
          - Metric variant  → absolute depth in METERS (raw model output).
          - Relative variant → unitless, min-max normalized to [0, 1] where
            higher value = closer (Depth Anything inverse-depth convention).

        Args:
            image: RGB uint8 numpy array (H, W, 3) or PIL Image.

        Returns:
            Depth map (H, W) float32. Units depend on self.is_metric.

        Raises:
            TypeError: If input is not ndarray or PIL Image.
            ValueError: If image is empty (0-size).
        """
        if isinstance(image, np.ndarray):
            # Handle grayscale: convert to RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            # Assumes input is RGB
            if image.size == 0:
                return np.array([], dtype=np.float32).reshape(0, 0)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
            if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                return np.array([], dtype=np.float32).reshape(0, 0)
        else:
            raise TypeError(f"Expected ndarray or PIL Image, got {type(image)}")

        original_size = pil_image.size[::-1]  # (H, W)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize to original
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = predicted_depth.cpu().numpy().astype(np.float32)

        if self.is_metric:
            # Metric variant: model output is already absolute depth in meters.
            # Do NOT normalize — that would destroy the metric scale.
            # Clamp non-finite values defensively.
            depth_map = np.where(np.isfinite(depth_map), depth_map, 0.0).astype(np.float32)
            return depth_map

        # Relative variant: min-max normalize to [0, 1] (higher = closer).
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map, dtype=np.float32)

        return depth_map
    
    def depth_to_distance(
        self,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Extract median depth value from depth map.

        Args:
            depth_map: (H, W) float depth map
            mask: Optional (H, W) bool/uint8, pixels > 0 are valid

        Returns:
            float: Median depth value, or NaN if no valid pixels
        """
        if mask is not None:
            valid_depths = depth_map[mask > 0]
        else:
            valid_depths = depth_map.flatten()

        if len(valid_depths) < 1:
            return float('nan')

        return float(np.median(valid_depths))
    
    def compute_direction(
        self,
        mask: np.ndarray,
        image_width: int
    ) -> Tuple[str, float, float]:
        """
        Compute direction and angle from mask centroid.

        Args:
            mask: (H, W) uint8/bool, pixels > 0 are ROI
            image_width: Image width in pixels

        Returns:
            (direction, angle_deg, centroid_x_norm): direction in ['left','center','right','unknown'],
                angle in [-FOV/2, +FOV/2], normalized x in [0, 1]
        """
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) == 0:
            return ('unknown', 0.0, 0.5)

        centroid_x = float(np.mean(x_indices))
        centroid_x_norm = centroid_x / max(image_width, 1)
        centroid_x_norm = np.clip(centroid_x_norm, 0.0, 1.0)

        # Calculate angle based on FOV
        fov = getattr(config, 'HORIZONTAL_FOV', 60.0)
        angle_deg = (centroid_x_norm - 0.5) * fov

        dir_left_thresh = getattr(config, 'DIR_LEFT', 0.33)
        dir_right_thresh = getattr(config, 'DIR_RIGHT', 0.67)

        if centroid_x_norm < dir_left_thresh:
            direction = 'left'
        elif centroid_x_norm > dir_right_thresh:
            direction = 'right'
        else:
            direction = 'center'

        return direction, float(angle_deg), float(centroid_x_norm)
    
    def scale_depth_to_meters(
        self,
        depth_map: np.ndarray,
        gt_depth: Optional[np.ndarray] = None,
        max_depth: Optional[float] = None,
        align_scale: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        Scale relative depth map to metric (meters).

        Args:
            depth_map: (H, W) relative depth [0, 1]
            gt_depth: Optional (H, W) ground-truth metric depth
            max_depth: Max depth in meters (default config.MAX_DEPTH_M)
            align_scale: If True and no GT, use median-based scaling

        Returns:
            (scaled_depth, scale, shift): metric_depth = scale * depth_map + shift
        """
        max_d = max_depth or getattr(config, 'MAX_DEPTH_M', 10.0)
        min_d = getattr(config, 'MIN_DEPTH_M', 0.1)

        if gt_depth is not None and np.isfinite(gt_depth).sum() > 10:
            # Least squares with valid pixels
            valid = (
                (gt_depth > min_d) &
                (gt_depth < max_d) &
                np.isfinite(depth_map) &
                np.isfinite(gt_depth)
            )

            if np.sum(valid) > 10:
                y = gt_depth[valid]
                x = depth_map[valid]

                # Least squares: y = scale * x + shift
                A = np.vstack([x, np.ones(len(x))]).T
                coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
                scale, shift = float(coeffs[0]), float(coeffs[1])

                metric_depth = scale * depth_map + shift
                metric_depth = np.clip(metric_depth, 0, max_d)
                return metric_depth, scale, shift

        # Fallback without GT: linear scaling
        scale = max_d
        shift = 0.0
        scaled_depth = np.clip(depth_map * scale, 0, max_d)
        return scaled_depth, float(scale), float(shift)


# ============================================================
# Region Aggregation (for YOLO bboxes / MobileSAM masks)
# ============================================================

# Depth Anything V2 (after min-max normalization in estimate_depth) uses
# the convention: LARGER value = CLOSER to camera. Flip closest_side="low"
# if the caller already converted to metric depth (smaller = closer).
_AGG_MODES = {"mean", "median", "max", "min", "top_k", "top_p"}


def _aggregate_values(
    values: np.ndarray,
    mode: str,
    k: Optional[int],
    p: Optional[float],
    closest_side: str,
) -> float:
    """Reduce a 1D array of depth samples to a single scalar."""
    if mode not in _AGG_MODES:
        raise ValueError(f"mode must be one of {_AGG_MODES}, got {mode!r}")
    if closest_side not in ("high", "low"):
        raise ValueError("closest_side must be 'high' or 'low'")

    values = np.asarray(values, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    if mode == "mean":
        return float(values.mean())
    if mode == "median":
        return float(np.median(values))
    if mode == "max":
        return float(values.max())
    if mode == "min":
        return float(values.min())

    if mode == "top_k":
        if k is None or k <= 0:
            raise ValueError("top_k mode requires positive integer k")
        k_eff = int(min(k, values.size))
        if closest_side == "high":
            kept = np.partition(values, -k_eff)[-k_eff:]
        else:
            kept = np.partition(values, k_eff - 1)[:k_eff]
        return float(kept.mean())

    # top_p
    if p is None or not (0.0 < p <= 1.0):
        raise ValueError("top_p mode requires p in (0, 1]")
    k_eff = max(1, int(np.ceil(values.size * p)))
    if closest_side == "high":
        kept = np.partition(values, -k_eff)[-k_eff:]
    else:
        kept = np.partition(values, k_eff - 1)[:k_eff]
    return float(kept.mean())


def aggregate_depth_in_bbox(
    depth_map: np.ndarray,
    bbox: Sequence[float],
    mode: str = "mean",
    k: Optional[int] = None,
    p: Optional[float] = None,
    closest_side: str = "high",
) -> float:
    """
    Aggregate depth inside an axis-aligned bounding box.

    Args:
        depth_map: (H, W) float array
        bbox: [x1, y1, x2, y2] in pixel coordinates (x2/y2 exclusive).
            Box is clipped to the image; float coords are rounded.
        mode: "mean" | "median" | "max" | "min" | "top_k" | "top_p"
        k: number of pixels to keep when mode="top_k"
        p: fraction in (0, 1] of pixels to keep when mode="top_p"
        closest_side: "high" if larger depth = closer (Depth Anything V2
            relative depth), "low" if smaller = closer (metric depth).

    Returns:
        Single aggregated depth value, or NaN if the region is empty.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")
    H, W = depth_map.shape

    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(W, int(round(float(x2))))
    y2 = min(H, int(round(float(y2))))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    region = depth_map[y1:y2, x1:x2]
    return _aggregate_values(region, mode, k, p, closest_side)


def aggregate_depth_in_mask(
    depth_map: np.ndarray,
    mask: np.ndarray,
    mode: str = "mean",
    k: Optional[int] = None,
    p: Optional[float] = None,
    closest_side: str = "high",
) -> float:
    """
    Aggregate depth inside a segmentation mask (e.g. MobileSAM).

    Args:
        depth_map: (H, W) float
        mask: (H, W) bool or uint8. Pixels > 0 are treated as ROI.
        Other args: see aggregate_depth_in_bbox.

    Returns:
        Single aggregated depth value, or NaN if mask is empty.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")
    if mask.shape != depth_map.shape:
        raise ValueError(
            f"mask shape {mask.shape} != depth_map shape {depth_map.shape}"
        )
    values = depth_map[np.asarray(mask) > 0]
    return _aggregate_values(values, mode, k, p, closest_side)


def aggregate_depth_in_bboxes(
    depth_map: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    mode: str = "mean",
    k: Optional[int] = None,
    p: Optional[float] = None,
    closest_side: str = "high",
) -> List[float]:
    """Batched version of aggregate_depth_in_bbox (one value per bbox)."""
    return [
        aggregate_depth_in_bbox(
            depth_map, b, mode=mode, k=k, p=p, closest_side=closest_side
        )
        for b in bboxes
    ]


def aggregate_depth_in_masks(
    depth_map: np.ndarray,
    masks: Union[Sequence[np.ndarray], np.ndarray],
    mode: str = "mean",
    k: Optional[int] = None,
    p: Optional[float] = None,
    closest_side: str = "high",
) -> List[float]:
    """
    Batched version of aggregate_depth_in_mask.

    Accepts either a list/tuple of (H, W) masks or a stacked (N, H, W) array
    (e.g. MobileSAM `masks` output).
    """
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        iterable = (masks[i] for i in range(masks.shape[0]))
    else:
        iterable = iter(masks)
    return [
        aggregate_depth_in_mask(
            depth_map, m, mode=mode, k=k, p=p, closest_side=closest_side
        )
        for m in iterable
    ]


# ============================================================
# Helper Functions
# ============================================================

def normalize_depth_map(
    depth_map: np.ndarray,
    percentile_range: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    p_min, p_max = np.percentile(depth_map, percentile_range)
    if p_max > p_min:
        norm_map = np.clip((depth_map - p_min) / (p_max - p_min), 0, 1)
    else:
        norm_map = np.zeros_like(depth_map)
    return norm_map

def compute_depth_uncertainty(
    depth_map: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    # Use cv2 blur to compute local variance: var(X) = E(X^2) - E(X)^2
    mean_x = cv2.blur(depth_map, (window_size, window_size))
    mean_x2 = cv2.blur(depth_map**2, (window_size, window_size))
    variance = np.clip(mean_x2 - mean_x**2, 0, None)
    return np.sqrt(variance)

def depth_map_to_3d_points(
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray = None
) -> np.ndarray:
    H, W = depth_map.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    if camera_intrinsics is not None:
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    else:
        # Generic camera assumption
        fx = fy = max(H, W)
        cx, cy = W / 2, H / 2
        
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points_3d
