"""
Depth Anything V2 Depth Estimation Module

This module provides monocular depth estimation using the Depth Anything V2 model.
It estimates relative depth maps and can convert them to metric depth.
"""

import numpy as np
import torch
from typing import Optional, Tuple
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
        
        if AutoImageProcessor is None:
            raise ImportError("Please install transformers: pip install transformers")
            
        print(f"Loading Depth Estimator ({self.model_name}) on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("Depth Estimator loaded successfully.")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            # Assumes input is RGB
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
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
        
        depth_map = predicted_depth.cpu().numpy()
        
        # Normalize to 0-1 for relative depth representation
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
            
        return depth_map
    
    def depth_to_distance(
        self,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        if mask is not None:
            valid_depths = depth_map[mask > 0]
        else:
            valid_depths = depth_map.flatten()
            
        if len(valid_depths) == 0:
            return float('nan')
            
        return float(np.median(valid_depths))
    
    def compute_direction(
        self,
        mask: np.ndarray,
        image_width: int
    ) -> Tuple[str, float, float]:
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) == 0:
            return ('unknown', 0.0, 0.5)
            
        centroid_x = np.mean(x_indices)
        centroid_x_norm = centroid_x / image_width
        
        # Calculate angle based on FOV (assuming 60 degrees default)
        fov = getattr(config, 'HORIZONTAL_FOV', 60.0)
        angle_deg = (centroid_x_norm - 0.5) * fov
        
        dir_left_thresh = getattr(config, 'DIR_LEFT', 0.33)
        dir_right_thresh = getattr(config, 'DIR_RIGHT', 0.66)
        
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
        max_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, float, float]:
        max_d = max_depth or getattr(config, 'MAX_DEPTH_M', 10.0)
        
        if gt_depth is not None:
            # Mask valid pixels
            valid = (gt_depth > 0.1) & (gt_depth < max_d) & np.isfinite(depth_map)
            
            if np.sum(valid) > 10:
                y = gt_depth[valid]
                x = depth_map[valid]
                
                # Least squares: y = scale * x + shift
                A = np.vstack([x, np.ones(len(x))]).T
                scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
                
                metric_depth = scale * depth_map + shift
                metric_depth = np.clip(metric_depth, 0, max_d)
                return metric_depth, float(scale), float(shift)
                
        # Fallback without GT
        scale = max_d
        shift = 0.0
        scaled_depth = depth_map * scale
        return scaled_depth, float(scale), float(shift)


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
