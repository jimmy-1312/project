"""
Depth-aware YOLO additions: distance-weighted loss + RGB-D first-conv surgery
+ custom dataset that reads our 6-column label format.

Everything in this package is OPTIONAL. The vanilla RGB fine-tune (variant A)
does NOT need any of this. Importing the package does not modify Ultralytics
state — call the explicit factory functions below to opt in.

Public surface:
    distance_weight(d, tau)
    WeightedV8DetectionLoss (subclass)
    convert_first_conv_to_rgbd(model, init=...)
    RGBDYoloDataset (subclass)
"""

from .loss import distance_weight, WeightedV8DetectionLoss
from .model import convert_first_conv_to_rgbd
from .dataset import RGBDYoloDataset

__all__ = [
    "distance_weight",
    "WeightedV8DetectionLoss",
    "convert_first_conv_to_rgbd",
    "RGBDYoloDataset",
]
