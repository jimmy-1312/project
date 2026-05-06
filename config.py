"""
Project Configuration File

This module contains all configuration constants and settings for the Visual Assistant Pipeline.
Setup all paths, model parameters, and hyperparameters here.
"""

import os
import torch

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training')
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'validation')
TESTING_DATA_DIR = os.path.join(DATA_DIR, 'testing')

SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'model', 'weights')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')
DEPTH_MAPS_DIR = os.path.join(RESULTS_DIR, 'depth_maps')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Create directories if they don't exist
for path in [CHECKPOINT_DIR, VISUALIZATIONS_DIR, DEPTH_MAPS_DIR, METRICS_DIR]:
    os.makedirs(path, exist_ok=True)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_AVAILABLE = torch.cuda.is_available()

# ============================================================
# MODEL NAMES & IDENTIFIERS
# ============================================================
# YOLO Detection Model
YOLO_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'yolov8m.pt')
YOLO_MODEL_NAME = YOLO_MODEL_PATH  # Use local path if available, otherwise model name
YOLO_CONFIDENCE = 0.3
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

# MobileSAM Segmentation
MOBILE_SAM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'mobile_sam.pt')
MOBILE_SAM_MODEL_TYPE = 'vit_t'  # Mobile Vision Transformer type

# Depth Anything V2
# Must be a valid Hugging Face model identifier (see https://huggingface.co/depth-anything).
#
# We use the Metric-Indoor variant: outputs are TRUE meters (not relative depth)
# and the model is fine-tuned on indoor metric data (Hypersim + NYU Depth v2),
# which matches our HK indoor assistive use case.
#
# Other options if you need to switch:
#   - 'depth-anything/Depth-Anything-V2-Small-hf'                  (relative depth, faster)
#   - 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf'     (more accurate, slower)
#   - 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'   (outdoor scenes)
DEPTH_MODEL_NAME = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
DEPTH_PROCESSOR_NAME = DEPTH_MODEL_NAME  # kept for backward compat with older scripts

# LLM Configuration
LLM_MODEL_NAME = 'gpt-3.5-turbo'  # or other model names
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 150

# ============================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================
# Direction classification thresholds (fraction of image width)
DIR_LEFT = 0.33
DIR_RIGHT = 0.67

# Depth estimation parameters
MAX_DEPTH_M = 10.0
MIN_DEPTH_M = 0.1

# FOV (Field of View) for angle calculation
HORIZONTAL_FOV = 60.0  # degrees

# ============================================================
# EVALUATION THRESHOLDS
# ============================================================
DISTANCE_TOLERANCE = 0.5  # meters
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.3

# ============================================================
# DATA LOADING PARAMETERS
# ============================================================
BATCH_SIZE = 8
NUM_WORKERS = 4
IMAGE_SIZE = (480, 640)  # (height, width)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================
# TRAINING PARAMETERS (if applicable)
# ============================================================
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
OPTIMIZER = 'adam'

# ============================================================
# HAZARD TAXONOMY (HK indoor assistive navigation)
# ============================================================
# Risk priority per hazard class [0.0, 1.0]. Higher = more dangerous.
# Used by src/hazard_scorer.py to rank detected objects for visually impaired users.
HAZARD_PRIORITY = {
    "stairs": 1.0,            # Fall hazard, critical
    "glass_door": 0.9,        # Invisible collision
    "pillar": 0.7,            # Frontal collision
    "hanging_obstacle": 0.7,  # Head-level signs/banners
    "low_obstacle": 0.5,      # Chairs, tables, bins, planters
    "person": 0.3,            # Dynamic, usually self-avoidant
}

# Mapping from vanilla COCO class names → our hazard taxonomy.
# Used as fallback before fine-tuned YOLO is available.
# Classes not in this dict are dropped from hazard ranking.
COCO_TO_HAZARD = {
    "person": "person",
    "chair": "low_obstacle",
    "couch": "low_obstacle",
    "bench": "low_obstacle",
    "potted plant": "low_obstacle",
    "vase": "low_obstacle",
    "dining table": "low_obstacle",
    "suitcase": "low_obstacle",
    "backpack": "low_obstacle",
    "handbag": "low_obstacle",
    # stairs / glass_door / pillar / hanging_obstacle are NOT in COCO →
    # require fine-tuned YOLO. See scripts/train_hazard_yolo.py
}

# Risk score components — relative weights for priority / proximity / area.
# risk = priority^w_p * proximity^w_d * mask_area_norm^w_a
HAZARD_WEIGHT_PRIORITY = 1.0
HAZARD_WEIGHT_PROXIMITY = 1.5
HAZARD_WEIGHT_AREA = 0.5

# Distance (m) at which proximity score = 0.5. Closer than this → score ↑.
HAZARD_PROXIMITY_HALF_M = 2.0

# How many top-risk objects to surface to the user.
HAZARD_TOP_K = 3

# Average adult step length (meters). Used to convert distance → "N steps".
# 0.7 m is a common figure for adult walking; tune if you have user-specific data.
STEP_LENGTH_M = 0.7

# ============================================================
# PROXIMITY ALERTING (Nearest Objects Feature)
# ============================================================
# Default top-K for proximity-based nearest-object ranking.
PROXIMITY_DEFAULT_TOP_K = 5

# ============================================================
# LOGGING & OUTPUT
# ============================================================
LOG_LEVEL = 'INFO'
SAVE_VISUALIZATIONS = True
SAVE_DEPTH_MAPS = True
SAVE_METRICS = True
VERBOSE = True
