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
# Other options: Depth-Anything-V2-Base-hf, Depth-Anything-V2-Large-hf
DEPTH_MODEL_NAME = 'depth-anything/Depth-Anything-V2-Small-hf'
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
# LOGGING & OUTPUT
# ============================================================
LOG_LEVEL = 'INFO'
SAVE_VISUALIZATIONS = True
SAVE_DEPTH_MAPS = True
SAVE_METRICS = True
VERBOSE = True
