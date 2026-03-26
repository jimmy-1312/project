# Visual Understanding Pipeline

## Project Overview

A comprehensive machine learning pipeline for **real-time visual understanding** that combines:
- **Object Detection** (YOLOv8m) - ✅ **Implemented** - Identify and locate objects in scenes
- **Instance Segmentation** (MobileSAM) - Instance-level object boundaries
- **Depth Estimation** (Depth Anything V2) - Spatial relationship understanding
- **LLM Integration** (Optional) - Natural language scene descriptions

**Status**: Core object detection pipeline with YOLOv8m is fully implemented and evaluated on COCO8 dataset.

**Use Case**: Accessibility applications, robotic vision, autonomous systems, scene understanding.

---

## 📁 Project Structure

```
project/
├── README.md                          # This file
├── config.py                          # Global configuration & hyperparameters
├── main.py                            # Entry point script
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── __init__.py                    # Module initialization
│   ├── detector.py                    # ✅ YOLOv8 object detection (IMPLEMENTED)
│   ├── segmentor.py                   # MobileSAM segmentation wrapper
│   ├── depth_estimator.py             # Depth Anything V2 wrapper
│   ├── llm_generator.py               # LLM text generation (optional)
│   ├── pipeline.py                    # Main orchestration pipeline
│   ├── visualizer.py                  # ✅ Visualization utilities (IMPLEMENTED)
│   ├── data_loader.py                 # ✅ Data loading & preprocessing (IMPLEMENTED)
│   └── evaluation.py                  # ✅ Evaluation metrics (IMPLEMENTED)
│
├── data/                              # Datasets
│   ├── coco8/                         # COCO8 evaluation dataset
│   │   ├── images/{train,val}/        # Images
│   │   └── labels/{train,val}/        # YOLO format annotations
│   ├── training/                      # Training images
│   ├── validation/                    # Validation images
│   └── testing/                       # Test images
│
├── model/                             # Model weights & checkpoints
│   ├── weights/
│   │   └── yolov8m.pt                 # YOLOv8m pretrained model
│   ├── depth_anything/
│   ├── mobileSAM/
│   ├── LLM/
│   └── Yolo/
│
├── tests/                             # Unit & integration tests
│   ├── __init__.py
│   ├── test_detector.py               # YOLODetector unit tests
│   ├── test_segmentor.py              # MobileSAMSegmentor tests
│   ├── test_depth_estimator.py        # DepthEstimator tests
│   ├── test_integration.py            # End-to-end tests
│   ├── test_coco8_milestone.py        # COCO8 evaluation test
│   ├── debug_coco8_coordinates.py     # Debug: coordinate matching analysis
│   ├── debug_coco8_image_042.py       # Debug: image 000000000042 analysis
│   └── debug_iou_calculation.py       # Debug: IoU calculation verification
│
├── scripts/                           # Utility scripts
│   ├── download_models.py             # Download model checkpoints
│   ├── prepare_data.py                # Prepare dataset splits
│   ├── evaluate_coco8.py              # ✅ COCO8 evaluation (improved version)
│   ├── evaluate_coco8_basic.py        # COCO8 basic evaluation
│   ├── create_test_image.py           # Generate synthetic test images
│   └── create_better_test_image.py    # Generate better synthetic test images
│
├── notebooks/                         # Jupyter notebooks for exploration
│   └── (examples to be created)
│
├── results/                           # Output directory
│   ├── visualizations/                # Annotated images with bounding boxes
│   │   ├── coco8_train/               # 4 annotated training images
│   │   └── coco8_val/                 # 4 annotated validation images
│   ├── depth_maps/                    # Depth estimation outputs
│   ├── metrics/                       # Evaluation results
│   │   └── coco8_evaluation_results.json
│   └── visualizations/                # Analysis visualizations
│
├── docs/                              # Documentation
│   ├── API.md                         # API reference
│   ├── IMPLEMENTATION_GUIDE.md        # Implementation guide
│   └── MILESTONE_REPORT.md            # COCO8 evaluation report
│
└── .gitignore                         # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project
cd project

# Create/activate conda environment
conda create -n project python=3.10 -y
conda activate project

# Install dependencies
pip install -r requirements.txt

# Download model weights (if not in model/weights/)
python scripts/download_models.py
```

### 2. Evaluate on COCO8 Dataset

```bash
# Run comprehensive evaluation with metrics
python scripts/evaluate_coco8.py

# Output:
# - Annotated images: results/visualizations/coco8_{train,val}/
# - Metrics JSON: results/metrics/coco8_evaluation_results.json
```

### 3. Basic Object Detection Usage

```python
import cv2
from src.detector import YOLODetector
from src.visualizer import Visualizer

# Initialize detector
detector = YOLODetector(device='cpu', confidence=0.3)

# Load and detect
image = cv2.imread('image.jpg')
detections = detector.detect(image)

# Visualize
annotated = Visualizer.visualize_detections(image, detections)
cv2.imshow('Detections', annotated)
cv2.waitKey(0)

# Print results
for detection in detections:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

### 4. Evaluate on Custom Dataset

```python
from src.data_loader import DataLoader
from src.detector import YOLODetector
from src.evaluation import calculate_metrics

# Load dataset
loader = DataLoader(image_dir='data/custom_images', annotation_dir='data/custom_labels')

# Run detector
detector = YOLODetector()
all_detections = []
ground_truth = []

for image, annotations in loader:
    detections = detector.detect(image)
    all_detections.append(detections)
    ground_truth.append(annotations)

# Calculate metrics
metrics = calculate_metrics(all_detections, ground_truth)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

---

## ✅ Implementation Status

### Completed Components

| Component | Status | Location |
|-----------|--------|----------|
| **Data Loader** | ✅ Complete | `src/data_loader.py` |
| **YOLOv8 Detector** | ✅ Complete | `src/detector.py` |
| **Visualizer** | ✅ Complete | `src/visualizer.py` |
| **Evaluation Metrics** | ✅ Complete | `src/evaluation.py` |
| **COCO8 Evaluation** | ✅ Complete | `scripts/evaluate_coco8.py` |
| Pipeline Orchestration | 🔄 In Progress | `src/pipeline.py` |
| MobileSAM Segmentation | 🔄 In Progress | `src/segmentor.py` |
| Depth Estimation | 🔄 In Progress | `src/depth_estimator.py` |
| LLM Integration | 📋 Planned | `src/llm_generator.py` |

### COCO8 Evaluation Results

**Training Dataset (4 images, 13 GT objects):**
- **Precision**: 0.90 (90% of detections were correct)
- **Recall**: 0.69 (69% of objects detected)
- **F1 Score**: 0.78
- True Positives: 9 | False Positives: 1 | False Negatives: 4

**Validation Dataset (4 images, 17 GT objects):**
- **Precision**: 1.00 (100% accuracy at IoU ≥ 0.5!)
- **Recall**: 0.59 (59% of objects detected)
- **F1 Score**: 0.74
- True Positives: 10 | False Positives: 0 | False Negatives: 7

See detailed results in [docs/MILESTONE_REPORT.md](docs/MILESTONE_REPORT.md)

---

## 🏗️ Architecture

Current implementation focus: **Object Detection Pipeline**

```
Input Image
    ↓
┌──────────────────────────────┐
│  YOLODetector.detect()       │  ✅ COMPLETE
│  (YOLOv8m inference)         │
└──────────────┬───────────────┘
               ↓
         [Detections]
    (class, bbox, confidence,
     class_name)
               ↓
┌──────────────────────────────┐
│  Visualizer.visualize()      │  ✅ COMPLETE
│  (Draw boxes & labels)       │
└──────────────┬───────────────┘
               ↓
      Annotated Image
               ↓
┌──────────────────────────────┐
│  evaluation.calculate_metrics │  ✅ COMPLETE
│  (IoU, Precision, Recall)    │
└──────────────┬───────────────┘
               ↓
         Evaluation Report
```

**Planned Extensions:**
- MobileSAM for instance segmentation
- Depth Anything V2 for depth estimation
- LLM integration for scene descriptions

---

## 🧪 Testing

Run unit tests:
```bash
# Test YOLODetector
pytest tests/test_detector.py -v

# Test Visualizer
pytest tests/test_segmentor.py -v

# Test evaluation metrics
pytest tests/test_integration.py -v

# Run all tests
pytest tests/ -v
```

Custom evaluation scripts:
```bash
# Full COCO8 evaluation with metrics
python scripts/evaluate_coco8.py

# Debug coordinate matching
python tests/debug_coco8_coordinates.py

# Debug IoU calculations
python tests/debug_iou_calculation.py
```

---

## 🔧 Configuration

Edit `config.py` to customize:
- YOLOv8 model variant (nano, small, medium, large)
- Detection confidence threshold (0.0-1.0)
- IoU threshold for NMS (0.0-1.0)
- Device: 'cpu' or 'cuda' (or 'mps' for Apple Silicon)
- Data directories and output paths

Example:
```python
# YOLOv8 Settings
YOLO_MODEL_NAME = 'yolov8m.pt'
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
DEVICE = 'cpu'  # or 'cuda'

# Input/Output
DATA_DIR = 'data/coco8'
OUTPUT_DIR = 'results'
```

---

## 📐 Data Formats

### Input
- **Images**: PNG, JPG, BMP (any size, auto-resized to 640x640)
- **Annotations (YOLO format)**: TXT files with normalized coordinates
  ```
  <class_id> <x_center> <y_center> <width> <height>
  # Example:
  14 0.5 0.5 0.3 0.4  # class_id=14 (bottle), center at (0.5, 0.5)
  ```

### YOLODetector Output
```python
detections = [
    {
        'class_id': int,          # COCO class ID (0-79)
        'class_name': str,        # Class name (e.g., 'person', 'dog')
        'bbox': [x1, y1, x2, y2], # Bounding box in pixels
        'confidence': float,      # Detection confidence (0.0-1.0)
    },
    ...
]
```

### Evaluation Output (JSON)
```json
{
  "train": {
    "total_images": 4,
    "total_detections": 10,
    "total_gt_objects": 13,
    "iou_thresholds": {
      "0.3": {
        "precision": 0.9,
        "recall": 0.6923,
        "f1_score": 0.7826,
        "tp": 9, "fp": 1, "fn": 4
      }
    }
  },
  "val": { ... }
}
```

---

## 📂 Directory Organization

After cleanup, the project structure is:
- **Root**: Only config, main, README, requirements
- **src/**: Core implementation modules (detector, visualizer, etc.)
- **scripts/**: Utility and evaluation scripts
- **tests/**: Unit tests and debugging scripts
- **data/coco8/**: COCO8 dataset (images + annotations)
- **model/weights/**: Pretrained model weights
- **results/**: Output directory for visualizations and metrics
- **docs/**: Documentation and reports

---

## 📚 Key References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **MobileSAM**: https://github.com/ChaoningZhang/MobileSAM
- **Depth Anything V2**: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- **COCO Dataset**: https://cocodataset.org/
- **YOLO Format Annotations**: https://docs.ultralytics.com/datasets/detect/

---

## 🚨 Known Issues & Fixes

### Issue: All metrics showing 0.0
**Status**: ✅ FIXED (v1.1)
- **Root Cause**: Duplicate `compute_box_iou()` function definition in evaluation.py
- **Solution**: Removed stub function at line 295-315, kept working implementation at line 15
- **Impact**: Metrics now report correctly (Precision, Recall, F1 Score)

---

## 📞 Support & Debugging

Check the debug scripts:
```bash
# Analyze coordinate matching between predictions and ground truth
python tests/debug_coco8_coordinates.py

# Check why specific images have no detections
python tests/debug_coco8_image_042.py

# Verify IoU calculation implementation
python tests/debug_iou_calculation.py
```

View detailed evaluation report:
```bash
cat results/metrics/coco8_evaluation_results.json
```

---

**Project Status**: Core object detection pipeline is complete and evaluated. ✅ Ready for next phase (segmentation & depth estimation).

**Last Updated**: March 2026
