# Visual Assistant Pipeline - Project Template

## Project Overview

A comprehensive machine learning pipeline for **real-time visual understanding** that combines:
- **Object Detection** (YOLOv8) - Identify what objects are in the scene
- **Instance Segmentation** (MobileSAM) - Get precise object boundaries
- **Depth Estimation** (Depth Anything V2) - Understand spatial relationships
- **LLM Integration** (Optional) - Generate natural language descriptions

**Use Case**: Accessibility applications, robotic vision, autonomous systems, scene understanding.

---

## 📁 Project Structure

```
project/
├── README.md                          # This file
├── config.py                          # Global configuration & hyperparameters
├── requirements.txt                   # Python dependencies
├── main.py                            # Demo/entry point script
│
├── src/                               # Source code (TEMPLATES - implement these)
│   ├── __init__.py                    # Module initialization
│   ├── detector.py                    # YOLOv8 object detection wrapper
│   ├── segmentor.py                   # MobileSAM segmentation wrapper
│   ├── depth_estimator.py             # Depth Anything V2 wrapper
│   ├── llm_generator.py               # LLM text generation (optional)
│   ├── pipeline.py                    # Main orchestration pipeline
│   ├── visualizer.py                  # Visualization utilities
│   ├── data_loader.py                 # Data loading & preprocessing
│   └── evaluation.py                  # Evaluation metrics
│
├── data/                              # Datasets
│   ├── training/                      # Training images
│   ├── validation/                    # Validation images
│   └── testing/                       # Test images
│
├── model/                             # Model weights & checkpoints
│   └── weights/                       # Downloaded model files
│
├── tests/                             # Unit & integration tests
│   ├── __init__.py
│   ├── test_detector.py               # YOLODetector tests (TEMPLATES)
│   ├── test_segmentor.py              # MobileSAMSegmentor tests (TEMPLATES)
│   ├── test_depth_estimator.py        # DepthEstimator tests (TEMPLATES)
│   └── test_integration.py            # End-to-end tests (TEMPLATES)
│
├── scripts/                           # Utility scripts
│   ├── download_models.py             # Download model checkpoints (TEMPLATE)
│   └── prepare_data.py                # Prepare dataset splits (TEMPLATE)
│
├── notebooks/                         # Jupyter notebooks for exploration
│   └── examples.ipynb                 # Example usage notebook (TO CREATE)
│
├── results/                           # Output directory
│   ├── visualizations/                # Saved visualizations
│   ├── depth_maps/                    # Saved depth maps
│   └── metrics/                       # Evaluation results
│
├── docs/                              # Documentation
│   └── API.md                         # API reference (TO CREATE)
│
└── .gitignore                         # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd project

# Install dependencies
pip install -r requirements.txt

# Download model weights (if checkpoints not included)
python scripts/download_models.py
```

### 2. Basic Usage

```python
from src.pipeline import VisualAssistantPipeline
from src.visualizer import Visualizer

# Initialize pipeline
pipeline = VisualAssistantPipeline(device='cuda')  # Use 'cpu' if no GPU

# Process an image
results = pipeline.process_image('path/to/image.jpg')

# Get summary
summary = pipeline.get_summary_text(results)
print(summary)

# Visualize results
Visualizer.visualize_full_results(results['image'], results)
```

### 3. Running Demo

```bash
# Process single image
python main.py --mode image --input samples/image.jpg --output ./results

# Batch process directory
python main.py --mode batch --input ./data/testing --output ./results

# Process video
python main.py --mode video --input samples/video.mp4 --output ./results

# Real-time webcam
python main.py --mode webcam --duration 30
```

---

## 📝 Template Implementation Guide

### This is a **Learning Project Template**

All Python files in `src/` are **TEMPLATES** with:
- ✅ Complete class and function signatures
- ✅ Comprehensive docstrings with input/output specifications
- ✅ TODO comments indicating what needs to be implemented
- ✅ Helper functions with implementation hints
- ❌ **NO actual implementations** (your coding homework!)

### How to Use:

1. **Read the TODO comments** in each function - they explain what you need to implement
2. **Follow the input/output specifications** in the docstrings
3. **Refer to the reference code** (`project.py` - Claude Opus generated code)
4. **Implement one function at a time** and test with unit tests
5. **Run tests** to verify your implementation:
   ```bash
   pytest tests/ -v
   ```

### Example: Implementing a Function

**Before (Template):**
```python
def detect(self, image: Union[np.ndarray, str]) -> List[Dict]:
    """
    Detect objects in an image.
    
    Args:
        image: Input image...
    
    Returns:
        List of detection dictionaries...
    
    TODO:
    1. Run YOLO inference on the image
    2. Extract bounding boxes and scores
    3. Filter by confidence threshold
    4. Format output as list of dicts
    """
    pass
```

**After (Your Implementation):**
```python
def detect(self, image: Union[np.ndarray, str]) -> List[Dict]:
    """..."""
    from ultralytics import YOLO
    
    # Run inference
    results = self.model(image, conf=self.confidence, iou=self.iou, device=self.device)
    
    # Extract and format detections
    detections = []
    if results[0].boxes is not None:
        # Process results...
        pass
    
    return detections
```

---

## 🏗️ Architecture

```
Input Image
    ↓
┌─────────────────────────────┐
│   YOLO Detector             │  → Detect objects & bounding boxes
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   MobileSAM Segmentor       │  → Get precise masks
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   Depth Estimator           │  → Compute depth map
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   Per-Object Analysis       │  → Distance, direction, angle
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   LLM Generator (Optional)  │  → Text description
└──────────────┬──────────────┘
               ↓
         Output Results
      (JSON, visualizations,
       text descriptions)
```

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
pytest tests/test_detector.py -v              # Test YOLO detector
pytest tests/test_segmentor.py -v              # Test segmentation
pytest tests/test_depth_estimator.py -v        # Test depth estimation
pytest tests/test_integration.py -v            # Test full pipeline
```

Test coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Key Modules to Implement

| Module | Purpose | Priority |
|--------|---------|----------|
| `detector.py` | YOLO object detection | 🔴 CRITICAL |
| `segmentor.py` | MobileSAM instance masks | 🔴 CRITICAL |
| `depth_estimator.py` | Depth map estimation | 🔴 CRITICAL |
| `pipeline.py` | Orchestrate all components | 🟠 HIGH |
| `visualizer.py` | Create visualizations | 🟠 HIGH |
| `data_loader.py` | Load training data | 🟡 MEDIUM |
| `evaluation.py` | Compute metrics | 🟡 MEDIUM |
| `llm_generator.py` | Text generation | 🟢 LOW (optional) |

---

## 🔧 Configuration

Edit `config.py` to customize:
- Model names and checkpoints
- Detection confidence thresholds
- Depth map parameters (max depth, FOV)
- Device (CPU/GPU)
- Data directories
- Output paths

Example:
```python
# Use smaller model for speed
YOLO_MODEL_NAME = 'yolov8n.pt'  # nano instead of medium

# Adjust depth range
MAX_DEPTH_M = 20.0  # extended range

# Change direction thresholds
DIR_LEFT = 0.25
DIR_RIGHT = 0.75
```

---

## 📐 Data Formats

### Input
- **Images**: PNG, JPG, BMP (any size, auto-resized)
- **Video**: MP4, AVI, MOV (any resolution)
- **Depth maps**: HDF5, NPZ, PNG

### Output
```python
results = {
    'objects': [
        {
            'class_name': str,
            'class_id': int,
            'confidence': float,
            'bbox': np.array([x1, y1, x2, y2]),
            'mask': np.array((H, W), dtype=bool),
            'distance_m': float,
            'direction': str,  # 'left', 'center', 'right'
            'angle_deg': float,
        },
        ...
    ],
    'depth_map': np.array((H, W), dtype=float),        # Relative depth
    'metric_depth_map': np.array((H, W), dtype=float), # In meters
    'image_shape': (H, W),
    'depth_scale': float,
    'depth_shift': float,
    'scene_description': str,  # If LLM enabled
}
```

---

## 📚 Reference Materials

- **YOLOv8**: https://docs.ultralytics.com/
- **MobileSAM**: https://github.com/ChaoningZhang/MobileSAM
- **Depth Anything V2**: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- **NYU Depth V2**: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

---

## 🧑‍💼 Implementation Workflow

1. **Day 1-2**: Implement detector.py with YOLOv8
2. **Day 2-3**: Implement segmentor.py with MobileSAM
3. **Day 3-4**: Implement depth_estimator.py
4. **Day 4-5**: Integrate into pipeline.py
5. **Day 5-6**: Add visualizations and tests
6. **Day 6-7**: Data loading and evaluation
7. **Day 7+**: Optimization and LLM integration

---

## 🤝 Contributing

1. Implement one module at a time
2. Write tests for your implementation
3. Run full test suite before committing
4. Document your changes
5. Update this README if needed

---

## 📝 Logging

Check `config.py` for logging settings:
```python
LOG_LEVEL = 'INFO'
VERBOSE = True
```

---

## 🆘 Troubleshooting

**CUDA out of memory:**
```python
# Use smaller model
YOLO_MODEL_NAME = 'yolov8n.pt'
# Or switch to CPU
DEVICE = 'cpu'
```

**Model download issues:**
```bash
# Manually download MobileSAM
python scripts/download_models.py
```

**Data loading errors:**
```bash
# Prepare and organize data
python scripts/prepare_data.py --dataset_path ./data
```

---

**Happy Coding! 🚀**
