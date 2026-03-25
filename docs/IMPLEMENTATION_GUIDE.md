# Implementation Guide

## Getting Started

This guide walks you through implementing the Visual Assistant Pipeline project step-by-step.

---

## Phase 1: Foundation (Days 1-2)

### Step 1: Setup  Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (try once, may trigger auto-downloads)
python scripts/download_models.py
```

### Step 2: Understand the Architecture

Read through:
- `README.md` - Overview
- `config.py` - Configuration
- `docs/API.md` - API reference
- Reference code in `project.py` (attached)

### Step 3: Explore Module Structure

Each module in `src/` follows a template pattern:
1. **Docstring**: Explains what the class/function does
2. **Signature**: Shows exact inputs/outputs
3. **TODO comments**: Lists implementation steps
4. **Pass statement**: Where your code goes

---

## Phase 2: Core Implementation (Days 2-5)

### Module Priority Order

Implement in this order to build incrementally:

#### 1️⃣ **detector.py** (Top Priority)

**What to do:**
- Implement `YOLODetector.__init__()` - Load YOLO model
- Implement `YOLODetector.detect()` - Run inference
- Implement `class_names` property
- Test with `pytest tests/test_detector.py`

**Key tips:**
- Use `ultralytics.YOLO` to load models
- Reference code example in `project.py` lines ~200-250
- Handle different image formats (numpy, PIL, string)
- Return consistently formatted dicts

**Test you implementation:**
```bash
pytest tests/test_detector.py::TestYOLODetector::test_detect_returns_list -v
```

---

#### 2️⃣ **segmentor.py** (High Priority)

**What to do:**
- Implement `MobileSAMSegmentor.__init__()` - Load segmentation model
- Implement `MobileSAMSegmentor.set_image()`
- Implement `MobileSAMSegmentor.segment_box()`
- Implement `MobileSAMSegmentor.segment_detections()`

**Key tips:**
- Handle two backends: mobile_sam and ultralytics
- Use detected boxes as SAM prompts
- Return (mask, confidence) tuples
- Reference code example in `project.py` lines ~300-400

**Test your implementation:**
```bash
pytest tests/test_segmentor.py -v
```

---

#### 3️⃣ **depth_estimator.py** (High Priority)

**What to do:**
- Implement `DepthEstimator.__init__()` - Load Depth Anything V2
- Implement `DepthEstimator.estimate_depth()`
- Implement `DepthEstimator.depth_to_distance()`
- Implement `DepthEstimator.compute_direction()`
- Implement `DepthEstimator.scale_depth_to_meters()`

**Key tips:**
- Use `transformers` library (_AutoImageProcessor, AutoModelForDepthEstimation_)
- Reference code in `project.py` lines ~500-650
- Handle depth scaling with/without ground truth
- FOV angle calculation: use horizontal FOV from config

**Test your implementation:**
```bash
pytest tests/test_depth_estimator.py -v
```

---

#### 4️⃣ **pipeline.py** (High Priority)

**What to do:**
- Implement `VisualAssistantPipeline.__init__()` - Initialize all components
- Implement `VisualAssistantPipeline.process_image()` - Orchestrate pipeline
- Implement `VisualAssistantPipeline.get_summary_text()`

**Key tips:**
- Call detector → segmentor → depth_estimator in sequence
- For each object: compute distance from mask + depth map
- Sort objects by distance
- Reference pipeline code in `project.py` lines ~700-900

**Test your implementation:**
```bash
pytest tests/test_integration.py::TestVisualAssistantPipeline::test_process_single_image -v
```

---

### Phase 3: Supporting Components (Days 5-6)

#### 5️⃣ **visualizer.py** (Medium Priority)

**What to do:**
- Implement visualization methods for debugging
- Draw bounding boxes, masks, depth maps
- Create comparison images

**How it helps:**
- See what your pipeline is detecting
- Debug issues visually
- Generate report images

---

#### 6️⃣ **data_loader.py** (Medium Priority)

**What to do:**
- Implement image loading from directories
- Support dataset splits (train/val/test)
- Handle different image formats

**Why it matters:**
- Needed for training/evaluation
- Keeps code organized

---

#### 7️⃣ **evaluation.py** (Medium Priority)

**What to do:**
- Implement `Evaluator.evaluate_depth_map()`
- Compute standard depth metrics (RMSE, AbsRel, δ<1.25)
- Implement detection/segmentation metrics if needed

**Why it matters:**
- Measure model performance
- Compare different configurations

---

#### 8️⃣ **llm_generator.py** (Low Priority - Optional)

**What to do:**
- Generate text descriptions of scenes
- Optional accessibility feature
- Can use OpenAI API or local LLM

**Only if:**
- Extra time available
- Want text generation features

---

## Testing Strategy

### Unit Tests

Each module has template tests. Fill in the TODO sections:

```bash
# Test single module
pytest tests/test_detector.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_detector.py::TestYOLODetector::test_detection_format -v
```

### Integration Tests

After completing core modules, test the full pipeline:

```bash
pytest tests/test_integration.py -v
```

### Manual Testing

```python
# Run your own tests
from src.pipeline import VisualAssistantPipeline

pipeline = VisualAssistantPipeline()
results = pipeline.process_image('sample_image.jpg')
print(results['objects'])
```

---

## Common Implementation Patterns

### Pattern 1: Initialization with Model Loading

```python
def __init__(self, model_name=None, device=None):
    self.model_name = model_name or config.YOLO_MODEL_NAME
    self.device = device or config.DEVICE
    
    print(f"  Loading {self.model_name}...")
    self.model = YOLO(self.model_name)  # Your actual loading code
    self.model.to(self.device)
    print(f"  ✅ Model loaded on {self.device}")
```

### Pattern 2: Inference with Error Handling

```python
def detect(self, image):
    try:
        results = self.model(image, conf=self.confidence, device=self.device)
        detections = []
        if results[0].boxes is not None:
            # Extract detections
            pass
        return detections
    except Exception as e:
        print(f"Error in detection: {e}")
        return []
```

### Pattern 3: Batch Processing

```python
def segment_detections(self, image_rgb, detections):
    self.set_image(image_rgb)
    results = []
    for det in detections:
        mask, score = self.segment_box(det['bbox'])
        det['mask'] = mask
        det['mask_score'] = score
        results.append(det)
    return results
```

---

## Debugging Tips

### Issue: Model fails to load
```python
# Check device availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Try CPU mode
from config import DEVICE
print(f"Using device: {DEVICE}")
```

### Issue: Output shapes wrong
```python
# Print shapes at each step
print(f"Image shape: {image.shape}")
print(f"Depth map shape: {depth.shape}")
print(f"Mask shape: {mask.shape}")

# Verify dtypes
print(f"Image dtype: {image.dtype}")
print(f"Mask dtype: {mask.dtype}")
```

### Issue: Detections not working
```python
# Check inputs
import cv2
img = cv2.imread('test.jpg')
print(f"Loaded image shape: {img.shape}")

# Test detector in isolation
detector = YOLODetector()
dets = detector.detect(img)
print(f"Found {len(dets)} detections")
```

---

## Performance Optimization

### For Development
- Use smaller YOLO model: `yolov8n.pt` (nano)
- Smaller images: 320×240 instead of 640×480
- Skip LLM generation initially

### For Production
- Use GPU: `device='cuda'`
- Larger model if accuracy matters: `yolov8x.pt`
- Batch process multiple images
- Use `frame_interval=3` to skip frames in video

---

## Submission Checklist

Before finishing:

- [ ] All modules implemented
- [ ] All unit tests pass: `pytest tests/ -v`
- [ ] No TODO comments left (except in docstrings)
- [ ] Code is documented and readable
- [ ] main.py demo works
- [ ] README is up to date
- [ ] Results directory has examples
- [ ] .gitignore properly configured

---

## Next Steps After Implementation

1. **Train/Fine-tune**: Adapt models to Hong Kong dataset
2. **Optimize**: Profile code and optimize bottlenecks
3. **Deploy**: Create API service or mobile app
4. **Evaluate**: Run comprehensive evaluation on test set
5. **Document**: Write usage guides and examples

---

## Reference Materials

- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Mobile SAM**: GitHub ChaoningZhang/MobileSAM
- **Depth Anything V2**: HuggingFace Hub depth-anything
- **PyTorch**: https://pytorch.org/docs/
- **Transformers**: https://huggingface.co/docs/transformers/

---

## Getting Help

1. Read the code comments (TODO sections)
2. Check the reference code: `project.py`
3. Review API documentation: `docs/API.md`
4. Run tests to see what's expected
5. Search for examples online

---

**Good luck! You've got this! 🚀**
