# API Documentation

## Overview

This document provides comprehensive API documentation for the Visual Assistant Pipeline. All classes and functions are templates that require implementation.

---

## Core Classes

### 1. YOLODetector

**Module**: `src.detector`

Object detection using YOLOv8 model.

#### Initialization
```python
detector = YOLODetector(
    model_name='yolov8m.pt',      # Model size
    confidence=0.3,                # Confidence threshold
    iou=0.45,                      # NMS IOU threshold
    device='cuda'                  # Device
)
```

#### Methods

**`detect(image) → List[Dict]`**
- Detects objects in an image
- **Input**: numpy array (H, W, 3) or file path
- **Output**: List of dicts with keys: `bbox`, `confidence`, `class_id`, `class_name`

**`class_names → Dict[int, str]`**
- Returns class ID to name mapping

**`get_num_classes() → int`**
- Returns number of detectable classes

---

### 2. MobileSAMSegmentor

**Module**: `src.segmentor`

Instance segmentation using MobileSAM.

#### Initialization
```python
segmentor = MobileSAMSegmentor(
    checkpoint='model/weights/mobile_sam.pt',
    model_type='vit_t',           # Vision Transformer type
    device='cuda'
)
```

#### Methods

**`set_image(image_rgb) → None`**
- Pre-processes image for segmentation
- **Input**: numpy array (H, W, 3) RGB uint8

**`segment_box(bbox) → Tuple[np.ndarray, float]`**
- Segments object from bounding box
- **Input**: bbox as [x1, y1, x2, y2]
- **Output**: (mask, confidence_score)

**`segment_detections(image_rgb, detections) → List[Dict]`**
- Batch segments all detected objects
- **Input**: image, list of detection dicts
- **Output**: Detections with added 'mask' and 'mask_score' keys

---

### 3. DepthEstimator

**Module**: `src.depth_estimator`

Monocular depth estimation using Depth Anything V2.

#### Initialization
```python
depth_estimator = DepthEstimator(
    model_name='depth-anything/Depth-Anything-V2-Small-hf',
    device='cuda'
)
```

#### Methods

**`estimate_depth(image) → np.ndarray`**
- Estimates relative depth map
- **Input**: numpy array (H, W, 3) RGB uint8 or PIL Image
- **Output**: (H, W) float32 - relative depth map

**`depth_to_distance(depth_map, mask=None) → float`**
- Computes median depth in region
- **Input**: depth_map, optional mask
- **Output**: float - distance value

**`compute_direction(mask, image_width) → Tuple[str, float, float]`**
- Determines object direction and angle
- **Output**: (direction, angle_deg, centroid_x_norm)
  - direction: 'left', 'center', or 'right'
  - angle_deg: negative=left, positive=right
  - centroid_x_norm: normalized x position [0,1]

**`scale_depth_to_meters(depth_map, gt_depth=None, max_depth=10.0) → Tuple[np.ndarray, float, float]`**
- Converts relative depth to metric depth
- **Output**: (metric_depth, scale, shift)
  - scale: applied scaling factor
  - shift: applied offset

---

### 4. VisualAssistantPipeline

**Module**: `src.pipeline`

Main pipeline orchestrating all components.

#### Initialization
```python
pipeline = VisualAssistantPipeline(
    device='cuda',
    load_llm=True  # Include LLM module
)
```

#### Methods

**`process_image(image, gt_depth=None, return_llm_description=True) → Dict`**
- Runs complete pipeline on single image
- **Input**: 
  - image: numpy array, PIL Image, or file path
  - gt_depth: optional ground truth depth for metric alignment
  - return_llm_description: whether to generate text
- **Output**: Comprehensive results dict containing:
  ```python
  {
      'objects': [
          {
              'class_name': str,
              'class_id': int,
              'confidence': float,
              'bbox': np.array([x1, y1, x2, y2]),
              'mask': (H, W) bool,
              'distance_m': float,
              'direction': str,
              'angle_deg': float,
          },
          ...
      ],
      'depth_map': (H, W) float,           # Relative
      'metric_depth_map': (H, W) float,    # Absolute in meters
      'image_shape': (H, W),
      'depth_scale': float,
      'depth_shift': float,
      'scene_description': str,            # If LLM enabled
      'warning': str or None,              # Safety warnings
  }
  ```

**`process_video(video_path, output_path=None, frame_interval=1, visualize=True) → List[Dict]`**
- Processes video frame by frame
- **Input**: 
  - video_path: path to video file
  - output_path: where to save annotated video (optional)
  - frame_interval: process every Nth frame
  - visualize: annotate with boxes/masks
- **Output**: List of results dicts, one per frame

**`process_webcam(duration_seconds=30, visualize=True, save_results=False) → List[Dict]`**
- Real-time processing from webcam
- **Input**:
  - duration_seconds: capture duration
  - visualize: display results
  - save_results: save frames
- **Output**: List of frame results

**`get_summary_text(results, include_details=False) → str`**
- Generates human-readable summary
- **Output**: String like "Detected 3 objects: 1. person (95%) - 2.1m ahead..."

---

### 5. Visualizer

**Module**: `src.visualizer`

Visualization utilities.

#### Static Methods

**`visualize_detections(image, objects, show_confidence=True) → np.ndarray`**
- Draws bounding boxes on image
- **Output**: Annotated image with boxes and labels

**`visualize_masks(image, objects, alpha=0.5) → np.ndarray`**
- Overlays masks as colored regions
- **Output**: Image with mask overlays

**`visualize_depth_map(depth_map, title='Depth Map', cmap='plasma', save_path=None) → None`**
- Displays depth map as heatmap
- Creates matplotlib figure with colorbar

**`visualize_full_results(image, results, title=None, save_path=None) → None`**
- Creates comprehensive multi-subplot visualization
- Shows: original, depth map, detections, masks

---

### 6. DataLoader

**Module**: `src.data_loader`

Data loading utilities.

#### DataLoader (Base)
```python
loader = DataLoader(data_dir='./data/training')

# Access by index
sample = loader[0]

# Iterate
for sample in loader:
    image = sample['image']
    path = sample['path']
```

#### CustomDataLoader
```python
loader = CustomDataLoader(
    images_dir='./data/images',
    annotations_dir='./data/annotations',
    depth_dir='./data/depth'
)

sample = loader[0]
# Returns: image, annotations, depth map
```

---

### 7. Evaluator

**Module**: `src.evaluation`

Evaluation metrics computation.

#### Static Methods

**`evaluate_depth_map(predicted_depth, gt_depth) → Dict[str, float]`**
- Computes depth estimation metrics
- **Output**: Dict with keys:
  - `abs_rel`: Absolute relative error
  - `sq_rel`: Squared relative error  
  - `rmse`: Root mean squared error
  - `rmse_log`: RMSE of log
  - `a1`, `a2`, `a3`: Accuracy thresholds
  - `mae`: Mean absolute error

**`evaluate_detections(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes) → Dict`**
- Computes mAP and related metrics
- **Output**: Dict with:
  - `mAP`: Mean Average Precision
  - `mAP_50`, `mAP_75`: At specific IoU thresholds
  - Size-based variants

**`evaluate_segmentation(pred_masks, gt_masks) → Dict`**
- Segmentation quality metrics
- **Output**: Dict with:
  - `mean_iou`: Mean Intersection over Union
  - `mean_dice`: Dice coefficient
  - `boundary_f1`: Boundary F1 score

---

### 8. LLMGenerator

**Module**: `src.llm_generator`

Large Language Model text generation.

#### Initialization
```python
llm = LLMGenerator(
    model_name='gpt-3.5-turbo',
    temperature=0.7,
    max_tokens=150,
    api_key='your-api-key'  # or from environment
)
```

#### Methods

**`generate_description(objects) → str`**
- Natural language description of scene
- Suitable for accessibility applications

**`generate_scene_summary(image_description=None, objects=None) → str`**
- Comprehensive high-level scene description

**`generate_directions(target_object, reference_objects=None) → str`**
- Step-by-step directions to target object

**`generate_warning(nearby_objects, threshold_distance=1.0) → Optional[str]`**
- Safety warnings for nearby objects

---

## Helper Functions

### detector.py

```python
def convert_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray
def filter_detections_by_class(detections: List[Dict], class_ids: List[int]) -> List[Dict]
def filter_detections_by_confidence(detections: List[Dict], threshold: float) -> List[Dict]
```

### segmentor.py

```python
def mask_to_bbox(mask: np.ndarray) -> np.ndarray
def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float
def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float]
```

### depth_estimator.py

```python
def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray
def compute_depth_uncertainty(depth_map: np.ndarray) -> np.ndarray
def depth_map_to_3d_points(depth_map: np.ndarray, camera_intrinsics=None) -> np.ndarray
```

### evaluation.py

```python
def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float
def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float
def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float
def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]
def compute_f1_score(precision: float, recall: float) -> float
def compute_average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float
```

---

## Configuration

All settings in `config.py`:

```python
# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
YOLO_MODEL_NAME = 'yolov8m.pt'
YOLO_CONFIDENCE = 0.3
YOLO_IOU = 0.45

MOBILE_SAM_CHECKPOINT = 'model/weights/mobile_sam.pt'
MOBILE_SAM_MODEL_TYPE = 'vit_t'

DEPTH_MODEL_NAME = 'depth-anything-v2-small'

# Thresholds
DIR_LEFT = 0.33    # Left third of image
DIR_RIGHT = 0.67   # Right third of image
MAX_DEPTH_M = 10.0

# Data
BATCH_SIZE = 8
IMAGE_SIZE = (480, 640)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
```

---

## Example Workflows

### Single Image Processing
```python
from src.pipeline import VisualAssistantPipeline
from src.visualizer import Visualizer

pipeline = VisualAssistantPipeline()
results = pipeline.process_image('photo.jpg')

print(pipeline.get_summary_text(results))
Visualizer.visualize_full_results(results['image'], results)
```

### Batch Processing
```python
from pathlib import Path

image_dir = Path('./data/testing')
for img_path in image_dir.glob('*.jpg'):
    results = pipeline.process_image(str(img_path))
    # Process results...
```

### Evaluation
```python
from src.evaluation import Evaluator

metrics = Evaluator.evaluate_depth_map(pred_depth, gt_depth)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"mAP: {metrics.get('mean_iou', 'N/A')}")
```

---

## Error Handling

All functions should handle errors gracefully:

```python
try:
    results = pipeline.process_image(image_path)
except FileNotFoundError:
    print(f"Image not found: {image_path}")
except Exception as e:
    print(f"Pipeline error: {e}")
```

---

## Performance Considerations

- **Resolution**: Larger images are slower. Typical: 640×480
- **GPU/CPU**: GPU ~50ms per image, CPU ~500ms
- **Batch processing**: Use frame_interval to skip frames for speed
- **Memory**: Adjust batch_size in config.py

---

## TODOs for Implementers

- [ ] Complete all TODO sections in module docstrings
- [ ] Implement each function following input/output specs
- [ ] Write unit tests for each component
- [ ] Optimize performance-critical sections
- [ ] Add error handling and validation
- [ ] Document any deviations from spec

---

End of API Documentation
