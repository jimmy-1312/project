"""
Visualization Module

Provides comprehensive visualization of pipeline results including detections,
masks, depth maps, and overlay combinations.

TODO: Implement visualization functions for analysis and debugging.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import config
import cv2
import matplotlib.pyplot as plt


class Visualizer:
    """
    Visualization utilities for pipeline results.
    
    This class provides static methods for visualizing various pipeline outputs
    including object detections, segmentation masks, depth maps, and combined views.
    """
    
    # Color palette for object visualization
    COLORS = [
        (255, 0, 0),       # Red
        (0, 255, 0),       # Green
        (0, 0, 255),       # Blue
        (255, 255, 0),     # Yellow
        (255, 0, 255),     # Magenta
        (0, 255, 255),     # Cyan
        (128, 0, 0),       # Maroon
        (0, 128, 0),       # Dark Green
        (0, 0, 128),       # Navy
        (128, 128, 0),     # Olive
        (128, 0, 128),     # Purple
        (0, 128, 128),     # Teal
        (255, 128, 0),     # Orange
        (255, 0, 128),     # Rose
        (128, 255, 0),     # Lime
        (0, 255, 128),     # Spring Green
    ]
    
    @staticmethod
    def visualize_detections(
        image: np.ndarray,
        objects: List[Dict],
        show_confidence: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Image array (H, W, 3) uint8 BGR
            objects: List of object dicts with 'bbox' and 'class_name'
            show_confidence: Whether to display confidence scores
            thickness: Line thickness for boxes
        
        Returns:
            Image with drawn bounding boxes (H, W, 3) uint8 BGR
        
        TODO:
        1. Create copy of image
        2. For each object:
           a. Get bbox [x1, y1, x2, y2]
           b. Assign color based on object index mod number of colors
           c. Draw rectangle on image
           d. Draw class_name and confidence% as text
        3. Return annotated image
        
        Tools needed: matplotlib or cv2 for drawing
        """
        # Create a copy (image is already BGR from DataLoader)
        img_display = image.copy()
        img_bgr = img_display
        
        for idx, obj in enumerate(objects):
            # Get bbox coordinates
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Get color for this object
            color = Visualizer.COLORS[idx % len(Visualizer.COLORS)]
            
            # Draw rectangle
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = obj['class_name']
            if show_confidence and 'confidence' in obj:
                label += f" {obj['confidence']:.2f}"
            
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # Draw background rectangle for text
            text_x, text_y = x1, max(y1 - 5, 20)
            cv2.rectangle(
                img_bgr,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                img_bgr,
                label,
                (text_x, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        # Convert back to RGB
        img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_display
    
    @staticmethod
    def visualize_masks(
        image: np.ndarray,
        objects: List[Dict],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay segmentation masks on image.
        
        Args:
            image: Image array (H, W, 3) uint8 RGB
            objects: List of object dicts with 'mask' and optional 'class_name'
            alpha: Transparency of mask overlay [0-1]
        
        Returns:
            Image with mask overlays (H, W, 3) uint8
        
        TODO:
        1. Create copy of image
        2. For each object:
           a. Get mask (H, W) bool
           b. Assign color based on index
           c. Overlay colored mask on image with alpha blending
           d. Optionally add border around mask
        3. Return combined image
        
        Note: Alpha blending: output = (1-alpha)*image + alpha*mask_overlay
        """
        pass
    
    @staticmethod
    def visualize_depth_map(
        depth_map: np.ndarray,
        title: str = "Depth Map",
        cmap: str = 'plasma',
        save_path: Optional[str] = None,
    ) -> None:
        """
        Display depth map as heatmap.
        
        Args:
            depth_map: Depth map (H, W) float
            title: Figure title
            cmap: Matplotlib colormap name
            save_path: Optional path to save figure
        
        TODO:
        1. Create figure with matplotlib
        2. Display depth_map as image with specified colormap
        3. Add colorbar showing depth scale
        4. Set title and remove axes
        5. If save_path provided, save figure
        6. Display figure
        
        Note: Use plt.imshow() and plt.colorbar()
        """
        pass
    
    @staticmethod
    def visualize_full_results(
        image: np.ndarray,
        results: Dict,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive visualization with 3+ subplots.
        
        Args:
            image: Original image (H, W, 3)
            results: Results dict from pipeline.process_image()
            title: Figure title
            save_path: Optional path to save figure
        
        Returns:
            None (displays figure)
        
        TODO:
        1. Create figure with 3-4 subplots:
           - Original image
           - Detections (boxes)
           - Masks overlay
           - Depth map
        2. For each subplot:
           a. Plot the visualization
           b. Add minimal labels
           c. Remove axes
        3. Add overall figure title
        4. Save if path provided
        5. Display figure
        
        Note: Use matplotlib.pyplot subplots
        """
        pass
    
    @staticmethod
    def create_detection_summary_image(
        image: np.ndarray,
        objects: List[Dict],
    ) -> np.ndarray:
        """
        Create annotated image with all detection info.
        
        Args:
            image: Original image (H, W, 3)
            objects: List of detected objects
        
        Returns:
            Annotated image with boxes, masks, labels, distances
        
        TODO:
        1. Draw detections (boxes)
        2. Draw masks (overlays)
        3. For each object, draw label with:
           - class_name
           - confidence%
           - distance_m
           - direction
        4. Use different positions to avoid overlap
        5. Return annotated image
        """
        pass
    
    @staticmethod
    def create_depth_analysis_image(
        image: np.ndarray,
        depth_map: np.ndarray,
        objects: List[Dict],
    ) -> np.ndarray:
        """
        Create visualization combining depth map with object distances.
        
        Args:
            image: Original image
            depth_map: Metric depth map
            objects: Detected objects with masks
        
        Returns:
            Composite image showing depth and detections
        
        TODO:
        1. Create colored depth map (use colormap)
        2. For each object:
           a. Draw mask boundaries
           b. Add distance label
        3. Combine original image with depth visualization
        4. Return composite
        """
        pass
    
    @staticmethod
    def create_comparison_image(
        original: np.ndarray,
        processed: np.ndarray,
    ) -> np.ndarray:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original image (H, W, 3)
            processed: Processed/annotated image (H, W, 3)
        
        Returns:
            Side-by-side composite (H, 2*W, 3)
        
        TODO:
        1. Ensure both images same height
        2. Resize if needed
        3. Concatenate horizontally
        4. Optionally add "Before | After" labels
        5. Return combined image
        """
        pass


# ============================================================
# Helper Functions (Optional)
# ============================================================

def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw single bounding box on image.
    
    Args:
        image: Image array (H, W, 3)
        bbox: [x1, y1, x2, y2]
        color: (B, G, R) color tuple for OpenCV or (R, G, B) for matplotlib
        thickness: Line thickness
    
    Returns:
        Image with drawn box
    
    TODO:
    1. Use cv2.rectangle() or similar
    2. Draw rectangle from (x1,y1) to (x2,y2)
    3. Return modified image
    """
    pass


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_size: int = 12,
) -> np.ndarray:
    """
    Draw text label on image.
    
    Args:
        image: Image array
        text: Text to draw
        position: (x, y) coordinates for text
        color: (R, G, B) color tuple
        font_size: Font size in pixels
    
    Returns:
        Image with text drawn
    
    TODO:
    1. Use cv2.putText() or PIL ImageDraw
    2. Handle font selection
    3. Return modified image
    """
    pass


def mask_to_contour_image(
    mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Convert mask to contour outline image.
    
    Args:
        mask: Binary mask (H, W) bool
        color: (R, G, B) color for outline
        thickness: Line thickness
    
    Returns:
        Image (H, W, 3) with contour drawn
    
    TODO:
    1. Find contours from mask using cv2.findContours()
    2. Create blank image
    3. Draw contours
    4. Return image
    """
    pass


def create_colormap_image(
    data: np.ndarray,
    cmap: str = 'plasma',
) -> np.ndarray:
    """
    Convert 2D data to RGB image using colormap.
    
    Args:
        data: 2D array (H, W)
        cmap: Matplotlib colormap name
    
    Returns:
        RGB image (H, W, 3) uint8
    
    TODO:
    1. Normalize data to [0, 1]
    2. Apply colormap using matplotlib.cm
    3. Convert to uint8 in [0, 255]
    4. Return as RGB image
    """
    pass
