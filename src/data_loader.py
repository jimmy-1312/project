"""
Data Loading Module

Provides utilities for loading training, validation, and test data.
Supports loading from various dataset formats.

TODO: Implement data loaders for different data sources.
"""

import os
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
from pathlib import Path
import config
import cv2


class DataLoader:
    """
    Base data loader class for image datasets.
    
    Handles loading images and optional ground truth labels/depth maps.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_extensions: List[str] = None,
    ) -> None:
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing images
            image_extensions: List of image file extensions to look for
                            Default: ['.jpg', '.jpeg', '.png', '.bmp']
        
        TODO:
        1. Store data_dir path
        2. Set default image_extensions if not provided
        3. Scan directory for all images
        4. Store list of image paths
        5. Print loading summary
        """
        self.data_dir = Path(data_dir)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Scan directory for images
        self.image_paths = []
        for ext in self.image_extensions:
            self.image_paths.extend(self.data_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.data_dir.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"DataLoader initialized with {len(self.image_paths)} images from {data_dir}")
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get single sample by index.
        
        Returns:
            Dict with keys:
                'image': numpy array (H, W, 3) uint8
                'path': str - image file path
                'filename': str - image filename
        """
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path))  # Loads as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        return {
            'image': image,
            'path': str(image_path),
            'filename': image_path.name,
        }
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate through all samples."""
        for i in range(len(self)):
            yield self[i]
    
    def load_batch(
        self,
        indices: List[int],
        resize: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        """
        Load multiple samples.
        
        Args:
            indices: List of sample indices
            resize: Optional (height, width) to resize images
        
        Returns:
            List of sample dicts
        
        TODO:
        1. For each index, load image
        2. Optionally resize
        3. Return list of dicts
        """
        samples = []
        for idx in indices:
            sample = self[idx]
            if resize:
                sample['image'] = cv2.resize(sample['image'], (resize[1], resize[0]))
            samples.append(sample)
        return samples


class CustomDataLoader(DataLoader):
    """
    Custom dataset loader with support for structured data.
    
    Loads images and associated annotations (depth maps, bounding boxes, etc.)
    from organized directory structure.
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize CustomDataLoader.
        
        Args:
            images_dir: Directory with images
            annotations_dir: Directory with bounding box annotations (JSON/XML)
            depth_dir: Directory with ground truth depth maps
        
        TODO:
        1. Call parent __init__
        2. Store annotation and depth directories
        3. Match images with annotations and depth maps
        4. Print dataset statistics
        """
        pass
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get sample with image, annotations, and depth.
        
        Returns:
            Dict with:
                'image': (H, W, 3) uint8
                'path': str
                'annotations': dict (if available)
                'depth': (H, W) float (if available)
        
        TODO:
        1. Load image
        2. Load annotations if exist
        3. Load depth map if exist
        4. Return combined dict
        """
        pass
    
    def load_annotations(
        self,
        annotation_path: str
    ) -> Dict:
        """
        Load bounding box annotations from file.
        
        Args:
            annotation_path: Path to annotation file (JSON or XML)
        
        Returns:
            Dict with boxes and class labels
        
        TODO:
        1. Detect file format (JSON or XML)
        2. Parse file appropriately
        3. Return structured dict with annotations
        4. Handle missing files
        """
        pass
    
    def load_depth_map(
        self,
        depth_path: str
    ) -> np.ndarray:
        """
        Load depth map from file.
        
        Args:
            depth_path: Path to depth map file
        
        Returns:
            Depth map (H, W) float in meters
        
        TODO:
        1. Load from HDF5, NPZ, or PNG
        2. Convert to float32
        3. Handle different depth value ranges
        4. Return depth map
        """
        pass


class NYUDepthV2Loader(DataLoader):
    """
    Loader for NYU Depth V2 dataset.
    
    Specialized loader for NYU Depth V2 benchmark dataset used for depth estimation.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = 'train',
    ) -> None:
        """
        Initialize NYU Depth V2 loader.
        
        Args:
            dataset_path: Path to NYU Depth V2 dataset
            split: 'train' or 'test'
        
        TODO:
        1. Download dataset if not present
        2. Load .mat file structure
        3. Index images and depth maps
        4. Store metadata
        """
        pass


# ============================================================
# Dataset Utilities
# ============================================================

def create_train_val_split(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Directory with all images
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
    
    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    
    TODO:
    1. List all image files
    2. Randomly split according to ratios
    3. Return three lists of file paths
    """
    pass


def augment_image(
    image: np.ndarray,
    augmentation_type: str = 'random',
) -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Image array (H, W, 3)
        augmentation_type: Type of augmentation
            - 'flip_h': horizontal flip
            - 'flip_v': vertical flip
            - 'rotate': random rotation
            - 'brightness': random brightness adjustment
            - 'random': apply random transformations
    
    Returns:
        Augmented image
    
    TODO:
    1. Implement each augmentation type
    2. For 'random', randomly select and apply transformations
    3. Return augmented image
    
    Useful library: albumentations or torchvision.transforms
    """
    pass


def normalize_image(
    image: np.ndarray,
    mean: List[float] = None,
    std: List[float] = None,
) -> np.ndarray:
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image: Image (H, W, 3) float in [0, 1] or uint8 in [0, 255]
        mean: Mean per channel (R, G, B). Default to ImageNet stats
        std: Std per channel (R, G, B). Default to ImageNet stats
    
    Returns:
        Normalized image (H, W, 3) float
    
    TODO:
    1. Convert to [0, 1] if needed
    2. Use provided statistics or defaults
    3. Apply normalization: (x - mean) / std
    4. Return normalized image
    """
    pass


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: str = 'bilinear',
) -> np.ndarray:
    """
    Resize image to specified size.
    
    Args:
        image: Image array (H, W, 3)
        size: Target (height, width)
        interpolation: 'bilinear', 'nearest', etc.
    
    Returns:
        Resized image
    
    TODO:
    1. Use cv2.resize() or PIL.Image.resize()
    2. Apply specified interpolation
    3. Maintain image dtype
    4. Return resized image
    """
    pass


def save_dataset_split(
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    split_file: str = 'dataset_split.txt',
) -> None:
    """
    Save dataset split information to files.
    
    Args:
        train_paths: List of training image paths
        val_paths: List of validation image paths
        test_paths: List of test image paths
        split_file: Base filename for splits
    
    TODO:
    1. Create three files: {split_file}_train.txt, etc.
    2. Write image paths to files (one per line)
    3. Print summary of split
    """
    pass
