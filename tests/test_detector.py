"""
Unit tests for YOLO Detector

TODO: Write unit tests to verify detector functionality.
Run tests with: pytest tests/test_detector.py -v
"""

import pytest
import numpy as np
from src.detector import YOLODetector
import config


class TestYOLODetector:
    """Test suite for YOLODetector class."""
    
    @pytest.fixture
    def detector(self):
        """Initialize detector fixture."""
        # TODO: Create detector instance
        # TODO: Yield detector for tests
        # TODO: Cleanup after tests
        pass
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        # TODO: Generate dummy image (H, W, 3) uint8
        # TODO: Return numpy array
        pass
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes properly."""
        # TODO: Assert detector is not None
        # TODO: Assert model is loaded
        pass
    
    def test_detect_returns_list(self, detector, sample_image):
        """Test that detect() returns list of dicts."""
        # TODO: Call detector.detect(sample_image)
        # TODO: Assert return type is list
        # TODO: Assert each element is dict with required keys
        pass
    
    def test_detection_format(self, detector, sample_image):
        """Test detection output format."""
        # TODO: Get detections
        # TODO: Verify each dict has: bbox, confidence, class_id, class_name
        # TODO: Verify bbox is numpy array with 4 elements
        # TODO: Verify confidence is float in [0, 1]
        pass
    
    def test_class_names_property(self, detector):
        """Test class_names property returns dict."""
        # TODO: Get class_names
        # TODO: Assert it's a dict
        # TODO: Verify mappings make sense
        pass
    
    def test_empty_image_handling(self, detector):
        """Test detector handles images with no objects."""
        # TODO: Create blank/empty image
        # TODO: Run detection
        # TODO: Assert returns empty list (not error)
        pass
    
    def test_different_image_sizes(self, detector):
        """Test detector handles various image sizes."""
        # TODO: Create images of different sizes
        # TODO: Run detection on each
        # TODO: Verify output is valid for each size
        pass
    
    def test_confidence_filtering(self):
        """Test confidence threshold filtering."""
        # TODO: Create detector with low confidence threshold
        # TODO: Get detections
        # TODO: Verify all have confidence >= threshold
        pass
