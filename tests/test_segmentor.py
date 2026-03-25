"""
Unit tests for MobileSAM Segmentor

TODO: Write tests for segmentation functionality.
Run tests with: pytest tests/test_segmentor.py -v
"""

import pytest
import numpy as np
from src.segmentor import MobileSAMSegmentor
import config


class TestMobileSAMSegmentor:
    """Test suite for MobileSAMSegmentor class."""
    
    @pytest.fixture
    def segmentor(self):
        """Initialize segmentor fixture."""
        # TODO: Create segmentor instance
        pass
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        # TODO: Generate test image (H, W, 3) uint8
        pass
    
    @pytest.fixture
    def sample_bbox(self):
        """Create sample bounding box."""
        # TODO: Return [x1, y1, x2, y2] bbox
        pass
    
    def test_segmentor_initialization(self, segmentor):
        """Test segmentor initializes."""
        # TODO: Assert segmentor is not None
        # TODO: Assert predictor is ready
        pass
    
    def test_set_image(self, segmentor, sample_image):
        """Test set_image pre-processing."""
        # TODO: Call set_image
        # TODO: Assert image is stored
        pass
    
    def test_segment_box_output(self, segmentor, sample_image, sample_bbox):
        """Test segment_box returns correct format."""
        # TODO: Set image
        # TODO: Call segment_box
        # TODO: Assert returns (mask, score)
        # TODO: Assert mask is (H, W) bool
        # TODO: Assert score is float in [0, 1]
        pass
    
    def test_mask_properties(self, segmentor, sample_image, sample_bbox):
        """Test mask output properties."""
        # TODO: Get mask
        # TODO: Assert dtype is bool
        # TODO: Assert shape matches image size
        # TODO: Assert contains True values
        pass
    
    def test_segment_detections(self, segmentor):
        """Test batch segmentation on multiple detections."""
        # TODO: Create mock detections with boxes
        # TODO: Call segment_detections
        # TODO: Verify all detections have masks added
        pass
    
    def test_empty_image_handling(self, segmentor):
        """Test handling of edge cases."""
        # TODO: Test with empty/blank image
        # TODO: Test with single-pixel region
        # TODO: Assert no crashes
        pass
