"""
Unit tests for Depth Estimator

TODO: Write tests for depth estimation.
Run tests with: pytest tests/test_depth_estimator.py -v
"""

import pytest
import numpy as np
from src.depth_estimator import DepthEstimator
import config


class TestDepthEstimator:
    """Test suite for DepthEstimator class."""
    
    @pytest.fixture
    def depth_estimator(self):
        """Initialize depth estimator fixture."""
        # TODO: Create DepthEstimator instance
        pass
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        # TODO: Generate (480, 640, 3) uint8 image
        pass
    
    @pytest.fixture
    def sample_mask(self):
        """Create sample segmentation mask."""
        # TODO: Generate (480, 640) bool mask
        pass
    
    def test_estimator_initialization(self, depth_estimator):
        """Test init loads model."""
        # TODO: Assert estimator is not None
        # TODO: Assert model is loaded
        pass
    
    def test_estimate_depth_output(self, depth_estimator, sample_image):
        """Test estimate_depth returns correct format."""
        # TODO: Call estimate_depth
        # TODO: Assert output is numpy array
        # TODO: Assert shape matches input (H, W)
        # TODO: Assert dtype is float
        pass
    
    def test_depth_map_properties(self, depth_estimator, sample_image):
        """Test depth map has expected properties."""
        # TODO: Get depth map
        # TODO: Assert all values are finite (no NaN/Inf)
        # TODO: Assert values in reasonable range
        pass
    
    def test_depth_to_distance(self, depth_estimator, sample_image, sample_mask):
        """Test distance computation."""
        # TODO: Estimate depth
        # TODO: Compute distance from mask
        # TODO: Assert return is float
        # TODO: Assert value is positive
        pass
    
    def test_compute_direction(self, depth_estimator, sample_mask):
        """Test direction computation."""
        # TODO: Call compute_direction
        # TODO: Verify returns (direction, angle, norm_x)
        # TODO: Assert direction is valid string
        # TODO: Assert angle is float
        pass
    
    def test_scale_depth_with_gt(self, depth_estimator, sample_image):
        """Test metric depth scaling with ground truth."""
        # TODO: Estimate relative depth
        # TODO: Create mock GT depth
        # TODO: Scale to metric
        # TODO: Assert output is in valid range
        pass
    
    def test_scale_depth_without_gt(self, depth_estimator, sample_image):
        """Test metric depth scaling without GT."""
        # TODO: Estimate depth
        # TODO: Scale without GT
        # TODO: Assert reasonable scaling applied
        pass
