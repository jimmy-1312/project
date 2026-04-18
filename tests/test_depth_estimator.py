"""
Unit tests for Depth Estimator

Run tests with: pytest tests/test_depth_estimator.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.depth_estimator import DepthEstimator
from src.evaluation import evaluate_depth_qualitative, Evaluator
import config


class TestDepthEstimator:
    """Test suite for DepthEstimator class."""

    @pytest.fixture
    def sample_image(self):
        """Create sample test image (480, 640, 3) uint8 RGB."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        return image

    @pytest.fixture
    def sample_mask(self):
        """Create sample segmentation mask (480, 640) bool."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:500] = 255
        return mask

    @pytest.fixture
    def sample_depth_map(self):
        """Create sample relative depth map (480, 640) float32 [0, 1]."""
        np.random.seed(42)
        depth = np.random.rand(480, 640).astype(np.float32)
        return depth

    @pytest.fixture
    def depth_estimator(self):
        """Initialize depth estimator with mocked model."""
        with patch('src.depth_estimator.AutoImageProcessor') as mock_processor_cls, \
             patch('src.depth_estimator.AutoModelForDepthEstimation') as mock_model_cls:
            # Mock processor
            mock_processor = MagicMock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            # Mock model
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model

            estimator = DepthEstimator(
                model_name='depth-anything/Depth-Anything-V2-Small-hf',
                device='cpu'
            )
            estimator.processor = mock_processor
            estimator.model = mock_model

            return estimator

    def test_estimator_initialization(self, depth_estimator):
        """Test init loads model."""
        assert depth_estimator is not None
        assert depth_estimator.model is not None
        assert depth_estimator.processor is not None
        assert depth_estimator.device == 'cpu'

    def test_estimate_depth_output(self, depth_estimator, sample_image):
        """Test estimate_depth returns correct format."""
        with patch('torch.nn.functional.interpolate') as mock_interp, \
             patch('torch.no_grad'):
            # Mock the depth output
            mock_output = MagicMock()
            mock_output.predicted_depth = MagicMock()
            depth_estimator.model.return_value = mock_output

            # Mock interpolate to return correct shape
            depth_tensor = MagicMock()
            depth_tensor.squeeze.return_value = MagicMock()
            depth_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = (
                np.ones((480, 640), dtype=np.float32)
            )
            mock_interp.return_value = depth_tensor

            result = depth_estimator.estimate_depth(sample_image)

            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.dtype == np.float32

    def test_depth_map_properties(self, depth_estimator, sample_depth_map):
        """Test depth map has expected properties."""
        assert np.all(np.isfinite(sample_depth_map))
        assert sample_depth_map.min() >= 0.0
        assert sample_depth_map.max() <= 1.0

    def test_depth_to_distance_with_mask(self, depth_estimator, sample_depth_map, sample_mask):
        """Test distance computation with mask."""
        distance = depth_estimator.depth_to_distance(sample_depth_map, sample_mask)
        assert isinstance(distance, float)
        assert 0 <= distance <= 1.0 or np.isnan(distance)

    def test_depth_to_distance_without_mask(self, depth_estimator, sample_depth_map):
        """Test distance computation without mask (full image)."""
        distance = depth_estimator.depth_to_distance(sample_depth_map)
        assert isinstance(distance, float)
        assert 0 <= distance <= 1.0

    def test_depth_to_distance_empty_mask(self, depth_estimator, sample_depth_map):
        """Test distance with empty mask."""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        distance = depth_estimator.depth_to_distance(sample_depth_map, empty_mask)
        assert np.isnan(distance)

    def test_compute_direction(self, depth_estimator, sample_mask):
        """Test direction computation."""
        direction, angle, norm_x = depth_estimator.compute_direction(sample_mask, 640)

        assert isinstance(direction, str)
        assert direction in ['left', 'center', 'right', 'unknown']
        assert isinstance(angle, float)
        assert isinstance(norm_x, float)
        assert 0 <= norm_x <= 1.0
        assert -30 <= angle <= 30  # FOV=60 degrees

    def test_compute_direction_empty_mask(self, depth_estimator):
        """Test direction with empty mask."""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        direction, angle, norm_x = depth_estimator.compute_direction(empty_mask, 640)

        assert direction == 'unknown'
        assert angle == 0.0
        assert norm_x == 0.5

    def test_compute_direction_left(self, depth_estimator):
        """Test left direction detection."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:150, 50:100] = 255  # Left side
        direction, angle, norm_x = depth_estimator.compute_direction(mask, 640)

        assert direction == 'left'

    def test_compute_direction_right(self, depth_estimator):
        """Test right direction detection."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:150, 550:600] = 255  # Right side
        direction, angle, norm_x = depth_estimator.compute_direction(mask, 640)

        assert direction == 'right'

    def test_scale_depth_without_gt(self, depth_estimator, sample_depth_map):
        """Test metric depth scaling without GT."""
        max_depth = 10.0
        scaled, scale, shift = depth_estimator.scale_depth_to_meters(
            sample_depth_map, gt_depth=None, max_depth=max_depth
        )

        assert isinstance(scaled, np.ndarray)
        assert scaled.shape == sample_depth_map.shape
        assert 0 <= scaled.min() <= scaled.max() <= max_depth
        assert scale == max_depth
        assert shift == 0.0

    def test_scale_depth_with_gt(self, depth_estimator, sample_depth_map):
        """Test metric depth scaling with ground truth."""
        # Create synthetic GT with linear relationship
        gt_depth = sample_depth_map * 5.0 + 2.0  # scale=5, shift=2

        scaled, scale, shift = depth_estimator.scale_depth_to_meters(
            sample_depth_map, gt_depth=gt_depth, max_depth=10.0
        )

        assert isinstance(scaled, np.ndarray)
        assert 0 <= scaled.min() <= scaled.max() <= 10.0

    def test_edge_case_single_pixel(self, depth_estimator):
        """Test single pixel image."""
        single_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)

        with patch('torch.nn.functional.interpolate') as mock_interp, \
             patch('torch.no_grad'):
            mock_output = MagicMock()
            mock_output.predicted_depth = MagicMock()
            depth_estimator.model.return_value = mock_output

            depth_tensor = MagicMock()
            depth_tensor.squeeze.return_value = MagicMock()
            depth_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = (
                np.ones((1, 1), dtype=np.float32)
            )
            mock_interp.return_value = depth_tensor

            result = depth_estimator.estimate_depth(single_pixel)
            assert result.shape == (1, 1)

    def test_edge_case_grayscale_conversion(self, depth_estimator):
        """Test grayscale to RGB conversion."""
        grayscale = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

        with patch('torch.nn.functional.interpolate') as mock_interp, \
             patch('torch.no_grad'):
            mock_output = MagicMock()
            mock_output.predicted_depth = MagicMock()
            depth_estimator.model.return_value = mock_output

            depth_tensor = MagicMock()
            depth_tensor.squeeze.return_value = MagicMock()
            depth_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = (
                np.ones((480, 640), dtype=np.float32)
            )
            mock_interp.return_value = depth_tensor

            result = depth_estimator.estimate_depth(grayscale)
            assert result.shape == (480, 640)
            assert result.dtype == np.float32

    def test_evaluate_depth_qualitative(self, sample_depth_map):
        """Test qualitative depth evaluation."""
        stats = evaluate_depth_qualitative(sample_depth_map)

        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'valid_ratio' in stats
        assert 'shape' in stats
        assert stats['shape'] == (480, 640)
        assert 0 <= stats['valid_ratio'] <= 1.0

    def test_evaluate_depth_map_with_gt(self):
        """Test NYU depth evaluation."""
        np.random.seed(42)
        pred = np.random.uniform(0.5, 8.0, (480, 640)).astype(np.float32)
        gt = pred + np.random.normal(0, 0.5, (480, 640))  # Add noise
        gt = np.clip(gt, 0.1, 10.0)

        metrics = Evaluator.evaluate_depth_map(pred, gt)

        assert 'abs_rel' in metrics
        assert 'sq_rel' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'delta_1' in metrics
        assert 'delta_2' in metrics
        assert 'delta_3' in metrics

        # Delta should be monotonically increasing
        assert metrics['delta_1'] <= metrics['delta_2'] <= metrics['delta_3']

    def test_evaluate_depth_map_empty(self):
        """Test evaluation with no valid pixels."""
        pred = np.full((480, 640), np.nan, dtype=np.float32)
        gt = np.full((480, 640), np.nan, dtype=np.float32)

        metrics = Evaluator.evaluate_depth_map(pred, gt)

        assert np.isnan(metrics['abs_rel'])
        assert np.isnan(metrics['rmse'])
