"""
Unit tests for Scene Analyzer Module

Run tests with: pytest tests/test_scene_analyzer.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.scene_analyzer import (
    analyze_scene,
    _parse_aggregation_modes,
    _extract_depth_in_region,
    _compute_centroid_from_mask_or_bbox,
    _aggregate_for_mode,
    _compute_direction_and_angle,
)
import config


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_image():
    """Create sample test image (480, 640, 3) uint8 RGB."""
    np.random.seed(42)
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_depth_map():
    """Create sample relative depth map (480, 640) float32 [0, 1]."""
    np.random.seed(42)
    depth = np.random.rand(480, 640).astype(np.float32)
    return depth


@pytest.fixture
def sample_mask():
    """Create sample segmentation mask (480, 640) bool."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:500] = True
    return mask


@pytest.fixture
def sample_detection(sample_mask):
    """Create a sample detection dict."""
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": np.array([200.0, 100.0, 500.0, 300.0], dtype=np.float32),
        "mask": sample_mask,
        "mask_score": 0.92,
    }


@pytest.fixture
def mock_detector():
    """Create mock YOLODetector."""
    detector = MagicMock()
    return detector


@pytest.fixture
def mock_segmentor():
    """Create mock MobileSAMSegmentor."""
    segmentor = MagicMock()
    return segmentor


@pytest.fixture
def mock_depth_estimator():
    """Create mock DepthEstimator."""
    depth_estimator = MagicMock()
    return depth_estimator


# ============================================================
# Tests: _parse_aggregation_modes
# ============================================================


class TestParseAggregationModes:
    """Test aggregation modes parsing."""

    def test_simple_modes(self):
        """Test parsing simple string modes."""
        modes = ["mean", "median", "max", "min"]
        result = _parse_aggregation_modes(modes)
        assert "mean" in result
        assert "median" in result
        assert "max" in result
        assert "min" in result
        assert result["mean"]["mode"] == "mean"

    def test_top_k_mode(self):
        """Test top_k mode parsing."""
        modes = [{"mode": "top_k", "k": 200}]
        result = _parse_aggregation_modes(modes)
        assert "top_k_200" in result
        assert result["top_k_200"]["mode"] == "top_k"
        assert result["top_k_200"]["k"] == 200

    def test_top_p_mode(self):
        """Test top_p mode parsing."""
        modes = [{"mode": "top_p", "p": 0.1}]
        result = _parse_aggregation_modes(modes)
        assert "top_p_0.1" in result
        assert result["top_p_0.1"]["mode"] == "top_p"
        assert result["top_p_0.1"]["p"] == 0.1

    def test_mixed_modes(self):
        """Test parsing mixed string and dict modes."""
        modes = ["mean", "median", {"mode": "top_k", "k": 50}, {"mode": "top_p", "p": 0.2}]
        result = _parse_aggregation_modes(modes)
        assert len(result) == 4
        assert "mean" in result
        assert "median" in result
        assert "top_k_50" in result
        assert "top_p_0.2" in result

    def test_invalid_simple_mode(self):
        """Test that unknown simple mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            _parse_aggregation_modes(["unknown_mode"])

    def test_invalid_dict_mode_name(self):
        """Test that invalid dict mode name raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'top_k' or 'top_p'"):
            _parse_aggregation_modes([{"mode": "invalid", "k": 10}])

    def test_missing_mode_key(self):
        """Test that dict without 'mode' key raises ValueError."""
        with pytest.raises(ValueError, match="'mode' key"):
            _parse_aggregation_modes([{"k": 10}])

    def test_top_k_missing_k(self):
        """Test that top_k without k parameter raises ValueError."""
        with pytest.raises(ValueError, match="top_k mode requires 'k'"):
            _parse_aggregation_modes([{"mode": "top_k"}])

    def test_top_p_missing_p(self):
        """Test that top_p without p parameter raises ValueError."""
        with pytest.raises(ValueError, match="top_p mode requires 'p'"):
            _parse_aggregation_modes([{"mode": "top_p"}])

    def test_invalid_k_value(self):
        """Test that invalid k value raises ValueError."""
        with pytest.raises(ValueError, match="positive int"):
            _parse_aggregation_modes([{"mode": "top_k", "k": 0}])

    def test_invalid_p_value(self):
        """Test that invalid p value raises ValueError."""
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            _parse_aggregation_modes([{"mode": "top_p", "p": 1.5}])

    def test_not_list_input(self):
        """Test that non-list input raises TypeError."""
        with pytest.raises(TypeError, match="list or tuple"):
            _parse_aggregation_modes("mean")


# ============================================================
# Tests: _extract_depth_in_region
# ============================================================


class TestExtractDepthInRegion:
    """Test depth extraction from regions."""

    def test_extract_with_mask(self, sample_depth_map, sample_mask):
        """Test extraction using mask."""
        depth_values = _extract_depth_in_region(
            sample_depth_map, sample_mask, np.array([0, 0, 640, 480], dtype=np.float32)
        )
        assert isinstance(depth_values, np.ndarray)
        assert len(depth_values) > 0
        assert all(np.isfinite(depth_values[np.isfinite(depth_values)]))

    def test_extract_with_empty_mask_fallback_to_bbox(self, sample_depth_map):
        """Test fallback to bbox when mask is empty."""
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        depth_values = _extract_depth_in_region(sample_depth_map, empty_mask, bbox)
        assert isinstance(depth_values, np.ndarray)
        assert len(depth_values) > 0

    def test_extract_with_none_mask(self, sample_depth_map):
        """Test extraction with None mask."""
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        depth_values = _extract_depth_in_region(sample_depth_map, None, bbox)
        assert isinstance(depth_values, np.ndarray)
        assert len(depth_values) > 0

    def test_extract_with_invalid_bbox(self, sample_depth_map):
        """Test extraction with bbox outside image."""
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([640.0, 480.0, 700.0, 500.0], dtype=np.float32)
        depth_values = _extract_depth_in_region(sample_depth_map, empty_mask, bbox)
        assert len(depth_values) == 0


# ============================================================
# Tests: _compute_centroid_from_mask_or_bbox
# ============================================================


class TestComputeCentroid:
    """Test centroid computation."""

    def test_centroid_from_mask(self, sample_mask):
        """Test centroid computation from mask."""
        cx, cy = _compute_centroid_from_mask_or_bbox(
            sample_mask, np.array([0, 0, 640, 480], dtype=np.float32)
        )
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert 200 <= cx <= 500  # Mask is at [200:500]
        assert 100 <= cy <= 300  # Mask is at [100:300]

    def test_centroid_from_empty_mask_fallback_to_bbox(self):
        """Test fallback to bbox when mask is empty."""
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        cx, cy = _compute_centroid_from_mask_or_bbox(empty_mask, bbox)
        assert cx == 200.0  # (100 + 300) / 2
        assert cy == 150.0  # (100 + 200) / 2

    def test_centroid_from_none_mask(self):
        """Test centroid computation with None mask."""
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        cx, cy = _compute_centroid_from_mask_or_bbox(None, bbox)
        assert cx == 200.0
        assert cy == 150.0


# ============================================================
# Tests: _safe_aggregate_depth
# ============================================================


class TestSafeAggregateDepth:
    """Test depth aggregation through _aggregate_for_mode wrapper."""

    def test_mean_aggregation(self):
        values = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "mean", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.4)

    def test_median_aggregation(self):
        values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "median", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.5)

    def test_max_aggregation(self):
        values = np.array([0.2, 0.4, 0.9], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "max", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.9)

    def test_min_aggregation(self):
        values = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "min", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.1)

    def test_top_k_aggregation_high(self):
        values = np.arange(10, dtype=np.float32) / 10.0  # [0.0, 0.1, ..., 0.9]
        result = _aggregate_for_mode(values, {"mode": "top_k", "k": 3, "p": None}, "high")
        # Should select [0.7, 0.8, 0.9] and mean = 0.8
        assert np.isclose(result, np.mean([0.7, 0.8, 0.9]))

    def test_top_p_aggregation_high(self):
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_p", "k": None, "p": 0.3}, "high")
        # top 30% = 3 values [0.7, 0.8, 0.9], mean = 0.8
        assert np.isclose(result, np.mean([0.7, 0.8, 0.9]))

    def test_top_k_aggregation_low(self):
        """top_k with closest_side='low' selects smallest values."""
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_k", "k": 3, "p": None}, "low")
        # Should select [0.0, 0.1, 0.2], mean = 0.1
        assert np.isclose(result, np.mean([0.0, 0.1, 0.2]))

    def test_empty_values_returns_nan(self):
        values = np.array([], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "mean", "k": None, "p": None}, "high")
        assert np.isnan(result)

    def test_nan_values_filtered(self):
        values = np.array([0.2, np.nan, 0.6, np.inf], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "mean", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.4)  # mean([0.2, 0.6])


# ============================================================
# Tests: _compute_direction_and_angle
# ============================================================


class TestComputeDirectionAndAngle:
    """Test direction and angle computation."""

    def test_center_direction(self):
        """Test center direction at x_norm=0.5."""
        direction, angle = _compute_direction_and_angle(0.5)
        assert direction == "center"
        assert np.isclose(angle, 0.0)

    def test_left_direction(self):
        """Test left direction at x_norm < DIR_LEFT."""
        direction, angle = _compute_direction_and_angle(0.2)
        assert direction == "left"
        assert angle < 0

    def test_right_direction(self):
        """Test right direction at x_norm > DIR_RIGHT."""
        direction, angle = _compute_direction_and_angle(0.8)
        assert direction == "right"
        assert angle > 0

    def test_angle_range(self):
        """Test that angle is within FOV range."""
        for x_norm in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _, angle = _compute_direction_and_angle(x_norm)
            fov = getattr(config, "HORIZONTAL_FOV", 60.0)
            assert -fov / 2 <= angle <= fov / 2


# ============================================================
# Tests: analyze_scene (Integration)
# ============================================================


class TestAnalyzeScene:
    """Test main analyze_scene function."""

    def test_happy_path_single_object(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test analyze_scene with single object and all stats."""
        # Setup mocks
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        aggregation_modes = ["mean", "median", {"mode": "top_k", "k": 10}]
        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            aggregation_modes,
        )

        assert len(result) == 1
        res = result[0]
        assert res["class_id"] == 0
        assert res["class_name"] == "person"
        assert res["confidence"] == 0.95
        assert res["direction"] in ["left", "center", "right"]
        assert "mean" in res["depth_stats"]
        assert "median" in res["depth_stats"]
        assert "top_k_10" in res["depth_stats"]

    def test_no_detections(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test analyze_scene with no detections."""
        mock_detector.detect.return_value = []
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
        )

        assert result == []

    def test_multiple_objects(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test analyze_scene with multiple objects."""
        det2 = sample_detection.copy()
        det2["class_id"] = 1
        det2["class_name"] = "car"

        mock_detector.detect.return_value = [sample_detection, det2]
        mock_segmentor.segment_detections.return_value = [sample_detection, det2]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
        )

        assert len(result) == 2
        assert result[0]["class_name"] == "person"
        assert result[1]["class_name"] == "car"

    def test_invalid_image_type(
        self, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that invalid image type raises TypeError."""
        with pytest.raises(TypeError, match="np.ndarray"):
            analyze_scene(
                [[[1, 2, 3]]],  # List instead of ndarray
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_invalid_image_shape(
        self, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that invalid image shape raises ValueError."""
        invalid_image = np.zeros((640,), dtype=np.uint8)  # 1D
        with pytest.raises(ValueError, match="shape"):
            analyze_scene(
                invalid_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_invalid_channel_count(
        self, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that invalid channel count raises ValueError."""
        invalid_image = np.zeros((480, 640, 4), dtype=np.uint8)  # RGBA
        with pytest.raises(ValueError, match="shape"):
            analyze_scene(
                invalid_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_zero_image_size(
        self, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that zero-size image raises ValueError."""
        invalid_image = np.zeros((0, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="dimensions"):
            analyze_scene(
                invalid_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_detector_error_wrapped(
        self, sample_image, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that detector errors are wrapped in RuntimeError."""
        mock_detector.detect.side_effect = RuntimeError("Detector failed")
        with pytest.raises(RuntimeError, match="YOLODetector"):
            analyze_scene(
                sample_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_segmentor_error_wrapped(
        self, sample_image, sample_detection, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that segmentor errors are wrapped in RuntimeError."""
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.side_effect = RuntimeError("Segmentor failed")
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        with pytest.raises(RuntimeError, match="MobileSAMSegmentor"):
            analyze_scene(
                sample_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_depth_estimator_error_wrapped(
        self, sample_image, sample_detection, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that depth estimator errors are wrapped in RuntimeError."""
        mock_detector.detect.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.side_effect = RuntimeError("Depth failed")

        with pytest.raises(RuntimeError, match="DepthEstimator"):
            analyze_scene(
                sample_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_scale_depth_to_meters(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test analyze_scene with depth scaling."""
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        # Mock scale_depth_to_meters
        scaled_depth = sample_depth_map * 10.0  # Simulated metric depth
        mock_depth_estimator.scale_depth_to_meters.return_value = (scaled_depth, 10.0, 0.0)

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
            scale_depth_to_meters=True,
        )

        assert len(result) == 1
        # Depth values should be scaled (roughly 10x larger)
        assert mock_depth_estimator.scale_depth_to_meters.called

    def test_empty_mask_fallback(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that empty mask falls back to bbox for depth."""
        det = {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32),
            "mask": np.zeros((480, 640), dtype=bool),  # Empty mask
            "mask_score": 0.0,
        }

        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
        )

        assert len(result) == 1
        assert not np.isnan(result[0]["depth_stats"]["mean"])

    def test_all_nan_depth_values(
        self, sample_image, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test handling of all-NaN depth region."""
        nan_depth = np.full((480, 640), np.nan, dtype=np.float32)

        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = nan_depth

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean", "median"],
        )

        assert len(result) == 1
        assert np.isnan(result[0]["depth_stats"]["mean"])
        assert np.isnan(result[0]["depth_stats"]["median"])

    def test_clipped_bbox(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that bbox is clipped to image bounds."""
        det = {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": np.array([600.0, 400.0, 700.0, 500.0], dtype=np.float32),  # Outside bounds
            "mask": np.zeros((480, 640), dtype=bool),
            "mask_score": 0.0,
        }

        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
        )

        assert len(result) == 1


# ============================================================
# Tests: Edge Cases & Centroid Normalization
# ============================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_centroid_x_norm_bounds(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that centroid_x_norm stays in [0, 1]."""
        # Create a mask at the far left
        left_mask = np.zeros((480, 640), dtype=bool)
        left_mask[100:200, 0:50] = True

        det = {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": np.array([0.0, 100.0, 50.0, 200.0], dtype=np.float32),
            "mask": left_mask,
            "mask_score": 0.9,
        }

        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            ["mean"],
        )

        assert len(result) == 1
        cx_norm = result[0]["centroid_x_norm"]
        assert 0.0 <= cx_norm <= 1.0

    def test_missing_detection_keys(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that missing detection keys raise RuntimeError."""
        incomplete_det = {
            "class_id": 0,
            "class_name": "person",
            # Missing: confidence, bbox, mask, mask_score
        }

        mock_detector.detect.return_value = [incomplete_det]
        mock_segmentor.segment_detections.return_value = [incomplete_det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        with pytest.raises(RuntimeError, match="missing keys"):
            analyze_scene(
                sample_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )

    def test_invalid_depth_map_shape(
        self, sample_image, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        """Test that 1D depth map raises ValueError."""
        bad_depth = np.ones(640, dtype=np.float32)

        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = bad_depth

        with pytest.raises(ValueError, match="2D"):
            analyze_scene(
                sample_image,
                mock_detector,
                mock_segmentor,
                mock_depth_estimator,
                ["mean"],
            )


# ============================================================
# Tests: scale_depth_to_meters auto-flip of closest_side
# ============================================================


class TestScaleDepthFlipsClosestSide:
    """When scale_depth_to_meters=True, default closest_side should flip."""

    def test_metric_flip_preserves_closest_pixel_semantics(
        self, sample_image, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        # Depth map: left half = 0.2 (closer, inverse-depth), right half = 0.8
        depth_map = np.zeros((480, 640), dtype=np.float32)
        depth_map[:, :320] = 0.2
        depth_map[:, 320:] = 0.8

        # scale_depth_to_meters flips: high relative -> low metric
        metric_depth = np.zeros_like(depth_map)
        metric_depth[:, :320] = 8.0   # originally 0.2 -> far in metric
        metric_depth[:, 320:] = 2.0   # originally 0.8 -> close in metric

        det = {
            "class_id": 0,
            "class_name": "obj",
            "confidence": 0.9,
            "bbox": np.array([0.0, 0.0, 640.0, 480.0], dtype=np.float32),
            "mask": np.ones((480, 640), dtype=bool),
            "mask_score": 0.9,
        }

        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = depth_map
        mock_depth_estimator.scale_depth_to_meters.return_value = (metric_depth, 1.0, 0.0)

        result = analyze_scene(
            sample_image,
            mock_detector,
            mock_segmentor,
            mock_depth_estimator,
            [{"mode": "top_k", "k": 100}],
            scale_depth_to_meters=True,
            # closest_side left as default "high" — should auto-flip to "low"
        )
        # top_k should now pick the smallest metric values = closest pixels = ~2.0
        top_k_val = result[0]["depth_stats"]["top_k_100"]
        assert top_k_val < 3.0, f"expected close-side (~2.0), got {top_k_val}"
