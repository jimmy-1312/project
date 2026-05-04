"""
Unit tests for Scene Analyzer Module

Run tests with: pytest tests/test_scene_analyzer.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
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
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_map():
    """Create sample relative depth map (480, 640) float32 [0, 1]."""
    np.random.seed(42)
    return np.random.rand(480, 640).astype(np.float32)


@pytest.fixture
def gradient_depth():
    """Left=0.0, Right=1.0 horizontal gradient."""
    x = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    return np.tile(x, (480, 1))


@pytest.fixture
def sample_mask():
    """Create sample segmentation mask (480, 640) bool."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:500] = True
    return mask


@pytest.fixture
def left_mask():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 10:60] = True
    return mask


@pytest.fixture
def right_mask():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 580:630] = True
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
    return MagicMock()


@pytest.fixture
def mock_segmentor():
    return MagicMock()


@pytest.fixture
def mock_depth_estimator():
    return MagicMock()


# ============================================================
# Tests: _parse_aggregation_modes
# ============================================================


class TestParseAggregationModes:

    def test_simple_modes(self):
        result = _parse_aggregation_modes(["mean", "median", "max", "min"])
        assert set(result.keys()) == {"mean", "median", "max", "min"}
        assert result["mean"]["mode"] == "mean"

    def test_simple_modes_have_none_k_p(self):
        result = _parse_aggregation_modes(["mean"])
        assert result["mean"]["k"] is None
        assert result["mean"]["p"] is None

    def test_top_k_mode(self):
        result = _parse_aggregation_modes([{"mode": "top_k", "k": 200}])
        assert "top_k_200" in result
        assert result["top_k_200"]["mode"] == "top_k"
        assert result["top_k_200"]["k"] == 200

    def test_top_p_mode(self):
        result = _parse_aggregation_modes([{"mode": "top_p", "p": 0.1}])
        assert "top_p_0.1" in result
        assert result["top_p_0.1"]["p"] == 0.1

    def test_mixed_modes(self):
        modes = ["mean", "median", {"mode": "top_k", "k": 50}, {"mode": "top_p", "p": 0.2}]
        result = _parse_aggregation_modes(modes)
        assert len(result) == 4

    def test_invalid_simple_mode(self):
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            _parse_aggregation_modes(["unknown_mode"])

    def test_invalid_dict_mode_name(self):
        with pytest.raises(ValueError, match="mode must be 'top_k' or 'top_p'"):
            _parse_aggregation_modes([{"mode": "invalid", "k": 10}])

    def test_missing_mode_key(self):
        with pytest.raises(ValueError, match="'mode' key"):
            _parse_aggregation_modes([{"k": 10}])

    def test_top_k_missing_k(self):
        with pytest.raises(ValueError, match="top_k mode requires 'k'"):
            _parse_aggregation_modes([{"mode": "top_k"}])

    def test_top_p_missing_p(self):
        with pytest.raises(ValueError, match="top_p mode requires 'p'"):
            _parse_aggregation_modes([{"mode": "top_p"}])

    def test_invalid_k_value(self):
        with pytest.raises(ValueError, match="positive int"):
            _parse_aggregation_modes([{"mode": "top_k", "k": 0}])

    def test_invalid_p_value(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            _parse_aggregation_modes([{"mode": "top_p", "p": 1.5}])

    def test_invalid_p_zero(self):
        with pytest.raises(ValueError):
            _parse_aggregation_modes([{"mode": "top_p", "p": 0.0}])

    def test_not_list_input(self):
        with pytest.raises(TypeError, match="list or tuple"):
            _parse_aggregation_modes("mean")

    def test_non_str_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="must be str or dict"):
            _parse_aggregation_modes([42])

    def test_empty_list_returns_empty_dict(self):
        assert _parse_aggregation_modes([]) == {}

    def test_duplicate_keys_last_wins(self):
        result = _parse_aggregation_modes([
            {"mode": "top_k", "k": 50},
            {"mode": "top_k", "k": 50},
        ])
        assert len(result) == 1


# ============================================================
# Tests: _extract_depth_in_region
# ============================================================


class TestExtractDepthInRegion:

    def test_extract_with_mask(self, sample_depth_map, sample_mask):
        values = _extract_depth_in_region(
            sample_depth_map, sample_mask,
            np.array([0, 0, 640, 480], dtype=np.float32)
        )
        assert isinstance(values, np.ndarray)
        assert len(values) > 0

    def test_mask_values_match_expected(self, gradient_depth, sample_mask):
        bbox = np.array([0.0, 0.0, 640.0, 480.0], dtype=np.float32)
        values = _extract_depth_in_region(gradient_depth, sample_mask, bbox)
        expected = gradient_depth[sample_mask]
        np.testing.assert_allclose(np.sort(values), np.sort(expected))

    def test_extract_with_empty_mask_fallback_to_bbox(self, sample_depth_map):
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        values = _extract_depth_in_region(sample_depth_map, empty_mask, bbox)
        assert len(values) > 0

    def test_extract_with_none_mask(self, sample_depth_map):
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        values = _extract_depth_in_region(sample_depth_map, None, bbox)
        assert len(values) > 0

    def test_extract_with_fully_outside_bbox_returns_empty(self, sample_depth_map):
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([640.0, 480.0, 700.0, 500.0], dtype=np.float32)
        values = _extract_depth_in_region(sample_depth_map, empty_mask, bbox)
        assert len(values) == 0

    def test_partial_out_of_bounds_bbox_clipped(self, sample_depth_map):
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([-10.0, -10.0, 50.0, 50.0], dtype=np.float32)
        values = _extract_depth_in_region(sample_depth_map, empty_mask, bbox)
        assert len(values) > 0


# ============================================================
# Tests: _compute_centroid_from_mask_or_bbox
# ============================================================


class TestComputeCentroid:

    def test_centroid_from_mask(self, sample_mask):
        cx, cy = _compute_centroid_from_mask_or_bbox(
            sample_mask, np.array([0, 0, 640, 480], dtype=np.float32)
        )
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert 200 <= cx <= 500
        assert 100 <= cy <= 300

    def test_centroid_from_empty_mask_fallback_to_bbox(self):
        empty_mask = np.zeros((480, 640), dtype=bool)
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        cx, cy = _compute_centroid_from_mask_or_bbox(empty_mask, bbox)
        assert cx == 200.0
        assert cy == 150.0

    def test_centroid_from_none_mask(self):
        bbox = np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32)
        cx, cy = _compute_centroid_from_mask_or_bbox(None, bbox)
        assert cx == 200.0
        assert cy == 150.0

    def test_left_mask_centroid_in_left_third(self, left_mask):
        bbox = np.array([0.0, 0.0, 640.0, 480.0], dtype=np.float32)
        cx, _ = _compute_centroid_from_mask_or_bbox(left_mask, bbox)
        assert cx < 640 * 0.33

    def test_right_mask_centroid_in_right_third(self, right_mask):
        bbox = np.array([0.0, 0.0, 640.0, 480.0], dtype=np.float32)
        cx, _ = _compute_centroid_from_mask_or_bbox(right_mask, bbox)
        assert cx > 640 * 0.67


# ============================================================
# Tests: _aggregate_for_mode
# ============================================================


class TestSafeAggregateDepth:

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
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_k", "k": 3, "p": None}, "high")
        assert np.isclose(result, np.mean([0.7, 0.8, 0.9]))

    def test_top_p_aggregation_high(self):
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_p", "k": None, "p": 0.3}, "high")
        assert np.isclose(result, np.mean([0.7, 0.8, 0.9]))

    def test_top_k_aggregation_low(self):
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_k", "k": 3, "p": None}, "low")
        assert np.isclose(result, np.mean([0.0, 0.1, 0.2]))

    def test_top_p_aggregation_low(self):
        values = np.arange(10, dtype=np.float32) / 10.0
        result = _aggregate_for_mode(values, {"mode": "top_p", "k": None, "p": 0.3}, "low")
        assert np.isclose(result, np.mean([0.0, 0.1, 0.2]))

    def test_empty_values_returns_nan(self):
        values = np.array([], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "mean", "k": None, "p": None}, "high")
        assert np.isnan(result)

    def test_nan_and_inf_values_filtered(self):
        values = np.array([0.2, np.nan, 0.6, np.inf], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "mean", "k": None, "p": None}, "high")
        assert np.isclose(result, 0.4)

    def test_top_k_larger_than_array_uses_all(self):
        values = np.array([0.3, 0.7], dtype=np.float32)
        result = _aggregate_for_mode(values, {"mode": "top_k", "k": 100, "p": None}, "high")
        assert np.isclose(result, np.mean([0.3, 0.7]))


# ============================================================
# Tests: _compute_direction_and_angle
# ============================================================


class TestComputeDirectionAndAngle:

    def test_center_direction(self):
        direction, angle = _compute_direction_and_angle(0.5)
        assert direction == "center"
        assert np.isclose(angle, 0.0)

    def test_left_direction(self):
        direction, angle = _compute_direction_and_angle(0.2)
        assert direction == "left"
        assert angle < 0

    def test_right_direction(self):
        direction, angle = _compute_direction_and_angle(0.8)
        assert direction == "right"
        assert angle > 0

    def test_angle_range(self):
        fov = getattr(config, "HORIZONTAL_FOV", 60.0)
        for x_norm in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _, angle = _compute_direction_and_angle(x_norm)
            assert -fov / 2 <= angle <= fov / 2

    def test_angle_sign_matches_direction(self):
        for x_norm, expected_dir in [(0.1, "left"), (0.9, "right"), (0.5, "center")]:
            direction, angle = _compute_direction_and_angle(x_norm)
            assert direction == expected_dir
            if direction == "left":
                assert angle < 0
            elif direction == "right":
                assert angle > 0

    def test_exact_boundary_values_do_not_crash(self):
        for norm_x in [0.0, config.DIR_LEFT, config.DIR_RIGHT, 1.0]:
            direction, angle = _compute_direction_and_angle(norm_x)
            assert direction in {"left", "center", "right"}
            assert isinstance(angle, float)


# ============================================================
# Tests: analyze_scene (Integration)
# ============================================================


class TestAnalyzeScene:

    def test_happy_path_single_object(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean", "median", {"mode": "top_k", "k": 10}],
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

    def test_required_keys_present(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean"],
        )
        required = {
            "class_id", "class_name", "confidence", "bbox",
            "mask", "mask_score", "depth_stats", "direction",
            "angle_deg", "centroid_x_norm",
        }
        for r in result:
            assert required.issubset(r.keys()), f"Missing: {required - r.keys()}"

    def test_no_detections(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = []
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean"],
        )
        assert result == []

    def test_multiple_objects(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        det2 = dict(sample_detection)
        det2["class_id"] = 1
        det2["class_name"] = "car"
        mock_detector.detect.return_value = [sample_detection, det2]
        mock_segmentor.segment_detections.return_value = [sample_detection, det2]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map

        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean"],
        )
        assert len(result) == 2
        assert result[0]["class_name"] == "person"
        assert result[1]["class_name"] == "car"

    def test_invalid_image_type(self, mock_detector, mock_segmentor, mock_depth_estimator):
        with pytest.raises(TypeError, match="np.ndarray"):
            analyze_scene(
                [[[1, 2, 3]]], mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_invalid_image_shape(self, mock_detector, mock_segmentor, mock_depth_estimator):
        with pytest.raises(ValueError, match="shape"):
            analyze_scene(
                np.zeros((640,), dtype=np.uint8),
                mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_invalid_channel_count(self, mock_detector, mock_segmentor, mock_depth_estimator):
        with pytest.raises(ValueError, match="shape"):
            analyze_scene(
                np.zeros((480, 640, 4), dtype=np.uint8),
                mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_zero_image_size(self, mock_detector, mock_segmentor, mock_depth_estimator):
        with pytest.raises(ValueError, match="dimensions"):
            analyze_scene(
                np.zeros((0, 640, 3), dtype=np.uint8),
                mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_detector_error_wrapped(
        self, sample_image, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.side_effect = RuntimeError("Detector failed")
        with pytest.raises(RuntimeError, match="YOLODetector"):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_segmentor_error_wrapped(
        self, sample_image, sample_detection, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.side_effect = RuntimeError("Segmentor failed")
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        with pytest.raises(RuntimeError, match="MobileSAMSegmentor"):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_depth_estimator_error_wrapped(
        self, sample_image, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.side_effect = RuntimeError("Depth failed")
        with pytest.raises(RuntimeError, match="DepthEstimator"):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_scale_depth_to_meters(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        mock_depth_estimator.scale_depth_to_meters.return_value = (
            sample_depth_map * 10.0, 10.0, 0.0
        )
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean"], scale_depth_to_meters=True,
        )
        assert len(result) == 1
        assert mock_depth_estimator.scale_depth_to_meters.called

    def test_empty_mask_fallback(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        det = {
            "class_id": 0, "class_name": "person", "confidence": 0.95,
            "bbox": np.array([100.0, 100.0, 300.0, 200.0], dtype=np.float32),
            "mask": np.zeros((480, 640), dtype=bool),
            "mask_score": 0.0,
        }
        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert not np.isnan(result[0]["depth_stats"]["mean"])

    def test_all_nan_depth_returns_nan_stats(
        self, sample_image, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = np.full((480, 640), np.nan, dtype=np.float32)
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            ["mean", "median"],
        )
        assert np.isnan(result[0]["depth_stats"]["mean"])
        assert np.isnan(result[0]["depth_stats"]["median"])

    def test_clipped_bbox_does_not_crash(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        det = {
            "class_id": 0, "class_name": "person", "confidence": 0.95,
            "bbox": np.array([600.0, 400.0, 700.0, 500.0], dtype=np.float32),
            "mask": np.zeros((480, 640), dtype=bool), "mask_score": 0.0,
        }
        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert len(result) == 1

    def test_mask_dtype_is_bool(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert result[0]["mask"].dtype == bool

    def test_bbox_dtype_is_float32(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert result[0]["bbox"].dtype == np.float32
        assert result[0]["bbox"].shape == (4,)

    def test_confidence_preserved(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert result[0]["confidence"] == pytest.approx(0.95)

    def test_centroid_x_norm_in_range(
        self, sample_image, sample_depth_map, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert 0.0 <= result[0]["centroid_x_norm"] <= 1.0

    def test_invalid_aggregation_modes_propagated(
        self, sample_image, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_depth_estimator.estimate_depth.return_value = np.zeros((480, 640), dtype=np.float32)
        with pytest.raises(ValueError):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
                ["not_a_real_mode"],
            )


# ============================================================
# Tests: Edge Cases
# ============================================================


class TestEdgeCases:

    def test_centroid_x_norm_bounds_far_left(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        left_mask = np.zeros((480, 640), dtype=bool)
        left_mask[100:200, 0:50] = True
        det = {
            "class_id": 0, "class_name": "person", "confidence": 0.95,
            "bbox": np.array([0.0, 100.0, 50.0, 200.0], dtype=np.float32),
            "mask": left_mask, "mask_score": 0.9,
        }
        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
        )
        assert 0.0 <= result[0]["centroid_x_norm"] <= 1.0

    def test_missing_detection_keys_raises(
        self, sample_image, sample_depth_map,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        incomplete_det = {"class_id": 0, "class_name": "person"}
        mock_detector.detect.return_value = [incomplete_det]
        mock_segmentor.segment_detections.return_value = [incomplete_det]
        mock_depth_estimator.estimate_depth.return_value = sample_depth_map
        with pytest.raises(RuntimeError, match="missing keys"):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )

    def test_invalid_depth_map_shape(
        self, sample_image, sample_detection,
        mock_detector, mock_segmentor, mock_depth_estimator
    ):
        mock_detector.detect.return_value = [sample_detection]
        mock_segmentor.segment_detections.return_value = [sample_detection]
        mock_depth_estimator.estimate_depth.return_value = np.ones(640, dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            analyze_scene(
                sample_image, mock_detector, mock_segmentor, mock_depth_estimator, ["mean"],
            )


# ============================================================
# Tests: scale_depth_to_meters auto-flip of closest_side
# ============================================================


class TestScaleDepthFlipsClosestSide:

    def test_metric_flip_preserves_closest_pixel_semantics(
        self, sample_image, mock_detector, mock_segmentor, mock_depth_estimator
    ):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        depth_map[:, :320] = 0.2
        depth_map[:, 320:] = 0.8

        metric_depth = np.zeros_like(depth_map)
        metric_depth[:, :320] = 8.0
        metric_depth[:, 320:] = 2.0

        det = {
            "class_id": 0, "class_name": "obj", "confidence": 0.9,
            "bbox": np.array([0.0, 0.0, 640.0, 480.0], dtype=np.float32),
            "mask": np.ones((480, 640), dtype=bool), "mask_score": 0.9,
        }
        mock_detector.detect.return_value = [det]
        mock_segmentor.segment_detections.return_value = [det]
        mock_depth_estimator.estimate_depth.return_value = depth_map
        mock_depth_estimator.scale_depth_to_meters.return_value = (metric_depth, 1.0, 0.0)

        result = analyze_scene(
            sample_image, mock_detector, mock_segmentor, mock_depth_estimator,
            [{"mode": "top_k", "k": 100}],
            scale_depth_to_meters=True,
        )

        top_k_val = result[0]["depth_stats"]["top_k_100"]
        assert top_k_val < 3.0, f"Expected ~2.0 m (close side), got {top_k_val}"
