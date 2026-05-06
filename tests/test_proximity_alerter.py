"""
Unit tests for Proximity Alerter Module.

Run with:  python3 -m pytest tests/test_proximity_alerter.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.proximity_alerter import (
    _direction_word,
    _extract_distance_m,
    detect_by_proximity,
    format_nearest_alert,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def detection_close_left():
    return {
        "class_name": "stairs",
        "direction": "left",
        "angle_deg": -22.0,
        "depth_stats": {"top_k_100": 0.5, "mean": 0.6, "median": 0.55},
    }


@pytest.fixture
def detection_mid_center():
    return {
        "class_name": "person",
        "direction": "center",
        "angle_deg": 0.0,
        "depth_stats": {"top_k_100": 1.0, "mean": 1.1, "median": 1.0},
    }


@pytest.fixture
def detection_far_right():
    return {
        "class_name": "chair",
        "direction": "right",
        "angle_deg": 18.0,
        "depth_stats": {"top_k_100": 2.0, "mean": 2.1, "median": 2.05},
    }


@pytest.fixture
def detections_three(detection_close_left, detection_mid_center, detection_far_right):
    # Returned in non-sorted input order on purpose.
    return [detection_mid_center, detection_far_right, detection_close_left]


# ============================================================
# _extract_distance_m
# ============================================================


class TestExtractDistance:
    def test_uses_top_k_100_when_available(self):
        assert _extract_distance_m({"top_k_100": 1.5, "mean": 2.0, "median": 1.8}) == 1.5

    def test_falls_back_to_mean(self):
        assert _extract_distance_m({"mean": 2.0, "median": 1.8}) == 2.0

    def test_falls_back_to_median(self):
        assert _extract_distance_m({"median": 1.8}) == 1.8

    def test_none_for_empty_dict(self):
        assert _extract_distance_m({}) is None

    def test_none_for_none_input(self):
        assert _extract_distance_m(None) is None

    def test_none_for_non_dict_input(self):
        assert _extract_distance_m("not a dict") is None
        assert _extract_distance_m(42) is None
        assert _extract_distance_m([1, 2, 3]) is None

    def test_skips_nan(self):
        assert _extract_distance_m({"top_k_100": float("nan"), "mean": 2.0}) == 2.0

    def test_skips_inf(self):
        assert _extract_distance_m({"top_k_100": float("inf"), "mean": 2.0}) == 2.0

    def test_skips_negative(self):
        assert _extract_distance_m({"top_k_100": -1.0, "mean": 2.0}) == 2.0

    def test_skips_zero(self):
        assert _extract_distance_m({"top_k_100": 0.0, "mean": 2.0}) == 2.0

    def test_returns_none_when_all_invalid(self):
        assert _extract_distance_m({"top_k_100": -1, "mean": 0, "median": float("nan")}) is None

    def test_handles_string_numeric_values(self):
        # Floats stored as strings should be coerced.
        assert _extract_distance_m({"top_k_100": "1.5"}) == 1.5

    def test_handles_non_numeric_strings(self):
        assert _extract_distance_m({"top_k_100": "abc", "mean": 2.0}) == 2.0


# ============================================================
# _direction_word
# ============================================================


class TestDirectionWord:
    def test_left(self):
        assert _direction_word("left") == "left"

    def test_center_to_ahead(self):
        assert _direction_word("center") == "ahead"

    def test_right(self):
        assert _direction_word("right") == "right"

    def test_unknown_returns_empty(self):
        assert _direction_word("unknown") == ""

    def test_none_returns_empty(self):
        assert _direction_word(None) == ""

    def test_unrecognized_returns_empty(self):
        assert _direction_word("backwards") == ""


# ============================================================
# format_nearest_alert
# ============================================================


class TestFormatAlert:
    def test_full_format(self):
        det = {"class_name": "stairs", "distance_m": 0.5, "direction": "left"}
        assert format_nearest_alert(det) == "left 0.5m stairs"

    def test_center_becomes_ahead(self):
        det = {"class_name": "person", "distance_m": 1.0, "direction": "center"}
        assert format_nearest_alert(det) == "ahead 1.0m person"

    def test_right(self):
        det = {"class_name": "chair", "distance_m": 2.0, "direction": "right"}
        assert format_nearest_alert(det) == "right 2.0m chair"

    def test_no_direction(self):
        det = {"class_name": "window", "distance_m": 3.0, "direction": "unknown"}
        assert format_nearest_alert(det) == "3.0m window"

    def test_no_distance(self):
        det = {"class_name": "wall", "direction": "left"}
        assert format_nearest_alert(det) == "left wall"

    def test_only_class_name(self):
        det = {"class_name": "tree"}
        assert format_nearest_alert(det) == "tree"

    def test_missing_class_uses_object_default(self):
        det = {"distance_m": 1.0, "direction": "left"}
        assert format_nearest_alert(det) == "left 1.0m object"

    def test_distance_rounded_to_one_decimal(self):
        det = {"class_name": "x", "distance_m": 1.234567, "direction": "left"}
        assert format_nearest_alert(det) == "left 1.2m x"

    def test_nan_distance_dropped(self):
        det = {"class_name": "x", "distance_m": float("nan"), "direction": "left"}
        assert format_nearest_alert(det) == "left x"

    def test_inf_distance_dropped(self):
        det = {"class_name": "x", "distance_m": float("inf"), "direction": "left"}
        assert format_nearest_alert(det) == "left x"

    def test_negative_distance_dropped(self):
        det = {"class_name": "x", "distance_m": -1.0, "direction": "left"}
        assert format_nearest_alert(det) == "left x"

    def test_string_distance_coerced(self):
        det = {"class_name": "x", "distance_m": "1.5", "direction": "left"}
        assert format_nearest_alert(det) == "left 1.5m x"


# ============================================================
# detect_by_proximity
# ============================================================


class TestDetectByProximity:
    def test_basic_three(self, detections_three):
        result = detect_by_proximity(detections_three, top_k=5)
        assert [r["class_name"] for r in result] == ["stairs", "person", "chair"]
        assert [r["distance_m"] for r in result] == [0.5, 1.0, 2.0]

    def test_alerts_match_expected(self, detections_three):
        result = detect_by_proximity(detections_three)
        assert result[0]["alert"] == "left 0.5m stairs"
        assert result[1]["alert"] == "ahead 1.0m person"
        assert result[2]["alert"] == "right 2.0m chair"

    def test_default_top_k_is_5(self):
        # With 7 valid detections, default top_k=5 should return 5
        dets = [
            {
                "class_name": f"obj{i}",
                "direction": "center",
                "angle_deg": 0.0,
                "depth_stats": {"top_k_100": float(i + 1)},
            }
            for i in range(7)
        ]
        result = detect_by_proximity(dets)
        assert len(result) == 5
        assert [r["distance_m"] for r in result] == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_top_k_smaller_than_count(self, detections_three):
        result = detect_by_proximity(detections_three, top_k=2)
        assert len(result) == 2
        assert [r["class_name"] for r in result] == ["stairs", "person"]

    def test_top_k_larger_than_count(self, detections_three):
        result = detect_by_proximity(detections_three, top_k=10)
        assert len(result) == 3

    def test_top_k_zero_means_no_limit(self, detections_three):
        result = detect_by_proximity(detections_three, top_k=0)
        assert len(result) == 3

    def test_top_k_negative_means_no_limit(self, detections_three):
        result = detect_by_proximity(detections_three, top_k=-1)
        assert len(result) == 3

    def test_empty_input(self):
        assert detect_by_proximity([]) == []

    def test_filters_detection_without_depth(self):
        dets = [
            {"class_name": "no_depth", "direction": "left", "angle_deg": 0.0},
            {
                "class_name": "with_depth",
                "direction": "center",
                "angle_deg": 0.0,
                "depth_stats": {"top_k_100": 1.0},
            },
        ]
        result = detect_by_proximity(dets)
        assert len(result) == 1
        assert result[0]["class_name"] == "with_depth"

    def test_filters_detection_with_invalid_depth(self):
        dets = [
            {
                "class_name": "nan",
                "depth_stats": {"top_k_100": float("nan")},
                "direction": "left",
                "angle_deg": 0.0,
            },
            {
                "class_name": "neg",
                "depth_stats": {"top_k_100": -1.0},
                "direction": "left",
                "angle_deg": 0.0,
            },
            {
                "class_name": "ok",
                "depth_stats": {"top_k_100": 1.0},
                "direction": "center",
                "angle_deg": 0.0,
            },
        ]
        result = detect_by_proximity(dets)
        assert len(result) == 1
        assert result[0]["class_name"] == "ok"

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError):
            detect_by_proximity("not a list")
        with pytest.raises(TypeError):
            detect_by_proximity(None)
        with pytest.raises(TypeError):
            detect_by_proximity(42)

    def test_returns_required_keys(self, detection_close_left):
        result = detect_by_proximity([detection_close_left])
        assert len(result) == 1
        for key in ("class_name", "direction", "distance_m", "angle_deg", "alert"):
            assert key in result[0]

    def test_distance_is_float(self, detection_close_left):
        result = detect_by_proximity([detection_close_left])
        assert isinstance(result[0]["distance_m"], float)
        assert isinstance(result[0]["angle_deg"], float)

    def test_handles_missing_optional_fields(self):
        # Only class_name + depth_stats — direction/angle_deg should default.
        det = {
            "class_name": "x",
            "depth_stats": {"top_k_100": 1.0},
        }
        result = detect_by_proximity([det])
        assert len(result) == 1
        assert result[0]["class_name"] == "x"
        assert result[0]["distance_m"] == 1.0
        assert result[0]["direction"] == "unknown"
        assert result[0]["angle_deg"] == 0.0
        # Direction "unknown" → no direction word → just distance + class.
        assert result[0]["alert"] == "1.0m x"

    def test_ordering_stable_at_equal_distance(self):
        # Two objects at the exact same distance: input order should be preserved.
        dets = [
            {"class_name": "a", "direction": "left", "angle_deg": 0.0,
             "depth_stats": {"top_k_100": 1.0}},
            {"class_name": "b", "direction": "right", "angle_deg": 0.0,
             "depth_stats": {"top_k_100": 1.0}},
        ]
        result = detect_by_proximity(dets)
        assert [r["class_name"] for r in result] == ["a", "b"]

    def test_extreme_distances(self):
        dets = [
            {"class_name": "very_close", "direction": "center", "angle_deg": 0.0,
             "depth_stats": {"top_k_100": 0.01}},
            {"class_name": "very_far", "direction": "center", "angle_deg": 0.0,
             "depth_stats": {"top_k_100": 100.0}},
        ]
        result = detect_by_proximity(dets)
        assert result[0]["class_name"] == "very_close"
        assert math.isclose(result[0]["distance_m"], 0.01)
        assert result[1]["class_name"] == "very_far"

    def test_does_not_mutate_input(self, detections_three):
        before = [dict(d) for d in detections_three]
        _ = detect_by_proximity(detections_three)
        for orig, after in zip(before, detections_three):
            assert orig.keys() == after.keys()
            assert orig["class_name"] == after["class_name"]


# ============================================================
# Performance smoke test
# ============================================================


class TestPerformance:
    def test_handles_large_detection_set_quickly(self):
        # 1000 detections — should be effectively instant.
        rng = np.random.default_rng(0)
        dists = rng.uniform(0.1, 20.0, size=1000)
        dets = [
            {
                "class_name": f"obj_{i}",
                "direction": ["left", "center", "right"][i % 3],
                "angle_deg": float(i % 30),
                "depth_stats": {"top_k_100": float(d)},
            }
            for i, d in enumerate(dists)
        ]
        result = detect_by_proximity(dets, top_k=5)
        assert len(result) == 5
        # Top-5 must be the 5 smallest distances overall.
        smallest_5 = sorted(dists)[:5]
        for got, exp in zip([r["distance_m"] for r in result], smallest_5):
            assert math.isclose(got, exp)
