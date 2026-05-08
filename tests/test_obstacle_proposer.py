"""
Unit tests for src/obstacle_proposer.py (depth-only fallback proposer).

Run:  python3 -m pytest tests/test_obstacle_proposer.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.obstacle_proposer import propose_obstacles, merge_with_detections


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_depth():
    """100x100 depth map, two close blobs and a far background."""
    H, W = 100, 100
    depth = np.full((H, W), 5.0, dtype=np.float32)  # background 5 m
    depth[10:40, 10:40] = 0.5   # close blob top-left, 0.5 m, area 900
    depth[60:90, 60:90] = 1.0   # close blob bottom-right, 1.0 m, area 900
    return depth


# ============================================================
# Basic behavior
# ============================================================


class TestBasic:
    def test_finds_close_blobs(self, simple_depth):
        proposals = propose_obstacles(simple_depth)
        assert len(proposals) == 2

    def test_sorted_closest_first(self, simple_depth):
        proposals = propose_obstacles(simple_depth)
        d0 = proposals[0]["depth_stats"]["top_k_100"]
        d1 = proposals[1]["depth_stats"]["top_k_100"]
        assert d0 <= d1
        assert d0 == pytest.approx(0.5, abs=0.05)
        assert d1 == pytest.approx(1.0, abs=0.05)

    def test_class_name_is_obstacle(self, simple_depth):
        for p in propose_obstacles(simple_depth):
            assert p["class_name"] == "obstacle"
            assert p["class_id"] == -1
            assert p["source"] == "obstacle_proposer"

    def test_returns_detection_shape(self, simple_depth):
        p = propose_obstacles(simple_depth)[0]
        for k in ("class_id", "class_name", "confidence", "bbox", "mask",
                 "mask_score", "depth_stats", "direction", "angle_deg",
                 "centroid_x_norm"):
            assert k in p
        assert p["bbox"].shape == (4,)
        assert p["mask"].dtype == bool

    def test_directions_left_and_right(self, simple_depth):
        proposals = propose_obstacles(simple_depth)
        # blob 1 (top-left) is on the left of image, blob 2 (bottom-right) on the right
        directions = sorted(p["direction"] for p in proposals)
        assert "left" in directions and "right" in directions

    def test_confidence_in_unit_range(self, simple_depth):
        for p in propose_obstacles(simple_depth):
            assert 0.0 <= p["confidence"] <= 1.0


# ============================================================
# Threshold + filtering
# ============================================================


class TestFiltering:
    def test_distance_threshold_excludes_far(self, simple_depth):
        # Threshold below the 1m blob → only the 0.5m blob survives
        proposals = propose_obstacles(simple_depth, distance_threshold_m=0.8)
        assert len(proposals) == 1
        assert proposals[0]["depth_stats"]["top_k_100"] == pytest.approx(0.5, abs=0.05)

    def test_min_area_drops_noise(self, simple_depth):
        # Add a 2x2 close noise pixel
        d = simple_depth.copy()
        d[0:2, 0:2] = 0.3
        # default min_area_frac=0.005 → 0.005 * 100*100 = 50 px → 2x2=4 px filtered out
        proposals = propose_obstacles(d)
        assert len(proposals) == 2  # not 3

    def test_min_area_zero_keeps_everything(self, simple_depth):
        d = simple_depth.copy()
        d[0:2, 0:2] = 0.3
        proposals = propose_obstacles(d, min_area_frac=0.0)
        assert len(proposals) == 3

    def test_max_proposals_cap(self):
        # 5 distinct close blobs → cap to 3
        depth = np.full((50, 100), 5.0, dtype=np.float32)
        for i, x in enumerate([5, 25, 45, 65, 85]):
            depth[10:30, x:x + 5] = 0.5 + i * 0.1   # increasing distance
        proposals = propose_obstacles(depth, max_proposals=3, min_area_frac=0.0)
        assert len(proposals) == 3
        # Closest first
        ds = [p["depth_stats"]["top_k_100"] for p in proposals]
        assert ds == sorted(ds)


# ============================================================
# Claimed masks
# ============================================================


class TestClaimedMasks:
    def test_claimed_region_is_excluded(self, simple_depth):
        H, W = simple_depth.shape
        claimed = np.zeros((H, W), dtype=bool)
        claimed[10:40, 10:40] = True   # exactly cover the 0.5m blob
        proposals = propose_obstacles(simple_depth, claimed_masks=[claimed])
        # Only the 1.0 m blob should remain
        assert len(proposals) == 1
        assert proposals[0]["depth_stats"]["top_k_100"] == pytest.approx(1.0, abs=0.05)

    def test_partial_claimed_still_proposes(self, simple_depth):
        # Claim only HALF of the close blob → component shrinks but remains
        H, W = simple_depth.shape
        claimed = np.zeros((H, W), dtype=bool)
        claimed[10:25, 10:40] = True
        proposals = propose_obstacles(simple_depth, claimed_masks=[claimed])
        assert len(proposals) == 2

    def test_empty_claimed_list(self, simple_depth):
        # Empty claimed list should be equivalent to no claimed masks at all.
        a = propose_obstacles(simple_depth, claimed_masks=[])
        b = propose_obstacles(simple_depth)
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert pa["depth_stats"]["top_k_100"] == pb["depth_stats"]["top_k_100"]
            assert pa["direction"] == pb["direction"]

    def test_none_claimed_handled(self, simple_depth):
        assert len(propose_obstacles(simple_depth, claimed_masks=None)) == 2

    def test_mismatched_mask_shape_skipped(self, simple_depth):
        # Wrong shape → warned + ignored (not crashed)
        wrong = np.zeros((10, 10), dtype=bool)
        proposals = propose_obstacles(simple_depth, claimed_masks=[wrong])
        # Both blobs still found, since wrong mask was ignored
        assert len(proposals) == 2


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_empty_depth_map(self):
        assert propose_obstacles(np.zeros((0, 0), dtype=np.float32)) == []

    def test_all_far_returns_nothing(self):
        far = np.full((50, 50), 10.0, dtype=np.float32)
        assert propose_obstacles(far) == []

    def test_nan_pixels_are_excluded(self):
        d = np.full((50, 50), 5.0, dtype=np.float32)
        d[10:30, 10:30] = np.nan       # invalid signal in middle
        d[35:45, 35:45] = 0.5          # close blob
        proposals = propose_obstacles(d, min_area_frac=0.0)
        assert len(proposals) == 1

    def test_negative_depth_excluded(self):
        d = np.full((50, 50), 5.0, dtype=np.float32)
        d[10:30, 10:30] = -1.0
        proposals = propose_obstacles(d, min_area_frac=0.0)
        assert proposals == []

    def test_3d_depth_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            propose_obstacles(np.zeros((10, 10, 3), dtype=np.float32))


# ============================================================
# merge helper
# ============================================================


class TestMerge:
    def test_concatenates_in_order(self):
        a = [{"class_name": "chair"}]
        b = [{"class_name": "obstacle"}]
        merged = merge_with_detections(a, b)
        assert len(merged) == 2
        assert merged[0]["class_name"] == "chair"
        assert merged[1]["class_name"] == "obstacle"

    def test_does_not_mutate_inputs(self):
        a = [{"class_name": "chair"}]
        b = [{"class_name": "obstacle"}]
        before_a = list(a)
        merged = merge_with_detections(a, b)
        merged.append({"new": True})
        assert a == before_a
