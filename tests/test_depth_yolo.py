"""
Unit tests for src/depth_yolo/* — distance weight, label parsing, dataset.

Loss class and model surgery are imported through `__init__.py` but their
actual ultralytics-dependent paths are NOT exercised here (no ultralytics
in the test sandbox). They have their own integration tests on the user's
training machine.

Run:  python3 -m pytest tests/test_depth_yolo.py -v
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from src.depth_yolo.dataset import parse_label_line, load_distances_for_split
from src.depth_yolo.loss import distance_weight


# ============================================================
# distance_weight — pure scalar function
# ============================================================


class TestDistanceWeight:
    def test_zero_returns_one_default(self):
        # Distance 0 is treated as "no info" → weight 1.0 (fail-safe)
        assert distance_weight(0.0) == 1.0

    def test_negative_returns_one(self):
        assert distance_weight(-1.0) == 1.0

    def test_nan_returns_one(self):
        assert distance_weight(float("nan")) == 1.0

    def test_inf_returns_one(self):
        assert distance_weight(float("inf")) == 1.0

    def test_decreases_with_distance(self):
        ws = [distance_weight(d) for d in (0.5, 1.0, 2.0, 5.0)]
        assert ws == sorted(ws, reverse=True)
        assert all(0.0 < w < 1.0 for w in ws)

    def test_known_values(self):
        # tau=2 default
        assert distance_weight(2.0) == pytest.approx(math.exp(-1.0), rel=1e-6)
        assert distance_weight(0.5) == pytest.approx(math.exp(-0.25), rel=1e-6)

    def test_tau_infinity_collapses_to_one(self):
        # For very large tau, weight → 1.0
        assert distance_weight(2.0, tau=1e9) == pytest.approx(1.0, abs=1e-6)

    def test_tau_zero_is_clipped(self):
        # tau=0 would divide by zero; we clip internally
        w = distance_weight(2.0, tau=0.0)
        assert math.isfinite(w)
        assert w >= 0.0


# ============================================================
# Label parsing
# ============================================================


class TestParseLabelLine:
    def test_five_column(self):
        cls, xywh = parse_label_line("3 0.5 0.5 0.4 0.4")
        assert cls == 3
        assert xywh == (0.5, 0.5, 0.4, 0.4)

    def test_extra_columns_ignored(self):
        # Forward-compat: extra columns past 5 are silently ignored.
        cls, xywh = parse_label_line("0 0.1 0.1 0.2 0.2 ignored 99")
        assert cls == 0
        assert xywh == (0.1, 0.1, 0.2, 0.2)

    def test_extra_whitespace(self):
        cls, xywh = parse_label_line("   2  0.5  0.5  0.5  0.5   ")
        assert cls == 2
        assert xywh == (0.5, 0.5, 0.5, 0.5)

    def test_too_few_columns_raises(self):
        with pytest.raises(ValueError, match="fewer than 5"):
            parse_label_line("0 0.1 0.1 0.2")


class TestLoadDistancesForSplit:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        out = load_distances_for_split(str(tmp_path), "train")
        assert out == {}

    def test_reads_split(self, tmp_path):
        (tmp_path / "distances.json").write_text(
            '{"train": {"img_a": [1.5, 2.5], "img_b": [0.7]}, "val": {}}'
        )
        out = load_distances_for_split(str(tmp_path), "train")
        assert out == {"img_a": [1.5, 2.5], "img_b": [0.7]}

    def test_unknown_split_returns_empty(self, tmp_path):
        (tmp_path / "distances.json").write_text('{"train": {"a": [1.0]}}')
        assert load_distances_for_split(str(tmp_path), "test") == {}

    def test_null_split_returns_empty(self, tmp_path):
        (tmp_path / "distances.json").write_text('{"train": null}')
        assert load_distances_for_split(str(tmp_path), "train") == {}
