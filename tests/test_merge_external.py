"""
Unit + integration tests for scripts/merge_external_dataset.py.

Covers:
  - class mapping loader: normal, null mappings, missing dataset, invalid target
  - source data.yaml parsing: list / dict `names` formats, missing file
  - label remapping: drop reasons logged correctly
  - end-to-end synthetic merge into a tmp HK-style dataset
  - distances.json gains null entries for public rows

Run:  python3 -m pytest tests/test_merge_external.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import merge_external_dataset as merge_mod  # noqa: E402


# ============================================================
# load_mapping
# ============================================================


def _write_mapping(path: Path, body: str) -> None:
    path.write_text(body)


class TestLoadMapping:
    def test_basic(self, tmp_path):
        _write_mapping(tmp_path / "m.yaml", """
datasets:
  ds1:
    sourceURL: "https://example.com"
    license: "CC-BY 4.0"
    notes: "demo"
    classes:
      sofa: couch
      desk: table
      lamp: null
""")
        m = merge_mod.load_mapping(str(tmp_path / "m.yaml"), "ds1")
        assert m.name == "ds1"
        assert m.license == "CC-BY 4.0"
        assert m.classes == {"sofa": "couch", "desk": "table", "lamp": None}

    def test_unknown_dataset_raises(self, tmp_path):
        _write_mapping(tmp_path / "m.yaml", "datasets: {ds1: {classes: {}}}\n")
        with pytest.raises(KeyError, match="ds_other"):
            merge_mod.load_mapping(str(tmp_path / "m.yaml"), "ds_other")

    def test_invalid_target_class_raises(self, tmp_path):
        _write_mapping(tmp_path / "m.yaml", """
datasets:
  ds1:
    classes:
      sofa: not_in_taxonomy
""")
        with pytest.raises(ValueError, match="not_in_taxonomy"):
            merge_mod.load_mapping(str(tmp_path / "m.yaml"), "ds1")

    def test_empty_mapping_block_ok(self, tmp_path):
        _write_mapping(tmp_path / "m.yaml", "datasets: {ds1: {}}\n")
        m = merge_mod.load_mapping(str(tmp_path / "m.yaml"), "ds1")
        assert m.classes == {}


# ============================================================
# load_source_class_names
# ============================================================


class TestLoadSourceClassNames:
    def test_list_format(self, tmp_path):
        (tmp_path / "data.yaml").write_text("names: ['chair', 'sofa', 'desk']\n")
        names = merge_mod.load_source_class_names(tmp_path)
        assert names == ["chair", "sofa", "desk"]

    def test_dict_format(self, tmp_path):
        (tmp_path / "data.yaml").write_text(
            "names:\n  0: chair\n  1: sofa\n  2: desk\n"
        )
        names = merge_mod.load_source_class_names(tmp_path)
        assert names == ["chair", "sofa", "desk"]

    def test_yml_extension_fallback(self, tmp_path):
        (tmp_path / "data.yml").write_text("names: ['chair']\n")
        names = merge_mod.load_source_class_names(tmp_path)
        assert names == ["chair"]

    def test_missing_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_mod.load_source_class_names(tmp_path)

    def test_missing_names_field_raises(self, tmp_path):
        (tmp_path / "data.yaml").write_text("nc: 2\n")
        with pytest.raises(ValueError, match="no 'names'"):
            merge_mod.load_source_class_names(tmp_path)


# ============================================================
# remap_label_file
# ============================================================


class TestRemapLabelFile:
    def _make_label(self, tmp_path, body: str) -> Path:
        p = tmp_path / "lbl.txt"
        p.write_text(body)
        return p

    def test_basic_remap(self, tmp_path):
        # Source has classes [chair, sofa]; we map chair→chair, sofa→couch
        lbl = self._make_label(tmp_path, "0 0.5 0.5 0.4 0.4\n1 0.6 0.6 0.3 0.3\n")
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="",
            classes={"chair": "chair", "sofa": "couch"},
        )
        drops = {}
        rows = merge_mod.remap_label_file(lbl, ["chair", "sofa"], m, drops)
        assert len(rows) == 2
        assert rows[0].cls_id == merge_mod.CLASS_TO_ID["chair"]
        assert rows[1].cls_id == merge_mod.CLASS_TO_ID["couch"]
        assert drops == {}

    def test_null_mapping_drops(self, tmp_path):
        lbl = self._make_label(tmp_path, "0 0.5 0.5 0.4 0.4\n1 0.6 0.6 0.3 0.3\n")
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="",
            classes={"chair": "chair", "lamp": None},
        )
        drops = {}
        rows = merge_mod.remap_label_file(lbl, ["chair", "lamp"], m, drops)
        assert len(rows) == 1
        assert "dropped:lamp" in drops

    def test_unmapped_logged(self, tmp_path):
        lbl = self._make_label(tmp_path, "0 0.5 0.5 0.4 0.4\n")
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="", classes={},
        )
        drops = {}
        rows = merge_mod.remap_label_file(lbl, ["chair"], m, drops)
        assert rows == []
        assert "unmapped:chair" in drops

    def test_degenerate_bbox_dropped(self, tmp_path):
        lbl = self._make_label(tmp_path, "0 0.5 0.5 0.0 0.4\n0 0.5 0.5 0.4 0.0\n")
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="",
            classes={"chair": "chair"},
        )
        drops = {}
        rows = merge_mod.remap_label_file(lbl, ["chair"], m, drops)
        assert rows == []
        assert drops.get("degenerate_bbox", 0) == 2

    def test_class_id_out_of_range(self, tmp_path):
        lbl = self._make_label(tmp_path, "5 0.5 0.5 0.4 0.4\n")  # only 1 class in src
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="",
            classes={"chair": "chair"},
        )
        drops = {}
        rows = merge_mod.remap_label_file(lbl, ["chair"], m, drops)
        assert rows == []
        assert drops.get("class_id_out_of_range", 0) == 1

    def test_missing_label_file(self):
        m = merge_mod.DatasetMapping(
            name="ds", source_url="", license="", notes="", classes={},
        )
        rows = merge_mod.remap_label_file(None, [], m, {})
        assert rows == []


# ============================================================
# update_distances_json
# ============================================================


class TestUpdateDistancesJson:
    def test_creates_file_if_missing(self, tmp_path):
        merge_mod.update_distances_json(tmp_path, {"foo": [None, None]})
        blob = json.loads((tmp_path / "distances.json").read_text())
        assert blob["train"]["foo"] == [None, None]
        assert blob["val"] == {}

    def test_merges_with_existing(self, tmp_path):
        (tmp_path / "distances.json").write_text(json.dumps({
            "train": {"hk_a": [1.0, 2.0]},
            "val":   {"hk_b": [0.5]},
            "test":  {},
        }))
        merge_mod.update_distances_json(tmp_path, {"public_x": [None]})
        blob = json.loads((tmp_path / "distances.json").read_text())
        assert blob["train"]["hk_a"] == [1.0, 2.0]
        assert blob["train"]["public_x"] == [None]
        assert blob["val"]["hk_b"] == [0.5]   # untouched

    def test_clean_existing_prefix_strips(self, tmp_path):
        (tmp_path / "distances.json").write_text(json.dumps({
            "train": {
                "hk_a":      [1.0],
                "ds__old":   [None],
                "ds__older": [None],
            },
        }))
        merge_mod.update_distances_json(
            tmp_path, {"ds__new": [None]}, keep_existing_dataset_prefix="ds"
        )
        blob = json.loads((tmp_path / "distances.json").read_text())
        assert "hk_a" in blob["train"]
        assert "ds__old" not in blob["train"]
        assert "ds__older" not in blob["train"]
        assert blob["train"]["ds__new"] == [None]


# ============================================================
# End-to-end smoke test
# ============================================================


def _build_synthetic_source(root: Path) -> None:
    """Create a minimal Roboflow-shaped source dataset with 2 train images."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['chair', 'sofa', 'lamp']\n")

    # Image 1: chair + sofa (mapped) + lamp (null) → keeps 2 rows
    (root / "train" / "images" / "img001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "train" / "labels" / "img001.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n1 0.7 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    )
    # Image 2: only lamp → all rows drop, image skipped
    (root / "train" / "images" / "img002.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "train" / "labels" / "img002.txt").write_text("2 0.5 0.5 0.3 0.3\n")


class TestEndToEnd:
    def test_synthetic_merge(self, tmp_path, monkeypatch):
        # Set up synthetic external source
        src = tmp_path / "external"
        _build_synthetic_source(src)

        # Mapping yaml
        mapping_path = tmp_path / "class_mapping.yaml"
        mapping_path.write_text("""
datasets:
  testset:
    sourceURL: "https://example.com/testset"
    license: "CC-BY 4.0"
    notes: "synthetic"
    classes:
      chair: chair
      sofa:  couch
      lamp:  null
""")

        # Empty HK output dir (with prior distances.json containing HK entries)
        out = tmp_path / "hk"
        out.mkdir()
        (out / "distances.json").write_text(json.dumps({
            "train": {"hk_existing": [1.5]},
            "val":   {"hk_v": [0.5]},
            "test":  {},
        }))

        # Build argv and call main()
        argv = [
            "merge_external_dataset.py",
            "--source", str(src),
            "--dataset-name", "testset",
            "--mapping", str(mapping_path),
            "--out-dir", str(out),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        merge_mod.main()

        # Assertions
        # img001 should have been copied with prefix
        copied = list((out / "images" / "train").iterdir())
        assert len(copied) == 1
        assert copied[0].name == "testset__img001.jpg"

        # Label file should have 2 rows (chair, sofa→couch); lamp dropped
        lbl = (out / "labels" / "train" / "testset__img001.txt").read_text().splitlines()
        assert len(lbl) == 2
        chair_id = merge_mod.CLASS_TO_ID["chair"]
        couch_id = merge_mod.CLASS_TO_ID["couch"]
        cls_ids = {int(line.split()[0]) for line in lbl}
        assert cls_ids == {chair_id, couch_id}

        # distances.json: HK entries preserved + new public entry with nulls
        dj = json.loads((out / "distances.json").read_text())
        assert dj["train"]["hk_existing"] == [1.5]
        assert dj["val"]["hk_v"] == [0.5]
        assert dj["train"]["testset__img001"] == [None, None]

        # licenses.txt appended
        licenses = (out / "licenses.txt").read_text()
        assert "testset" in licenses
        assert "CC-BY 4.0" in licenses

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        src = tmp_path / "external"
        _build_synthetic_source(src)
        mapping_path = tmp_path / "m.yaml"
        mapping_path.write_text("""
datasets:
  testset:
    classes:
      chair: chair
      sofa:  couch
      lamp:  null
""")
        out = tmp_path / "hk"
        out.mkdir()

        argv = [
            "merge_external_dataset.py",
            "--source", str(src),
            "--dataset-name", "testset",
            "--mapping", str(mapping_path),
            "--out-dir", str(out),
            "--dry-run",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        merge_mod.main()

        # Nothing should have been written
        assert not (out / "images").exists() or list((out / "images").iterdir()) == []
        assert not (out / "distances.json").exists()
        assert not (out / "licenses.txt").exists()
