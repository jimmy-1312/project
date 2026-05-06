"""
Custom YOLO dataset that:
  1) reads STANDARD 5-column ultralytics labels (cls cx cy w h), and
  2) looks up per-target distances from an out-of-band distances.json,
  3) optionally stacks a precomputed depth map as a 4th input channel.

Why distances are out of band:
  Ultralytics 8.4.x silently rejects any detection label row with more
  than 5 columns. Earlier versions of this code put distance as a 6th
  column; the result was training at "0 instances" with box_loss=0.

We deliberately do NOT subclass `ultralytics.data.dataset.YOLODataset` —
its augmentation pipeline is tightly coupled and changes between minor
versions. Instead we provide a thin PyTorch-Dataset-compatible class
`RGBDYoloDataset` for use in our own short training loops.

Public surface:
    parse_label_line(line) → (cls_id, (cx, cy, w, h))
    load_distances_for_split(root, split) → Dict[stem, List[float]]
    RGBDYoloDataset (PyTorch Dataset, for B variant)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Label parsing (no torch/ultralytics needed)
# ============================================================


def parse_label_line(line: str) -> Tuple[int, Tuple[float, float, float, float]]:
    """
    Parse one row of a standard 5-column YOLO label.

    Returns (class_id, (cx, cy, w, h)). Raises ValueError if fewer than 5
    columns are present. Extra columns are ignored (forward-compat).
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Label line has fewer than 5 columns: {line!r}")
    cls = int(float(parts[0]))
    cx, cy, w, h = (float(p) for p in parts[1:5])
    return cls, (cx, cy, w, h)


def load_distances_for_split(root: str, split: str) -> Dict[str, List[float]]:
    """
    Read `distances.json` at the dataset root and return the per-stem distances
    dict for a given split. Returns {} if the file is missing.

    distances.json is written by scripts/convert_to_yolo.py and has shape:
        {"train": {"<stem>": [d1, d2, ...]}, "val": {...}, "test": {...}}
    """
    p = Path(root) / "distances.json"
    if not p.is_file():
        logger.warning(f"  distances.json not found at {p}; distances will be NaN")
        return {}
    blob = json.loads(p.read_text())
    return blob.get(split, {}) or {}


# ============================================================
# Standalone RGB-D PyTorch Dataset (variant B)
# ============================================================


class RGBDYoloDataset:
    """
    Minimal PyTorch-Dataset-compatible loader for RGB-D YOLO experiments.

    NOT a drop-in replacement for ultralytics' YOLODataset. Used by our
    short, controlled RGB-D training loop where we don't need their full
    augmentation rig. Augmentations applied here:
        - random horizontal flip (depth flipped together)
        - resize to imgsz (square)
        - depth normalized to [0, 1] by /max_depth_m, clamped

    Layout expected (matches scripts/convert_to_yolo.py + scripts/precompute_depth.py):
        <root>/images/<split>/<stem>.{jpg,png}
        <root>/labels/<split>/<stem>.txt          (6-column)
        <root>/depth/<split>/<stem>.npy           (float32 meters)

    Yields:
        rgbd:      torch.float32 (4, imgsz, imgsz)        normalized to [0, 1]
        targets:   dict {
                        "cls":       (n,) int,
                        "bboxes":    (n, 4) float [cx, cy, w, h] (normalized),
                        "distances": (n,) float (meters or NaN),
                   }
        meta:      dict with image_path
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        root: str,
        split: str,
        imgsz: int = 640,
        max_depth_m: float = 10.0,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.imgsz = int(imgsz)
        self.max_depth_m = float(max_depth_m)
        self.augment = bool(augment)

        img_dir = self.root / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        self.images: List[Path] = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.images:
            raise RuntimeError(f"No images found under {img_dir}")

        # Out-of-band distances keyed by image stem
        self.distances_by_stem: Dict[str, List[float]] = load_distances_for_split(
            str(self.root), split
        )

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, img_path: Path) -> Path:
        return self.root / "labels" / self.split / (img_path.stem + ".txt")

    def _depth_path(self, img_path: Path) -> Path:
        return self.root / "depth" / self.split / (img_path.stem + ".npy")

    def _load_labels(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read 5-column labels and pair with distances from distances.json."""
        if not label_path.is_file():
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        cls_l: List[int] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        for raw in label_path.read_text().splitlines():
            if not raw.strip():
                continue
            c, xywh = parse_label_line(raw)
            cls_l.append(c)
            bboxes.append(xywh)

        stem = label_path.stem
        dists_list = self.distances_by_stem.get(stem, [])
        # Pad/trim to len(cls_l) so shapes always align (NaN for missing).
        if len(dists_list) != len(cls_l):
            logger.warning(
                f"  {stem}: {len(dists_list)} distances vs {len(cls_l)} labels — "
                "padding with NaN"
            )
            dists_arr = np.full(len(cls_l), float("nan"), dtype=np.float32)
            for i, d in enumerate(dists_list[: len(cls_l)]):
                dists_arr[i] = d
        else:
            dists_arr = np.asarray(dists_list, dtype=np.float32)

        return (
            np.asarray(cls_l, dtype=np.int64),
            np.asarray(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            dists_arr,
        )

    def __getitem__(self, idx: int):
        import torch  # lazy
        from PIL import Image

        img_path = self.images[idx]
        label_path = self._label_path(img_path)
        depth_path = self._depth_path(img_path)

        # Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = img.shape[:2]

        # Depth (or zeros if missing — log once)
        if depth_path.is_file():
            depth = np.load(depth_path).astype(np.float32)
            if depth.shape != (H0, W0):
                # Resize depth to image size with bilinear
                from PIL import Image as _Image
                depth = np.array(
                    _Image.fromarray(depth).resize((W0, H0), resample=_Image.BILINEAR),
                    dtype=np.float32,
                )
        else:
            logger.warning(f"  depth missing: {depth_path} (using zeros)")
            depth = np.zeros((H0, W0), dtype=np.float32)

        # Labels
        cls, bboxes, dists = self._load_labels(label_path)

        # Augmentation: horizontal flip
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            depth = depth[:, ::-1].copy()
            if bboxes.size > 0:
                bboxes = bboxes.copy()
                bboxes[:, 0] = 1.0 - bboxes[:, 0]  # cx flips

        # Resize image to (imgsz, imgsz)
        from PIL import Image as _Image
        img_pil = _Image.fromarray(img).resize((self.imgsz, self.imgsz), resample=_Image.BILINEAR)
        img = np.array(img_pil, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
        depth = np.array(
            _Image.fromarray(depth).resize((self.imgsz, self.imgsz), resample=_Image.BILINEAR),
            dtype=np.float32,
        )
        # Normalize depth to roughly [0, 1] like RGB
        depth = np.clip(depth / self.max_depth_m, 0.0, 1.0)

        # Stack channel-first (4, H, W)
        rgbd = np.concatenate([img.transpose(2, 0, 1), depth[np.newaxis, ...]], axis=0)

        return (
            torch.from_numpy(rgbd).float(),
            {
                "cls": torch.from_numpy(cls),
                "bboxes": torch.from_numpy(bboxes),
                "distances": torch.from_numpy(dists),
            },
            {"image_path": str(img_path)},
        )


def collate_rgbd(batch):
    """
    Custom collate for RGBDYoloDataset because target counts vary per image.

    Returns a tuple (images_tensor, targets_dict, metas_list) where
    `targets_dict` flattens per-image targets and adds a `batch_idx`
    tensor pointing each target back to its image — same convention
    Ultralytics uses internally.
    """
    import torch  # lazy

    images = torch.stack([b[0] for b in batch], dim=0)

    cls_list, bbox_list, dist_list, idx_list = [], [], [], []
    for i, (_img, t, _m) in enumerate(batch):
        if t["cls"].numel() == 0:
            continue
        cls_list.append(t["cls"])
        bbox_list.append(t["bboxes"])
        dist_list.append(t["distances"])
        idx_list.append(torch.full((t["cls"].numel(),), i, dtype=torch.long))

    targets = {
        "cls": torch.cat(cls_list) if cls_list else torch.zeros(0, dtype=torch.long),
        "bboxes": torch.cat(bbox_list) if bbox_list else torch.zeros(0, 4),
        "distances": torch.cat(dist_list) if dist_list else torch.zeros(0),
        "batch_idx": torch.cat(idx_list) if idx_list else torch.zeros(0, dtype=torch.long),
    }
    metas = [b[2] for b in batch]
    return images, targets, metas
