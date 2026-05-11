import cv2
import torch
import numpy as np
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class MetricDepth:
    def __init__(self, encoder="vits", dataset="hypersim", max_depth=16, device=None):
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
        ckpt = f"metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    def get_depth_map(self, img_path, user_xfov=73, base_xfov=60.0):
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        depth = self.model.infer_image(raw_img)
        scale_factor = np.tan(np.radians(user_xfov / 2)) / np.tan(np.radians(base_xfov / 2))
        return depth / scale_factor

    def get_obstacle_mask(self, depth, threshold=1):
        return (depth > 0) & (depth < threshold)

    def analyze_obstacles(self, depth, yolo_result=None, threshold=1, min_area=2000):
        mask = self.get_obstacle_mask(depth, threshold).astype(np.uint8) * 255

        if yolo_result is not None and yolo_result.boxes is not None:
            for box in yolo_result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = 0
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        obstacles = []
        h, w = depth.shape
        label_map = labels.copy()
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= min_area:
                label_map[label_map == i] = 0
                continue

            cluster_depths = depth[labels == i]
            dist = np.median(cluster_depths)

            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = bh / bw if bw else 0

            if aspect_ratio > 2.5:
                obs_type = "slender obstacle"
            elif aspect_ratio < 0.4:
                obs_type = "flat obstacle"
            elif area > (h * w * 0.15):
                obs_type = "large obstacle"
            else:
                obs_type = "obstacle"

            cx, _ = centroids[i]
            direction = self._clock_direction(cx, w)

            obstacles.append(
                {
                    "name": obs_type,
                    "dist": round(float(dist), 2),
                    "dir": direction,
                    "label": i,
                }
            )

        return obstacles, label_map

    @staticmethod
    def _clock_direction(cx, width):
        bins = ["10 oclock", "11 oclock", "12 oclock", "1 oclock", "2 oclock"]
        idx = int(np.clip((cx / width) * len(bins), 0, len(bins) - 1))
        return bins[idx]

    def save_depth_map(self, depth, out_path):
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        cv2.imwrite(out_path, color)

    def save_masked_depth_map(self, depth, out_path, threshold=1):
        mask = self.get_obstacle_mask(depth, threshold)
        bg = np.full((*depth.shape, 3), 230, dtype=np.uint8)
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        bg[mask] = color[mask]
        cv2.imwrite(out_path, bg)

    def overlay_clusters(self, image, label_map, labels=None, alpha=0.6):
        if image is None:
            raise ValueError("Image is None")
        if labels is not None:
            keep = np.isin(label_map, labels)
            label_map = np.where(keep, label_map, 0)

        palette = np.array(
            [
                [0, 0, 0],
                [255, 255, 153],
                [204, 255, 204],
                [204, 229, 255],
                [255, 204, 229],
                [255, 230, 204],
            ],
            dtype=np.uint8,
        )
        max_label = int(label_map.max())
        colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
        for i in range(1, max_label + 1):
            colors[i] = palette[i % len(palette)]
        overlay = colors[label_map]

        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        out = image.copy()
        mask = label_map > 0
        out[mask] = blended[mask]
        return out

    def save_cluster_overlay(self, image, label_map, out_path, labels=None, alpha=0.6):
        out = self.overlay_clusters(image, label_map, labels=labels, alpha=alpha)
        cv2.imwrite(out_path, out)