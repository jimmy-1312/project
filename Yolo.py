from ultralytics import YOLO
import cv2
import numpy as np
from Depth import MetricDepth


def build_models(yolo_weights="yolo11s.pt", yolo_seg_weights="yolo11s-seg.pt", use_seg=False):
    yolo_model = YOLO(yolo_weights)
    seg_model = YOLO(yolo_seg_weights) if use_seg else None
    return yolo_model, seg_model, MetricDepth()


def _clock_direction(cx, width):
    bins = ["10 oclock", "11 oclock", "12 oclock", "1 oclock", "2 oclock"]
    idx = int(np.clip((cx / width) * len(bins), 0, len(bins) - 1))
    return bins[idx]


def analyze_detectable_objects(yolo_result, depth_map, masks=None, top_ratio=0.2):
    detected_list = []
    h, w = depth_map.shape
    names = yolo_result.names

    if yolo_result.boxes is None:
        return detected_list

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    classes = yolo_result.boxes.cls.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()

    for i, (b, cls, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = b.astype(int)
        mask_crop = None
        if masks is not None and i < len(masks):
            mask_crop = masks[i]
            if mask_crop.shape != depth_map.shape:
                mask_crop = cv2.resize(mask_crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_crop = mask_crop.astype(bool)

        if mask_crop is not None:
            crop = depth_map[mask_crop]
        else:
            crop = depth_map[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].flatten()

        if crop.size == 0:
            continue

        sorted_depths = np.sort(crop)
        top_n = max(1, int(len(sorted_depths) * top_ratio))
        dist = sorted_depths[:top_n].mean()

        cx = (x1 + x2) / 2
        item = {
            "name": names[int(cls)],
            "dist": round(float(dist), 2),
            "dir": _clock_direction(cx, w),
            "conf": round(float(conf), 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
        if masks is not None:
            item["mask_index"] = i
        detected_list.append(item)

    return detected_list


def analyze_image(
    img_path,
    yolo_model,
    depth_model,
    seg_model=None,
    top_ratio=0.2,
    mask_top_ratio=0.1,
):
    results = yolo_model(img_path)
    depth = depth_model.get_depth_map(img_path)

    masks = None
    ratio = top_ratio
    det_result = results[0]
    if seg_model is not None:
        seg_results = seg_model(img_path)
        if seg_results:
            det_result = seg_results[0]
            if det_result.masks is not None:
                masks = det_result.masks.data.cpu().numpy().astype(bool)
                ratio = mask_top_ratio

    non_detectable, label_map = depth_model.analyze_obstacles(depth, det_result)
    detectable = analyze_detectable_objects(det_result, depth, masks=masks, top_ratio=ratio)
    return det_result, depth, label_map, masks, detectable, non_detectable


def save_yolo_masks(img_path, masks, out_path, alpha=0.6):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    out = overlay_yolo_masks(image, masks, alpha=alpha)
    cv2.imwrite(out_path, out)


def overlay_yolo_masks(image, masks, indices=None, alpha=0.6):
    if masks is None or len(masks) == 0:
        return image

    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)
    palette = np.array(
        [
            [255, 255, 153],
            [204, 255, 204],
            [204, 229, 255],
            [255, 204, 229],
            [255, 230, 204],
        ],
        dtype=np.uint8,
    )
    pick = set(indices) if indices is not None else None
    for i, mask in enumerate(masks):
        if pick is not None and i not in pick:
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
        color = palette[i % len(palette)]
        overlay[mask] = color

    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    out = image.copy()
    mask_any = overlay.sum(axis=2) > 0
    out[mask_any] = blended[mask_any]
    return out


def save_yolo_boxes(img_path, yolo_result, depth_map, out_path, top_ratio=0.2, boxes=None):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    h, w = depth_map.shape
    names = yolo_result.names
    if yolo_result.boxes is None:
        cv2.imwrite(out_path, image)
        return

    for b, cls in zip(
        yolo_result.boxes.xyxy.cpu().numpy(),
        yolo_result.boxes.cls.cpu().numpy(),
    ):
        x1, y1, x2, y2 = b.astype(int)
        if boxes is not None and [x1, y1, x2, y2] not in boxes:
            continue
        crop = depth_map[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].flatten()
        if crop.size == 0:
            continue
        sorted_depths = np.sort(crop)
        top_n = max(1, int(len(sorted_depths) * top_ratio))
        dist = sorted_depths[:top_n].mean()
        label = f"{names[int(cls)]}, distance={dist:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        pad = 4
        max_w = max(10, (x2 - x1) - 2 * pad - 4)
        max_h = max(10, (y2 - y1) - 2 * pad - 4)

        text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]
        scale = min(max_w / text_w, max_h / text_h, 1.0)
        font_scale = font_scale * scale
        text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]

        box_x1 = x1 + 2
        box_x2 = min(x2 - 2, box_x1 + text_w + 2 * pad)
        box_y1 = y1 + 2
        box_y2 = min(y2 - 2, box_y1 + text_h + 2 * pad)

        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            (box_x1 + pad, box_y2 - pad),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, image)
