import cv2
import numpy as np
from yolo import build_models, analyze_image, save_yolo_boxes, save_yolo_masks, overlay_yolo_masks


def format_output(detectable, non_detectable, top_k=3):
    items = sorted(detectable + non_detectable, key=lambda x: x["dist"])
    result = []
    for obj in items[:top_k]:
        result.append({obj["name"]: [obj["dist"], obj["dir"]]})
    return result


def top_items(detectable, non_detectable, top_k=3):
    return sorted(detectable + non_detectable, key=lambda x: x["dist"])[:top_k]


def draw_label_box(image, bbox, label):
    x1, y1, x2, y2 = bbox
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


def label_bbox(label_map, label_id):
    ys, xs = np.where(label_map == label_id)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2]


def main():
    while True:
        pipeline = input("Pipeline (1=bbox+depth, 2=seg+bbox+depth, blank to exit): ").strip()
        if not pipeline:
            break
        use_seg = pipeline == "2"
        yolo_model, seg_model, depth_model = build_models(use_seg=use_seg)

        while True:
            img_num = input("Image Number(1-10) (blank to exit): ").strip()
            if not img_num:
                break
            try:
                img_path = f"{img_num}.jpg"
                yolo_result, depth, label_map, masks, detectable, non_detectable = analyze_image(
                    img_path, yolo_model, depth_model, seg_model=seg_model
                )
                base_img = cv2.imread(img_path)
                if base_img is None:
                    raise FileNotFoundError(f"Could not load image: {img_path}")

                while True:
                    print("Options:")
                    if use_seg:
                        print("  1: Save YOLO mask -> *_mask.jpg")
                    else:
                        print("  1: Save YOLO boxes -> *_boxes.jpg")
                    print("  2: Save depth map  -> *_depthmap.jpg")
                    print("  3: Save cluster map -> *_clusters.jpg")
                    if use_seg:
                        print("  4: Save top-3 map  -> *_result(yolo+seg+depth).jpg")
                    else:
                        print("  4: Save top-3 map  -> *_result(yolo+depth).jpg")
                    print("  5: Show top-3 list")
                    print("  0: Back to image selection")
                    choice = input("Choose: ").strip()
                    if choice == "0":
                        break
                    if choice == "1":
                        if use_seg:
                            save_yolo_masks(img_path, masks, img_path.replace(".jpg", "_mask.jpg"))
                        else:
                            save_yolo_boxes(
                                img_path,
                                yolo_result,
                                depth,
                                img_path.replace(".jpg", "_boxes.jpg"),
                            )
                        continue
                    if choice == "2":
                        depth_model.save_depth_map(depth, img_path.replace(".jpg", "_depthmap.jpg"))
                        continue
                    if choice == "3":
                        depth_model.save_cluster_overlay(
                            base_img, label_map, img_path.replace(".jpg", "_clusters.jpg")
                        )
                        continue
                    if choice == "4":
                        items = top_items(detectable, non_detectable)
                        out = base_img.copy()
                        if not use_seg:
                            norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            depth_color = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
                            for obj in items:
                                if "bbox" in obj:
                                    x1, y1, x2, y2 = obj["bbox"]
                                    out[y1:y2, x1:x2] = depth_color[y1:y2, x1:x2]

                        mask_indices = [obj["mask_index"] for obj in items if "mask_index" in obj]
                        if masks is not None and mask_indices:
                            out = overlay_yolo_masks(out, masks, indices=mask_indices)
                        labels = [obj["label"] for obj in items if "label" in obj]
                        if labels:
                            out = depth_model.overlay_clusters(out, label_map, labels=labels)

                        for obj in items:
                            if "bbox" in obj:
                                label = (
                                    f"[{obj['name']}, distance={obj['dist']:.2f}, "
                                    f"direction={obj['dir']}]"
                                )
                                draw_label_box(out, obj["bbox"], label)
                            elif "label" in obj:
                                bbox = label_bbox(label_map, obj["label"])
                                if bbox is not None:
                                    label = (
                                        f"[{obj['name']}, distance={obj['dist']:.2f}, "
                                        f"direction={obj['dir']}]"
                                    )
                                    draw_label_box(out, bbox, label)

                        suffix = "_result(yolo+seg+depth).jpg" if use_seg else "_result(yolo+depth).jpg"
                        cv2.imwrite(img_path.replace(".jpg", suffix), out)
                        continue
                    if choice == "5":
                        print(format_output(detectable, non_detectable))
                        continue
                    print("Unknown option.")
            except Exception as exc:
                print(f"Error: {exc}")


if __name__ == "__main__":
    main()