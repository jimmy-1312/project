import cv2


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


img_path = "3.jpg"
image = cv2.imread(img_path)
if image is None:
	raise FileNotFoundError(f"Could not load image: {img_path}")

# Update only bbox and label text below
draw_label_box(
	image,
	bbox=[12, 2275, 2601, 3010],
	label="dining table, distance=0.95, direction=12 oclock",
)
draw_label_box(
	image,
	bbox=[1669, 2431, 3054, 4609],
	label="chair, distance=0.89, direction=11 oclock",
)

cv2.imwrite("updated_image.jpg", image)