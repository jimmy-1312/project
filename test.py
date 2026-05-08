#pip install -r requirements.txt
from Yolo import Yolo26
yolo = Yolo26()
img_path = "1.jpg"
out_path = "result.jpg"
yolo.label(["1.jpg"],["result.jpg"])
print(yolo.boxes(img_path).xyxy)
"""
boxes.xyxy returns a pytorch tensor of shape (N,4)

    N: the number of detected object
    For every detected object, the row structure is [xmin, ymin, xmax, ymax] in pixel scale, so y_max - y_min = H_pixel that we need
    it give the boundary box cordinates.
"""