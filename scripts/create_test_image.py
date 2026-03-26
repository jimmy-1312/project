"""
Create a sample test image for milestone testing
"""

import cv2
import numpy as np
import os

# Create a sample image with some objects
output_dir = r"c:\Users\user\Desktop\project\project\data\testing"
output_path = os.path.join(output_dir, "tbc.jpg")

# Create a 640x480 image with some content
image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background

# Add some colored rectangles to represent objects that YOLO can detect
cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), -1)  # Green rectangle
cv2.rectangle(image, (200, 100), (350, 250), (255, 0, 0), -1)  # Blue rectangle
cv2.rectangle(image, (400, 150), (550, 300), (0, 0, 255), -1)  # Red rectangle

# Add some text
cv2.putText(image, "Test Image for YOLO Detection", (150, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, "Milestone Testing 2026", (180, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Save the image
cv2.imwrite(output_path, image)

print(f"Test image created: {output_path}")
print(f"Image size: {image.shape}")
print(f"File size: {os.path.getsize(output_path)} bytes")
