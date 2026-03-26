"""
Create a more realistic test image with objects for YOLO detection
Using webcam capture or PIL-based generation with more detail
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import ssl

output_dir = r"c:\Users\user\Desktop\project\project\data\testing"
output_path = os.path.join(output_dir, "tbc.jpg")

# Try to download a real image with objects, or create a more realistic one
try:
    # Create a more detailed image that looks more realistic
    # Using PIL to create a scene with objects
    
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Draw sky and ground
    draw.rectangle([(0, 0), (640, 240)], fill=(135, 206, 235))  # Sky blue
    draw.rectangle([(0, 240), (640, 480)], fill=(34, 139, 34))  # Grass green
    
    # Draw some buildings/houses (rectangles)
    # Building 1
    draw.rectangle([(50, 150), (150, 280)], fill=(139, 69, 19), outline=(100, 50, 0), width=2)
    draw.rectangle([(70, 170), (90, 190)], fill=(0, 0, 0))  # Window
    draw.rectangle([(110, 170), (130, 190)], fill=(0, 0, 0))  # Window
    
    # Building 2
    draw.rectangle([(200, 120), (350, 310)], fill=(192, 192, 192), outline=(128, 128, 128), width=2)
    draw.rectangle([(220, 140), (240, 160)], fill=(0, 0, 0))  # Window
    draw.rectangle([(260, 140), (280, 160)], fill=(0, 0, 0))  # Window
    draw.rectangle([(300, 140), (320, 160)], fill=(0, 0, 0))  # Window
    draw.rectangle([(220, 200), (240, 220)], fill=(0, 0, 0))  # Window
    draw.rectangle([(260, 200), (280, 220)], fill=(0, 0, 0))  # Window
    
    # Draw simple car shapes
    # Car 1
    draw.rectangle([(380, 250), (500, 290)], fill=(255, 0, 0), outline=(0, 0, 0), width=2)  # Car body
    draw.ellipse([(390, 280), (410, 300)], fill=(0, 0, 0))  # Wheel
    draw.ellipse([(480, 280), (500, 300)], fill=(0, 0, 0))  # Wheel
    
    # Tree shapes (circles)
    draw.ellipse([(550, 80), (600, 130)], fill=(34, 139, 34), outline=(0, 0, 0), width=2)  # Tree 1
    draw.ellipse([(580, 200), (630, 250)], fill=(34, 139, 34), outline=(0, 0, 0), width=2)  # Tree 2
    
    # Add some text
    try:
        draw.text((150, 20), "YOLO Test Image - Milestone", fill=(255, 255, 255))
        draw.text((150, 400), "Detecting: Buildings, Cars, Trees", fill=(255, 255, 255))
    except:
        pass  # Font might not be available
    
    # Convert to numpy array and save with cv2
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_cv)
    
    print(f"Test image created: {output_path}")
    print(f"Image size: {img_cv.shape}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
except Exception as e:
    print(f"Error creating image: {e}")
    # Fallback: create a simple image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 150
    cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(image, (250, 100), (450, 280), (255, 0, 0), -1)
    cv2.rectangle(image, (500, 150), (600, 300), (0, 0, 255), -1)
    cv2.putText(image, "YOLO Test Image", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Fallback test image created: {output_path}")
