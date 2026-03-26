"""
Debug script to check raw YOLO detections on image 000000000042
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.detector import YOLODetector
from src.data_loader import DataLoader

detector = YOLODetector(device=config.DEVICE)
data_loader = DataLoader(os.path.join(config.DATA_DIR, 'coco8', 'images', 'val'))

for idx, sample in enumerate(data_loader):
    if '000000000042' in sample['filename']:
        print(f"Image: {sample['filename']}")
        image = sample['image']
        
        # Run YOLO without confidence threshold
        results = detector.model(image, device=config.DEVICE, verbose=False, conf=0.01)
        
        if len(results) > 0:
            boxes = results[0].boxes
            print(f"Total raw detections (conf >= 0.01): {len(boxes)}")
            
            if len(boxes) > 0:
                print("\nTop detections:")
                # Sort by confidence
                confs = [float(b.conf[0]) for b in boxes]
                sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
                
                for rank, i in enumerate(sorted_indices[:15]):
                    conf = float(boxes[i].conf[0])
                    cls_id = int(boxes[i].cls[0])
                    cls_name = detector.class_names.get(cls_id, f'Class {cls_id}')
                    print(f"  [{rank+1}] {cls_name:20s} | confidence = {conf:.4f}")
            else:
                print("No detections at all")
        else:
            print("No results returned")
        break
