# statement
Our team project aims to build a model pipeline to identify the object and distance from mobile photo in real time

## Current Project Structure

```
project/
├── README.md                          # Project description
├── src/
│   ├── detector.py                    # YOLOv8 object detection
│   ├── segmentor.py                   # MobileSAM segmentation
│   ├── depth_estimator.py             # Depth Anything V2
│   ├── pipeline.py                    # End-to-end pipeline (Combine every model)
│   ├── visualizer.py                  # Color-coded depth maps, overlays (for report display)
│   ├── evaluation.py                  # RMSE, AbsRel, δ<1.25, P/R/F1
│   ├── LLMgeneration.py               # optional LLM output
│   ├── data_loader.py                 # data loader (tentative --> NYU Depth V2 & custom HK)
│   └── main.py                        # input(image) --> output(every object + corresponding distance)(map)
├── data/
│   ├── training/                      # training data (probably from NYU Depth V2)
│   ├── evaluation/                    # evaluation data (probably seperate from training set)
│   ├── testing/                       # test data (as well)
│   └── HK_custom_for_finetuning/      # see if can use for finetune
├── model/
│   ├── Yolo/                          # YOLOv8 weights/checkpoint
│   ├── Depth_Anything/                # Depth Anything V2 weights/checkpoint
│   ├── MobileSAM/                     # MobileSAM weights/checkpoint
│   └── LLM/                           # LLM weights/checkpoint (optional)
```

## Project Overview

This project aims to:
- Detect objects in mobile photos using YOLOv8
- Estimate depth information for each detected object
- Provide real-time distance estimation
- Generate visualizations with color-coded depth maps

## Models Used

- **YOLOv8**: Object detection
- **MobileSAM**: Image segmentation
- **Depth Anything V2**: Depth estimation
- **LLM**: Optional description generation

## Data

- **Training**: NYU Depth V2 dataset
- **Custom**: Hong Kong custom dataset for fine-tuning
- **Evaluation & Testing**: Separate test sets
