# project
Our team project aims to build a model pipeline to identify the object and distance from mobile photo in real time

Current project structure:

├── README.md                          # Project description
├── src/
│   ├── detector.py                    # YOLOv8 object detection
│   ├── segmentor.py                   # MobileSAM segmentation
│   ├── depth_estimator.py             # Depth Anything V2
│   ├── pipeline.py                    # End-to-end pipeline (Combine every model)
│   ├── visualizer.py                  # Color-coded depth maps, overlays (for report display)
│   ├── evaluation.py                  # RMSE, AbsRel, δ<1.25, P/R/F1
│   ├── LLMgeneration.py               # optional LLM output
│   └── data_loader.py                 # data loader (tentative --> NYU Depth V2 & custom HK)
│   └── main.py                        # input(image) --> output(every object + corresponding distance)(map)
├── data/
│   └── training/                      # training data (probably from NYU Depth V2)
│   └── evaluation/                    # evaluation data (probably seperate from training set)
│   └── testing/                       # test data (as well)
|   └── HK_custom_for_finetuning/      # see if can use for finetune 
└── model
|   └── Yolo/                          #see if we need a independent record file for every model to store their weight/checkpoint
|   └── Depth Anything/
|   └── mobileSAM
|   └── LLM/
