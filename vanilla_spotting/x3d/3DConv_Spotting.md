## Anomaly Action Spotting Model Implementation Plan

### Overview

 Implement a real-time Video Anomaly Detection (VAD) "spotting" model using X3D-M as the backbone feature extractor. The model
 performs binary classification (Normal vs. Abnormal) on 1-second video units from untrimmed clips.

### Architecture Summary

```
 Input Video (untrimmed)
     ↓
 Sliding Window (1-second units)
     ↓ [16 frames per unit: even indices 0,2,...,28 + frame 29]
 X3D-M Backbone (pretrained from pytorchvideo)
     ↓ [blocks 0-4: feature extraction]
 Binary Classification Head
     ↓ [blocks[5].proj → Linear(2048, 2)]
 Anomaly Score per Segment
 ```

 ### Componetns
 - config.py
    - SpottingModelConfig
    - SpottingDataConfig
    - SpottingTrainConfig
 - dataset.py
    - UCFCrimeSpottingDataset
    - Details:

        Inference Unit: 1 second of video
        Frame Sampling: Extract 16 frames from 30-frame window
            Even indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            Plus final frame: [29]
            Total: 16 frames
        Labeling Logic:
            Parse CSV annotations (file_name, start_time, end_time)
            If 1-second window overlaps with any annotated event → label=1
            Otherwise → label=0
        Training Mode: stride=8 frames (0.5s, 50% overlap)
        Inference Mode: stride=16 frames (1.0s, non-overlapping)
 - model.py
    - SpottingModel Wrapper
    - Details:

        Load X3D-M backbone from pytorchvideo.models.hub
        Replace final `model.blocks[5].proj` with nn.Linear(2048, 2) (Feature extracted before head)
        Input shape: (B, 3, 16, 224, 224)
        Output shape: (B, 2)

- train.py
- evaluate.py
- inference.py
- utils.py