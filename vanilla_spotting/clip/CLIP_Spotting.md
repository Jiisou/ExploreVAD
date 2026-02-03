## Anomaly Action Spotting Model Implementation Plan

- quick start
`python train.py --model-name MobileCLIP-S2 --batch-size 16 --phase1-epochs 6 --log-dir ./logs`

### Overview

 Implement a real-time Video Anomaly Detection (VAD) "spotting" model using CLIP variants (e.g., openai or mobile clip series) as the backbone feature extractor. 
 The model performs binary classification (Normal vs. Abnormal) on 2-second video units from untrimmed clips.

### Architecture Summary

```
Video (untrimmed) → 2s sliding window → 10 frames sampled
    → CLIP vision encoder (per-frame) → [B, 10, D]
    → temporal mean pooling → [B, D]
    → dropout → Linear(D, 2) → Normal/Abnormal
 ```

 ### Componetns
 - config.py
    - SpottingModelConfig
    - SpottingDataConfig
    - SpottingTrainConfig
 - dataset.py
    - UCFCrimeSpottingDataset
    - CustomSpottingDataset (c. ETRI)
    - Details:

        Inference Unit: 2 second of video
        Frame Sampling: Extract 10 frames from 60-frame window
            Even indices: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
            Total: 10 frames
        Labeling Logic:
            Parse CSV annotations (file_name, start_time, end_time)
            If 2-second window overlaps with any annotated event → label=1
            Otherwise → label=0
        Training Mode: stride=5 frames (0.5s, 50% overlap)
        Inference Mode: stride=10 frames (1.0s, non-overlapping)
 - model.py
    - SpottingModel Wrapper
    - Details:

        Load Pre-trained CLIP backbone from open_clip 
        Add linear layer on top of the backbone as a CLS HEAD.
        (Feature extracted before head)
        Input shape: (B, 3, 10, 224, 224)
        Output shape: (B, 2)

- train.py
- evaluate.py
- inference.py
- utils.py


---

1. Training

cd vanilla_spotting/clip

# Train with default ViT-B/32 (OpenAI CLIP)
python train.py \
    --train-root /path/to/UCF_Crimes/Videos/train \
    --train-annotation /path/to/UCF_Crimes/Videos/train/00_timestamp \
    --batch-size 8

# Train with MobileCLIP-S2
python train.py \
    --model-name MobileCLIP-S2 \
    --batch-size 16 \
    --phase1-epochs 5 \
    --log-dir ./logs

# Train with attention-based temporal aggregation
python train.py \
    --model-name ViT-B-16 \
    --temporal-agg attention \
    --optimizer adamw
2. Evaluation

python evaluate.py \
    --checkpoint ./checkpoints/spotting_clip_best.pth \
    --model-name ViT-B-32 \
    --test-root /path/to/UCF_Crimes/Videos/test \
    --test-annotation /path/to/UCF_Crimes/Videos/test/00_timestamp \
    --output-dir ./evaluation
3. Inference

# Single video
python inference.py \
    --checkpoint ./checkpoints/spotting_clip_best.pth \
    --model-name ViT-B-32 \
    --video /path/to/video.mp4

# Batch (directory of videos)
python inference.py \
    --checkpoint ./checkpoints/spotting_clip_best.pth \
    --model-name ViT-B-32 \
    --video-dir /path/to/videos/ \
    --output-dir ./inference_output


Available Models
Key	Backbone	Embed Dim	Notes
ViT-B-32	CLIP ViT-B/32	512	Default, fastest
ViT-B-16	CLIP ViT-B/16	512	Better accuracy
ViT-L-14	CLIP ViT-L/14	768	Largest OpenAI
MobileCLIP-S0	MobileCLIP-S0	512	Lightest mobile
MobileCLIP-S2	MobileCLIP-S2	1024	Best mobile
MobileCLIP-B	MobileCLIP-B	512	Mobile base


> The --model-name flag must match between training and inference. The --temporal-agg must also match (mean by default).