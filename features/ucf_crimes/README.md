- Install dependencies
```
pip install ftfy regex tqdm
# CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python

```

- Extract features
Single video:
```
python features/ucf_crimes/extract_clip_vit.py \
    --video /mnt/c/JJS/UCF_Crimes/Videos/train/Abuse/Abuse001_x264.mp4 \
    --output_dir /mnt/c/Users/USER/Desktop/ExploreVAD/features/ucf_crimes \
    --model_name openai/clip-vit-base-patch16
```

- Directory of videos:
```
python features/ucf_crimes/extract_clip_vit.py \
    --video_dir /mnt/c/JJS/UCF_Crimes/Videos \
    --output_dir /mnt/c/JJS/UCF_Crimes/Features/CLIP-ViT-B32 \
    --video_ext .mp4
```
---
- Single GPU (default)
```
python extract_clip_vit.py --video_dir /path/to/videos --output_dir /path/to/features
```

- Multiple GPUs
```
python extract_clip_vit.py --video_dir /path/to/videos --output_dir /path/to/features --num_gpus 4
```