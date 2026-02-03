"""
Utility functions for CLIP-based Anomaly Action Spotting Model.

Key differences from X3D variant:
- CLIP normalization (not Kinetics)
- Simple resize + center crop (no ShortSideScale from pytorchvideo)
- No 3D-specific transforms
"""

import random
import collections
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from config import SpottingModelConfig, SpottingDataConfig, SPOTTING_CLASSES


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_spotting_transform(
    model_cfg: SpottingModelConfig,
    data_cfg: SpottingDataConfig,
    is_train: bool = True,
):
    """
    Create transform pipeline for CLIP-based spotting.

    Input: (T, H, W, C) uint8 [0, 255] from dataset
    Output: (C, T, H, W) float32, normalized for CLIP

    The transform:
    1. Permute to (C, T, H, W) -- treat T as batch-like dim
    2. Scale to [0, 1]
    3. Resize shortest side to image_size
    4. Center crop to (image_size, image_size)
    5. Normalize with CLIP mean/std

    Args:
        model_cfg: Model config with image_size.
        data_cfg: Data config with normalization params.
        is_train: Whether to create train transform.

    Returns:
        Transform callable.
    """
    image_size = model_cfg.image_size
    mean = data_cfg.mean
    std = data_cfg.std

    def transform_fn(video: torch.Tensor) -> torch.Tensor:
        """
        Transform video tensor.

        Args:
            video: (T, H, W, C) float tensor [0, 255].

        Returns:
            (C, T, H, W) normalized tensor.
        """
        # (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2)

        # Scale to [0, 1]
        video = video / 255.0

        T = video.shape[0]

        # Resize shortest side to image_size and center crop
        resized_frames = []
        for t in range(T):
            frame = video[t]  # (C, H, W)

            # Resize: shortest side to image_size
            _, h, w = frame.shape
            if h < w:
                new_h = image_size
                new_w = int(w * image_size / h)
            else:
                new_w = image_size
                new_h = int(h * image_size / w)

            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            # Center crop
            _, h, w = frame.shape
            top = (h - image_size) // 2
            left = (w - image_size) // 2
            frame = frame[:, top:top + image_size, left:left + image_size]

            # Normalize with CLIP mean/std
            for c in range(3):
                frame[c] = (frame[c] - mean[c]) / std[c]

            resized_frames.append(frame)

        # (T, C, H, W) -> (C, T, H, W)
        video = torch.stack(resized_frames, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        return video

    return transform_fn


def freeze_backbone(model: nn.Module):
    """
    Freeze CLIP vision encoder, keep head + temporal aggregation trainable.
    """
    # Freeze visual encoder
    for param in model.visual.parameters():
        param.requires_grad = False

    # Keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.dropout.parameters():
        param.requires_grad = True

    # Keep temporal attention trainable if present
    if hasattr(model, 'temporal_attn'):
        for param in model.temporal_attn.parameters():
            param.requires_grad = True

    print("Backbone (CLIP visual) frozen, head trainable")


def unfreeze_all(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen")


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        labels: List of labels in the dataset.
        num_classes: Number of classes.

    Returns:
        Tensor of class weights.
    """
    counts = collections.Counter(labels)
    total = len(labels)

    weights = []
    for idx in range(num_classes):
        count = counts.get(idx, 1)
        weights.append(total / (num_classes * count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {weights_tensor.tolist()}")
    return weights_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    auc: float,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'auc': auc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None,
) -> dict:
    """Load training checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    if model is not None and 'model_state_dict' in checkpoint:
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"  AUC: {checkpoint.get('auc', 'unknown'):.4f}")

    return checkpoint


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""

    def __init__(self, patience: int = 5, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
