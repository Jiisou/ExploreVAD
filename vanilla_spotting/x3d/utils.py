"""
Utility functions for Anomaly Action Spotting Model.
"""

import random
import collections
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ShortSideScale

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
    is_train: bool = True
):
    """
    This transform expects video tensor already extracted (T, H, W, C)
    and converts it to (C, T, H, W) format required by X3D.

    Args:
        model_cfg: Model configuration with crop sizes.
        data_cfg: Data configuration with normalization parameters.
        is_train: Whether to create train transform.

    Returns:
        Compose transform for video data.
    """
    crop_size = model_cfg.crop_size if is_train else model_cfg.test_crop_size

    transform = Compose([
        # Input: (T, H, W, C) uint8 [0, 255]
        Lambda(lambda x: x.permute(3, 0, 1, 2)),  # (C, T, H, W) 
        Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
        NormalizeVideo(data_cfg.mean, data_cfg.std),
        ShortSideScale(size=model_cfg.side_size),
        CenterCropVideo(crop_size=(crop_size, crop_size)),
    ])
    return transform


def freeze_backbone(model: nn.Module):
    """
    Freeze backbone layers (blocks 0-4), keep head (block 5) trainable.

    Args:
        model: SpottingModel or X3D model with blocks attribute.
    """
    # Handle SpottingModel wrapper, avoid to access model.blocks directly
    if hasattr(model, 'backbone'):
        blocks = model.backbone.blocks
    else:
        blocks = model.blocks

    for param in blocks[:5].parameters():
        param.requires_grad = False
    for param in blocks[5].parameters():
        param.requires_grad = True
    print("Backbone frozen, head trainable")


def unfreeze_all(model: nn.Module):
    """
    Unfreeze all model parameters.

    Args:
        model: Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen")


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters.

    Args:
        model: Model to count parameters for.

    Returns:
        Dictionary with total, trainable, and frozen counts.
    """
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
    Useful for handling class imbalance in loss functions.

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
        count = counts.get(idx, 1)  # Avoid division by zero
        weights.append(total / (num_classes * count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights for imbalanced data: {weights_tensor.tolist()}")
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
    """
    Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        auc: Current AUC score.
        path: Path to save checkpoint.
        scheduler: Optional scheduler state to save.
    """
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
    device: torch.device = None
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to map checkpoint to.

    Returns:
        Checkpoint dictionary with epoch, loss, auc, etc.
    """
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
        """
        Args:
            patience: Number of epochs to wait before stopping.
            mode: 'max' or 'min' for metric improvement direction.
            delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric.

        Returns:
            True if training should stop, False otherwise.
        """
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
