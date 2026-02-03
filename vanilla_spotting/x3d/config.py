"""
Configuration for Anomaly Action Spotting Model using X3D-M backbone.

Key differences from original X3D project:
- Binary classification (Normal vs Abnormal)
- Sliding window on untrimmed videos with variable FPS support
- Fixed 16-frame sampling uniformly from each window (regardless of window size)
- Controllable overlap ratio for training vs inference
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SpottingModelConfig:
    """X3D-M model configuration for spotting."""
    name: str = "x3d_m"
    num_classes: int = 2  # Binary: Normal vs Abnormal
    dropout_rate: float = 0.5
    pretrained: bool = True

    # X3D-M specific parameters
    side_size: int = 256
    crop_size: int = 224
    test_crop_size: int = 256
    num_frames: int = 16  # Fixed: 16 frames per 1-second unit

    # Head dimension (for reference)
    head_dim: int = 2048


@dataclass
class SpottingDataConfig:
    """Dataset configuration for spotting."""
    # Dataset paths
    train_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/train"
    train_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp"
    test_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/test"
    test_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp"

    # Normalization (Kinetics pretrained values)
    mean: Tuple[float, ...] = (0.45, 0.45, 0.45)
    std: Tuple[float, ...] = (0.225, 0.225, 0.225)

    # Video parameters
    fallback_fps: int = 30  # Fallback if video FPS detection fails
    unit_duration: float = 2.0  # seconds per unit

    # Sliding window overlap ratios (0.0 = no overlap, 0.5 = 50% overlap, etc.)
    train_overlap_ratio: float = 0.5  # 50% overlap for training
    infer_overlap_ratio: float = 0.0  # No overlap for inference

    # Validation split
    val_split: float = 0.1

    # Video balancing: limit videos per class (None = no limit, use every video)
    max_videos_per_class: Optional[int] = None


@dataclass
class SpottingTrainConfig:
    """Training configuration for spotting model."""
    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Phase 1: Head-only training
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2: Full fine-tuning
    phase2_epochs: int = 10
    phase2_backbone_lr: float = 1e-5
    phase2_head_lr: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_name: str = "spotting_x3d"
    save_best_only: bool = True

    # Random seed
    seed: int = 42

    # Early stopping
    patience: int = 5


# Constants for frame sampling
FRAMES_PER_UNIT = 16  # Total frames to extract per 1-second unit
UNIT_FRAMES = 30  # Frames in 1 second at 30 fps

# Frame indices to sample from 30-frame window
# Even indices [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28] + frame 29
SAMPLE_INDICES: List[int] = list(range(0, 29, 2)) + [29]  # Length: 16

# Class names for binary classification
SPOTTING_CLASSES: List[str] = ["Normal", "Abnormal"]


def get_spotting_config(
    model_name: str = "x3d_m",
    num_classes: int = 2
) -> SpottingModelConfig:
    """
    Get spotting model configuration.

    Args:
        model_name: Model variant (currently only x3d_m supported).
        num_classes: Number of output classes (default: 2 for binary).

    Returns:
        SpottingModelConfig instance.
    """
    if model_name != "x3d_m":
        raise ValueError(f"Only x3d_m is supported for spotting, got: {model_name}")

    return SpottingModelConfig(
        name=model_name,
        num_classes=num_classes,
    )

#  Calculate which 16 indices to extract (-> (dataset.py)._extract_frames)
def get_frame_indices(window_size: int, num_frames: int = FRAMES_PER_UNIT) -> List[int]:
    """
    Get frame indices to uniformly sample from a window.

    Supports variable window sizes (for variable FPS videos). Always returns
    exactly `num_frames` indices uniformly distributed across the window.

    Examples:
    - 30-frame window (1s @ 30fps) → indices spaced ~2 frames apart
    - 48-frame window (2s @ 24fps) → indices spaced ~3 frames apart
    - 60-frame window (2s @ 30fps) → indices spaced ~4 frames apart

    Args:
        window_size: Size of the window in frames.
        num_frames: Number of frames to sample (default: 16).

    Returns:
        List of frame indices to sample (length = num_frames).
    """
    if window_size < num_frames:
        # If window is smaller than requested frames, sample with repetition
        indices = [int(i * (window_size - 1) / (num_frames - 1)) for i in range(num_frames)]
        return indices

    # Uniform sampling across the window
    indices = [int(i * (window_size - 1) / (num_frames - 1)) for i in range(num_frames)]
    return indices
