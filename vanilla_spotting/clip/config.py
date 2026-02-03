"""
Configuration for Anomaly Action Spotting Model using CLIP/MobileCLIP backbone.

Key differences from X3D variant:
- CLIP vision encoder processes frames individually (2D, not 3D)
- Temporal aggregation via mean pooling over frame-level features
- CLIP-specific normalization (not Kinetics values)
- 10 frames per 2-second unit (from 60-frame window at 30fps)
- Embedding dimension depends on CLIP variant (512 for ViT-B, 1280 for ViT-H, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# CLIP model registry: model_name -> (open_clip_name, pretrained_tag, embed_dim)
CLIP_MODEL_REGISTRY: Dict[str, Tuple[str, str, int]] = {
    # OpenAI CLIP models via open_clip
    "ViT-B-32": ("ViT-B-32", "openai", 512),
    "ViT-B-16": ("ViT-B-16", "openai", 512),
    "ViT-L-14": ("ViT-L-14", "openai", 768),
    # LAION-trained variants
    "ViT-B-16-laion2b": ("ViT-B-16", "laion2b_s34b_b88k", 512),
    "ViT-L-14-laion2b": ("ViT-L-14", "laion2b_s32b_b82k", 768),
    # MobileCLIP models via open_clip
    "MobileCLIP-S0": ("MobileCLIP-S0", "datacompdr", 512),
    "MobileCLIP-S1": ("MobileCLIP-S1", "datacompdr", 512),
    "MobileCLIP-S2": ("MobileCLIP-S2", "datacompdr", 1024),
    "MobileCLIP-B": ("MobileCLIP-B", "datacompdr_lt", 512),
}

# CLIP normalization constants (shared across all CLIP variants)
CLIP_MEAN: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)

# Frame sampling
FRAMES_PER_UNIT = 10  # 10 frames per 2-second unit
UNIT_DURATION = 2.0   # seconds

# Class names for binary classification
SPOTTING_CLASSES: List[str] = ["Normal", "Abnormal"]


@dataclass
class SpottingModelConfig:
    """CLIP-based model configuration for spotting."""
    # Model selection (key into CLIP_MODEL_REGISTRY, or custom open_clip name)
    name: str = "ViT-B-32"
    pretrained: str = "openai"  # open_clip pretrained tag

    num_classes: int = 2  # Binary: Normal vs Abnormal
    dropout_rate: float = 0.5

    # Image size for CLIP (all CLIP variants use 224x224)
    image_size: int = 224

    # Temporal
    num_frames: int = FRAMES_PER_UNIT  # 10 frames per unit

    # Embedding dimension (auto-resolved from registry)
    embed_dim: int = 512

    # Temporal aggregation method: "mean", "max", "attention"
    temporal_agg: str = "mean"

    def __post_init__(self):
        """Resolve embed_dim from registry if model is registered."""
        if self.name in CLIP_MODEL_REGISTRY:
            _, _, self.embed_dim = CLIP_MODEL_REGISTRY[self.name]


@dataclass
class SpottingDataConfig:
    """Dataset configuration for spotting."""
    # Dataset paths
    train_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/train"
    train_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp"
    test_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/test"
    test_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp"

    # CLIP normalization
    mean: Tuple[float, ...] = CLIP_MEAN
    std: Tuple[float, ...] = CLIP_STD

    # Video parameters
    fallback_fps: int = 30
    unit_duration: float = UNIT_DURATION  # seconds per unit

    # Sliding window overlap ratios
    train_overlap_ratio: float = 0.5  # 50% overlap for training
    infer_overlap_ratio: float = 0.0  # No overlap for inference

    # Validation split
    val_split: float = 0.1

    # Video balancing
    max_videos_per_class: Optional[int] = None


@dataclass
class SpottingTrainConfig:
    """Training configuration for spotting model."""
    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Phase 1: Head-only training (backbone frozen)
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2: Full fine-tuning with differential LR
    phase2_epochs: int = 10
    phase2_backbone_lr: float = 1e-6  # Lower than X3D since CLIP is well-pretrained
    phase2_head_lr: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_name: str = "spotting_clip"
    save_best_only: bool = True
    save_interval: int = 2

    # TensorBoard
    log_dir: str = "./runs"

    # Random seed
    seed: int = 42

    # Early stopping
    patience: int = 3


def get_frame_indices(window_size: int, num_frames: int = FRAMES_PER_UNIT) -> List[int]:
    """
    Get frame indices to uniformly sample from a window.

    Always returns exactly `num_frames` indices uniformly distributed
    across the window. Supports variable window sizes (for variable FPS).

    Args:
        window_size: Size of the window in frames.
        num_frames: Number of frames to sample (default: 10).

    Returns:
        List of frame indices (length = num_frames).
    """
    if window_size <= 1:
        return [0] * num_frames

    if window_size < num_frames:
        # Window smaller than requested: sample with repetition
        indices = [int(i * (window_size - 1) / (num_frames - 1)) for i in range(num_frames)]
    else:
        # Uniform sampling
        indices = [int(i * (window_size - 1) / (num_frames - 1)) for i in range(num_frames)]

    return indices


def get_spotting_config(model_name: str = "ViT-B-32") -> SpottingModelConfig:
    """
    Get spotting model configuration.

    Args:
        model_name: Model variant (key in CLIP_MODEL_REGISTRY or open_clip model name).

    Returns:
        SpottingModelConfig instance.
    """
    if model_name in CLIP_MODEL_REGISTRY:
        oc_name, pretrained, embed_dim = CLIP_MODEL_REGISTRY[model_name]
        return SpottingModelConfig(
            name=oc_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
        )

    # Fallback: use as raw open_clip model name
    return SpottingModelConfig(name=model_name)
