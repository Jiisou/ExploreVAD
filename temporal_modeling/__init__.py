"""
Temporal Modeling Module for Video Anomaly Detection.

This module implements temporal self-attention models for processing
pre-extracted frame-level video features with sliding window aggregation.
"""

try:
    from .model import (
        # VadCLIP components
        LayerNorm,
        ResidualAttentionBlock,
        Transformer,
        # Original components
        TemporalSelfAttentionModel,
        SimplifiedTemporalAttention,
        WindowSelfAttention,
        PositionalEncoding,
        QuickGELU,
    )
except ImportError:
    pass

from .dataset import (
    FrameLevelFeatureDataset,
    VideoLevelFeatureDataset,
    collate_variable_length,
)

__all__ = [
    # VadCLIP components
    'LayerNorm',
    'ResidualAttentionBlock',
    'Transformer',
    # Models
    'TemporalSelfAttentionModel',
    'SimplifiedTemporalAttention',
    'WindowSelfAttention',
    'PositionalEncoding',
    'QuickGELU',
    # Datasets
    'FrameLevelFeatureDataset',
    'VideoLevelFeatureDataset',
    'collate_variable_length',
]
