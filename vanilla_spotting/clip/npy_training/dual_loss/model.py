"""
SpottingModel - CLIP/MobileCLIP based binary classifier for anomaly action spotting.
Modified for Dual Loss training (segment-level outputs).
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Optional

# OpenCLIP/MobileCLIP imports kept for compatibility if needed, but mainly for ResidualMLPSpottingModel
import open_clip

class QuickGELU(nn.Module):
    """CLIP's GELU approximation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

# SpottingModel removed/skipped as it's not the target of modification (user asked for resmlp)
# Keeping FeatureSpottingModel for reference or compatibility if needed, but mainly ResidualMLPSpottingModel

class ResidualMLPSpottingModel(nn.Module):
    """
    CLIP-style residual MLP head for pre-extracted features.
    Modified to output both video-level and segment-level logits.

    Architecture:
        Input: (B, T, D)
            ↓
        Branch 1 (Segment):
            Linear(D, 1) -> (B, T, 1) segment logits
            
        Branch 2 (Video):
            Temporal Aggregation -> (B, D)
            Residual MLP: x + MLP(x)
            Classifier: Linear(D, 1) -> (B, 1) video logit
    """

    def __init__(
        self,
        embed_dim: int = 512,
        dropout_rate: float = 0.5,
        temporal_agg: str = "mean",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.temporal_agg = temporal_agg

        if temporal_agg == "attention":
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True,
            )

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
        ]))

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, 1)
        
        # New Segment Classifier
        self.segment_classifier = nn.Linear(embed_dim, 1)

        print(f"ResidualMLPSpottingModel created (Dual Loss):")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Temporal aggregation: {temporal_agg}")
        print(f"  MLP: Linear({embed_dim}, {embed_dim*4}) → QuickGELU → Linear({embed_dim*4}, {embed_dim})")
        print(f"  Video Classifier: Linear({embed_dim}, 1)")
        print(f"  Segment Classifier: Linear({embed_dim}, 1)")

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Forward pass.

        Args:
            x: (B, T, D) pre-extracted feature tensor.
            return_all: If True, return (video_logit, segment_logits).
                        video_logit: (B, 1)
                        segment_logits: (B, T, 1)

        Returns:
            (B, 1) logit or tuple if return_all=True.
        """
        # Segment Branch
        segment_logits = self.segment_classifier(x) # (B, T, 1)

        # Video Branch
        # Temporal aggregation
        if self.temporal_agg == "mean":
            pooled = x.mean(dim=1)
        elif self.temporal_agg == "max":
            pooled = x.max(dim=1).values
        elif self.temporal_agg == "attention":
            attn_out, _ = self.temporal_attn(x, x, x)
            pooled = attn_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal aggregation: {self.temporal_agg}")

        pooled = self.dropout(pooled)

        # Residual MLP + classifier
        video_logit = self.classifier(pooled + self.mlp(pooled))
        
        if return_all:
            return video_logit, segment_logits
        
        return video_logit
