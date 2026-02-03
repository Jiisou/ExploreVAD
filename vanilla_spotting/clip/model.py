"""
SpottingModel - CLIP/MobileCLIP based binary classifier for anomaly action spotting.

Architecture:
    Input: (B, 3, T, H, W) = (B, 3, 10, 224, 224)
        ↓ reshape to (B*T, 3, H, W)
    CLIP Vision Encoder (pretrained, per-frame)
        ↓ (B*T, D)  e.g. D=512 for ViT-B
    Reshape → (B, T, D)
        ↓
    Temporal Aggregation (mean pooling)
        ↓ (B, D)
    Classification Head: Linear(D, num_classes)
        ↓
    Output: (B, 2) logits for Normal/Abnormal
"""

import torch
import torch.nn as nn
from typing import Optional

import open_clip


class SpottingModel(nn.Module):
    """
    Anomaly Action Spotting Model using CLIP vision encoder backbone.

    Processes video frames individually through a frozen/fine-tunable CLIP
    vision encoder, aggregates temporal features, and classifies as
    Normal vs Abnormal.

    Args:
        model_name: open_clip model name (e.g. "ViT-B-32", "MobileCLIP-S2").
        pretrained: open_clip pretrained tag (e.g. "openai", "datacompdr").
        num_classes: Number of output classes (default: 2).
        embed_dim: Embedding dimension of the CLIP model.
        dropout_rate: Dropout rate before classification head.
        temporal_agg: Temporal aggregation method ("mean", "max", "attention").
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout_rate: float = 0.5,
        temporal_agg: str = "mean",
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temporal_agg = temporal_agg

        # Load CLIP model via open_clip (only need vision encoder)
        print(f"Loading {model_name} (pretrained={pretrained}) via open_clip...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )

        # Extract vision encoder only
        self.visual = clip_model.visual

        # Verify embed_dim matches
        # Try a dummy forward to get actual output dim
        actual_dim = self._get_visual_output_dim()
        if actual_dim != embed_dim:
            print(f"Warning: expected embed_dim={embed_dim}, got {actual_dim}. Updating.")
            self.embed_dim = actual_dim

        # Temporal aggregation
        if temporal_agg == "attention":
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=4, batch_first=True,
            )

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Discard text encoder to save memory
        del clip_model

        print(f"CLIP vision encoder loaded: {model_name}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Temporal aggregation: {temporal_agg}")
        print(f"  Head: Linear({self.embed_dim}, {num_classes})")

    def _get_visual_output_dim(self) -> int:
        """Probe the visual encoder to determine output dimension."""
        self.visual.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.visual(dummy)
            return out.shape[-1]

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode individual frames through CLIP vision encoder.

        Args:
            x: (B, C, T, H, W) video tensor.

        Returns:
            (B, T, D) frame-level feature tensor.
        """
        B, C, T, H, W = x.shape

        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Forward through CLIP vision encoder
        features = self.visual(x)  # (B*T, D)

        # Reshape back: (B*T, D) -> (B, T, D)
        features = features.view(B, T, -1)

        return features

    def aggregate_temporal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate frame-level features into a single video-level representation.

        Args:
            features: (B, T, D) frame-level features.

        Returns:
            (B, D) video-level features.
        """
        if self.temporal_agg == "mean":
            return features.mean(dim=1)
        elif self.temporal_agg == "max":
            return features.max(dim=1).values
        elif self.temporal_agg == "attention":
            attn_out, _ = self.temporal_attn(features, features, features)
            return attn_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal aggregation: {self.temporal_agg}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W).
               Expected: (B, 3, 10, 224, 224).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # Per-frame CLIP features
        features = self.encode_frames(x)  # (B, T, D)

        # Temporal aggregation
        pooled = self.aggregate_temporal(features)  # (B, D)

        # Classification
        pooled = self.dropout(pooled)
        logits = self.head(pooled)  # (B, num_classes)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head.

        Args:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        features = self.encode_frames(x)
        pooled = self.aggregate_temporal(features)
        return pooled

    def get_head_parameters(self):
        """Get parameters of the classification head."""
        params = list(self.head.parameters()) + list(self.dropout.parameters())
        if self.temporal_agg == "attention":
            params += list(self.temporal_attn.parameters())
        return params

    def get_backbone_parameters(self):
        """Get parameters of the CLIP vision encoder."""
        return self.visual.parameters()


def create_spotting_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    num_classes: int = 2,
    embed_dim: int = 512,
    dropout_rate: float = 0.5,
    temporal_agg: str = "mean",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SpottingModel:
    """
    Create and optionally load a SpottingModel.

    Args:
        model_name: open_clip model name.
        pretrained: open_clip pretrained tag.
        num_classes: Number of output classes.
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        temporal_agg: Temporal aggregation method.
        checkpoint_path: Optional path to load trained weights from.
        device: Device to load model to.

    Returns:
        SpottingModel instance.
    """
    model = SpottingModel(
        model_name=model_name,
        pretrained=pretrained if checkpoint_path is None else "",
        num_classes=num_classes,
        embed_dim=embed_dim,
        dropout_rate=dropout_rate,
        temporal_agg=temporal_agg,
    )

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys}")

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    print("Testing SpottingModel (CLIP)...")

    model = SpottingModel(
        model_name="ViT-B-32",
        pretrained="openai",
        num_classes=2,
        embed_dim=512,
    )

    # Test forward pass: (B, C, T, H, W)
    dummy_input = torch.randn(2, 3, 10, 224, 224)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    assert features.shape[0] == 2
    assert features.shape[1] == model.embed_dim

    print("\nAll tests passed!")
