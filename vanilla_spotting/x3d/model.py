"""
SpottingModel - X3D-M based binary classifier for anomaly action spotting.

Architecture:
    Input: (B, C, T, H, W) = (B, 3, 16, 224, 224)
    ↓
    X3D-M Backbone (blocks 0-4: feature extraction)
    ↓
    X3D-M Head (blocks[5]: pooling + projection)
    ↓
    Output: (B, 2) logits for Normal/Abnormal
"""

import torch
import torch.nn as nn
from typing import Optional


class SpottingModel(nn.Module):
    """
    Anomaly Action Spotting Model using X3D-M backbone.

    This model wraps the X3D-M architecture from pytorchvideo and replaces
    the classification head for binary anomaly detection (Normal vs Abnormal).

    Args:
        num_classes: Number of output classes (default: 2 for binary).
        pretrained: Whether to load pretrained Kinetics weights.
        dropout_rate: Dropout rate in the head (default: 0.5).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load X3D-M from torch hub
        print(f"Loading X3D-M from torch hub (pretrained={pretrained})...")
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'x3d_m',
            pretrained=pretrained,
        )

        # Get head dimensions
        # X3D-M head structure: blocks[5] = ResNetBasicHead
        # blocks[5].proj = Linear(2048, original_num_classes)
        self.head_dim = self.backbone.blocks[5].proj.in_features  # 2048

        # Replace classification head
        self.backbone.blocks[5].proj = nn.Linear(self.head_dim, num_classes)
        print(f"Replaced classification head: {self.head_dim} -> {num_classes} classes")

        # Update dropout if needed
        if hasattr(self.backbone.blocks[5], 'dropout') and dropout_rate > 0:
            self.backbone.blocks[5].dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W).
               Expected: (B, 3, 16, 224, 224) for X3D-M.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head.

        This runs the input through blocks 0-4 and the pooling part of
        block 5, but stops before the final linear projection.

        Args:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Feature tensor of shape (B, head_dim).
        """
        # Run through feature extraction blocks (0-4)
        for block in self.backbone.blocks[:5]:
            x = block(x)

        # Run through head pooling but not projection
        head = self.backbone.blocks[5]

        # Pool
        if head.pool is not None:
            x = head.pool(x)

        # Dropout
        if head.dropout is not None:
            x = head.dropout(x)

        # Output pool (global average)
        if head.output_pool is not None:
            x = head.output_pool(x)

        # Flatten
        x = x.flatten(1)

        return x

    def get_head_parameters(self):
        """Get parameters of the classification head (block 5)."""
        return self.backbone.blocks[5].parameters()

    def get_backbone_parameters(self):
        """Get parameters of the backbone (blocks 0-4)."""
        return self.backbone.blocks[:5].parameters()


def create_spotting_model(
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SpottingModel:
    """
    Create and optionally load a SpottingModel.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained Kinetics weights.
        checkpoint_path: Optional path to load trained weights from.
        device: Device to load model to.

    Returns:
        SpottingModel instance.
    """
    model = SpottingModel(
        num_classes=num_classes,
        pretrained=pretrained if checkpoint_path is None else False,
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


def load_pretrained_x3d_for_spotting(
    model_name: str = "x3d_m",
    num_classes: int = 2,
) -> nn.Module:
    """
    Load pretrained X3D model and replace head for spotting.

    This is a simpler alternative to SpottingModel class.

    Args:
        model_name: X3D variant (only "x3d_m" supported).
        num_classes: Number of output classes.

    Returns:
        Modified X3D model.
    """
    if model_name != "x3d_m":
        raise ValueError(f"Only x3d_m is supported, got: {model_name}")

    print(f"Loading pretrained {model_name} from torch hub...")
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

    # Replace final classification layer
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    print(f"Replaced classification head: {in_features} -> {num_classes} classes")

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing SpottingModel...")

    model = SpottingModel(num_classes=2, pretrained=True)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    assert features.shape == (2, 2048), f"Expected (2, 2048), got {features.shape}"

    print("\nAll tests passed!")
