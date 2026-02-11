"""
Temporal Self-Attention Model for Video Anomaly Detection.

This module implements a temporal self-attention model that:
1. Takes pre-extracted frame-level features (npy)
2. Applies sliding window with configurable duration and overlap
3. Performs self-attention within each window for temporal aggregation
4. Uses ResidualMLP classification head (CLIP-style)

Architecture:
    Input: (B, T, D) pre-extracted features
        ↓
    Frame Sampling (skip duplicated frames from extraction stage)
        ↓ (B, T', D) where T' = T // frame_skip
    Sliding Window (window_size frames, overlap_ratio)
        ↓ (B, num_windows, window_size, D)
    Self-Attention per window
        ↓ (B, num_windows, D)
    Window Aggregation (mean/max/attention)
        ↓ (B, D)
    Residual MLP: x + MLP(x)
        ↓ (B, D)
    Classifier: Linear(D, 1)
        ↓
    Output: (B, 1) logit
"""

from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuickGELU(nn.Module):
    """CLIP's GELU approximation for representation space consistency."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# =============================================================================
# VadCLIP Components (from VadCLIP/src/model.py)
# =============================================================================

class LayerNorm(nn.LayerNorm):
    """LayerNorm with float32 casting for mixed precision compatibility."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    """
    VadCLIP-style Residual Attention Block.

    Pre-LN architecture with support for attention mask (local attention).
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    """
    VadCLIP-style Transformer.

    Stacked ResidualAttentionBlocks with optional attention mask.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        return self.resblocks((x, padding_mask))


# =============================================================================
# Basic Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tensor
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalSelfAttention(nn.Module):
    """
    Self-attention block for temporal windows.

    Applies multi-head self-attention within each window and aggregates
    to a single representation per window.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_pos_encoding = use_pos_encoding

        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) window features
        Returns:
            (B, D) aggregated window representation
        """
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        # Aggregate: mean pooling over time
        return x.mean(dim=1)


class TemporalSelfAttentionModel(nn.Module):
    """
    Temporal Self-Attention Model for Video Anomaly Detection.

    Processes pre-extracted frame-level features with:
    1. Frame sampling to skip duplicated frames (from extraction stage)
    2. Sliding window extraction with configurable overlap
    3. Self-attention within each window
    4. Window-level aggregation
    5. ResidualMLP classification head

    Args:
        embed_dim: Feature embedding dimension (default: 512)
        window_frames: Number of unique frames per window (default: 10)
                      30fps video, 6 frame skip → 5fps effective
                      2 seconds * 5fps = 10 frames per window
        frame_skip: Skip factor for duplicated frames (default: 6)
                   Features are copied 6 times at extraction, so skip 6
        overlap_ratio: Window overlap ratio (default: 0.5 for 50%)
        num_heads: Number of attention heads (default: 1)
        num_layers: Number of transformer layers per window (default: 2)
        dropout: Dropout rate (default: 0.1)
        window_agg: How to aggregate windows ("mean", "max", "attention")
    """

    def __init__(
        self,
        embed_dim: int = 512,
        window_frames: int = 10,
        frame_skip: int = 6,
        overlap_ratio: float = 0.5,
        num_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_agg: str = "mean",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_frames = window_frames
        self.frame_skip = frame_skip
        self.overlap_ratio = overlap_ratio
        self.window_agg = window_agg

        # Stride for sliding window (50% overlap = stride of window_frames // 2)
        self.window_stride = int(window_frames * (1 - overlap_ratio))

        # Input projection (optional, identity if same dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # self-attention layers for window processing
        self.window_attention_layers = nn.ModuleList([
            TemporalSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_pos_encoding=(i == 0),  # Only first layer has pos encoding
            )
            for i in range(num_layers)
        ])

        # Final window-level self-attention (if using attention aggregation)
        if window_agg == "attention":
            self.window_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Residual MLP (CLIP-style)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
        ]))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

        self._print_config()

    def _print_config(self):
        print(f"TemporalSelfAttentionModel created:")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Window frames: {self.window_frames} (after skip)")
        print(f"  Frame skip: {self.frame_skip}")
        print(f"  Window stride: {self.window_stride} frames ({self.overlap_ratio*100:.0f}% overlap)")
        print(f"  Window aggregation: {self.window_agg}")
        print(f"  Attention layers: {len(self.window_attention_layers)}")

    def _extract_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Extract sliding windows from input features.

        Args:
            x: (B, T, D) input features (already frame-skipped)

        Returns:
            windows: (B, num_windows, window_frames, D)
            num_windows: number of windows extracted
        """
        B, T, D = x.shape

        # Calculate number of windows
        if T < self.window_frames:
            # Pad if sequence is shorter than window
            pad_size = self.window_frames - T
            x = F.pad(x, (0, 0, 0, pad_size), mode='replicate')
            T = self.window_frames

        num_windows = max(1, (T - self.window_frames) // self.window_stride + 1)

        # Extract windows using unfold
        windows = []
        for i in range(num_windows):
            start = i * self.window_stride
            end = start + self.window_frames
            if end <= T:
                windows.append(x[:, start:end, :])
            else:
                # Last window: pad if needed
                window = x[:, start:, :]
                pad_size = self.window_frames - window.size(1)
                window = F.pad(window, (0, 0, 0, pad_size), mode='replicate')
                windows.append(window)

        windows = torch.stack(windows, dim=1)  # (B, num_windows, window_frames, D)
        return windows, num_windows

    def _process_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Process each window through self-attention layers.

        Args:
            windows: (B, num_windows, window_frames, D)

        Returns:
            (B, num_windows, D) window representations
        """
        B, num_windows, window_frames, D = windows.shape

        # Reshape for batch processing: (B * num_windows, window_frames, D)
        x = windows.view(B * num_windows, window_frames, D)

        # Apply stacked attention layers
        for layer in self.window_attention_layers:
            x = layer(x)  # (B * num_windows, D)
            # For subsequent layers, we need to unsqueeze back
            if layer != self.window_attention_layers[-1]:
                x = x.unsqueeze(1).expand(-1, window_frames, -1)

        # Reshape back: (B, num_windows, D)
        x = x.view(B, num_windows, D)
        return x

    def _aggregate_windows(self, window_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate window-level features into a single video representation.

        Args:
            window_features: (B, num_windows, D)

        Returns:
            (B, D) video representation
        """
        if self.window_agg == "mean":
            return window_features.mean(dim=1)
        elif self.window_agg == "max":
            return window_features.max(dim=1).values
        elif self.window_agg == "attention":
            attn_out, _ = self.window_attn(
                window_features, window_features, window_features
            )
            return attn_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown window aggregation: {self.window_agg}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, D) pre-extracted frame features
               T is the total number of frames (with duplicates from extraction)

        Returns:
            (B, 1) logit for binary classification
        """
        B, T, D = x.shape

        # 1. Frame sampling: skip duplicated frames
        if self.frame_skip > 1:
            x = x[:, ::self.frame_skip, :]  # (B, T//frame_skip, D)

        # 2. Input projection
        x = self.input_proj(x)

        # 3. Extract sliding windows
        windows, num_windows = self._extract_windows(x)  # (B, num_windows, window_frames, D)

        # 4. Process windows through self-attention
        window_features = self._process_windows(windows)  # (B, num_windows, D)

        # 5. Aggregate windows
        video_features = self._aggregate_windows(window_features)  # (B, D)

        # 6. Residual MLP + classification
        video_features = self.dropout(video_features)
        logit = self.classifier(video_features + self.mlp(video_features))

        return logit

    def forward_with_window_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns per-window scores (for visualization).

        Args:
            x: (B, T, D) pre-extracted frame features

        Returns:
            logit: (B, 1) final classification logit
            window_scores: (B, num_windows) per-window anomaly scores
        """
        B, T, D = x.shape

        # Frame sampling
        if self.frame_skip > 1:
            x = x[:, ::self.frame_skip, :]

        # Input projection
        x = self.input_proj(x)

        # Extract and process windows
        windows, num_windows = self._extract_windows(x)
        window_features = self._process_windows(windows)  # (B, num_windows, D)

        # Get per-window scores
        window_features_dropped = self.dropout(window_features)
        window_logits = self.classifier(
            window_features_dropped + self.mlp(window_features_dropped)
        )  # (B, num_windows, 1)
        window_scores = torch.sigmoid(window_logits.squeeze(-1))  # (B, num_windows)

        # Aggregate for final prediction
        video_features = self._aggregate_windows(window_features)
        video_features = self.dropout(video_features)
        logit = self.classifier(video_features + self.mlp(video_features))

        return logit, window_scores


class SimplifiedTemporalAttention(nn.Module):
    """
    Simplified version with single-layer attention for efficiency.

    Input: (B, T, D) → sample frames → window attention → aggregate → classify
    """

    def __init__(
        self,
        embed_dim: int = 512,
        window_frames: int = 10,
        frame_skip: int = 6,
        overlap_ratio: float = 0.5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_frames = window_frames
        self.frame_skip = frame_skip
        self.overlap_ratio = overlap_ratio
        self.window_stride = int(window_frames * (1 - overlap_ratio))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        # Single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Residual MLP
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
        ]))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

        print(f"SimplifiedTemporalAttention created:")
        print(f"  Embed dim: {embed_dim}, Window: {window_frames}, Skip: {frame_skip}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) pre-extracted features
        Returns:
            (B, 1) logit
        """
        B, T, D = x.shape

        # Sample frames
        if self.frame_skip > 1:
            x = x[:, ::self.frame_skip, :]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Take CLS token output
        cls_out = x[:, 0, :]

        # Residual MLP + classifier
        cls_out = self.dropout(cls_out)
        logit = self.classifier(cls_out + self.mlp(cls_out))

        return logit


# For backward compatibility with existing code
Transformer = SimplifiedTemporalAttention


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TemporalSelfAttentionModel")
    print("=" * 60)

    # Test parameters matching the use case:
    # - Original video: 30fps
    # - 2 second window = 60 raw frames
    # - 6 frame skip (duplicates) → 10 unique frames per window
    # - 50% overlap = 5 frame stride (1 second)

    model = TemporalSelfAttentionModel(
        embed_dim=512,
        window_frames=10,  # 2s * (30fps / 6 skip) = 10 unique frames
        frame_skip=6,      # skip duplicated frames
        overlap_ratio=0.5, # 50% overlap = 1 second stride
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        window_agg="attention",
    )

    # Simulate input: 10 seconds of video at 30fps (with 6x duplication)
    # Raw features: 10s * 30fps = 300 frames
    # After skip: 300 / 6 = 50 unique frames
    B = 4
    T = 300  # total frames with duplication (10s * 30fps)
    D = 512

    dummy_input = torch.randn(B, T, D)

    print(f"\nInput shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test with window scores
    logit, window_scores = model.forward_with_window_scores(dummy_input)
    print(f"Logit shape: {logit.shape}")
    print(f"Window scores shape: {window_scores.shape}")

    print("\n" + "=" * 60)
    print("Testing SimplifiedTemporalAttention")
    print("=" * 60)

    simple_model = SimplifiedTemporalAttention(
        embed_dim=512,
        window_frames=10,
        frame_skip=6,
        overlap_ratio=0.5,
    )

    output = simple_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\nAll tests passed!")
