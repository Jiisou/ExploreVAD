"""
Model Registry for Vision-Language Similarity Dashboard.

Provides a unified TextEncoder interface across multiple CLIP backends:
- HuggingFace transformers (CLIPModel)
- OpenCLIP (ViT, MobileCLIP v2)
- MobileCLIP v1 (apple/ml-mobileclip package)
"""

import numpy as np
import torch
import torch.nn.functional as F


MODEL_REGISTRY = {
    # --- HuggingFace transformers ---
    "CLIP ViT-B/32 (HF)": {
        "backend": "huggingface",
        "model_name": "openai/clip-vit-base-patch32",
        "embedding_dim": 512,
        "description": "OpenAI CLIP ViT-B/32 via HuggingFace transformers",
    },
    "CLIP ViT-B/16 (HF)": {
        "backend": "huggingface",
        "model_name": "openai/clip-vit-base-patch16",
        "embedding_dim": 512,
        "description": "OpenAI CLIP ViT-B/16 via HuggingFace transformers",
    },
    # --- OpenCLIP ---
    "ViT-B-16 (OpenCLIP, openai)": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "embedding_dim": 512,
        "description": "ViT-B/16 loaded via open_clip with OpenAI weights",
    },
    "ViT-B-16 (OpenCLIP, laion2b)": {
        "backend": "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b79k",
        "embedding_dim": 512,
        "description": "ViT-B/16 loaded via open_clip with LAION-2B weights",
    },
    # --- MobileCLIP v1 (mobileclip package) ---
    "MobileCLIP-S0 (v1)": {
        "backend": "mobileclip_v1",
        "model_name": "mobileclip_s0",
        "pretrained": None,
        "embedding_dim": 512,
        "description": "Apple MobileCLIP v1 S0 via mobileclip package (requires .pt path)",
    },
    "MobileCLIP-S1 (v1)": {
        "backend": "mobileclip_v1",
        "model_name": "mobileclip_s1",
        "pretrained": None,
        "embedding_dim": 512,
        "description": "Apple MobileCLIP v1 S1 via mobileclip package (requires .pt path)",
    },
    "MobileCLIP-S2 (v1)": {
        "backend": "mobileclip_v1",
        "model_name": "mobileclip_s2",
        "pretrained": None,
        "embedding_dim": 512,
        "description": "Apple MobileCLIP v1 S2 via mobileclip package (requires .pt path)",
    },
    # --- MobileCLIP v2 (open_clip) ---
    "MobileCLIP2-S0": {
        "backend": "open_clip",
        "model_name": "MobileCLIP2-S0",
        "pretrained": "dfndr2b",
        "embedding_dim": 512,
        "description": "Apple MobileCLIP2 S0 via open_clip (default pretrained: dfndr2b)",
    },
    "MobileCLIP2-S2": {
        "backend": "open_clip",
        "model_name": "MobileCLIP2-S2",
        "pretrained": "dfndr2b",
        "embedding_dim": 512,
        "description": "Apple MobileCLIP2 S2 via open_clip (default pretrained: dfndr2b)",
    },
}


def get_available_models():
    """Return list of model keys from MODEL_REGISTRY."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_key):
    """Return metadata dict for the given model key."""
    return MODEL_REGISTRY[model_key]


class TextEncoder:
    """Uniform interface for text encoding across all CLIP backends."""

    def __init__(self, backend, model, tokenizer_or_processor, device):
        self.backend = backend
        self.model = model
        self.tokenizer_or_processor = tokenizer_or_processor
        self.device = device

    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode a list of text strings into L2-normalized embeddings.

        Args:
            texts: list of str

        Returns:
            numpy array of shape [len(texts), embedding_dim], float32, L2-normalized.
        """
        if self.backend == "huggingface":
            inputs = self.tokenizer_or_processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            return text_features.cpu().float().numpy()

        elif self.backend in ("open_clip", "mobileclip_v1"):
            tokenizer = self.tokenizer_or_processor
            tokens = tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
            return text_features.cpu().float().numpy()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")


def load_text_encoder(model_key, device="cpu", pretrained_path=None):
    """
    Load the text encoder for the given model key.

    Args:
        model_key: Key from MODEL_REGISTRY
        device: 'cpu' or 'cuda'
        pretrained_path: Override pretrained path (required for MobileCLIP v1,
                         optional override for MobileCLIP v2)

    Returns:
        TextEncoder instance
    """
    info = MODEL_REGISTRY[model_key]
    backend = info["backend"]

    if backend == "huggingface":
        from transformers import CLIPModel, CLIPProcessor

        model_name = info["model_name"]
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        return TextEncoder(backend, model, processor, device)

    elif backend == "mobileclip_v1":
        import mobileclip
        from mobileclip.modules.common.mobileone import reparameterize_model

        model_name = info["model_name"]
        pretrained = pretrained_path or info.get("pretrained")

        if pretrained is None:
            raise ValueError(
                f"{model_key} requires a pretrained weights path (.pt). "
                "Please provide the path in the sidebar."
            )

        model, _, _ = mobileclip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()

        tokenizer = mobileclip.get_tokenizer(model_name)
        return TextEncoder(backend, model, tokenizer, device)

    elif backend == "open_clip":
        import open_clip

        model_name = info["model_name"]
        pretrained = pretrained_path or info.get("pretrained")

        if pretrained is None:
            raise ValueError(
                f"{model_key} requires a pretrained weights path. "
                "Please provide the .pt file path."
            )

        # MobileCLIP-specific image normalization kwargs
        model_kwargs = {}
        if "MobileCLIP" in model_name and not (
            model_name.endswith("S3")
            or model_name.endswith("S4")
            or model_name.endswith("L-14")
        ):
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device, **model_kwargs
        )
        model.eval()

        # Reparameterize MobileCLIP for faster inference
        if "MobileCLIP" in model_name:
            try:
                from mobileclip.modules.common.mobileone import reparameterize_model

                model = reparameterize_model(model)
            except ImportError:
                pass

        tokenizer = open_clip.get_tokenizer(model_name)
        return TextEncoder(backend, model, tokenizer, device)

    else:
        raise ValueError(f"Unknown backend: {backend}")
