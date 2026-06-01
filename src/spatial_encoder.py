"""
VIPER — spatial_encoder.py
EfficientNet-B4 fine-tuned on face crops for spatial artifact detection.

Architecture:
  - EfficientNet-B4 pretrained on ImageNet
  - Freeze first 6 blocks, fine-tune last 2 blocks + classifier
  - Output: 1792-dim embedding per frame, mean-pooled across 16 frames
  - Final head: binary classifier (real/fake)

Training time: ~50 minutes on free Colab T4 with CelebDF-v2.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
import cv2
from typing import Optional


# ── Image transforms ──────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

class VIPERSpatialEncoder(nn.Module):
    """
    EfficientNet-B4 backbone with a binary classification head.
    The backbone produces a 1792-dim embedding; the head maps to real/fake.

    The embedding is also exposed for use in the fusion classifier.
    """

    def __init__(self, freeze_blocks: int = 6, dropout: float = 0.3):
        super().__init__()

        # Load pretrained EfficientNet-B4
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Freeze early blocks
        # EfficientNet-B4 has 9 MBConv blocks (features[1] through features[8])
        # freeze_blocks=9 means freeze entire backbone (recommended for small datasets)
        # freeze_blocks=6 means fine-tune last 2 blocks (recommended for larger datasets)
        for i, block in enumerate(base.features):
            if i <= freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.backbone = base.features  # (B, 1792, 7, 7) output for 224×224 input
        self.pool     = nn.AdaptiveAvgPool2d(1)  # → (B, 1792, 1, 1)
        self.flatten  = nn.Flatten()             # → (B, 1792)

        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # logit
        )

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1792-dim embedding from a batch of face crops."""
        feat = self.backbone(x)       # (B, 1792, 7, 7)
        feat = self.pool(feat)        # (B, 1792, 1, 1)
        feat = self.flatten(feat)     # (B, 1792)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) batch of face crops
        Returns: (B,) logits (before sigmoid)
        """
        feat  = self.forward_embedding(x)
        feat  = self.dropout(feat)
        logit = self.head(feat).squeeze(-1)
        return logit

    def forward_video(self, frames: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a list of frame tensors, mean-pool embeddings across frames.

        frames: list of (C, H, W) tensors
        Returns: (logit scalar, 1792-dim video embedding)
        """
        if not frames:
            raise ValueError("No frames provided.")

        batch = torch.stack(frames)  # (T, C, H, W)
        embs  = self.forward_embedding(batch)  # (T, 1792)
        video_emb = embs.mean(dim=0)           # (1792,)

        logit = self.head(self.dropout(video_emb.unsqueeze(0))).squeeze()
        return logit, video_emb


# ── Preprocessing helper ──────────────────────────────────────────────────────

def crops_to_tensors(
    crops: list[np.ndarray],
    train: bool = False,
) -> list[torch.Tensor]:
    """
    Convert list of BGR face crops (224×224 uint8) to normalized tensors.
    """
    transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))
    return tensors


# ── Model loading ─────────────────────────────────────────────────────────────

def load_spatial_encoder(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> VIPERSpatialEncoder:
    """
    Load the spatial encoder. If checkpoint_path is None, returns
    the pretrained-only model (no fine-tuning applied yet).
    """
    model = VIPERSpatialEncoder()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model
