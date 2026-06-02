"""
VIPER — clip_model.py
CLIP ViT-L/14 based deepfake detection model (v3 — production model).

Architecture:
  - CLIP ViT-L/14 visual encoder (frozen, 768-dim output)
  - Mean pool across 16 temporal frames
  - Fusion MLP: [768 CLIP + 16 hand_feats] = 784-dim → 512 → 128 → 1

Achieves:
  - Test AUC: 0.9909
  - Accuracy: 95.2%
  - Training time: ~25 minutes on Colab T4
  - Inference: ~4s per video end-to-end
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class VIPERv3(nn.Module):
    """
    VIPER v3: CLIP ViT-L/14 + Fusion MLP.

    The CLIP visual encoder is completely frozen. Only the fusion MLP
    trains on the task-specific data. This achieves extreme data efficiency
    (0.99 AUC from 530 videos) by leveraging CLIP's pretrained representations.

    Input:
        crops:      (B, T, 3, 224, 224) — face crops from T frames
        hand_feats: (B, 16) — GIR(3) + TFR(3) + BCR(3) + flags(7)

    Output:
        (B,) logits — apply sigmoid for probability
    """

    def __init__(self, clip_model, dropout: float = 0.4):
        super().__init__()
        self.clip_visual = clip_model.visual  # frozen

        # Freeze CLIP entirely
        for p in self.clip_visual.parameters():
            p.requires_grad = False

        # Fusion MLP: 768 (CLIP) + 16 (hand features) = 784
        self.head = nn.Sequential(
            nn.Linear(768 + 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

        # Stable initialization
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, crops: torch.Tensor, hand_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            crops: (B, T, C, H, W) face crops
            hand_feats: (B, 16) analytical signal features

        Returns:
            (B,) logits
        """
        B, T, C, H, W = crops.shape

        # Encode all frames with frozen CLIP
        frames_flat = crops.view(B * T, C, H, W)
        with torch.no_grad():
            embs_flat = self.clip_visual(frames_flat)  # (B*T, 768)

        # Mean pool across temporal dimension
        embs = embs_flat.view(B, T, -1)        # (B, T, 768)
        video_emb = embs.mean(dim=1)           # (B, 768)

        # Concatenate with hand-crafted features
        features = torch.cat([video_emb.float(), hand_feats], dim=1)  # (B, 784)

        # Classify
        return self.head(features).squeeze(-1)  # (B,)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to fake probability."""
        return torch.sigmoid(logits)


def load_clip_model(device: str = "cuda"):
    """Load CLIP ViT-L/14 and return the model object for VIPERv3."""
    try:
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        clip_model = clip_model.to(device).eval()
        return clip_model, preprocess
    except ImportError:
        raise ImportError(
            "open_clip_torch required. Install: pip install open_clip_torch"
        )


def load_viper_v3(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> VIPERv3:
    """
    Load the production VIPER v3 model.

    Args:
        checkpoint_path: path to viper_best_v3_clip.pt
        device: 'cuda' or 'cpu'

    Returns:
        VIPERv3 model ready for inference
    """
    clip_model, _ = load_clip_model(device)
    model = VIPERv3(clip_model, dropout=0.4).to(device)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model
