"""
VIPER — fusion_classifier.py
Lightweight MLP that fuses spatial, biomechanical, and texture signals.

Input features (1808-dim total):
  - Spatial embedding from EfficientNet-B4:  1792-dim
  - Biomechanical stats (GIR + BCR + TFR):    10-dim
  - Anchor quality score:                      1-dim
  - Spatial logit from EfficientNet head:      1-dim
  - BCR available flag:                        1-dim
  - Padding to round number:                   3-dim
  Total:                                      1808-dim

Output: single logit → sigmoid → REAL/FAKE probability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ── Feature assembly ──────────────────────────────────────────────────────────

def assemble_features(
    spatial_emb: torch.Tensor,          # (1792,)
    spatial_logit: torch.Tensor,        # scalar
    residuals: dict,                    # output of displacement_probe.compute_all_residuals
    anchor_quality: float,
) -> torch.Tensor:
    """
    Assemble all signals into a single feature vector.

    Biomechanical stats (10 values):
      GIR: MR, TV, DSR  (3)
      TFR: MR, TV, DSR  (3)
      BCR: MR, TV, DSR  (3) — zeros if BCR unavailable
      BCR available flag (1)

    Returns (1808,) float32 tensor.
    """
    # GIR stats
    g = residuals["gir_stats"]
    gir_feats = [g["MR"], g["TV"], g["DSR"]]

    # TFR stats
    t = residuals["tfr_stats"]
    tfr_feats = [t["MR"], t["TV"], t["DSR"]]

    # BCR stats
    if residuals["bcr_stats"] is not None:
        b = residuals["bcr_stats"]
        bcr_feats = [b["MR"], b["TV"], b["DSR"], 1.0]
    else:
        bcr_feats = [0.0, 0.0, 0.0, 0.0]

    # Scalar features
    scalar_feats = [
        float(spatial_logit.item()),
        float(anchor_quality),
        0.0, 0.0, 0.0,  # padding
    ]

    hand_feats = gir_feats + tfr_feats + bcr_feats + scalar_feats  # 16 values

    hand_tensor = torch.tensor(hand_feats, dtype=torch.float32)

    # Concatenate with spatial embedding
    features = torch.cat([spatial_emb.cpu().float(), hand_tensor], dim=0)  # (1808,)
    return features


# ── Fusion MLP ────────────────────────────────────────────────────────────────

class VIPERFusionClassifier(nn.Module):
    """
    Lightweight 3-layer MLP that fuses all VIPER signals.

    Architecture:
      1808 → 512 → 128 → 1
      BatchNorm + ReLU + Dropout at each hidden layer.

    Designed to be fast to train (~10 minutes on T4) and
    interpretable — the hand-crafted features have known meaning.
    """

    def __init__(self, input_dim: int = 1808, dropout: float = 0.4):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Output
            nn.Linear(128, 1),
        )

        # Initialize output layer with small weights for stable training start
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1808) feature vectors
        Returns: (B,) logits
        """
        return self.net(x).squeeze(-1)


# ── Combined VIPER model ──────────────────────────────────────────────────────

class VIPERModel(nn.Module):
    """
    Full VIPER model combining spatial encoder + fusion classifier.
    Used for end-to-end training and inference.
    """

    def __init__(self, freeze_blocks: int = 6, dropout: float = 0.3):
        super().__init__()
        from .spatial_encoder import VIPERSpatialEncoder
        self.spatial_encoder = VIPERSpatialEncoder(
            freeze_blocks=freeze_blocks,
            dropout=dropout,
        )
        self.fusion = VIPERFusionClassifier(input_dim=1808, dropout=dropout + 0.1)

    def forward(
        self,
        frame_tensors: torch.Tensor,   # (B, T, C, H, W) — batch of videos
        hand_features: torch.Tensor,   # (B, 16) — biomech + texture + scalar stats
    ) -> torch.Tensor:
        """
        Forward pass for training.

        frame_tensors: (B, T, C, H, W)
        hand_features: (B, 16)
        Returns: (B,) logits
        """
        B, T, C, H, W = frame_tensors.shape

        # Process all frames through backbone
        frames_flat = frame_tensors.view(B * T, C, H, W)
        embs_flat   = self.spatial_encoder.forward_embedding(frames_flat)  # (B*T, 1792)
        embs        = embs_flat.view(B, T, 1792)
        video_embs  = embs.mean(dim=1)  # (B, 1792) — mean pool across frames

        # Spatial logit from EfficientNet head
        spatial_logits = self.spatial_encoder.head(
            self.spatial_encoder.dropout(video_embs)
        ).squeeze(-1)  # (B,)

        # Assemble full feature vector
        features = torch.cat([
            video_embs,                          # (B, 1792)
            hand_features,                       # (B, 16)
        ], dim=1)  # (B, 1808)

        # Fusion classifier
        logits = self.fusion(features)  # (B,)
        return logits

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to fake probability."""
        return torch.sigmoid(logits)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_viper_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> VIPERModel:
    model = VIPERModel()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model
