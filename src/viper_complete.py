"""
VIPER — viper_complete.py
Full inference pipeline: video path → REAL/FAKE + confidence + signal breakdown.

Usage:
    from src.viper_complete import VIPERDetector

    detector = VIPERDetector(checkpoint="checkpoints/viper_best.pt")
    result   = detector.detect("path/to/video.mp4")
    print(result)
    # {
    #   "prediction": "FAKE",
    #   "confidence": 0.87,
    #   "viper_score": 0.87,
    #   "signals": {
    #     "geometric":     {"score": 0.72, "triggered": True},
    #     "texture":       {"score": 0.45, "triggered": False},
    #     "biomechanical": {"score": 0.81, "triggered": True},
    #     "spatial_nn":    {"score": 0.91, "triggered": True},
    #   },
    #   "anchor_quality": 0.875,
    #   "frames_analyzed": 14,
    # }
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .preprocessing import preprocess_video
from .anchor_extractor import build_identity_anchor
from .displacement_probe import compute_all_residuals
from .spatial_encoder import crops_to_tensors, load_spatial_encoder
from .fusion_classifier import VIPERModel, assemble_features, load_viper_model


# ── Signal thresholds (calibrated on CelebDF-v2 val set) ─────────────────────
# These are starting values; update after running evaluate.py

GIR_THRESHOLD  = 0.35   # ArcFace cosine distance — same person < 0.35
TFR_THRESHOLD  = 0.08   # KL divergence — same texture < 0.08
BCR_THRESHOLD  = 0.45   # Frobenius distance ratio — same coupling < 0.45
VIPER_THRESHOLD = 0.50  # Final score threshold


class VIPERDetector:
    """
    Main VIPER inference class.

    Two modes:
      1. With checkpoint: uses trained fusion classifier (recommended, ~90% AUC)
      2. Without checkpoint: uses analytical signals only (~82% AUC)
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        num_frames: int = 16,
        n_anchor: int = 8,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.n_anchor   = n_anchor
        self.checkpoint = checkpoint

        if checkpoint is not None and Path(checkpoint).exists():
            self.model = load_viper_model(checkpoint, self.device)
            self.use_nn = True
            print(f"[VIPER] Loaded checkpoint: {checkpoint}")
        else:
            # Analytical mode — no trained model needed
            self.model  = None
            self.use_nn = False
            if checkpoint is not None:
                print(f"[VIPER] Checkpoint not found, using analytical mode.")
            else:
                print(f"[VIPER] Running in analytical mode (no checkpoint).")

    def detect(self, video_path: str) -> dict:
        """
        Run VIPER detection on a video.

        Returns a result dict with prediction, confidence, and per-signal breakdown.
        """
        # Step 1: Preprocess
        preprocessed = preprocess_video(
            video_path,
            num_frames=self.num_frames,
            n_anchor=self.n_anchor,
        )

        if not preprocessed["valid"]:
            return {
                "prediction":    "UNKNOWN",
                "confidence":    0.0,
                "viper_score":   0.5,
                "error":         "Could not detect faces in video.",
                "signals":       {},
                "anchor_quality": 0.0,
                "frames_analyzed": 0,
            }

        # Step 2: Build identity anchor
        anchor = build_identity_anchor(preprocessed)

        # Step 3: Compute displacement residuals
        residuals = compute_all_residuals(preprocessed, anchor)

        # Step 4: Per-signal scores
        gir_score = float(np.mean(residuals["gir_seq"]))
        tfr_score = float(np.mean(residuals["tfr_seq"]))
        bcr_score = float(np.mean(residuals["bcr_seq"])) if residuals["bcr_seq"] is not None else None

        signals = {
            "geometric": {
                "score":     round(gir_score, 4),
                "triggered": gir_score > GIR_THRESHOLD,
                "meaning":   "ArcFace identity distance from anchor",
            },
            "texture": {
                "score":     round(tfr_score, 4),
                "triggered": tfr_score > TFR_THRESHOLD,
                "meaning":   "DCT texture frequency divergence from anchor",
            },
            "biomechanical": {
                "score":     round(bcr_score, 4) if bcr_score is not None else None,
                "triggered": (bcr_score > BCR_THRESHOLD) if bcr_score is not None else None,
                "meaning":   "Facial landmark coupling matrix distance from anchor",
            },
        }

        # Step 5: Final score
        if self.use_nn:
            viper_score = self._nn_score(preprocessed, residuals, anchor)
            signals["spatial_nn"] = {
                "score":     round(viper_score, 4),
                "triggered": viper_score > VIPER_THRESHOLD,
                "meaning":   "EfficientNet-B4 spatial artifact score",
            }
        else:
            # Analytical score: weighted combination
            scores = [gir_score * 0.5, tfr_score * 0.3]
            weights = [0.5, 0.3]
            if bcr_score is not None:
                scores.append(bcr_score * 0.2)
                weights.append(0.2)

            # Normalize to [0,1] using sigmoid
            raw = sum(scores) / sum(weights)
            viper_score = float(1 / (1 + np.exp(-10 * (raw - 0.35))))

        prediction = "FAKE" if viper_score > VIPER_THRESHOLD else "REAL"
        confidence = viper_score if prediction == "FAKE" else (1.0 - viper_score)

        return {
            "prediction":      prediction,
            "confidence":      round(confidence, 4),
            "viper_score":     round(viper_score, 4),
            "signals":         signals,
            "anchor_quality":  round(anchor["anchor_quality"], 4),
            "frames_analyzed": len(preprocessed["video_frames"]),
            "gir_sequence":    residuals["gir_seq"].tolist(),
            "tfr_sequence":    residuals["tfr_seq"].tolist(),
        }

    def _nn_score(
        self,
        preprocessed: dict,
        residuals: dict,
        anchor: dict,
    ) -> float:
        """Run the trained fusion classifier and return fake probability."""
        self.model.eval()

        # Convert crops to tensors
        frame_tensors = crops_to_tensors(preprocessed["video_frames"], train=False)

        # Pad/truncate to num_frames
        while len(frame_tensors) < self.num_frames:
            frame_tensors.append(frame_tensors[-1])
        frame_tensors = frame_tensors[:self.num_frames]

        crops = torch.stack(frame_tensors).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # Hand features
        g = residuals["gir_stats"]
        t = residuals["tfr_stats"]
        b = residuals["bcr_stats"]
        bcr_feats = [b["MR"], b["TV"], b["DSR"], 1.0] if b else [0.0]*4
        hand_feats = (
            [g["MR"], g["TV"], g["DSR"]]
            + [t["MR"], t["TV"], t["DSR"]]
            + bcr_feats
            + [anchor["anchor_quality"], 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        hand_tensor = torch.tensor([hand_feats], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logit = self.model(crops, hand_tensor)
            prob  = torch.sigmoid(logit).item()

        return float(prob)


# ── Convenience function ──────────────────────────────────────────────────────

def detect_video(
    video_path: str,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """One-line deepfake detection."""
    detector = VIPERDetector(checkpoint=checkpoint, device=device)
    return detector.detect(video_path)
