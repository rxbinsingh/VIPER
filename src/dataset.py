"""
VIPER — dataset.py
Dataset loader for the custom 580-video dataset.

Structure:
    dataset_production/
        real/               label=0
        face_swap/          label=1
        expression_swap/    label=1
        fullbody_gan/       label=1
        metadata.csv

Supports:
  - Loading from metadata.csv (recommended — includes quality metadata)
  - 70/10/20 train/val/test split (deterministic, seed=42)
  - Per-fake-type breakdown for ablation analysis
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from typing import Optional

from .preprocessing import preprocess_video
from .anchor_extractor import build_identity_anchor
from .displacement_probe import compute_all_residuals


# ── Label mapping ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    "real":            0,
    "face_swap":       1,
    "expression_swap": 1,
    "fullbody_gan":    1,
}

FAKE_TYPES = ["face_swap", "expression_swap", "fullbody_gan"]


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
    T.RandomGrayscale(p=0.03),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Sample loading ────────────────────────────────────────────────────────────

def load_samples(
    data_dir: str,
    split: str = "train",
    seed: int = 42,
) -> list[tuple[str, int, str]]:
    """
    Load (video_path, label, label_str) tuples from metadata.csv.

    split: 'train' (70%), 'val' (10%), 'test' (20%)
    """
    meta_path = Path(data_dir) / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")

    meta = pd.read_csv(meta_path)
    samples = []

    for _, row in meta.iterrows():
        label_str  = row["label"]
        label      = LABEL_MAP.get(label_str, 1)
        video_path = str(Path(data_dir) / label_str / row["filename"])

        if Path(video_path).exists():
            samples.append((video_path, label, label_str))

    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)

    splits = {
        "train": samples[:int(0.7 * n)],
        "val":   samples[int(0.7 * n):int(0.8 * n)],
        "test":  samples[int(0.8 * n):],
        "all":   samples,
    }

    if split not in splits:
        raise ValueError(f"split must be one of {list(splits.keys())}")

    return splits[split]


def print_split_stats(samples: list, split_name: str = ""):
    """Print label distribution for a split."""
    total = len(samples)
    real  = sum(1 for _, l, _ in samples if l == 0)
    fake  = sum(1 for _, l, _ in samples if l == 1)
    print(f"  {split_name}: {total} videos ({real} real, {fake} fake)")
    for ft in FAKE_TYPES:
        n = sum(1 for _, _, ls in samples if ls == ft)
        if n > 0:
            print(f"    {ft}: {n}")


# ── Cached dataset ────────────────────────────────────────────────────────────

class VIPERDataset(Dataset):
    """
    PyTorch Dataset for VIPER training.

    Preprocesses videos on first access and caches features to disk.
    Subsequent runs load from cache — safe to restart Colab session.

    Cache format per video: .npz with keys:
        crops      (T, 224, 224, 3) uint8 — face crops
        hand_feats (16,) float32          — GIR+TFR+BCR statistics
        label      scalar int32
    """

    def __init__(
        self,
        samples: list[tuple[str, int, str]],
        cache_dir: str,
        num_frames: int = 16,
        n_anchor: int = 8,
        train: bool = True,
    ):
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.n_anchor   = n_anchor
        self.transform  = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

        # Only keep samples that are cached or can be processed
        self.samples = samples
        self._preprocess_all()

        # Filter to only cached samples
        self.samples = [
            s for s in self.samples
            if self._cache_path(s[0]).exists()
        ]

    def _cache_path(self, video_path: str) -> Path:
        return self.cache_dir / f"{Path(video_path).stem}.npz"

    def _preprocess_all(self):
        """Preprocess all videos and cache. Skip already cached."""
        from tqdm import tqdm
        to_process = [
            s for s in self.samples
            if not self._cache_path(s[0]).exists()
        ]
        if not to_process:
            return

        print(f"  Preprocessing {len(to_process)} videos...")
        ok, fail = 0, 0
        for video_path, label, _ in tqdm(to_process, leave=False):
            if self._process_one(video_path, label):
                ok += 1
            else:
                fail += 1
        print(f"  Done: {ok} cached, {fail} failed")

    def _process_one(self, video_path: str, label: int) -> bool:
        try:
            preprocessed = preprocess_video(
                video_path,
                num_frames=self.num_frames,
                n_anchor=self.n_anchor,
            )
            if not preprocessed["valid"]:
                return False

            anchor    = build_identity_anchor(preprocessed)
            residuals = compute_all_residuals(preprocessed, anchor)

            g = residuals["gir_stats"]
            t = residuals["tfr_stats"]
            b = residuals["bcr_stats"]
            bcr_feats = [b["MR"], b["TV"], b["DSR"], 1.0] if b else [0.0] * 4
            hand_feats = (
                [g["MR"], g["TV"], g["DSR"]]
                + [t["MR"], t["TV"], t["DSR"]]
                + bcr_feats
                + [anchor["anchor_quality"], 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # 16 values

            crops_rgb = []
            for crop in preprocessed["video_frames"]:
                crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            np.savez_compressed(
                self._cache_path(video_path),
                crops=np.stack(crops_rgb),
                hand_feats=np.array(hand_feats, dtype=np.float32),
                label=np.array(label, dtype=np.int32),
            )
            return True

        except Exception as e:
            print(f"    Failed: {Path(video_path).name} — {e}")
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_path, label, label_str = self.samples[idx]
        data = np.load(self._cache_path(video_path))

        crops    = data["crops"]   # (T, 224, 224, 3)
        hand     = data["hand_feats"]
        T_actual = min(len(crops), self.num_frames)

        tensors = [
            self.transform(Image.fromarray(crops[i]))
            for i in range(T_actual)
        ]
        while len(tensors) < self.num_frames:
            tensors.append(tensors[-1])

        return {
            "crops":      torch.stack(tensors[:self.num_frames]),  # (T, 3, 224, 224)
            "hand_feats": torch.tensor(hand, dtype=torch.float32),
            "label":      torch.tensor(label, dtype=torch.float32),
            "label_str":  label_str,
        }
