"""
VIPER — train.py
Full training pipeline for VIPER on CelebDF-v2.

Usage:
    python train.py --data_dir /path/to/celeb-df-v2 --epochs 15 --batch_size 16

Training time: ~2.5 hours on free Colab T4.

Dataset structure expected:
    celeb-df-v2/
        Celeb-real/       ← real videos
        Celeb-synthesis/  ← fake videos
        YouTube-real/     ← additional real videos
"""

import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.preprocessing import preprocess_video
from src.anchor_extractor import build_identity_anchor
from src.displacement_probe import compute_all_residuals
from src.spatial_encoder import crops_to_tensors
from src.fusion_classifier import VIPERModel, assemble_features


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CelebDFDataset(Dataset):
    """
    CelebDF-v2 dataset loader.
    Preprocesses videos on-the-fly and caches features to disk.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: str = "./cache",
        num_frames: int = 16,
        n_anchor: int = 8,
        train: bool = True,
    ):
        self.data_dir   = Path(data_dir)
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.num_frames = num_frames
        self.n_anchor   = n_anchor
        self.train      = train

        # Collect video paths and labels
        self.samples = []
        self._collect_samples(split)

        print(f"[Dataset] {split}: {len(self.samples)} videos "
              f"({sum(1 for _,l in self.samples if l==0)} real, "
              f"{sum(1 for _,l in self.samples if l==1)} fake)")

    def _collect_samples(self, split: str):
        """Collect video paths with labels. 70/10/20 train/val/test split."""
        all_samples = []

        # Real videos
        for folder in ["Celeb-real", "YouTube-real"]:
            folder_path = self.data_dir / folder
            if folder_path.exists():
                for f in folder_path.glob("*.mp4"):
                    all_samples.append((str(f), 0))  # 0 = real

        # Fake videos
        fake_path = self.data_dir / "Celeb-synthesis"
        if fake_path.exists():
            for f in fake_path.glob("*.mp4"):
                all_samples.append((str(f), 1))  # 1 = fake

        # Deterministic split
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        if split == "train":
            self.samples = all_samples[:int(0.7 * n)]
        elif split == "val":
            self.samples = all_samples[int(0.7 * n):int(0.8 * n)]
        else:  # test
            self.samples = all_samples[int(0.8 * n):]

    def _cache_path(self, video_path: str) -> Path:
        name = Path(video_path).stem
        return self.cache_dir / f"{name}.npz"

    def _process_video(self, video_path: str) -> dict | None:
        """Process video and return feature dict, using cache if available."""
        cache = self._cache_path(video_path)

        if cache.exists():
            try:
                data = np.load(cache, allow_pickle=True)
                return {k: data[k] for k in data.files}
            except Exception:
                pass  # reprocess if cache is corrupt

        try:
            preprocessed = preprocess_video(
                video_path,
                num_frames=self.num_frames,
                n_anchor=self.n_anchor,
            )
            if not preprocessed["valid"]:
                return None

            anchor    = build_identity_anchor(preprocessed)
            residuals = compute_all_residuals(preprocessed, anchor)

            # Assemble hand features (16-dim)
            g = residuals["gir_stats"]
            t = residuals["tfr_stats"]
            b = residuals["bcr_stats"]
            bcr_feats = [b["MR"], b["TV"], b["DSR"], 1.0] if b else [0.0]*4
            hand_feats = (
                [g["MR"], g["TV"], g["DSR"]]
                + [t["MR"], t["TV"], t["DSR"]]
                + bcr_feats
                + [anchor["anchor_quality"], 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # 16 values

            # Convert face crops to tensors (stored as numpy for caching)
            crops_rgb = []
            import cv2
            for crop in preprocessed["video_frames"]:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops_rgb.append(rgb)

            result = {
                "crops": np.stack(crops_rgb),          # (T, 224, 224, 3) uint8
                "hand_feats": np.array(hand_feats, dtype=np.float32),  # (16,)
            }

            np.savez_compressed(cache, **result)
            return result

        except Exception as e:
            print(f"[Warning] Failed to process {video_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        data = self._process_video(video_path)

        if data is None:
            # Return a zero sample if processing failed
            return {
                "crops":      torch.zeros(self.num_frames, 3, 224, 224),
                "hand_feats": torch.zeros(16),
                "label":      torch.tensor(label, dtype=torch.float32),
                "valid":      False,
            }

        # Convert crops to tensors
        from torchvision import transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.train:
            transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        from PIL import Image
        crops = data["crops"]  # (T, 224, 224, 3)
        T_actual = min(len(crops), self.num_frames)

        tensors = []
        for i in range(T_actual):
            img = Image.fromarray(crops[i])
            tensors.append(transform(img))

        # Pad to num_frames if needed
        while len(tensors) < self.num_frames:
            tensors.append(tensors[-1])

        crops_tensor = torch.stack(tensors[:self.num_frames])  # (T, 3, 224, 224)

        return {
            "crops":      crops_tensor,
            "hand_feats": torch.tensor(data["hand_feats"], dtype=torch.float32),
            "label":      torch.tensor(label, dtype=torch.float32),
            "valid":      True,
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        crops      = batch["crops"].to(device)       # (B, T, 3, 224, 224)
        hand_feats = batch["hand_feats"].to(device)  # (B, 16)
        labels     = batch["label"].to(device)       # (B,)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(crops, hand_feats)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        crops      = batch["crops"].to(device)
        hand_feats = batch["hand_feats"].to(device)
        labels     = batch["label"].to(device)

        with torch.cuda.amp.autocast():
            logits = model(crops, hand_feats)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VIPER on CelebDF-v2")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--cache_dir",  type=str, default="./cache")
    parser.add_argument("--save_dir",   type=str, default="./checkpoints")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VIPER] Training on {device}")

    Path(args.save_dir).mkdir(exist_ok=True)

    # Datasets
    train_ds = CelebDFDataset(args.data_dir, "train", args.cache_dir,
                               args.num_frames, train=True)
    val_ds   = CelebDFDataset(args.data_dir, "val",   args.cache_dir,
                               args.num_frames, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = VIPERModel(freeze_blocks=6, dropout=0.3).to(device)
    print(f"[VIPER] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[VIPER] Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training setup
    # Weighted BCE to handle class imbalance (more fakes than reals in CelebDF-v2)
    pos_weight = torch.tensor([590 / 5639 * 5639 / 590]).to(device)  # ~9.5x more fakes
    criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    history  = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss,     "val_auc": val_auc,
        })

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{args.save_dir}/viper_best.pt")
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")

    # Save final model and history
    torch.save(model.state_dict(), f"{args.save_dir}/viper_final.pt")
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[VIPER] Training complete. Best Val AUC: {best_auc:.4f}")
    print(f"[VIPER] Checkpoints saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
