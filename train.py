"""
VIPER — train.py
Full training pipeline for VIPER.

Usage:
    python train.py --data_dir dataset_production --epochs 15 --batch_size 8

Training time: ~2.5 hours on free Colab T4.

Dataset structure expected:
    dataset_production/
        real/               ← real videos (label=0)
        face_swap/          ← face-swap deepfakes (label=1)
        expression_swap/    ← expression-swap deepfakes (label=1)
        fullbody_gan/       ← GAN deepfakes (label=1)
        metadata.csv        ← video metadata with labels
"""

import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import VIPERDataset, load_samples, print_split_stats
from src.fusion_classifier import VIPERModel


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        crops      = batch["crops"].to(device)
        hand_feats = batch["hand_feats"].to(device)
        labels     = batch["label"].to(device)

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
    parser = argparse.ArgumentParser(description="Train VIPER")
    parser.add_argument("--data_dir",   type=str, default="dataset_production")
    parser.add_argument("--cache_dir",  type=str, default="./cache")
    parser.add_argument("--save_dir",   type=str, default="./checkpoints")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VIPER] Training on {device}")
    Path(args.save_dir).mkdir(exist_ok=True)

    # Load splits
    print("[VIPER] Loading dataset splits...")
    train_samples = load_samples(args.data_dir, split="train", seed=args.seed)
    val_samples   = load_samples(args.data_dir, split="val",   seed=args.seed)
    test_samples  = load_samples(args.data_dir, split="test",  seed=args.seed)
    print_split_stats(train_samples, "Train")
    print_split_stats(val_samples,   "Val")
    print_split_stats(test_samples,  "Test")

    # Datasets
    print("[VIPER] Building datasets (preprocessing + caching)...")
    train_ds = VIPERDataset(train_samples, args.cache_dir, args.num_frames, train=True)
    val_ds   = VIPERDataset(val_samples,   args.cache_dir, args.num_frames, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = VIPERModel(freeze_blocks=6, dropout=0.3).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VIPER] Parameters: {total:,} total, {trainable:,} trainable")

    # Class balance: 250 real vs 330 fake
    n_real = sum(1 for _, l, _ in train_samples if l == 0)
    n_fake = sum(1 for _, l, _ in train_samples if l == 1)
    pos_weight = torch.tensor([n_real / max(n_fake, 1)]).to(device)
    print(f"[VIPER] Class balance — real: {n_real}, fake: {n_fake}, pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    history  = []

    print(f"\n[VIPER] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        flag = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{args.save_dir}/viper_best.pt")
            flag = "  ← best"

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss,     "val_auc": val_auc,
        })

    torch.save(model.state_dict(), f"{args.save_dir}/viper_final.pt")
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[VIPER] Training complete. Best Val AUC: {best_auc:.4f}")
    print(f"[VIPER] Checkpoints saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
