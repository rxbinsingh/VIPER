"""
VIPER — evaluate.py
Evaluate VIPER on CelebDF-v2 test split.

Usage:
    python eval/evaluate.py --data_dir /path/to/celeb-df-v2 \
                            --checkpoint checkpoints/viper_best.pt
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve,
)
from tqdm import tqdm

from src.viper_complete import VIPERDetector


def collect_test_videos(data_dir: str) -> list[tuple[str, int]]:
    """Collect test split videos (last 20% of shuffled dataset)."""
    data_dir = Path(data_dir)
    all_samples = []

    for folder in ["Celeb-real", "YouTube-real"]:
        p = data_dir / folder
        if p.exists():
            for f in p.glob("*.mp4"):
                all_samples.append((str(f), 0))

    fake_path = data_dir / "Celeb-synthesis"
    if fake_path.exists():
        for f in fake_path.glob("*.mp4"):
            all_samples.append((str(f), 1))

    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    return all_samples[int(0.8 * n):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output",     type=str, default="eval/results.json")
    args = parser.parse_args()

    detector = VIPERDetector(checkpoint=args.checkpoint)
    test_samples = collect_test_videos(args.data_dir)
    print(f"[Eval] Test set: {len(test_samples)} videos")

    labels, scores, predictions = [], [], []
    failed = 0

    for video_path, label in tqdm(test_samples, desc="Evaluating"):
        try:
            result = detector.detect(video_path)
            if result["prediction"] == "UNKNOWN":
                failed += 1
                continue
            labels.append(label)
            scores.append(result["viper_score"])
            predictions.append(1 if result["prediction"] == "FAKE" else 0)
        except Exception as e:
            print(f"[Warning] {video_path}: {e}")
            failed += 1

    print(f"\n[Eval] Processed: {len(labels)}, Failed: {failed}")

    auc      = roc_auc_score(labels, scores)
    acc      = accuracy_score(labels, predictions)
    cm       = confusion_matrix(labels, predictions)
    report   = classification_report(labels, predictions,
                                     target_names=["Real", "Fake"])

    print(f"\n{'='*50}")
    print(f"VIPER Results on CelebDF-v2 Test Set")
    print(f"{'='*50}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{report}")

    results = {
        "auc":        auc,
        "accuracy":   acc,
        "n_samples":  len(labels),
        "n_failed":   failed,
        "confusion_matrix": cm.tolist(),
    }

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
