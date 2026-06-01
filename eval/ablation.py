"""
VIPER — ablation.py
Per-signal ablation study: test each component independently.

Shows contribution of each signal to final AUC.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.preprocessing import preprocess_video
from src.anchor_extractor import build_identity_anchor
from src.displacement_probe import compute_all_residuals


def collect_test_videos(data_dir: str) -> list[tuple[str, int]]:
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
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    test_samples = collect_test_videos(args.data_dir)
    print(f"[Ablation] Test set: {len(test_samples)} videos")

    labels = []
    gir_scores, tfr_scores, bcr_scores = [], [], []

    for video_path, label in tqdm(test_samples[:200], desc="Ablation"):
        try:
            preprocessed = preprocess_video(video_path)
            if not preprocessed["valid"]:
                continue
            anchor    = build_identity_anchor(preprocessed)
            residuals = compute_all_residuals(preprocessed, anchor)

            labels.append(label)
            gir_scores.append(float(np.mean(residuals["gir_seq"])))
            tfr_scores.append(float(np.mean(residuals["tfr_seq"])))
            bcr = residuals["bcr_seq"]
            bcr_scores.append(float(np.mean(bcr)) if bcr is not None else 0.5)
        except Exception:
            continue

    print(f"\n{'='*50}")
    print(f"VIPER Ablation Study")
    print(f"{'='*50}")
    print(f"GIR alone (ArcFace):        AUC = {roc_auc_score(labels, gir_scores):.4f}")
    print(f"TFR alone (DCT texture):    AUC = {roc_auc_score(labels, tfr_scores):.4f}")
    print(f"BCR alone (Biomechanical):  AUC = {roc_auc_score(labels, bcr_scores):.4f}")

    combined = [0.5*g + 0.3*t + 0.2*b
                for g, t, b in zip(gir_scores, tfr_scores, bcr_scores)]
    print(f"GIR+TFR+BCR combined:       AUC = {roc_auc_score(labels, combined):.4f}")


if __name__ == "__main__":
    main()
