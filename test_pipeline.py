"""
VIPER — test_pipeline.py
Quick smoke test: run the full analytical pipeline on a few videos
from the local dataset_production/ folder.

Run this BEFORE the Colab training to catch any bugs.

Usage:
    python test_pipeline.py

Expected output:
    - No crashes
    - GIR, TFR, BCR values printed for each video
    - Real videos should have lower GIR than fake videos
    - Takes ~30-60 seconds per video (CPU)
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_video
from src.anchor_extractor import build_identity_anchor
from src.displacement_probe import compute_all_residuals

DATA_DIR = "dataset_production"

# Pick 2 real and 2 fake videos for the smoke test
def get_test_videos(data_dir: str, n_per_class: int = 2) -> list:
    samples = []
    for folder, label in [("real", 0), ("face_swap", 1),
                           ("expression_swap", 1), ("fullbody_gan", 1)]:
        folder_path = Path(data_dir) / folder
        if not folder_path.exists():
            continue
        videos = list(folder_path.glob("*.mp4"))[:n_per_class]
        for v in videos:
            samples.append((str(v), label, folder))
    return samples


def test_video(video_path: str, label: int, label_str: str):
    print(f"\n{'='*60}")
    print(f"Video:  {Path(video_path).name}")
    print(f"Label:  {label_str} ({'FAKE' if label == 1 else 'REAL'})")

    # Step 1: Preprocess
    print("  [1/3] Preprocessing...")
    preprocessed = preprocess_video(video_path, num_frames=16, n_anchor=8)

    if not preprocessed["valid"]:
        print("  FAILED: Could not detect faces")
        return None

    print(f"  Anchor frames: {len(preprocessed['anchor_frames'])}")
    print(f"  Video frames:  {len(preprocessed['video_frames'])}")

    # Step 2: Build anchor
    print("  [2/3] Building identity anchor...")
    anchor = build_identity_anchor(preprocessed)
    print(f"  Anchor quality: {anchor['anchor_quality']:.3f}")
    print(f"  Geometric anchor norm: {np.linalg.norm(anchor['geometric_anchor']):.4f}")
    print(f"  Texture anchor sum:    {anchor['texture_anchor'].sum():.4f}")
    print(f"  Biomech anchor:        {'available' if anchor['biomech_anchor'] is not None else 'unavailable (MediaPipe not installed)'}")

    # Step 3: Compute residuals
    print("  [3/3] Computing displacement residuals...")
    residuals = compute_all_residuals(preprocessed, anchor)

    gir = residuals["gir_stats"]
    tfr = residuals["tfr_stats"]
    bcr = residuals["bcr_stats"]

    print(f"\n  RESULTS:")
    print(f"  GIR (ArcFace geometry):  MR={gir['MR']:.4f}  TV={gir['TV']:.4f}  DSR={gir['DSR']:.4f}")
    print(f"  TFR (DCT texture):       MR={tfr['MR']:.4f}  TV={tfr['TV']:.4f}  DSR={tfr['DSR']:.4f}")
    if bcr:
        print(f"  BCR (Biomechanics):      MR={bcr['MR']:.4f}  TV={bcr['TV']:.4f}  DSR={bcr['DSR']:.4f}")
    else:
        print(f"  BCR (Biomechanics):      N/A (install mediapipe for this signal)")

    # Analytical score
    gir_score = gir["MR"]
    tfr_score = tfr["MR"]
    bcr_score = bcr["MR"] if bcr else 0.3
    raw = 0.5 * gir_score + 0.3 * tfr_score + 0.2 * bcr_score
    viper_score = float(1 / (1 + np.exp(-10 * (raw - 0.35))))

    prediction = "FAKE" if viper_score > 0.5 else "REAL"
    correct    = (prediction == "FAKE") == (label == 1)
    status     = "✓ CORRECT" if correct else "✗ WRONG"

    print(f"\n  VIPER Score: {viper_score:.4f}  →  {prediction}  {status}")

    return {
        "label":       label_str,
        "gir_mr":      gir["MR"],
        "tfr_mr":      tfr["MR"],
        "bcr_mr":      bcr["MR"] if bcr else None,
        "viper_score": viper_score,
        "prediction":  prediction,
        "correct":     correct,
    }


def main():
    print("VIPER Pipeline Smoke Test")
    print("="*60)

    if not Path(DATA_DIR).exists():
        print(f"ERROR: {DATA_DIR}/ not found.")
        print("Run this from the VIPER/ directory.")
        sys.exit(1)

    samples = get_test_videos(DATA_DIR, n_per_class=2)
    if not samples:
        print("ERROR: No videos found in dataset_production/")
        sys.exit(1)

    print(f"Testing {len(samples)} videos...")

    results = []
    for video_path, label, label_str in samples:
        r = test_video(video_path, label, label_str)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<20} {'GIR':>8} {'TFR':>8} {'BCR':>8} {'Score':>8} {'Result'}")
    print("-"*60)
    for r in results:
        bcr_str = f"{r['bcr_mr']:.4f}" if r["bcr_mr"] is not None else "  N/A  "
        status  = "✓" if r["correct"] else "✗"
        print(f"{r['label']:<20} {r['gir_mr']:>8.4f} {r['tfr_mr']:>8.4f} "
              f"{bcr_str:>8} {r['viper_score']:>8.4f}  {r['prediction']} {status}")

    n_correct = sum(1 for r in results if r["correct"])
    print(f"\nAccuracy: {n_correct}/{len(results)} correct")

    # Signal sanity check
    real_gir  = [r["gir_mr"] for r in results if r["label"] == "real"]
    fake_gir  = [r["gir_mr"] for r in results if r["label"] != "real"]
    if real_gir and fake_gir:
        print(f"\nSignal sanity check:")
        print(f"  Mean GIR — real: {np.mean(real_gir):.4f}  fake: {np.mean(fake_gir):.4f}")
        if np.mean(fake_gir) > np.mean(real_gir):
            print("  ✓ GIR is higher for fakes — signal is working correctly")
        else:
            print("  ⚠ GIR is NOT higher for fakes — check face detection quality")

    print("\nSmoke test complete. If no crashes and signal sanity check passes,")
    print("proceed to Colab for full training.")


if __name__ == "__main__":
    main()
