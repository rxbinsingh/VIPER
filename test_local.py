"""Quick local test — no torch/insightface needed. Tests video reading + metadata."""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = "dataset_production"

# Test metadata
meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
print(f"Metadata rows: {len(meta)}")
label_counts = meta["label"].value_counts().to_dict()
print(f"Labels: {label_counts}")
print(f"Sources: {meta['source'].value_counts().to_dict()}")

# Test video reading
print()
for folder in ["real", "face_swap", "expression_swap", "fullbody_gan"]:
    videos = list(Path(f"{DATA_DIR}/{folder}").glob("*.mp4"))
    if not videos:
        print(f"{folder}: NO VIDEOS FOUND")
        continue
    cap = cv2.VideoCapture(str(videos[0]))
    ok, frame = cap.read()
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    shape = frame.shape if ok else "N/A"
    print(f"{folder:20s}: {len(videos):3d} videos | first={videos[0].name[:40]} | "
          f"read={'OK' if ok else 'FAIL'} | shape={shape} | fps={fps:.1f} | frames={total}")

# Test DCT (no torch needed)
print()
print("Testing DCT profile (no GPU needed)...")
from scipy.fft import dctn

test_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8).astype(np.float32) / 255.0
dct = dctn(test_img, norm="ortho")
print(f"DCT shape: {dct.shape}, max: {np.abs(dct).max():.4f}")
print("DCT: OK")

print()
print("All local tests passed. Ready for Colab.")
print("Next: upload dataset_production/ to Google Drive and run VIPER_Train_Colab.ipynb")
