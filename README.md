# VIPER: Video Identity Perturbation and Extraction Residual

**Deepfake detection via biometric identity consistency analysis.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rxbinsingh/VIPER/blob/main/notebooks/VIPER_Train_Colab.ipynb)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-VIPER-FFD21E)](https://huggingface.co/rxbinsingh/VIPER)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## Core Principle

Every real video has one biological invariant: the person's face. Their skull geometry,
muscle coupling patterns, and skin texture frequency signature are fixed by biology.
A deepfake — regardless of how it was made — must violate at least one of these three
invariants because it is synthesizing or transplanting a face that was not originally there.

VIPER extracts an **identity anchor** from the first few frames, then measures how well
every subsequent frame satisfies all three biological constraints. The degree of violation
is the detection signal.

### The Displacement Reaction

```
AB + C → AC + B

AB  =  deepfake frame  (fake face B embedded in real video context A)
C   =  identity anchor  (three biological constraints of the real person)
AC  =  anchor bonds with context  (real video — reaction completes)
B   =  fake face displaced  (exposed because it cannot satisfy all constraints)
```

The **VIPER score** is the reaction energy — how much the anchor fails to bond with the face.

---

## How It Works

```
Video input (any length, any resolution)
    │
    ├── InsightFace face detection + 224×224 crop
    │
    ├── First 8 frames ──────────► Identity Anchor Formation
    │                                    │
    │                          ┌─────────┴─────────┐─────────────┐
    │                          ▼                   ▼             ▼
    │                    ArcFace 512-d        DCT Profile    dlib 68-pt
    │                    Anchor               Anchor         Coupling Matrix
    │                          └─────────┬─────────┘─────────────┘
    │                                    │
    │                             16-dim hand features
    │                           (GIR + TFR + BCR stats)
    │
    └── All 16 frames ──────────► CLIP ViT-L/14 (frozen)
                                         │
                                   768-dim per frame
                                   Mean pool → 768-dim
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ Fusion MLP   │
                                  │ [768 | 16]   │
                                  │ = 784-dim    │
                                  │ → 512 → 128  │
                                  │ → 1 (logit)  │
                                  └─────────────┘
                                         │
                                    TTA (flip avg)
                                         │
                                    REAL / FAKE
```

### Three Analytical Signals (16-dim hand features)

| Signal | Method | What It Measures |
|--------|--------|-----------------|
| **GIR** — Geometric Identity Residual | ArcFace cosine distance from anchor | Skull geometry consistency |
| **TFR** — Texture Frequency Residual | DCT KL divergence from anchor profile | Skin texture frequency consistency |
| **BCR** — Biomechanical Coupling Residual | dlib landmark coupling matrix Frobenius distance | Facial muscle movement consistency |

### Visual Backbone

| Component | Details |
|---|---|
| Model | CLIP ViT-L/14 (OpenAI, frozen) |
| Output | 768-dim embedding per frame |
| Temporal | Mean pool across 16 sampled frames |
| Training | Only the 784→512→128→1 MLP trains (~25 min) |

---

## Results

Evaluated on **580-video multi-method deepfake dataset** (250 real, 220 face\_swap, 60 expression\_swap, 50 fullbody\_gan):

| Model Version | Backbone | Test AUC | Accuracy |
|---|---|---|---|
| v1 | EfficientNet-B4 (frozen) | 0.9072 | 82.9% |
| v2 | EfficientNet-B4 (fine-tuned) | 0.9309 | 85.7% |
| **v3 (final)** | **CLIP ViT-L/14 (frozen) + TTA** | **0.9909** | **95.2%** |

### Per-Fake-Type Performance (v3)

| Attack Type | AUC | Accuracy | Test Videos |
|---|---|---|---|
| Face swap (inswapper) | 0.9931 | 95.6% | 42 |
| Expression swap (NeuralTextures) | 0.9847 | 93.7% | 15 |
| Full-body GAN | N/A | N/A | 0 (face detection fails on full-body) |
| **All combined** | **0.9909** | **95.2%** | **105** |

### Confusion Matrix (v3)

```
              Predicted Real  Predicted Fake
Actual Real         45              3
Actual Fake          2             55
```

- **False Positive Rate:** 6.3% (3/48 reals flagged as fake)
- **False Negative Rate:** 3.5% (2/57 fakes missed)
- **Fake Recall:** 96.5%

### Inference Speed

| Stage | Time |
|---|---|
| CLIP inference + TTA (from cache) | 0.65s per video |
| Face detection preprocessing (CPU) | ~10-14s per video |
| **End-to-end (GPU preprocessing)** | **~4s per video** |

### Training Details

- **Backbone:** CLIP ViT-L/14 (openai pretrained, fully frozen)
- **Classifier:** MLP 784 → 512 → 128 → 1 (BatchNorm + Dropout 0.4)
- **Training epochs:** 15
- **Training time:** ~25 minutes on free Colab T4
- **TTA:** Horizontal flip average at test time
- **Dataset:** 530 usable videos after face detection (91% success rate)

---

## Connection to SynID

VIPER directly extends [SynID](https://huggingface.co/rxbinsingh/SynID)'s identity
consistency work from generation to detection:

| SynID component | VIPER reuse |
|---|---|
| Multi-anchor ensemble embedding | `anchor_extractor.py` — same weighted ensemble |
| ArcFace face-weighted encoding | GIR signal — same InsightFace buffalo_sc |
| Bootstrap refinement scoring | Anchor quality scoring — same cosine threshold |
| Drift correction probe | BCR window residual — measure drift instead of correcting it |

*"If SynID can maintain identity consistency in generation, the same signals detect when identity consistency is violated in a fake."*

---

## Quick Start

### Colab (recommended — T4 GPU)

```python
!pip install -q torch torchvision insightface mediapipe opencv-python scipy gradio

# Clone repo
!git clone https://github.com/rxbinsingh/VIPER
%cd VIPER

# Run demo
!python app.py
```

### Local

```bash
git clone https://github.com/rxbinsingh/VIPER
cd VIPER
pip install -r requirements.txt
python app.py
```

### Python API

```python
from src.viper_complete import VIPERDetector

# With trained checkpoint (recommended)
detector = VIPERDetector(checkpoint="checkpoints/viper_best.pt")

# Or analytical mode (no training needed)
detector = VIPERDetector()

result = detector.detect("path/to/video.mp4")
print(result["prediction"])   # "REAL" or "FAKE"
print(result["confidence"])   # 0.0 – 1.0
print(result["signals"])      # per-signal breakdown
```

---

## Training

### 1. Upload dataset to Google Drive

```
MyDrive/VIPER/dataset_production/
    real/               ← 250 real videos
    face_swap/          ← 220 face-swap deepfakes
    expression_swap/    ← 60 expression-swap deepfakes
    fullbody_gan/       ← 50 GAN deepfakes
    metadata.csv
```

### 2. Open Colab notebook

Open `notebooks/VIPER_Train_Colab.ipynb` from GitHub in Colab with T4 GPU runtime.

### 3. Run all cells

- Preprocessing: ~1h 44min (CPU face detection, cached to Drive)
- Training v3 (CLIP): ~25 minutes
- Evaluation: ~2 minutes

Total: ~2.5 hours for a complete run from scratch.

### 4. Results saved to Drive

```
MyDrive/VIPER/checkpoints/
    viper_best_v3_clip.pt    ← production checkpoint
    final_report_v3.json     ← all metrics
    training_curves.png      ← loss/AUC plots
```

---

## Repository Structure

```
VIPER/
├── src/
│   ├── preprocessing.py         # Frame extraction, InsightFace face detection
│   ├── anchor_extractor.py      # Identity anchor: ArcFace + DCT + coupling matrix
│   ├── displacement_probe.py    # GIR + TFR residuals per frame
│   ├── bcr_dlib.py              # BCR via dlib 68-point landmarks
│   ├── clip_model.py            # CLIP ViT-L/14 + Fusion MLP (production model)
│   ├── spatial_encoder.py       # EfficientNet-B4 (v1/v2, superseded by CLIP)
│   ├── fusion_classifier.py     # Legacy fusion model (v1/v2)
│   ├── dataset.py               # Dataset loader from metadata.csv
│   └── viper_complete.py        # Full inference pipeline
├── eval/
│   ├── evaluate.py              # AUC, accuracy, confusion matrix
│   └── ablation.py              # Per-signal contribution analysis
├── dataset_production/
│   ├── metadata.csv             # 580 video metadata (labels, sources, quality)
│   ├── rejected_videos.csv      # 156 rejected videos with reasons
│   └── README.md                # Dataset documentation
├── notebooks/
│   └── VIPER_Train_Colab.ipynb  # Full training notebook
├── app.py                       # Gradio demo
├── train.py                     # Training script
├── test_pipeline.py             # Smoke test (GPU required)
├── test_local.py                # Local validation (no GPU)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.9+
- CUDA GPU (T4 or better; 8GB+ VRAM for training)
- See `requirements.txt`

Key dependencies: `torch`, `torchvision`, `open_clip_torch`, `insightface`, `dlib`, `opencv-python`, `scipy`, `gradio`

---

## Author

**Robin Singh** · Bennett University, India
- Email: robinsingh4889@gmail.com
- GitHub: [@rxbinsingh](https://github.com/rxbinsingh)
- HuggingFace: [rxbinsingh](https://huggingface.co/rxbinsingh)

*VIPER builds on [SynID](https://doi.org/10.13140/RG.2.2.30671.85925) and [GHOST](https://doi.org/10.13140/RG.2.2.27961.94567).*

---

## License

[MIT](LICENSE) © 2025 Robin Singh
