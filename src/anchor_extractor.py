"""
VIPER — anchor_extractor.py
Identity anchor formation from the first N frames of a video.
Directly adapted from SynID's multi-anchor ensemble embedding.

The anchor encodes three biological invariants of the claimed identity:
  1. Geometric anchor  — ArcFace embedding (skull geometry)
  2. Texture anchor    — DCT frequency profile (skin texture statistics)
  3. Biomech anchor    — mean coupling matrix (muscle movement patterns)
"""

import numpy as np
from scipy.fft import dctn
from typing import Optional
import cv2


# ── Geometric anchor (ArcFace) ────────────────────────────────────────────────

def build_geometric_anchor(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Build a softmax-weighted ensemble ArcFace anchor from N embeddings.
    Directly mirrors SynID's multi-anchor ensemble logic.

    Weights are proportional to embedding norm (higher norm = more confident detection).
    Returns a single L2-normalized 512-dim anchor embedding.
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings provided for anchor formation.")

    embs = np.stack(embeddings)  # (N, 512)

    # Softmax weights based on embedding norm (proxy for detection confidence)
    norms  = np.linalg.norm(embs, axis=1)          # (N,)
    weights = np.exp(norms) / np.sum(np.exp(norms)) # softmax

    # Weighted sum
    anchor = np.sum(weights[:, None] * embs, axis=0)  # (512,)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-8)  # L2 normalize

    return anchor.astype(np.float32)


# ── Texture anchor (DCT frequency profile) ───────────────────────────────────

def compute_dct_profile(face_crop: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    Compute the radial DCT frequency profile of a face crop.

    Steps:
      1. Convert to grayscale
      2. Apply 2D DCT
      3. Compute radial mean (average energy at each frequency radius)
      4. Normalize to sum to 1 (probability distribution)

    Returns a (n_bins,) frequency profile vector.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray / 255.0

    # 2D DCT
    dct = dctn(gray, norm="ortho")  # (224, 224)
    magnitude = np.abs(dct)

    # Radial binning
    H, W = magnitude.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    max_r = np.sqrt(cy**2 + cx**2)
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            profile[i] = magnitude[mask].mean()

    # Normalize to probability distribution
    total = profile.sum()
    if total > 0:
        profile = profile / total

    return profile


def build_texture_anchor(face_crops: list[np.ndarray], n_bins: int = 64) -> np.ndarray:
    """
    Build texture anchor as the mean DCT frequency profile across anchor frames.
    Returns a (n_bins,) anchor profile.
    """
    profiles = [compute_dct_profile(crop, n_bins) for crop in face_crops]
    anchor = np.mean(profiles, axis=0)
    return anchor.astype(np.float32)


# ── Biomechanical anchor (coupling matrix) ───────────────────────────────────

def compute_landmark_displacements(
    frames: list[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Extract MediaPipe facial landmarks and compute frame-to-frame displacements.

    Returns:
        displacements: (T-1, 136) array — flattened (x,y) displacements of 68 landmarks
        Returns None if MediaPipe fails to detect faces.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
    except ImportError:
        return None

    # Use 68 key landmarks (subset of MediaPipe's 468 — matches dlib convention)
    KEY_LANDMARKS = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        # eye landmarks
        33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160,
        159, 158, 157, 173,
        # mouth landmarks
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
        321, 405, 314, 17, 84, 181, 91, 146,
    ][:68]

    landmark_seq = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                continue

            lm = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            pts = np.array(
                [[lm[i].x * w, lm[i].y * h] for i in KEY_LANDMARKS],
                dtype=np.float32,
            )  # (68, 2)
            landmark_seq.append(pts)

    if len(landmark_seq) < 3:
        return None

    landmark_seq = np.stack(landmark_seq)  # (T, 68, 2)

    # Frame-to-frame displacements
    displacements = landmark_seq[1:] - landmark_seq[:-1]  # (T-1, 68, 2)
    displacements = displacements.reshape(len(displacements), -1)  # (T-1, 136)

    return displacements.astype(np.float32)


def build_biomech_anchor(frames: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Build biomechanical anchor as the mean coupling matrix from anchor frames.

    The coupling matrix C[i,j] = correlation between displacement of landmark i
    and displacement of landmark j across frames.

    Returns a (136, 136) coupling matrix, or None if landmarks unavailable.
    """
    displacements = compute_landmark_displacements(frames)
    if displacements is None or displacements.shape[0] < 2:
        return None

    # Correlation matrix across landmark dimensions
    # displacements: (T-1, 136)
    # We want correlation between columns (landmark dimensions)
    C = np.corrcoef(displacements.T)  # (136, 136)

    # Replace NaN (zero-variance landmarks) with 0
    C = np.nan_to_num(C, nan=0.0)

    return C.astype(np.float32)


# ── Full anchor formation ─────────────────────────────────────────────────────

def build_identity_anchor(preprocessed: dict) -> dict:
    """
    Build all three anchors from preprocessed video data.

    Input: output of preprocessing.preprocess_video()
    Returns dict with:
        geometric_anchor : (512,) ArcFace anchor embedding
        texture_anchor   : (64,)  DCT frequency profile anchor
        biomech_anchor   : (136, 136) coupling matrix anchor, or None
        anchor_quality   : float in [0,1] — confidence in anchor quality
    """
    anchor = {}

    # 1. Geometric anchor
    anchor["geometric_anchor"] = build_geometric_anchor(
        preprocessed["anchor_embeddings"]
    )

    # 2. Texture anchor
    anchor["texture_anchor"] = build_texture_anchor(
        preprocessed["anchor_frames"]
    )

    # 3. Biomechanical anchor
    anchor["biomech_anchor"] = build_biomech_anchor(
        preprocessed["anchor_frames"]
    )

    # Anchor quality: based on number of valid anchor frames
    n_valid = len(preprocessed["anchor_embeddings"])
    anchor["anchor_quality"] = min(1.0, n_valid / 8.0)

    return anchor
