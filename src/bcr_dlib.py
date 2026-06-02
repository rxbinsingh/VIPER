"""
VIPER — bcr_dlib.py
Biomechanical Coupling Residual using dlib 68-point face landmarks.

Computes the coupling matrix — how each facial landmark's displacement
correlates with every other landmark's displacement across frames.
This encodes person-specific muscle dynamics.

Requires:
  - dlib (pip install dlib)
  - shape_predictor_68_face_landmarks.dat (auto-downloaded)
"""

import os
import cv2
import bz2
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional

# ── Model download ────────────────────────────────────────────────────────────

PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PREDICTOR_PATH = os.path.join(os.path.expanduser("~"), ".viper", "shape_predictor_68_face_landmarks.dat")


def ensure_predictor() -> str:
    """Download dlib 68-point landmark predictor if not present."""
    if os.path.exists(PREDICTOR_PATH):
        return PREDICTOR_PATH

    os.makedirs(os.path.dirname(PREDICTOR_PATH), exist_ok=True)
    print(f"[BCR] Downloading dlib landmark predictor...")
    bz2_path = PREDICTOR_PATH + ".bz2"
    urllib.request.urlretrieve(PREDICTOR_URL, bz2_path)
    with open(PREDICTOR_PATH, "wb") as f_out, bz2.open(bz2_path) as f_in:
        f_out.write(f_in.read())
    os.remove(bz2_path)
    print(f"[BCR] Downloaded: {os.path.getsize(PREDICTOR_PATH) / 1e6:.1f} MB")
    return PREDICTOR_PATH


# ── Landmark extraction ───────────────────────────────────────────────────────

_detector = None
_predictor = None


def get_dlib_models():
    """Lazy-load dlib detector and predictor."""
    global _detector, _predictor
    if _detector is None:
        import dlib
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(ensure_predictor())
    return _detector, _predictor


def extract_landmarks_from_frames(frames: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Extract 68 facial landmarks from each frame using dlib.

    Args:
        frames: list of BGR frames

    Returns:
        (T, 68, 2) array of landmark coordinates, or None if <4 frames detected.
    """
    detector, predictor = get_dlib_models()
    landmarks = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if not faces:
            continue
        # Take largest face
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)
        pts = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float32,
        )
        landmarks.append(pts)

    if len(landmarks) < 4:
        return None

    return np.stack(landmarks)  # (T, 68, 2)


# ── Coupling matrix computation ───────────────────────────────────────────────

def compute_coupling_matrix(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute the biomechanical coupling matrix from a landmark sequence.

    The coupling matrix C[i,j] = correlation between displacement of
    landmark dimension i and displacement of landmark dimension j across frames.

    Input: (T, 68, 2) landmark positions
    Output: (136, 136) correlation matrix
    """
    # Frame-to-frame displacements
    deltas = landmarks[1:] - landmarks[:-1]  # (T-1, 68, 2)
    deltas_flat = deltas.reshape(len(deltas), -1)  # (T-1, 136)

    # Correlation matrix
    C = np.corrcoef(deltas_flat.T)  # (136, 136)
    C = np.nan_to_num(C, nan=0.0)

    return C.astype(np.float32)


# ── BCR score computation ─────────────────────────────────────────────────────

def compute_bcr_score(
    anchor_frames: list[np.ndarray],
    video_frames: list[np.ndarray],
) -> Optional[dict]:
    """
    Compute BCR: coupling matrix distance between anchor and video frames.

    Returns dict with:
        MR  — mean Frobenius distance (normalized)
        TV  — temporal variance of per-window distances
        DSR — fraction of windows exceeding 2σ threshold
        available — True if computation succeeded
    """
    # Anchor coupling matrix
    anchor_lms = extract_landmarks_from_frames(anchor_frames)
    if anchor_lms is None:
        return None

    C_anchor = compute_coupling_matrix(anchor_lms)
    anchor_norm = np.linalg.norm(C_anchor, "fro") + 1e-8

    # Video coupling matrix (full sequence)
    video_lms = extract_landmarks_from_frames(video_frames)
    if video_lms is None:
        return None

    C_video = compute_coupling_matrix(video_lms)

    # Overall distance
    frob_dist = np.linalg.norm(C_video - C_anchor, "fro")
    mr = frob_dist / anchor_norm

    # Windowed distances for TV and DSR
    window_size = max(4, len(video_lms) // 3)
    stride = max(2, window_size // 2)
    T = len(video_lms)

    window_dists = []
    t = 0
    while t + window_size <= T:
        window_lms = video_lms[t:t + window_size]
        C_w = compute_coupling_matrix(window_lms)
        d = np.linalg.norm(C_w - C_anchor, "fro") / anchor_norm
        window_dists.append(d)
        t += stride

    if not window_dists:
        window_dists = [mr]

    dists = np.array(window_dists)
    tv = float(np.std(dists))
    mu = float(np.mean(dists))
    sig = tv + 1e-8
    dsr = float(np.mean(dists > mu + 2 * sig))

    return {"MR": float(mr), "TV": tv, "DSR": dsr, "available": True}
