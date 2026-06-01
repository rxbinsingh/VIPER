"""
VIPER — displacement_probe.py
Computes the three displacement residuals per frame/window.

The displacement reaction:  AB + C → AC + B
  C   = identity anchor (geometric + texture + biomech)
  AB  = video frame (face B in context A)
  r   = residual energy when C tries to bond with AB

Three residuals:
  GIR(t) — Geometric Identity Residual    (ArcFace cosine distance)
  TFR(t) — Texture Frequency Residual     (KL divergence of DCT profiles)
  BCR(t) — Biomechanical Coupling Residual (Frobenius distance of coupling matrices)
"""

import numpy as np
from scipy.special import rel_entr
from typing import Optional
from .anchor_extractor import (
    compute_dct_profile,
    compute_landmark_displacements,
    build_biomech_anchor,
)


# ── Signal 1: Geometric Identity Residual ────────────────────────────────────

def compute_gir_sequence(
    video_embeddings: list[np.ndarray],
    geometric_anchor: np.ndarray,
) -> np.ndarray:
    """
    Compute GIR(t) = 1 - cosine_similarity(ArcFace(frame_t), anchor)
    for each frame.

    ArcFace embeddings are already L2-normalized by InsightFace,
    so cosine similarity = dot product.

    Returns (T,) array of residuals in [0, 2].
    Typical values:
      Same person:      0.05 – 0.35
      Different person: 0.55 – 1.20
    """
    anchor = geometric_anchor / (np.linalg.norm(geometric_anchor) + 1e-8)
    residuals = []

    for emb in video_embeddings:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        cos_sim  = float(np.dot(emb_norm, anchor))
        gir      = 1.0 - cos_sim  # 0 = identical, 2 = opposite
        residuals.append(gir)

    return np.array(residuals, dtype=np.float32)


# ── Signal 2: Texture Frequency Residual ─────────────────────────────────────

def compute_tfr_sequence(
    video_frames: list[np.ndarray],
    texture_anchor: np.ndarray,
    n_bins: int = 64,
) -> np.ndarray:
    """
    Compute TFR(t) = KL_divergence(DCT_profile(frame_t) || texture_anchor)
    for each frame.

    KL divergence measures how different the frame's texture frequency
    distribution is from the anchor's distribution.

    Returns (T,) array of residuals >= 0.
    Typical values:
      Same person:      0.001 – 0.05
      Different texture: 0.05 – 0.30
    """
    # Add small epsilon to avoid log(0)
    anchor = texture_anchor + 1e-8
    anchor = anchor / anchor.sum()

    residuals = []
    for crop in video_frames:
        profile = compute_dct_profile(crop, n_bins=n_bins)
        profile = profile + 1e-8
        profile = profile / profile.sum()

        # KL divergence: sum(p * log(p/q))
        kl = float(np.sum(rel_entr(profile, anchor)))
        residuals.append(kl)

    return np.array(residuals, dtype=np.float32)


# ── Signal 3: Biomechanical Coupling Residual ─────────────────────────────────

def compute_bcr_sequence(
    raw_frames: list[np.ndarray],
    biomech_anchor: Optional[np.ndarray],
    window_size: int = 8,
    stride: int = 4,
) -> Optional[np.ndarray]:
    """
    Compute BCR(t) = ||C_window(t) - C_anchor||_F / ||C_anchor||_F
    using a sliding window over the video frames.

    The coupling matrix C[i,j] encodes how landmark i's movement
    correlates with landmark j's movement — a person-specific biomechanical
    signature determined by their muscle anatomy.

    Returns (W,) array of residuals where W = number of windows,
    or None if MediaPipe is unavailable or too few landmarks detected.
    """
    if biomech_anchor is None:
        return None

    displacements = compute_landmark_displacements(raw_frames)
    if displacements is None or displacements.shape[0] < window_size:
        return None

    T = displacements.shape[0]
    anchor_norm = np.linalg.norm(biomech_anchor, "fro") + 1e-8

    residuals = []
    t = 0
    while t + window_size <= T:
        window = displacements[t : t + window_size]  # (W, 136)
        C_window = np.corrcoef(window.T)              # (136, 136)
        C_window = np.nan_to_num(C_window, nan=0.0)

        frob_dist = np.linalg.norm(C_window - biomech_anchor, "fro")
        bcr = frob_dist / anchor_norm
        residuals.append(bcr)
        t += stride

    if not residuals:
        return None

    return np.array(residuals, dtype=np.float32)


# ── Reaction curve statistics ─────────────────────────────────────────────────

def reaction_curve_stats(residuals: np.ndarray) -> dict:
    """
    Compute the three statistics from a residual sequence R = [r(1)...r(T)].

    MR  — Mean Residual: average displacement energy
    TV  — Temporal Variance: instability of the reaction
    DSR — Displacement Spike Ratio: fraction of anomalous frames

    These map to the displacement reaction:
      MR  = total reaction energy (high = fake)
      TV  = reaction stability   (high = fake — generator sampling noise)
      DSR = fraction of failed bonds (high = fake)
    """
    if len(residuals) == 0:
        return {"MR": 0.5, "TV": 0.5, "DSR": 0.5}

    mr  = float(np.mean(residuals))
    tv  = float(np.std(residuals))
    mu  = mr
    sig = tv + 1e-8
    dsr = float(np.mean(residuals > mu + 2 * sig))

    return {"MR": mr, "TV": tv, "DSR": dsr}


# ── Full displacement probe ───────────────────────────────────────────────────

def compute_all_residuals(
    preprocessed: dict,
    anchor: dict,
) -> dict:
    """
    Compute all three residual sequences for a video.

    Returns dict with:
        gir_seq   : (T,) GIR per frame
        tfr_seq   : (T,) TFR per frame
        bcr_seq   : (W,) BCR per window, or None
        gir_stats : {MR, TV, DSR} for GIR
        tfr_stats : {MR, TV, DSR} for TFR
        bcr_stats : {MR, TV, DSR} for BCR, or None
    """
    result = {}

    # GIR
    gir_seq = compute_gir_sequence(
        preprocessed["video_embeddings"],
        anchor["geometric_anchor"],
    )
    result["gir_seq"]   = gir_seq
    result["gir_stats"] = reaction_curve_stats(gir_seq)

    # TFR
    tfr_seq = compute_tfr_sequence(
        preprocessed["video_frames"],
        anchor["texture_anchor"],
    )
    result["tfr_seq"]   = tfr_seq
    result["tfr_stats"] = reaction_curve_stats(tfr_seq)

    # BCR
    bcr_seq = compute_bcr_sequence(
        preprocessed["raw_frames"],
        anchor["biomech_anchor"],
    )
    result["bcr_seq"]   = bcr_seq
    result["bcr_stats"] = reaction_curve_stats(bcr_seq) if bcr_seq is not None else None

    return result
