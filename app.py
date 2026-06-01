"""
VIPER — app.py
Gradio demo for deepfake detection.
Same UI pattern as SynID.

Run locally:  python app.py
HuggingFace:  deploy this file as app.py on a Space
"""

import gradio as gr
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os

from src.viper_complete import VIPERDetector

# ── Load detector ─────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/viper_best.pt"
detector   = VIPERDetector(checkpoint=CHECKPOINT if os.path.exists(CHECKPOINT) else None)


# ── Reaction curve plot ───────────────────────────────────────────────────────

def plot_reaction_curve(gir_seq: list, tfr_seq: list, prediction: str) -> str:
    """Plot GIR and TFR sequences as the 'reaction curve'."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    color = "#d62728" if prediction == "FAKE" else "#2ca02c"

    for ax, seq, title, threshold in zip(
        axes,
        [gir_seq, tfr_seq],
        ["Geometric Identity Residual (GIR)", "Texture Frequency Residual (TFR)"],
        [0.35, 0.08],
    ):
        frames = list(range(len(seq)))
        ax.plot(frames, seq, color=color, linewidth=2, label="Residual")
        ax.axhline(threshold, color="gray", linestyle="--",
                   linewidth=1.2, label=f"Threshold ({threshold})")
        ax.fill_between(frames, seq, threshold,
                        where=[s > threshold for s in seq],
                        alpha=0.25, color=color, label="Violation")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"VIPER Reaction Curve — {prediction}",
        fontsize=12, fontweight="bold",
        color=color,
    )
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close()
    return tmp.name


# ── Detection function ────────────────────────────────────────────────────────

def detect(video_path: str) -> tuple:
    if video_path is None:
        return "Please upload a video.", None, None

    try:
        result = detector.detect(video_path)
    except Exception as e:
        return f"Error: {str(e)}", None, None

    if result["prediction"] == "UNKNOWN":
        return "Could not detect faces in this video.", None, None

    # Format output
    pred   = result["prediction"]
    conf   = result["confidence"] * 100
    score  = result["viper_score"]
    emoji  = "🔴 FAKE" if pred == "FAKE" else "🟢 REAL"

    summary = f"""## {emoji}

**Confidence:** {conf:.1f}%
**VIPER Score:** {score:.4f}  *(>0.5 = FAKE)*
**Frames Analyzed:** {result['frames_analyzed']}
**Anchor Quality:** {result['anchor_quality']:.3f}

---

### Signal Breakdown

| Signal | Score | Triggered |
|--------|-------|-----------|"""

    for name, sig in result["signals"].items():
        if sig["score"] is not None:
            triggered = "⚠️ Yes" if sig["triggered"] else "✅ No"
            summary += f"\n| {name.title()} | {sig['score']:.4f} | {triggered} |"

    summary += f"""

---

### Displacement Reaction
`AB + C → AC + B`
- **A** = video context
- **B** = face in video  
- **C** = identity anchor (first {result.get('n_anchor', 8)} frames)

{'The anchor **failed to bond** with the face — identity displaced. **FAKE detected.**' if pred == 'FAKE' else 'The anchor **bonded successfully** with all frames. **REAL video.**'}
"""

    # Reaction curve plot
    plot_path = None
    if "gir_sequence" in result and "tfr_sequence" in result:
        plot_path = plot_reaction_curve(
            result["gir_sequence"],
            result["tfr_sequence"],
            pred,
        )

    # JSON details
    details = json.dumps({
        k: v for k, v in result.items()
        if k not in ["gir_sequence", "tfr_sequence"]
    }, indent=2)

    return summary, plot_path, details


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="VIPER — Deepfake Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔬 VIPER — Video Identity Perturbation and Extraction Residual
    **Deepfake detection via biometric identity consistency analysis.**

    Upload a video to detect whether it contains a deepfake face.
    VIPER measures three biological invariants: **geometry** (ArcFace),
    **texture** (DCT frequency), and **biomechanics** (facial landmark coupling).

    *Based on the displacement reaction principle: AB + C → AC + B*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", height=300)
            detect_btn  = gr.Button("🔍 Detect Deepfake", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_md = gr.Markdown(label="Result")

    with gr.Row():
        plot_out   = gr.Image(label="Reaction Curve", type="filepath")
        detail_out = gr.Code(label="Full Result (JSON)", language="json")

    detect_btn.click(
        fn=detect,
        inputs=[video_input],
        outputs=[result_md, plot_out, detail_out],
    )

    gr.Markdown("""
    ---
    **Robin Singh** · Bennett University · 2025
    | [SynID](https://huggingface.co/rxbinsingh/SynID)
    | [GHOST](https://huggingface.co/rxbinsingh/GHOST)
    | [GitHub](https://github.com/rxbinsingh)
    """)


if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
