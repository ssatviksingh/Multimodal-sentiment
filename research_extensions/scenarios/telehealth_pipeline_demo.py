"""
telehealth_pipeline_demo.py

End-to-end simulation of a telehealth emotional wellbeing monitoring
pipeline using the existing hybrid fusion model as the backend engine.

This script:
- Segments a video/audio/text example into fixed time windows
- Uses pre-extracted modality embeddings from data/features/
- Runs the HybridFusion model per segment
- Converts sentiment predictions into wellbeing risk states
- Saves a timeline plot and a JSON-like summary of the session

Note: This is a high-level demo and assumes that:
- Features were extracted using extract_pretrained_embeddings.py
- A compatible manifest CSV exists describing filename and label
"""

import os
import json
from typing import List, Tuple

import torch
import pandas as pd

from src.models.fusion_variants.hybrid_fusion_vit_ast import (
    HybridFusionModel,
)

from .wellbeing_state_mapping import (
    WindowPrediction,
    derive_risk_sequence,
    summarize_risk,
)
from .timeline_visualization import plot_risk_timeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = "data/features"
MANIFEST = "data/custom/manifest_train_expanded.csv"
RESULTS_DIR = os.path.join("research_extensions", "results")
DEMO_OUT_DIR = os.path.join(RESULTS_DIR, "telehealth_demo")

MODEL_PATH = "results/hybrid_fusion_best.pt"


def load_session_example(manifest_path: str, session_id: str) -> pd.DataFrame:
    """
    Filter the expanded manifest for a single session/video.
    We group by audio_path as a proxy for underlying clip.
    """
    df = pd.read_csv(manifest_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "audio_path" not in df.columns:
        raise ValueError("Expected 'audio_path' column in manifest for grouping.")
    session_rows = df[df["audio_path"] == session_id].reset_index(drop=True)
    if session_rows.empty:
        raise ValueError(f"No rows found for audio_path={session_id}")
    return session_rows


def load_features_for_rows(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load text/audio/video embeddings for each filename in the given DataFrame.
    Assumes .pt features were precomputed and match 'filename' ids.
    """
    t_list: List[torch.Tensor] = []
    a_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []

    for _, row in df.iterrows():
        sid = str(row["filename"])
        t = torch.load(os.path.join(FEATURE_DIR, "text", f"{sid}.pt")).float().squeeze()
        a = torch.load(os.path.join(FEATURE_DIR, "audio", f"{sid}.pt")).float().squeeze()
        v = torch.load(os.path.join(FEATURE_DIR, "video", f"{sid}.pt")).float().squeeze()

        if t.dim() > 1:
            t = t.mean(dim=0)
        if a.dim() > 1:
            a = a.mean(dim=0)
        if v.dim() > 1:
            v = v.mean(dim=0)

        t_list.append(t)
        a_list.append(a)
        v_list.append(v)

    T = torch.stack(t_list, dim=0)  # (N, D_t)
    A = torch.stack(a_list, dim=0)  # (N, D_a)
    V = torch.stack(v_list, dim=0)  # (N, D_v)
    return T, A, V


def run_session_demo(session_audio_path: str, window_duration: float = 10.0) -> None:
    """
    Run the full demo on a single telehealth-like session identified
    by its audio_path in the expanded manifest (as a proxy for session id).
    """
    os.makedirs(DEMO_OUT_DIR, exist_ok=True)

    df_session = load_session_example(MANIFEST, session_audio_path)
    T, A, V = load_features_for_rows(df_session)

    # Infer embedding dims
    text_dim = T.shape[-1]
    audio_dim = A.shape[-1]
    video_dim = V.shape[-1]

    # Load model
    model = HybridFusionModel(text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Forward pass per "segment" (row); here each row is treated as a window.
    with torch.no_grad():
        logits = model(T.to(DEVICE), A.to(DEVICE), V.to(DEVICE))
        probs = torch.softmax(logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1)

    # Map numeric labels to sentiment categories (assumes 0=neg,1=neu,2=pos)
    idx_to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}

    window_preds: List[WindowPrediction] = []
    for i, (p_idx, p_vec) in enumerate(zip(preds, probs)):
        sentiment = idx_to_sentiment[int(p_idx)]
        confidence = float(p_vec[int(p_idx)])
        start_t = i * window_duration
        end_t = (i + 1) * window_duration
        window_preds.append(
            WindowPrediction(
                start_time=start_t,
                end_time=end_t,
                sentiment=sentiment,  # type: ignore[arg-type]
                confidence=confidence,
            )
        )

    # Derive wellbeing / risk states
    risk_states = derive_risk_sequence(window_preds)
    calm_pct, mild_pct, high_pct = summarize_risk(risk_states)

    # Timeline plot
    mid_times = [(w.start_time + w.end_time) / 2.0 for w in window_preds]
    timeline_path = os.path.join(DEMO_OUT_DIR, "telehealth_risk_timeline.png")
    plot_risk_timeline(mid_times, risk_states, timeline_path)

    # Session summary JSON
    summary = {
        "session_audio_path": session_audio_path,
        "num_windows": len(window_preds),
        "risk_distribution": {
            "calm_percent": calm_pct,
            "mild_concern_percent": mild_pct,
            "high_concern_percent": high_pct,
        },
        "windows": [
            {
                "start_time": w.start_time,
                "end_time": w.end_time,
                "sentiment": w.sentiment,
                "confidence": w.confidence,
                "risk_state": rs,
            }
            for w, rs in zip(window_preds, risk_states)
        ],
    }

    summary_path = os.path.join(DEMO_OUT_DIR, "telehealth_session_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved telehealth session summary → {summary_path}")


def main():
    # For demo purposes, pick the first unique audio_path from the expanded manifest.
    df = pd.read_csv(MANIFEST)
    df.columns = [c.strip().lower() for c in df.columns]
    if "audio_path" not in df.columns:
        raise ValueError("Manifest must contain 'audio_path' column.")
    first_session = str(df["audio_path"].iloc[0])
    print(f"🔍 Running telehealth pipeline demo for session: {first_session}")
    run_session_demo(first_session)


if __name__ == "__main__":
    main()

