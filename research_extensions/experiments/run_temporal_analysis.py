"""
run_temporal_analysis.py

Temporal analysis of session-level emotion:
- Uses the expanded manifest as a proxy for time windows
- Runs the HybridFusion model per window
- Saves per-window predictions and basic trend statistics
"""

import os
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.models.fusion_variants.hybrid_fusion_vit_ast import (
    MultimodalDataset,
    HybridFusionModel,
    DEVICE,
)


DATA_DIR = "data/features"
MANIFEST = "data/custom/manifest_train_expanded.csv"
RESULTS_DIR = os.path.join("research_extensions", "results")
OUT_JSON = os.path.join(RESULTS_DIR, "temporal_predictions.jsonl")

MODEL_PATH = "results/hybrid_fusion_best.pt"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    df.columns = [c.strip().lower() for c in df.columns]

    # Group rows by underlying clip (audio_path)
    if "audio_path" not in df.columns:
        raise ValueError("Expected 'audio_path' column in manifest.")
    groups = df.groupby("audio_path")

    dataset = MultimodalDataset(MANIFEST, DATA_DIR)

    # Infer dims
    t_sample, a_sample, v_sample, _ = dataset[0]
    text_dim, audio_dim, video_dim = t_sample.shape[-1], a_sample.shape[-1], v_sample.shape[-1]

    model = HybridFusionModel(text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    idx_to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}

    with open(OUT_JSON, "w", encoding="utf-8") as f_out:
        for audio_path, group in groups:
            indices: List[int] = group.index.tolist()
            # build a mini-loader over this session
            loader = DataLoader(
                torch.utils.data.Subset(dataset, indices),
                batch_size=8,
                shuffle=False,
            )

            window_preds: List[Dict[str, Any]] = []
            order = 0
            with torch.no_grad():
                for t, a, v, lbl in loader:
                    t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
                    out = model(t, a, v)
                    probs = torch.softmax(out, dim=-1).cpu()
                    preds = probs.argmax(dim=-1)

                    for p_idx, p_vec, l in zip(preds, probs, lbl.cpu()):
                        sentiment = idx_to_sentiment[int(p_idx)]
                        conf = float(p_vec[int(p_idx)])
                        window_preds.append(
                            {
                                "order": order,
                                "sentiment": sentiment,
                                "confidence": conf,
                                "label": int(l),
                            }
                        )
                        order += 1

            record = {
                "audio_path": audio_path,
                "num_windows": len(window_preds),
                "windows": window_preds,
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"✅ Saved temporal predictions → {OUT_JSON}")


if __name__ == "__main__":
    main()

