"""
plot_temporal_trajectories.py

Plot example temporal sentiment trajectories from temporal_predictions.jsonl.
"""

import os
import json
from typing import List, Dict, Any

import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join("research_extensions", "results")
PRED_PATH = os.path.join(RESULTS_DIR, "temporal_predictions.jsonl")


def load_records(n_examples: int = 3) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= n_examples:
                break
    return records


def main():
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(f"Temporal predictions not found at: {PRED_PATH}")

    records = load_records()

    for idx, rec in enumerate(records):
        windows = rec["windows"]
        steps = [w["order"] for w in windows]
        sentiments = [w["sentiment"] for w in windows]

        # Map to numeric for plotting
        mapping = {"negative": -1, "neutral": 0, "positive": 1}
        y = [mapping[s] for s in sentiments]

        plt.figure(figsize=(8, 3))
        plt.step(steps, y, where="mid")
        plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        plt.xlabel("Window index")
        plt.ylabel("Sentiment")
        plt.title(f"Temporal Sentiment Trajectory – Example {idx+1}")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"temporal_trajectory_{idx+1}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved temporal trajectory → {out_path}")


if __name__ == "__main__":
    main()

