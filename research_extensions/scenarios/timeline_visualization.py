"""
timeline_visualization.py

Plot utilities for visualizing wellbeing / risk state over time
for a simulated telehealth session.
"""

import os
from typing import List

import matplotlib.pyplot as plt

from .wellbeing_state_mapping import RiskState


def plot_risk_timeline(
    times: List[float],
    states: List[RiskState],
    out_path: str,
) -> None:
    """
    Plot discrete risk states over time as a step-wise timeline.

    Args:
        times: list of window midpoints in seconds.
        states: list of risk states for each window.
        out_path: where to save the PNG figure.
    """
    assert len(times) == len(states), "times and states must have same length"

    # Map states to numeric levels for visualization
    level_map = {"Calm": 0, "Mild Concern": 1, "High Concern": 2}
    y = [level_map[s] for s in states]

    plt.figure(figsize=(10, 3))
    plt.step(times, y, where="mid", linewidth=2)
    plt.yticks(
        [0, 1, 2],
        ["Calm", "Mild Concern", "High Concern"],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Risk Level")
    plt.title("Telehealth Session – Emotional Risk Timeline")
    plt.ylim(-0.5, 2.5)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved risk timeline → {out_path}")

