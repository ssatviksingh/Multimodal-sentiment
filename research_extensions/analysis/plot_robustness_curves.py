"""
plot_robustness_curves.py

Visualize robustness study results:
- Accuracy / Macro-F1 vs. condition
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join("research_extensions", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "robustness_results.csv")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Robustness CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    plt.figure(figsize=(9, 4))
    x = range(len(df))
    plt.plot(x, df["accuracy"], marker="o", label="Accuracy (%)")
    plt.plot(x, df["macro_f1"], marker="s", label="Macro-F1 (%)")
    plt.xticks(x, df["condition"], rotation=30, ha="right")
    plt.ylabel("Metric (%)")
    plt.title("Robustness under Telehealth-like Degradations")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "robustness_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved robustness curves → {out_path}")


if __name__ == "__main__":
    main()

