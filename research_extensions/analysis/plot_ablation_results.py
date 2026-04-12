"""
plot_ablation_results.py

Utility script to visualize the ablation study results.

It:
- Loads research_extensions/results/ablation_summary.csv
- Plots:
    1) Bar chart comparing validation Accuracy for all models
    2) Bar chart comparing validation Macro-F1 for all models
- Saves the figures as PNG files in research_extensions/results/
- Prints a nicely formatted table of the metrics to the console
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join("research_extensions", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "ablation_summary.csv")


def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary CSV not found at: {path}")
    df = pd.read_csv(path)
    # Ensure expected columns exist
    required_cols = [
        "experiment_name",
        "model_type",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_macro_f1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in summary CSV: {missing}")
    return df


def plot_metric_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    labels = df["experiment_name"]
    values = df[metric]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="#4C72B0", alpha=0.85)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with values (rounded to 2 decimals)
    for bar, v in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot → {out_path}")


def print_table(df: pd.DataFrame) -> None:
    # Reorder / subset columns for readability
    cols = [
        "experiment_name",
        "model_type",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_macro_f1",
    ]
    view = df[cols].copy()

    # Round numeric metrics for display
    for c in ["val_accuracy", "val_precision", "val_recall", "val_macro_f1"]:
        view[c] = view[c].map(lambda x: f"{x:.2f}")

    print("\n📊 Ablation Summary:")
    print(view.to_string(index=False))


def main():
    df = load_summary(SUMMARY_CSV)

    # Plot Accuracy comparison
    plot_metric_bar(
        df,
        metric="val_accuracy",
        ylabel="Validation Accuracy (%)",
        title="Ablation Study – Validation Accuracy",
        filename="ablation_accuracy.png",
    )

    # Plot Macro-F1 comparison
    plot_metric_bar(
        df,
        metric="val_macro_f1",
        ylabel="Validation Macro-F1 (%)",
        title="Ablation Study – Validation Macro-F1",
        filename="ablation_macro_f1.png",
    )

    # Console table
    print_table(df)


if __name__ == "__main__":
    main()

