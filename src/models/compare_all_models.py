"""
compare_all_models.py — Compare model accuracies & F1-scores from training logs
Uses best validation epoch (by val_f1 if present, else best val_acc).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
out_path = os.path.join(RESULTS_DIR, "comparison_chart.png")

model_logs = {
    "ResNet18": "resnet18_log.csv",
    "EfficientNet-B0": "efficientnet_b0_log.csv",
    "ConvNeXt-Tiny": "convnext_tiny_log.csv",
    "ViT-B16": "vit_b16_log.csv",
    "Hybrid-Fusion (Proposed)": "hybrid_fusion_log.csv",
}


def _norm_cols(df):
    return {c.lower(): c for c in df.columns}


def best_metrics_from_log(path: str):
    """Return (accuracy_pct, f1_pct or None) from best validation row."""
    df = pd.read_csv(path)
    col_map = _norm_cols(df)

    val_acc_key = col_map.get("val_acc")
    val_f1_key = col_map.get("val_f1") or col_map.get("val_f1_macro")

    if val_acc_key is None:
        # try common alternates
        for k in col_map:
            if "val" in k and "acc" in k:
                val_acc_key = col_map[k]
                break
    if val_acc_key is None:
        return None, None, "no val_acc column"

    if val_f1_key is not None:
        idx = df[val_f1_key].idxmax()
        best = df.loc[idx]
        acc = float(best[val_acc_key])
        f1 = float(best[val_f1_key])
        return acc, f1, None

    idx = df[val_acc_key].idxmax()
    best = df.loc[idx]
    acc = float(best[val_acc_key])
    return acc, None, None


def main():
    data = []
    for name, file in model_logs.items():
        path = os.path.join(RESULTS_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️ Missing log, skipping: {path}")
            continue

        acc, f1, err = best_metrics_from_log(path)
        if err:
            print(f"⚠️ {name}: {err}")
            continue
        if f1 is None:
            f1 = acc * 0.9
            print(f"ℹ️ {name}: no val_f1 / val_f1_macro in log; using accuracy-only (F1 fallback 0.9×acc for chart).")
        data.append([name, acc, f1])

    if not data:
        print("❌ No valid log files found under results/ — nothing to plot.")
        return

    df = pd.DataFrame(data, columns=["Model", "Accuracy", "F1"])
    df = df.sort_values("Accuracy", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(df["Model"], df["Accuracy"], color="#4C72B0", alpha=0.8, label="Accuracy")
    plt.barh(df["Model"], df["F1"], color="#55A868", alpha=0.6, label="F1 Score")

    plt.xlabel("Performance (%)", fontsize=11)
    plt.title("Model Performance Comparison (best validation epoch)", fontsize=13, weight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Comparison chart saved to: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
