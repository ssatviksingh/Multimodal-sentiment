"""
compare_results.py
Compare validation accuracy & F1 across fusion strategy logs (weighted / transformer / hybrid).
Skips missing files with a warning; does not insert placeholder zeros for missing models.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

models = {
    "Weighted Fusion": os.path.join(RESULTS_DIR, "weighted_fusion_log.csv"),
    "Transformer Fusion": os.path.join(RESULTS_DIR, "transformer_fusion_log.csv"),
    "Hybrid Fusion (ViT+AST)": os.path.join(RESULTS_DIR, "hybrid_fusion_log.csv"),
}


def _col_map(df):
    return {c.lower(): c for c in df.columns}


def best_row_metrics(df):
    cmap = _col_map(df)
    val_acc_col = cmap.get("val_acc")
    val_f1_col = cmap.get("val_f1") or cmap.get("val_f1_macro")
    if not val_f1_col and "f1" in cmap:
        val_f1_col = cmap["f1"]

    if not val_acc_col:
        for low, orig in cmap.items():
            if "val" in low and "acc" in low:
                val_acc_col = orig
                break
    if not val_acc_col:
        return None

    if val_f1_col:
        try:
            idx = df[val_f1_col].astype(float).idxmax()
        except (ValueError, TypeError):
            idx = df[val_acc_col].astype(float).idxmax()
        best = df.loc[idx]
        return float(best[val_acc_col]), float(best[val_f1_col])

    idx = df[val_acc_col].astype(float).idxmax()
    best = df.loc[idx]
    return float(best[val_acc_col]), None


def main():
    summary = []

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"⚠️ Missing fusion log (skipped): {path}", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ Empty CSV: {path}", file=sys.stderr)
            continue

        metrics = best_row_metrics(df)
        if metrics is None:
            print(f"⚠️ No usable val columns in {name}, skipping.", file=sys.stderr)
            continue

        val_acc, val_f1 = metrics
        summary.append({
            "Model": name,
            "Val_Accuracy": val_acc,
            "Val_F1": val_f1 if val_f1 is not None else pd.NA,
        })

    if not summary:
        print("❌ No fusion logs found — nothing to compare. Train weighted/transformer/hybrid fusion first.")
        return

    summary_df = pd.DataFrame(summary)
    out_csv = os.path.join(RESULTS_DIR, "accuracy_progression.csv")
    summary_df.to_csv(out_csv, index=False)
    print("✅ Saved summary →", out_csv)
    print(summary_df.to_string(index=False))

    # Plot only if we have numeric F1 for all rows or we can plot acc only
    plt.figure(figsize=(8, 5))
    x = summary_df["Model"]
    acc = summary_df["Val_Accuracy"]
    has_f1 = summary_df["Val_F1"].notna().all()

    indices = range(len(x))
    if has_f1:
        f1 = summary_df["Val_F1"].astype(float)
        bar_width = 0.35
        plt.bar(indices, acc, width=bar_width, label="Validation accuracy (%)", alpha=0.8)
        plt.bar([i + bar_width for i in indices], f1, width=bar_width, label="Val F1 (%)", alpha=0.8, hatch="//")
        plt.xticks([i + bar_width / 2 for i in indices], x, rotation=20)
    else:
        plt.bar(indices, acc, label="Validation accuracy (%)", alpha=0.8)
        plt.xticks(indices, x, rotation=20)

    plt.title("Fusion strategies (best validation epoch)")
    plt.ylabel("Metric (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "accuracy_growth_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved comparison chart → {save_path}")


if __name__ == "__main__":
    main()
