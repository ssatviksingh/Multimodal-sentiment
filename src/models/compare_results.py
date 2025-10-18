"""
compare_results.py
Compare model accuracies & F1 across different fusion strategies.
Auto-handles missing columns and generates a clean chart.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

models = {
    "weighted_fusion": os.path.join(RESULTS_DIR, "weighted_fusion_log.csv"),
    "transformer_fusion": os.path.join(RESULTS_DIR, "transformer_fusion_log.csv"),
    "hybrid_fusion_vit_ast": os.path.join(RESULTS_DIR, "hybrid_fusion_log.csv"),
}

summary = []

for name, path in models.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
        continue

    df = pd.read_csv(path)
    cols = df.columns.str.lower()

    # Find available metric columns
    val_acc_col = next((c for c in cols if "val_acc" in c), None)
    val_f1_col = next((c for c in cols if "val_f1" in c), None)

    if val_acc_col is None:
        print(f"⚠️ No val_acc in {name}, skipping.")
        continue

    # Get best row (by F1 if present, else by val_acc)
    if val_f1_col and val_f1_col in df.columns:
        best = df.iloc[df[val_f1_col].idxmax()]
        val_acc = best[val_acc_col]
        val_f1 = best[val_f1_col]
    else:
        best = df.iloc[df[val_acc_col].idxmax()]
        val_acc = best[val_acc_col]
        val_f1 = None

    summary.append({
        "Model": name,
        "Val_Accuracy": val_acc,
        "F1": val_f1 if val_f1 is not None else 0.0
    })

# Save summary CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_DIR, "accuracy_progression.csv"), index=False)
print("✅ Saved summary → results/accuracy_progression.csv")
print(summary_df)

# ---- Visualization ----
plt.figure(figsize=(8, 5))
x = summary_df["Model"]
acc = summary_df["Val_Accuracy"]
f1 = summary_df["F1"]

bar_width = 0.35
indices = range(len(x))

plt.bar(indices, acc, width=bar_width, label="Validation Accuracy (%)", alpha=0.8)
plt.bar([i + bar_width for i in indices], f1, width=bar_width, label="F1 Score (%)", alpha=0.8, hatch="//")

plt.xticks([i + bar_width / 2 for i in indices], x, rotation=20)
plt.title("Model Performance Comparison")
plt.ylabel("Metric (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "accuracy_growth_comparison.png")
plt.savefig(save_path)
plt.show()

print(f"📊 Saved comparison chart → {save_path}")
