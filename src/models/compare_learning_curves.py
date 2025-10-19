"""
compare_learning_curves.py
Generates side-by-side plots for Validation Accuracy & F1 Score with annotations.
Perfect for paper/report visualization.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

models = {
    "Weighted Fusion": os.path.join(RESULTS_DIR, "weighted_fusion_log.csv"),
    "Transformer Fusion": os.path.join(RESULTS_DIR, "transformer_fusion_log.csv"),
    "Hybrid Fusion (ViT-AST)": os.path.join(RESULTS_DIR, "hybrid_fusion_log.csv"),
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- Create figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax_acc, ax_f1 = axes

for (name, path), color in zip(models.items(), colors):
    if not os.path.exists(path):
        print(f"⚠️ Missing log file: {path}")
        continue

    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]

    val_acc_col = next((c for c in df.columns if "val_acc" in c.lower()), None)
    val_f1_col = next((c for c in df.columns if "val_f1" in c.lower()), None)

    if val_acc_col is None:
        print(f"⚠️ Skipping {name}: no validation accuracy column.")
        continue

    epoch_col = df["epoch"].values if "epoch" in df.columns else range(1, len(df) + 1)

    # --- Accuracy Plot ---
    acc_values = df[val_acc_col].values
    ax_acc.plot(epoch_col, acc_values, label=name, color=color, linewidth=2)
    # Annotate final accuracy
    ax_acc.text(epoch_col[-1] + 0.2, acc_values[-1],
                f"{acc_values[-1]:.1f}%", color=color, fontsize=9, va='center')

    # --- F1 Plot ---
    if val_f1_col:
        f1_values = df[val_f1_col].values
        ax_f1.plot(epoch_col, f1_values, label=name, color=color, linewidth=2, linestyle='--')
        # Annotate final F1 score
        ax_f1.text(epoch_col[-1] + 0.2, f1_values[-1],
                   f"{f1_values[-1]:.1f}%", color=color, fontsize=9, va='center')
    else:
        print(f"⚠️ No F1 column found for {name}")

# --- Styling Left Plot (Accuracy) ---
ax_acc.set_title("Validation Accuracy per Epoch", fontsize=11)
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.grid(alpha=0.4)
ax_acc.legend()

# --- Styling Right Plot (F1) ---
ax_f1.set_title("Validation F1 Score per Epoch", fontsize=11)
ax_f1.set_xlabel("Epoch")
ax_f1.set_ylabel("F1 Score (%)")
ax_f1.grid(alpha=0.4)
ax_f1.legend()

# --- Global Title ---
plt.suptitle("📈 Multimodal Sentiment Analysis — Model Learning Curves with Final Performance", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

save_path = os.path.join(RESULTS_DIR, "learning_curves_annotated.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved annotated side-by-side learning curves → {save_path}")
