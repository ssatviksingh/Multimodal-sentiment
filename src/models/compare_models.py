"""
compare_models.py

Evaluates all *_best.pt models in the results/ directory and compares:
 - Accuracy
 - Macro F1
 - Weighted F1
 - Best Val F1 (from *_log.csv if available)

Saves results as:
 - results/model_comparison.csv
 - results/model_comparison_chart.png
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.models.fusion_variants.hybrid_fusion_vit_ast import HybridFusionModel, MultimodalDataset

# -------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/features"
TEST_MANIFEST = "data/manifest_test.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 4

# -------- EVALUATION FUNCTION ----------
def evaluate_model(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for t, a, v, lbl in dataloader:
            t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
            out = model(t, a, v)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, f1_macro * 100, f1_weighted * 100

# -------- TRAINING LOG PARSER ----------
def get_best_val_f1(model_name):
    """Finds *_log.csv for the given model and extracts best val_f1."""
    log_name = model_name.replace("_best.pt", "_log.csv")
    log_path = os.path.join(RESULTS_DIR, log_name)
    if not os.path.exists(log_path):
        return None
    try:
        df = pd.read_csv(log_path)
        if "val_f1" in df.columns:
            return float(df["val_f1"].max())
    except Exception as e:
        print(f"⚠️ Could not read {log_path}: {e}")
    return None

# -------- MAIN ----------
def main():
    print(f"🔍 Using device: {DEVICE}")
    print(f"📁 Scanning for models in: {RESULTS_DIR}")

    test_data = MultimodalDataset(TEST_MANIFEST, DATA_DIR)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    models = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_best.pt")]
    if not models:
        print("❌ No *_best.pt models found in results/.")
        return

    results = []

    for model_name in models:
        model_path = os.path.join(RESULTS_DIR, model_name)
        print(f"\n🚀 Evaluating {model_name} ...")

        model = HybridFusionModel(text_dim=768, audio_dim=768, video_dim=768)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        acc, f1_macro, f1_weighted = evaluate_model(model, test_loader)
        best_val_f1 = get_best_val_f1(model_name)

        results.append([
            model_name,
            acc,
            f1_macro,
            f1_weighted,
            best_val_f1 if best_val_f1 is not None else "N/A"
        ])
        print(f"✅ {model_name}: Accuracy={acc:.2f}% | Macro F1={f1_macro:.2f}% | Weighted F1={f1_weighted:.2f}% | Best Val F1={best_val_f1}")

    # Save to CSV
    df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Macro F1 (%)", "Weighted F1 (%)", "Best Val F1 (%)"])
    out_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n📊 Comparison results saved to: {out_csv}")

    # Plot
    plt.figure(figsize=(10, 6))
    x = range(len(df))
    plt.bar(x, df["Accuracy (%)"], width=0.2, label="Accuracy", align="center")
    plt.bar([i + 0.25 for i in x], df["Macro F1 (%)"], width=0.2, label="Macro F1", align="center")
    plt.bar([i + 0.5 for i in x], df["Weighted F1 (%)"], width=0.2, label="Weighted F1", align="center")

    plt.xticks([i + 0.25 for i in x], df["Model"], rotation=45, ha="right")
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison (Test + Training F1)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_chart.png"))
    plt.show()

    print("✅ Chart saved as results/model_comparison_chart.png")


if __name__ == "__main__":
    main()
