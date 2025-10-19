"""
evaluate_model.py

Loads the best saved HybridFusion model and evaluates it on the test set.
Outputs accuracy, precision, recall, F1 (macro + weighted), classification report,
and saves a confusion matrix heatmap + CSV report.

Works directly with your extracted features under data/features/.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.models.fusion_variants.hybrid_fusion_vit_ast import HybridFusionModel, MultimodalDataset

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/hybrid_fusion_best.pt"
DATA_DIR = "data/features"
TEST_MANIFEST = "data/manifest_test.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 4

# ---------------- EVALUATION ----------------
def evaluate_model():
    print(f"🔍 Loading model from {MODEL_PATH} on {DEVICE.upper()}")

    # Load dataset
    test_data = MultimodalDataset(TEST_MANIFEST, DATA_DIR)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model (ensure same dims as during training)
    model = HybridFusionModel(text_dim=768, audio_dim=768, video_dim=768)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for t, a, v, lbl in test_loader:
            t, a, v, lbl = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), lbl.to(DEVICE)
            out = model(t, a, v)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n📊 Evaluation Results:")
    print(f"Accuracy       : {acc:.2f}%")
    print(f"Macro F1-score : {f1_macro*100:.2f}%")
    print(f"Weighted F1    : {f1_weight*100:.2f}%")
    print(f"Macro Precision: {p_macro*100:.2f}%")
    print(f"Macro Recall   : {r_macro*100:.2f}%")

    # Classification Report
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(OUT_DIR, "classification_report.csv")
    df_report.to_csv(report_csv, index=True)
    print(f"✅ Saved detailed report to: {report_csv}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - HybridFusion")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")

    # Summary log
    summary_path = os.path.join(OUT_DIR, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"Macro F1: {f1_macro*100:.2f}%\n")
        f.write(f"Weighted F1: {f1_weight*100:.2f}%\n")
        f.write(f"Macro Precision: {p_macro*100:.2f}%\n")
        f.write(f"Macro Recall: {r_macro*100:.2f}%\n")
    print(f"📁 Summary saved to: {summary_path}")

    print("\n🎯 Evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_model()
