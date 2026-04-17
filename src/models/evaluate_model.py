"""
evaluate_model.py

Loads the best saved HybridFusion model and evaluates it on the test manifest.
Outputs accuracy, precision, recall, F1 (macro + weighted), classification report,
and saves a confusion matrix heatmap + CSV report.

Only HybridFusionModel + data/features is supported.

Works directly with extracted features under data/features/.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.models.fusion_variants.hybrid_fusion_vit_ast import MultimodalDataset, load_hybrid_for_eval
from src.models.fusion_variants.training_cli import resolve_device
from src.models.fusion_variants.feature_io import dataloader_kwargs
DEFAULT_CHECKPOINT = "results/hybrid_fusion_best.pt"
DEFAULT_DATA_DIR = "data/features"
DEFAULT_TEST_MANIFEST = "data/manifest_test.csv"
OUT_DIR = "results"
DEFAULT_BATCH_SIZE = 4


def evaluate_model(
    model_path: str,
    test_manifest: str,
    data_dir: str,
    out_dir: str,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    dev = resolve_device(device)
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔍 [TEST SET] HybridFusion — checkpoint: {model_path}")
    print(f"   Device: {dev.upper()} | Manifest: {test_manifest} | Features: {data_dir}")

    test_data = MultimodalDataset(test_manifest, data_dir)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **dataloader_kwargs(dev)
    )

    model = load_hybrid_for_eval(model_path, map_location=dev)
    model.to(dev)
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for t, a, v, lbl in test_loader:
            t, a, v, lbl = t.to(dev), a.to(dev), v.to(dev), lbl.to(dev)
            out = model(t, a, v)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n📊 Test-set evaluation results (HybridFusion):")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1-score  : {f1_macro*100:.2f}%")
    print(f"   Weighted F1     : {f1_weight*100:.2f}%")
    print(f"   Macro Precision : {p_macro*100:.2f}%")
    print(f"   Macro Recall    : {r_macro*100:.2f}%")

    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(out_dir, "classification_report.csv")
    df_report.to_csv(report_csv, index=True)
    print(f"✅ Saved detailed report to: {report_csv}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - HybridFusion (test set)")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    try:
        plt.savefig(cm_path, bbox_inches="tight")
    except OSError as e:
        alt = os.path.join(out_dir, "confusion_matrix_alt.png")
        plt.savefig(os.path.normpath(os.path.abspath(alt)), bbox_inches="tight")
        print(f"⚠️ Saved confusion matrix to {alt} (primary path failed: {e})")
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")

    summary_path = os.path.join(out_dir, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Metric source: test set (manifest below)\n")
        f.write(f"Device: {dev}\n")
        f.write(f"Model checkpoint: {model_path}\n")
        f.write(f"Test manifest: {test_manifest}\n")
        f.write(f"Feature dir: {data_dir}\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"Macro F1: {f1_macro*100:.2f}%\n")
        f.write(f"Weighted F1: {f1_weight*100:.2f}%\n")
        f.write(f"Macro Precision: {p_macro*100:.2f}%\n")
        f.write(f"Macro Recall: {r_macro*100:.2f}%\n")
    print(f"📁 Summary saved to: {summary_path}")

    print("\n🎯 Evaluation completed successfully!")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate HybridFusion on the test manifest (only supported architecture)."
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to HybridFusion weights (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_TEST_MANIFEST,
        help=f"Test manifest CSV (default: {DEFAULT_TEST_MANIFEST})",
    )
    p.add_argument(
        "--features-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory with text/audio/video .pt features (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help=f"Output directory for reports (default: {OUT_DIR})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or mps (same as fusion training)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="DataLoader batch size (default matches fusion training default)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        model_path=args.checkpoint,
        test_manifest=args.manifest,
        data_dir=args.features_dir,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
