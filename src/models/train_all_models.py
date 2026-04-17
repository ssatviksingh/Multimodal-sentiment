"""
train_all_models.py

Unified multimodal benchmark pipeline.
Trains ResNet18, EfficientNet-B0, ConvNeXt-Tiny, ViT, HybridFusion.
Auto-detects already trained models (unless --force-retrain).
Runs full test-set evaluation only for hybrid fusion (evaluate_model supports HybridFusion only).
"""

import argparse
import os
import sys
import subprocess
import time
import torch

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running benchmark pipeline on {DEVICE.upper()}")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "resnet18": "src.models.cnn_variants.resnet18_train",
    "efficientnet_b0": "src.models.cnn_variants.efficientnet_b0_train",
    "convnext_tiny": "src.models.cnn_variants.convnext_tiny_train",
    "vit_b16": "src.models.transformer_variants.vit_b16_train",
    "hybrid_fusion": "src.models.fusion_variants.hybrid_fusion_vit_ast",
}

EVAL_SCRIPT = "src.models.evaluate_model"
COMPARE_SCRIPT = "src.models.compare_all_models"


def run_script(module_name, desc):
    """Run a python module via subprocess"""
    print(f"\n⚙️ Running {desc} → {module_name}")
    start = time.time()
    try:
        subprocess.run([sys.executable, "-m", module_name], check=True)
        print(f"✅ {desc} finished in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {desc}: {e}")
        raise


def model_checkpoint_exists(name):
    checkpoint_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")
    return os.path.exists(checkpoint_path)


def parse_args():
    p = argparse.ArgumentParser(description="Train all benchmark models; test eval runs for hybrid only.")
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing results/*_best.pt and train every model from scratch.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("\n📚 Starting multimodal benchmark pipeline")
    if args.force_retrain:
        print("🔁 --force-retrain: will not skip training when checkpoints exist.")
    print("--------------------------------------------------")

    for name, module in MODELS.items():
        ckpt = os.path.join(RESULTS_DIR, f"{name}_best.pt")
        should_train = args.force_retrain or not model_checkpoint_exists(name)

        if should_train:
            print(f"\n🏋️ Training {name.upper()} model...")
            run_script(module, f"{name} training")
        else:
            print(f"⏩ Skipping {name.upper()} (checkpoint found: {ckpt})")

        if name == "hybrid_fusion":
            print(f"\n🧪 Test-set evaluation (HybridFusion only) → {EVAL_SCRIPT}")
            run_script(EVAL_SCRIPT, "hybrid fusion test evaluation")
        else:
            print(
                f"\n📋 {name.upper()}: test metrics via evaluate_model are not defined for this baseline "
                f"(only HybridFusion). See validation metrics in results/{name}_log.csv."
            )

    print("\n📈 Generating performance comparison chart...")
    run_script(COMPARE_SCRIPT, "comparison graph")

    print("\n🎯 Benchmark complete!")
    print(f"📊 All results stored under: {RESULTS_DIR}/")
    print("📌 Hybrid test headline: results/evaluation_summary.txt (from evaluate_model)")


if __name__ == "__main__":
    main()
