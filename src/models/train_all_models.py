"""
train_all_models.py

Unified multimodal benchmark pipeline.
✅ Trains & evaluates all models (ResNet18, EfficientNet-B0, ConvNeXt-Tiny, ViT, HybridFusion)
✅ Auto-detects already trained models → skips them
✅ Evaluates all → generates combined performance chart
✅ Compatible with existing extractors & manifests
"""

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


# ---------------- HELPERS ----------------
def run_script(module_name, desc):
    """Run a python module via subprocess"""
    print(f"\n⚙️ Running {desc} → {module_name}")
    start = time.time()
    try:
        subprocess.run([sys.executable, "-m", module_name], check=True)
        print(f"✅ {desc} finished in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {desc}: {e}")


def model_checkpoint_exists(name):
    """Check if model checkpoint already exists"""
    checkpoint_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")
    return os.path.exists(checkpoint_path)


# ---------------- MAIN ----------------
def main():
    print("\n📚 Starting multimodal benchmark pipeline")
    print("--------------------------------------------------")

    for name, module in MODELS.items():
        ckpt = os.path.join(RESULTS_DIR, f"{name}_best.pt")
        if model_checkpoint_exists(name):
            print(f"⏩ Skipping {name.upper()} (checkpoint found: {ckpt})")
        else:
            print(f"\n🏋️ Training {name.upper()} model...")
            run_script(module, f"{name} training")

        print(f"\n🧪 Evaluating {name.upper()} model...")
        run_script(EVAL_SCRIPT, f"{name} evaluation")

    print("\n📈 Generating performance comparison chart...")
    run_script(COMPARE_SCRIPT, "comparison graph")

    print("\n🎯 Benchmark complete!")
    print(f"📊 All results stored under: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
