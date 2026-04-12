"""
run_realtime_benchmark.py

Benchmark end-to-end inference latency and memory usage for the
HybridFusion model on representative samples.
"""

import os

import torch
from torch.utils.data import DataLoader

from src.models.fusion_variants.hybrid_fusion_vit_ast import (
    MultimodalDataset,
    HybridFusionModel,
    DEVICE,
)

from ..utils.timing_utils import time_forward_pass


DATA_DIR = "data/features"
MANIFEST = "data/manifest_test.csv"
RESULTS_DIR = os.path.join("research_extensions", "results")
OUT_TXT = os.path.join(RESULTS_DIR, "realtime_benchmark.txt")

MODEL_PATH = "results/hybrid_fusion_best.pt"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds = MultimodalDataset(MANIFEST, DATA_DIR)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    t_sample, a_sample, v_sample, _ = next(iter(loader))
    text_dim, audio_dim, video_dim = t_sample.shape[-1], a_sample.shape[-1], v_sample.shape[-1]

    model = HybridFusionModel(text_dim=text_dim, audio_dim=audio_dim, video_dim=video_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    batch = (t_sample.to(DEVICE), a_sample.to(DEVICE), v_sample.to(DEVICE))

    def forward():
        with torch.no_grad():
            model(*batch)

    mean_ms, p95_ms = time_forward_pass(forward, warmup=5, runs=20)

    # Memory (GPU only)
    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Batch size: {batch[0].shape[0]}\n")
        f.write(f"Mean latency per batch (ms): {mean_ms:.2f}\n")
        f.write(f"p95 latency per batch (ms): {p95_ms:.2f}\n")
        if gpu_mem_mb > 0:
            f.write(f"Peak GPU memory (MB): {gpu_mem_mb:.2f}\n")

    print(f"✅ Saved real-time benchmark → {OUT_TXT}")


if __name__ == "__main__":
    main()

