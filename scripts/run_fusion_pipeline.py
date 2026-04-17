"""
Full fusion benchmark: hybrid -> weighted -> transformer -> evaluate -> compare scripts.

Uses GPU when available (--device auto on each trainer). Force CPU: -- --device cpu

Usage (repo root):
  python scripts/run_fusion_pipeline.py
  python scripts/run_fusion_pipeline.py -- --smoke
  python scripts/run_fusion_pipeline.py -- --device cuda --epochs 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    p = argparse.ArgumentParser(description="Train fusion models and refresh evaluation / charts.")
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Args after -- are forwarded to each fusion trainer (e.g. -- --epochs 2 --smoke)",
    )
    args = p.parse_args()
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    py = [sys.executable, "-u"]
    fusion_mods = [
        "src.models.fusion_variants.hybrid_fusion_vit_ast",
        "src.models.fusion_variants.weighted_fusion",
        "src.models.fusion_variants.transformer_fusion",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")

    for mod in fusion_mods:
        cmd = py + ["-m", mod] + extra
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)

    for mod in ("src.models.evaluate_model", "src.models.compare_all_models", "src.models.compare_results"):
        cmd = py + ["-m", mod]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)

    print("\nDone. See results/ and results/evaluation_summary.txt", flush=True)


if __name__ == "__main__":
    main()
