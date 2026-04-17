"""Shared argparse helpers for fusion training scripts."""

from __future__ import annotations

import argparse
from typing import Any, Tuple


def resolve_device(choice: str) -> str:
    """Map --device to a torch device string. Raises if CUDA/MPS requested but unavailable."""
    import torch

    c = (choice or "auto").strip().lower()
    if c == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if c in ("cpu",):
        return "cpu"
    if c in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--device cuda) but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build for your GPU, or use --device cpu."
            )
        return "cuda"
    if c.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        return c
    if c == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this system.")
        return "mps"
    raise ValueError(f"Unknown --device {choice!r}; use auto, cpu, cuda, cuda:N, or mps.")


def build_fusion_train_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--train-manifest", default="data/manifest_train.csv")
    p.add_argument("--val-manifest", default="data/manifest_val.csv")
    p.add_argument("--data-dir", default="data/features")
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--device",
        default="auto",
        metavar="STR",
        help="auto (CUDA if available, else MPS on Apple Silicon, else CPU), or cpu | cuda | cuda:0 | mps",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (defaults: model-specific; --smoke uses 2 if omitted)",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use only the first N rows of the train manifest (after shuffle seed in DataLoader)",
    )
    p.add_argument("--max-val-samples", type=int, default=None, help="First N rows of val manifest")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: 2 epochs and 128 train / 32 val samples unless overridden",
    )
    p.add_argument(
        "--modality-dropout",
        type=float,
        default=0.1,
        help="Training-only: probability to drop one random modality embedding (0 disables)",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val macro-F1 does not improve for this many epochs (0 = disabled)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="AdamW learning rate (default: model-specific)",
    )
    return p


def resolve_train_config(
    args: Any,
    default_epochs: int,
    default_batch: int,
) -> Tuple[int, int, int | None, int | None]:
    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else default_epochs)
    batch = args.batch_size if args.batch_size is not None else default_batch
    max_tr = args.max_train_samples
    max_va = args.max_val_samples
    if args.smoke:
        if max_tr is None:
            max_tr = 128
        if max_va is None:
            max_va = 32
    return epochs, batch, max_tr, max_va


def apply_out_dir(out_dir: str) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
