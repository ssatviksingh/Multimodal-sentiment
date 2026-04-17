"""Load embedding .pt files (torch.save(tensor)) without pickle FutureWarning spam."""

from __future__ import annotations

import os
import platform
import warnings

import torch


def load_feature_pt(path: str) -> torch.Tensor:
    """
    Embeddings from extract_pretrained_embeddings.py use torch.save(tensor).
    Prefer weights_only=True (PyTorch 2.0+); fall back for legacy pickles / old torch.
    """
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            x = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.float()


def load_checkpoint_dict(path: str, map_location):
    """Training checkpoints may be dicts with tensors + metadata; needs full unpickle."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def dataloader_kwargs(device_str: str) -> dict:
    """Pin memory on GPU/MPS; use worker processes on Linux/macOS (Windows: 0 workers)."""
    pin = str(device_str).startswith("cuda") or str(device_str) == "mps"
    if platform.system() == "Windows":
        return {"num_workers": 0, "pin_memory": pin}
    nw = min(4, max(1, (os.cpu_count() or 4) - 1))
    return {"num_workers": nw, "pin_memory": pin, "persistent_workers": True}
