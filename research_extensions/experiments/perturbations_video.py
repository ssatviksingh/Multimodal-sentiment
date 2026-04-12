"""
perturbations_video.py

Video perturbation utilities to simulate webcam artifacts in telehealth:
- Blur
- Frame drop
- Partial occlusion
"""

from typing import Tuple

import torch
import torchvision.transforms as T


def apply_blur(frames: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply spatial Gaussian blur to video frames.

    Args:
        frames: (B, F, C, H, W) tensor in [0, 1] or [0, 255]
    """
    B, F, C, H, W = frames.shape
    blur = T.GaussianBlur(kernel_size=kernel_size)
    flat = frames.view(B * F, C, H, W)
    out = blur(flat)
    return out.view(B, F, C, H, W)


def apply_frame_dropout(frames: torch.Tensor, drop_prob: float = 0.3) -> torch.Tensor:
    """
    Randomly zero out a subset of frames to simulate stutter / freeze.

    Args:
        frames: (B, F, C, H, W)
    """
    B, F, C, H, W = frames.shape
    mask = (torch.rand(B, F, 1, 1, 1, device=frames.device) > drop_prob).float()
    return frames * mask


def apply_partial_occlusion(
    frames: torch.Tensor,
    box_fraction: Tuple[float, float] = (0.3, 0.3),
) -> torch.Tensor:
    """
    Apply a black rectangle over the central region of the frame.
    This approximates face occlusion (e.g., hand over mouth).

    Args:
        frames: (B, F, C, H, W)
        box_fraction: (height_frac, width_frac) of the occlusion box.
    """
    B, F, C, H, W = frames.shape
    h_frac, w_frac = box_fraction
    box_h = int(H * h_frac)
    box_w = int(W * w_frac)
    y1 = H // 2 - box_h // 2
    y2 = y1 + box_h
    x1 = W // 2 - box_w // 2
    x2 = x1 + box_w

    frames = frames.clone()
    frames[:, :, :, y1:y2, x1:x2] = 0.0
    return frames

