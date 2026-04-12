"""
modality_dropout.py

Utilities to simulate missing modality scenarios at inference time:
- Drop audio
- Drop video
- Drop text

We approximate missing modality by zeroing features and (optionally)
providing a mask that can be used by downstream models.
"""

from typing import Literal, Tuple

import torch


DropType = Literal["none", "no_text", "no_audio", "no_video"]


def apply_modality_dropout(
    t: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    drop: DropType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero out one modality to simulate it being unavailable.
    """
    t_out, a_out, v_out = t.clone(), a.clone(), v.clone()

    if drop == "no_text":
        t_out.zero_()
    elif drop == "no_audio":
        a_out.zero_()
    elif drop == "no_video":
        v_out.zero_()

    return t_out, a_out, v_out

