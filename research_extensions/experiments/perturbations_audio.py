"""
perturbations_audio.py

Audio perturbation utilities to simulate real-world telehealth noise
conditions at test time only.
"""

from typing import Literal

import numpy as np
import torch


NoiseType = Literal["white", "office", "cafe"]


def _generate_noise_like(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Generate white noise with a given SNR relative to the input waveform.

    Args:
        waveform: (B, T) tensor
        snr_db: desired signal-to-noise ratio in dB
    """
    # Compute signal power
    sig_power = waveform.pow(2).mean(dim=1, keepdim=True)
    snr = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr

    noise = torch.randn_like(waveform)
    noise = noise * torch.sqrt(noise_power / (noise.pow(2).mean(dim=1, keepdim=True) + 1e-8))
    return noise


def add_background_noise(
    waveform: torch.Tensor,
    snr_db: float,
    noise_type: NoiseType = "white",
) -> torch.Tensor:
    """
    Add synthetic background noise to an audio waveform tensor.

    For now, we approximate different noise types using colored variants of
    white noise (e.g., low-pass / band-pass shaping could be added later).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    noise = _generate_noise_like(waveform, snr_db)

    if noise_type == "office":
        # crude low-pass: average with a smoothed version
        kernel = torch.ones(1, 1, 5, device=waveform.device) / 5.0
        noise_ = noise.unsqueeze(1)
        noise = torch.nn.functional.conv1d(noise_, kernel, padding=2).squeeze(1)
    elif noise_type == "cafe":
        # simple high-pass emphasis
        noise = noise - torch.nn.functional.avg_pool1d(noise.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)

    return waveform + noise

