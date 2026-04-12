"""
timing_utils.py

Helpers for measuring inference latency and throughput.
"""

import time
from typing import Callable, Any, Tuple

import torch


def time_forward_pass(
    fn: Callable[[], Any],
    warmup: int = 3,
    runs: int = 10,
) -> Tuple[float, float]:
    """
    Time a forward pass function on the current device.

    Args:
        fn: zero-argument callable that runs a single forward pass.
        warmup: number of warmup iterations.
        runs: number of timed runs.

    Returns:
        (mean_latency_ms, p95_latency_ms)
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # ms

    times_sorted = sorted(times)
    mean_ms = sum(times_sorted) / len(times_sorted)
    p95_ms = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    return mean_ms, p95_ms

