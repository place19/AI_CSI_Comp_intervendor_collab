"""Level-vector builders for uniform quantizers."""
from __future__ import annotations

from typing import Tuple

import torch


def build_uniform(bits: int, value_range: Tuple[float, float]) -> torch.Tensor:
    """Uniformly-spaced bin midpoints.

    For bits=2, range=(-1, 1) → step = 2/4 = 0.5, levels = (-0.75, -0.25, 0.25, 0.75).
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        raise ValueError(f"bad value_range: {value_range}")
    n = 1 << int(bits)
    step = (hi - lo) / n
    idx = torch.arange(n, dtype=torch.float32)
    return lo + (idx + 0.5) * step
