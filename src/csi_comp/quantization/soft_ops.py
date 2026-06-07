"""Soft scoring over quantization levels — the shared primitive.

`level_logits` produces the per-level scores ``-(x - level)^2 / T``. The softmax
of those is the per-level assignment distribution (`soft_assign`), and the
assignment-weighted level sum is the differentiable soft-quantized value
(`soft_value`).

The same scores are reused several ways, so they live here once:
  * soft *forward* value — the value fed to the decoder (`quant_forward: soft`),
  * soft *backward* surrogate — the gradient path to the encoder (`quant_backward: soft`),
  * cross-entropy over levels (`losses/cross_entropy_levels.py`) — `level_logits`
    *are* the per-element CE logits (one class per level); with the teacher's level
    index as label, `F.cross_entropy(level_logits, idx)` supervises the encoder.
    Use `level_logits` directly here, not `soft_assign`, to avoid a double softmax.

All consumers read one `temperature`, owned by the `Quantizer`, so a single
annealing schedule drives every soft-scoring path.
"""
from __future__ import annotations

import torch


def level_logits(
    x: torch.Tensor, levels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Per-level scores ``-(x - level)^2 / T``. Shape ``(*x.shape, N_levels)``."""
    return -((x.unsqueeze(-1) - levels) ** 2) / temperature


def soft_assign(
    x: torch.Tensor, levels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """softmax over levels of `level_logits` → assignment probabilities ``(..., N)``."""
    return torch.softmax(level_logits(x, levels, temperature), dim=-1)


def soft_value(
    x: torch.Tensor, levels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Assignment-weighted level sum ``sum_i softmax(...)_i * level_i``. Shape ``x.shape``.

    As ``temperature → 0`` this collapses to the nearest level.
    """
    return (soft_assign(x, levels, temperature) * levels).sum(dim=-1)
