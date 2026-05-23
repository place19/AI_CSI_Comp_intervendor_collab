"""Linear warmup followed by cosine annealing — iteration-based.

Implemented on top of `torch.optim.lr_scheduler.LambdaLR` so per-group base
learning rates and `state_dict()` (including resume) behave automatically.
"""
from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ...registry import register


def _make_lambda(warmup_steps: int, total_steps: int, floor: float):
    """Return f(step) → multiplicative factor in [floor, 1].

    floor = min_lr / base_lr (per group). We assume groups share base_lr; if
    they don't, LambdaLR still applies the same factor to each group, which is
    the desired behaviour for relative scaling.
    """
    warmup_steps = max(1, int(warmup_steps))
    decay_steps = max(1, int(total_steps) - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if step >= total_steps:
            return float(floor)
        progress = (step - warmup_steps) / float(decay_steps)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(floor + (1.0 - floor) * cos_factor)

    return lr_lambda


@register("scheduler", "warmup_cosine")
def build_warmup_cosine(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    **_ignored: Any,
):
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if total_steps < warmup_steps:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
        )

    # Use the first group's base lr as the reference for the floor ratio. All
    # groups share the same multiplicative factor; this matches LambdaLR semantics.
    base_lr = float(optimizer.param_groups[0]["lr"])
    floor = float(min_lr) / base_lr if base_lr > 0 else 0.0

    sched = LambdaLR(optimizer, lr_lambda=_make_lambda(warmup_steps, total_steps, floor))
    sched.step_unit = "iter"
    # Persist these so resume / debugging can see them
    sched.warmup_steps = int(warmup_steps)
    sched.total_steps = int(total_steps)
    sched.min_lr = float(min_lr)
    return sched
