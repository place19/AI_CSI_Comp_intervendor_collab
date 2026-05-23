"""Registry-based factories for the PyTorch built-in schedulers we already use.

Each factory tags the returned scheduler with `.step_unit = 'epoch'` so the
Trainer steps it at end-of-epoch, matching the prior behaviour.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ...registry import register


def _tag_epoch(sched):
    sched.step_unit = "epoch"
    return sched


@register("scheduler", "cosine")
def build_cosine(
    optimizer: Optimizer,
    *,
    T_max: int | None = None,
    t_max: int | None = None,
    eta_min: float = 0.0,
):
    tmax = T_max if T_max is not None else t_max
    if tmax is None:
        raise ValueError("cosine scheduler requires T_max")
    return _tag_epoch(CosineAnnealingLR(optimizer, T_max=int(tmax), eta_min=float(eta_min)))


@register("scheduler", "step")
def build_step(
    optimizer: Optimizer,
    *,
    step_size: int | None = None,
    gamma: float = 0.1,
):
    if step_size is None:
        raise ValueError("step scheduler requires step_size")
    return _tag_epoch(StepLR(optimizer, step_size=int(step_size), gamma=float(gamma)))


@register("scheduler", "none")
def build_none(optimizer: Optimizer):
    # Returning None lets build_scheduler short-circuit; but we still register so
    # the dispatch table is complete and `name: none` is a valid choice.
    return None
