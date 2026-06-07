"""Pluggable quantizer *forward* (value) and *backward* (gradient) axes.

The training-time quantizer output is assembled from two independently chosen
pieces via the straight-through identity::

    out = surrogate + (forward_value - surrogate).detach()

so the forward *value* (what the decoder receives) and the *gradient* that flows
back to the encoder are decoupled:

  forward  (``quant_forward``)  : ``hard`` snap        | ``soft`` assignment-weighted value
  backward (``quant_backward``) : ``identity`` (STE)   | ``soft`` surrogate | ``none`` (no grad)

This gives the full forward×backward matrix from two short option lists; the
legacy presets are just three of its cells (resolved in `Quantizer`):

  ste  = forward ``hard`` + backward ``identity``
  soft = forward ``soft`` + backward ``soft``
  hard = forward ``hard`` + backward ``none``

Adding a new forward (e.g. stochastic rounding) or backward (e.g. a gumbel
surrogate) option automatically combines with every existing one — no new
combined classes. All strategies take ``(x, levels, temperature)`` and ignore
the args they don't need (hard / identity ignore ``temperature``).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..registry import register
from .base import snap_to_nearest
from .soft_ops import soft_value


# ----- forward (value sent to the decoder) -----

@register("quant_forward", "hard")
class HardForward(nn.Module):
    """Snap to the nearest level (the deployed / eval value)."""

    def forward(self, x: torch.Tensor, levels: torch.Tensor, temperature: float) -> torch.Tensor:
        return snap_to_nearest(x, levels)


@register("quant_forward", "soft")
class SoftForward(nn.Module):
    """Assignment-weighted soft value (continuous; → nearest level as T → 0)."""

    def forward(self, x: torch.Tensor, levels: torch.Tensor, temperature: float) -> torch.Tensor:
        return soft_value(x, levels, temperature)


# ----- backward (gradient surrogate; None = no gradient) -----

@register("quant_backward", "identity")
class IdentityBackward(nn.Module):
    """Straight-through estimator: the surrogate is ``x`` itself (identity grad)."""

    def forward(self, x: torch.Tensor, levels: torch.Tensor, temperature: float) -> Optional[torch.Tensor]:
        return x


@register("quant_backward", "soft")
class SoftBackward(nn.Module):
    """Soft surrogate: gradient flows through the assignment-weighted soft value."""

    def forward(self, x: torch.Tensor, levels: torch.Tensor, temperature: float) -> Optional[torch.Tensor]:
        return soft_value(x, levels, temperature)


@register("quant_backward", "none")
class NoneBackward(nn.Module):
    """No gradient — the forward value passes through detached (export / pure hard)."""

    def forward(self, x: torch.Tensor, levels: torch.Tensor, temperature: float) -> Optional[torch.Tensor]:
        return None
