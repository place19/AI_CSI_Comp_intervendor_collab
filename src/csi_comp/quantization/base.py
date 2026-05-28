"""Quantizer module: snap-to-nearest with a pluggable gradient strategy."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..registry import get as reg_get
from ..registry import register
from .uniform import build_uniform


def build_levels(
    type: str,
    bits: int,
    value_range: Tuple[float, float],
    unit_spaced: bool = True,
) -> torch.Tensor:
    if type != "uniform":
        raise ValueError(f"unknown quantizer type: {type!r}")
    if not unit_spaced:
        raise NotImplementedError(
            "non-uniformly-spaced levels not implemented yet; "
            "add a new builder + register in quantization/."
        )
    return build_uniform(bits, value_range)


def snap_to_nearest(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Map each element of x to the closest entry of `levels` (1-D, sorted)."""
    # (..., 1) - (N,)  →  (..., N)
    sq = (x.unsqueeze(-1) - levels) ** 2
    idx = sq.argmin(dim=-1)
    return levels[idx]


@register("quantizer", "uniform")
class Quantizer(nn.Module):
    """Quantize an input to a configurable set of levels.

    Gradient behaviour is delegated to a `grad` strategy module registered under
    `registry['grad']` (e.g. 'ste', 'soft', 'hard').

    When `encoder_value_range` differs from `value_range`, a linear transform is
    applied to the encoder output before quantization so that the quantization
    levels (defined in `value_range`) align with the decoder's expected input range.
    The quantizer output is always in `value_range` regardless of input range.
    """

    levels: torch.Tensor

    def __init__(
        self,
        bits: int,
        value_range: Tuple[float, float],
        unit_spaced: bool = True,
        grad: Union[str, Mapping[str, Any]] = "ste",
        type: str = "uniform",
        encoder_value_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.bits = int(bits)
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.unit_spaced = bool(unit_spaced)
        self.type = type
        levels = build_levels(type, bits, value_range, unit_spaced)
        self.register_buffer("levels", levels)

        # Linear transform: encoder output range → decoder input range (value_range).
        # Precompute scalar alpha/beta so forward is a single fused multiply-add.
        if encoder_value_range is not None:
            enc_lo, enc_hi = float(encoder_value_range[0]), float(encoder_value_range[1])
            if enc_hi <= enc_lo:
                raise ValueError(f"bad encoder_value_range: {encoder_value_range}")
            dec_lo, dec_hi = self.value_range
            self.encoder_value_range: Optional[Tuple[float, float]] = (enc_lo, enc_hi)
            self._alpha: Optional[float] = (dec_hi - dec_lo) / (enc_hi - enc_lo)
            self._beta: Optional[float] = dec_lo - enc_lo * self._alpha
        else:
            self.encoder_value_range = None
            self._alpha = None
            self._beta = None

        if isinstance(grad, str):
            grad_cfg: dict[str, Any] = {"name": grad}
        else:
            grad_cfg = dict(grad)
        grad_name = grad_cfg.pop("name")
        self.grad_name = grad_name
        self.grad = reg_get("grad", grad_name)(**grad_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._alpha is not None:
            x = self._alpha * x + self._beta
        return self.grad(x, self.levels)

    def to_hard(self) -> "Quantizer":
        """Swap the gradient strategy to the hard / no-grad path. Used for ONNX export."""
        self.grad = reg_get("grad", "hard")()
        self.grad_name = "hard"
        return self


def build_quantizer(cfg: Mapping[str, Any]) -> Quantizer:
    """Construct a Quantizer from a YAML-style config block."""
    cfg = dict(cfg)
    qtype = cfg.pop("type", "uniform")
    cls = reg_get("quantizer", qtype)
    return cls(type=qtype, **cfg)
