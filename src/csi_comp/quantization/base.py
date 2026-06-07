"""Quantizer module: decoupled forward (value) / backward (gradient) axes."""
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


# Legacy single-name presets → (forward, backward) on the two-axis scheme.
_GRAD_PRESETS: dict[str, Tuple[str, str]] = {
    "ste":  ("hard", "identity"),
    "soft": ("soft", "soft"),
    "hard": ("hard", "none"),
}
_DEFAULT_TEMPERATURE = 1.0


def _resolve_grad_cfg(grad: Union[str, Mapping[str, Any]]) -> Tuple[str, str, float]:
    """Normalise a ``quantizer.grad`` config into ``(forward, backward, temperature)``.

    Accepts three forms (all backward compatible):
      * preset string ``"ste" | "soft" | "hard"``;
      * legacy mapping ``{name: "soft", temperature: 0.01}``;
      * two-axis mapping ``{forward: "hard", backward: "soft", temperature: 1.0}``.
    """
    if isinstance(grad, str):
        if grad not in _GRAD_PRESETS:
            raise ValueError(
                f"unknown grad preset {grad!r}; expected one of {sorted(_GRAD_PRESETS)} "
                f"or a mapping with 'forward'/'backward'"
            )
        fwd, bwd = _GRAD_PRESETS[grad]
        return fwd, bwd, _DEFAULT_TEMPERATURE

    cfg = dict(grad)
    temperature = float(cfg.pop("temperature", _DEFAULT_TEMPERATURE))
    if "name" in cfg:
        name = cfg.pop("name")
        if cfg:
            raise ValueError(f"unexpected keys in grad mapping: {sorted(cfg)}")
        if name not in _GRAD_PRESETS:
            raise ValueError(
                f"unknown grad preset {name!r}; expected one of {sorted(_GRAD_PRESETS)}"
            )
        fwd, bwd = _GRAD_PRESETS[name]
        return fwd, bwd, temperature

    try:
        fwd = cfg.pop("forward")
        bwd = cfg.pop("backward")
    except KeyError as e:
        raise ValueError(
            "grad mapping must have either 'name' (preset) or both 'forward' and "
            f"'backward'; got keys {sorted(grad)}"
        ) from e
    if cfg:
        raise ValueError(f"unexpected keys in grad mapping: {sorted(cfg)}")
    return fwd, bwd, temperature


def snap_to_nearest(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Map each element of x to the closest entry of `levels`.

    Assumes **uniformly-spaced ascending** levels (as produced by `build_uniform`
    — the only supported case). Computes the nearest index by rounding instead of
    materialising the full ``(..., N_levels)`` squared-distance tensor, so peak
    memory stays O(numel(x)) regardless of bit-width (the old argmin form was
    O(numel(x) * 2**bits)). Uses `ceil(z - 0.5)` rather than `round` so exact ties
    resolve to the lower level — matching the previous ``argmin`` (first-minimum)
    behaviour — and so it stays ONNX-exportable (`searchsorted` is not, and
    `round` is banker's-rounding which would break ties differently). Output keeps
    `levels`' dtype.
    """
    # build_uniform guarantees >= 2 levels (bits >= 1), so levels[1] is always valid.
    xf = x.to(levels.dtype)
    step = levels[1] - levels[0]
    # z = position of x on the level grid; ceil(z - 0.5) = round-half-down.
    idx = torch.ceil((xf - levels[0]) / step - 0.5)
    idx = idx.clamp(0, levels.numel() - 1).to(torch.long)
    return levels[idx]


@register("quantizer", "uniform")
class Quantizer(nn.Module):
    """Quantize an input to a configurable set of levels.

    The forward *value* (passed to the decoder) and the *gradient* surrogate
    (flowing back to the encoder) are two independent axes, registered under
    `registry['quant_forward']` (``hard`` | ``soft``) and
    `registry['quant_backward']` (``identity`` | ``soft`` | ``none``). They are
    combined with the straight-through identity
    ``out = surrogate + (forward_value - surrogate).detach()``. The legacy
    ``grad`` presets are three cells of that matrix: ``ste`` = hard+identity,
    ``soft`` = soft+soft, ``hard`` = hard+none. ``temperature`` is owned here and
    shared by every soft-scoring path (soft forward / soft backward / future
    cross-entropy over levels).

    When `encoder_value_range` differs from `value_range`, a linear transform is
    applied to the encoder output before quantization so that the quantization
    levels (defined in `value_range`) align with the decoder's expected input range.
    The quantizer output is always in `value_range` regardless of input range.

    Eval vs. train: in **eval mode** (`self.training is False`) the forward always
    hard-snaps to the nearest level, regardless of the configured forward/backward
    axes. This keeps validation / test / inference faithful to the deployed (hard)
    quantizer — otherwise a soft *forward* would emit continuous, un-snapped values
    at eval and overstate SGCS. Hard-forward configs (ste / hard / hard+soft) already
    produce the snapped value in train too, so eval is value-identical for them;
    only soft-forward configs (soft / soft+identity) differ at eval.
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

        forward_name, backward_name, temperature = _resolve_grad_cfg(grad)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.forward_name = forward_name
        self.backward_name = backward_name
        self.temperature = float(temperature)
        self.forward_fn = reg_get("quant_forward", forward_name)()
        self.backward_fn = reg_get("quant_backward", backward_name)()

    def rescale_to_value_range(self, x: torch.Tensor) -> torch.Tensor:
        """Map the encoder output into `value_range` (the pre-quantization affine).

        Identity when `encoder_value_range` is unset (`_alpha is None`). Shared by
        `forward` and by callers (e.g. the trainer) that want the rescaled-but-not-
        yet-quantized latent for a latent-space loss, so the affine has one source
        of truth.
        """
        if self._alpha is not None:
            return self._alpha * x + self._beta
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rescale_to_value_range(x)
        # Eval / inference: hard-snap regardless of strategy (see class docstring).
        if not self.training:
            return snap_to_nearest(x, self.levels)

        forward_value = self.forward_fn(x, self.levels, self.temperature)
        if self.backward_name == "none":
            return forward_value.detach()
        # Reuse the forward tensor when both axes name the same strategy (currently
        # only soft+soft) so the soft value isn't computed twice. Safe because the
        # only name shared by both registries ("soft") maps to the same computation.
        if self.backward_name == self.forward_name:
            surrogate = forward_value
        else:
            surrogate = self.backward_fn(x, self.levels, self.temperature)
        # Straight-through: value == forward_value, gradient == d surrogate / dx.
        return surrogate + (forward_value - surrogate).detach()

    def to_hard(self) -> "Quantizer":
        """Force the hard / no-grad path (forward=hard, backward=none). Used for ONNX export."""
        self.forward_name = "hard"
        self.backward_name = "none"
        self.forward_fn = reg_get("quant_forward", "hard")()
        self.backward_fn = reg_get("quant_backward", "none")()
        return self


def build_quantizer(cfg: Mapping[str, Any]) -> Quantizer:
    """Construct a Quantizer from a YAML-style config block."""
    cfg = dict(cfg)
    qtype = cfg.pop("type", "uniform")
    cls = reg_get("quantizer", qtype)
    return cls(type=qtype, **cfg)
