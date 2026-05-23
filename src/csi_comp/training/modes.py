"""Training mode specifications: who is trainable, who is frozen, what loss needs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

MODE_NAMES = (
    "joint",
    "encoder_only",
    "decoder_only",
    "encoder_only_frozen_decoder",
)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    trainable: Set[str] = field(default_factory=set)        # {'encoder','quantizer','decoder'}
    frozen_inference: Set[str] = field(default_factory=set) # decoder is loaded but in eval()
    needs_encoder: bool = True
    needs_decoder: bool = True
    needs_quantizer: bool = True


_SPECS: dict[str, ModeSpec] = {
    "joint": ModeSpec(
        name="joint",
        trainable={"encoder", "quantizer", "decoder"},
        needs_encoder=True, needs_decoder=True, needs_quantizer=True,
    ),
    "encoder_only": ModeSpec(
        name="encoder_only",
        trainable={"encoder", "quantizer"},
        needs_encoder=True, needs_decoder=False, needs_quantizer=True,
    ),
    "decoder_only": ModeSpec(
        name="decoder_only",
        trainable={"decoder"},
        needs_encoder=False, needs_decoder=True, needs_quantizer=False,
    ),
    "encoder_only_frozen_decoder": ModeSpec(
        name="encoder_only_frozen_decoder",
        trainable={"encoder", "quantizer"},
        frozen_inference={"decoder"},
        needs_encoder=True, needs_decoder=True, needs_quantizer=True,
    ),
}


def get_mode_spec(name: str) -> ModeSpec:
    if name not in _SPECS:
        raise KeyError(f"unknown training mode {name!r}; available: {sorted(_SPECS)}")
    return _SPECS[name]
