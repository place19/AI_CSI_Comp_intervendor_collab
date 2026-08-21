"""Encoder builder + module."""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..data.layout import LayoutAdapter
from ..registry import get as reg_get
from .trace import BlockTraceEntry


def build_block_list(
    specs: Sequence[dict[str, Any]],
    in_shape: Tuple[int, ...],
) -> Tuple[nn.ModuleList, Tuple[int, ...], List[BlockTraceEntry]]:
    """Instantiate a sequence of registered blocks, propagating shape.

    Mirrors the loop used by `build_encoder` / `build_decoder` so nested
    constructs (e.g. `ResidualBlock`'s main/skip paths) can reuse the same
    `name + kwargs → cls(in_shape=..., **kwargs)` flow.
    """
    blocks: List[nn.Module] = []
    trace: List[BlockTraceEntry] = []
    cur: Tuple[int, ...] = tuple(in_shape)
    for i, entry in enumerate(specs):
        cfg: dict[str, Any] = dict(entry)
        name = cfg.pop("name")
        cls = reg_get("block", name)
        block = cls(in_shape=cur, **cfg)
        blocks.append(block)
        trace.append(
            BlockTraceEntry(
                idx=i,
                name=name,
                in_shape=tuple(cur),
                out_shape=tuple(block.out_shape),
                num_params=sum(p.numel() for p in block.parameters()),
            )
        )
        cur = tuple(block.out_shape)
    return nn.ModuleList(blocks), cur, trace


class Encoder(nn.Module):
    """Layout adapter → quant stub → user blocks → dequant stub. Returns the latent.

    `quant` / `dequant` are `nn.Identity()` in a normal float model and are swapped
    for `QuantStub` / `DeQuantStub` by `training.qat.prepare_encoder_qat`. Eager-mode
    QAT only attaches an observer to each QAT module's *output*, so without a stub the
    model's very first input would be the one tensor that never sees fake-quant — yet
    the deployment toolchain quantizes it like any other activation. `nn.Identity` has
    no parameters, so the slots add no `state_dict` keys and existing checkpoints keep
    loading unchanged.
    """

    def __init__(self, blocks: List[nn.Module], layout: str):
        super().__init__()
        self.layout = layout
        self.layout_adapter = LayoutAdapter(layout)
        self.quant: nn.Module = nn.Identity()
        self.blocks = nn.ModuleList(blocks)
        self.dequant: nn.Module = nn.Identity()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        x = self.quant(self.layout_adapter(real, imag))
        for b in self.blocks:
            x = b(x)
        return self.dequant(x)


def _initial_shape(layout: str, max_S: int, max_P: int) -> Tuple[int, ...]:
    if layout == "cnn":
        return (2, max_S, max_P)
    if layout == "transformer":
        return (max_S, max_P * 2)
    raise ValueError(f"unknown layout: {layout!r}")


def build_encoder(
    model_cfg: dict, data_cfg: dict
) -> Tuple[Encoder, List[BlockTraceEntry]]:
    """Construct the encoder from a config block list and return its block trace.

    The last block is the encoder's terminal — its output is the latent that
    feeds the quantizer. The user is responsible for ensuring its output is in
    the quantizer's `value_range` (typically by ending with a bounded activation
    such as `{name: activation, activation: tanh}`).
    """
    layout = data_cfg["layout"]
    cur_shape = _initial_shape(layout, data_cfg["max_subband"], data_cfg["max_port"])

    enc_cfg = model_cfg["encoder"]
    module_list, _, trace = build_block_list(enc_cfg.get("blocks", []), cur_shape)
    return Encoder(list(module_list), layout), trace
