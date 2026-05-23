"""Decoder builder + module."""
from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn

from .encoder import build_block_list
from .trace import BlockTraceEntry


class Decoder(nn.Module):
    """User-defined block sequence terminating in (max_S, max_P, 2)."""

    def __init__(self, blocks: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for b in self.blocks:
            x = b(x)
        return x


def build_decoder(
    model_cfg: dict,
    data_cfg: dict,
    latent_shape: Tuple[int, ...],
) -> Tuple[Decoder, List[BlockTraceEntry]]:
    cur_shape: Tuple[int, ...] = tuple(latent_shape)
    dec_cfg = model_cfg["decoder"]
    module_list, cur_shape, trace = build_block_list(
        dec_cfg.get("blocks", []), cur_shape
    )

    expected = (int(data_cfg["max_subband"]), int(data_cfg["max_port"]), 2)
    if tuple(cur_shape) != expected:
        raise ValueError(
            f"decoder must terminate in (max_subband, max_port, 2) = {expected}, "
            f"got {tuple(cur_shape)}. Add a terminal block such as "
            f"{{name: reshape_head, max_subband: ${{data.max_subband}}, max_port: ${{data.max_port}}}} "
            f"or {{name: complex_ffn_head, max_port: ${{data.max_port}}}} at the end of "
            f"decoder.blocks."
        )

    return Decoder(list(module_list)), trace
