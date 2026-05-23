"""Flexible residual block: user injects main and skip paths via the registry."""
from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation


@register("block", "residual")
class ResidualBlock(Block):
    """`y = post_act(main_blocks(x) + skip_blocks(x))`.

    Both `main_blocks` and `skip_blocks` are explicit lists of sub-block
    specs (same `{name, ...kwargs}` format as the top-level encoder/decoder
    config). An empty `skip_blocks: []` means identity skip — that's still
    required to be present in YAML so it's obvious there's no implicit
    projection happening. A shape mismatch at the add raises `ValueError`.

    The block doesn't declare any `fusion_pairs` of its own — any fusion
    intent lives inside the sub-blocks (e.g. `dw_sep_conv` pairs its own
    BNs with their preceding convs), and the profiler walks them recursively.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        main_blocks: Sequence[dict[str, Any]],
        skip_blocks: Sequence[dict[str, Any]],
        post_activation: str = "identity",
    ):
        super().__init__(in_shape)
        if main_blocks is None or skip_blocks is None:
            raise ValueError(
                "residual: both main_blocks and skip_blocks must be specified "
                "(use skip_blocks: [] for an identity skip)"
            )
        # Local import to avoid a top-level cycle (encoder → blocks → encoder).
        from ..encoder import build_block_list

        main_mods, main_out, _ = build_block_list(main_blocks, self.in_shape)
        if len(skip_blocks) == 0:
            skip_mods = nn.ModuleList()
            skip_out: Tuple[int, ...] = tuple(self.in_shape)
        else:
            skip_mods, skip_out, _ = build_block_list(skip_blocks, self.in_shape)

        if tuple(main_out) != tuple(skip_out):
            raise ValueError(
                f"residual: main path out_shape {tuple(main_out)} does not "
                f"match skip path out_shape {tuple(skip_out)}; add projection "
                f"blocks to `skip_blocks` to align them"
            )

        self.main_blocks = main_mods
        self.skip_blocks = skip_mods
        self.post_act = make_activation(post_activation)
        self.out_shape = tuple(main_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x
        for b in self.main_blocks:
            m = b(m)

        s = x
        for b in self.skip_blocks:
            s = b(s)

        return self.post_act(m + s)
