"""Average pooling block: thin wrapper around nn.AvgPool2d with Conv2d-style knobs."""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, to_int_pair
from .cnn import _conv_out_dim, _resolve_padding


IntOrPair = Union[int, List[int], Tuple[int, int]]


@register("block", "avg_pool")
class AvgPoolBlock(Block):
    """nn.AvgPool2d wrapper with the same kernel/stride/padding conventions as cnn_block.

    Knobs:
        kernel:            int OR (h, w). Required.
        stride:            int OR (h, w). Default 1 (matches cnn_block; PyTorch's
                           default is stride=kernel — overridden here for consistency).
        padding:           int OR (h, w) OR "same". Default 0.
                           "same" is only valid for stride=1 (same constraint as cnn_block).
        ceil_mode:         bool. Default False. Forwarded to nn.AvgPool2d.
        count_include_pad: bool. Default True. Forwarded to nn.AvgPool2d.

    Input  shape (excl. batch): (C, S, P)
    Output shape (excl. batch): (C, S_out, P_out) — channels unchanged.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        kernel: IntOrPair,
        stride: IntOrPair = 1,
        padding: Union[IntOrPair, str] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 3:
            raise ValueError(f"avg_pool expects (C, S, P), got {self.in_shape}")
        c_in, S, P = self.in_shape

        kh, kw = to_int_pair(kernel, name="kernel")
        sh, sw = to_int_pair(stride, name="stride")
        # AvgPool2d does not support dilation; hard-code (1, 1) for padding resolution.
        ph, pw_ = _resolve_padding(padding, (kh, kw), (sh, sw), (1, 1))

        self.pool = nn.AvgPool2d(
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw_),
            ceil_mode=bool(ceil_mode),
            count_include_pad=bool(count_include_pad),
        )

        s_out = _pool_out_dim(S, kh, sh, ph, ceil_mode)
        p_out = _pool_out_dim(P, kw, sw, pw_, ceil_mode)
        if s_out <= 0 or p_out <= 0:
            raise ValueError(
                f"avg_pool: computed non-positive output spatial dims "
                f"({s_out}, {p_out}) from in=({S},{P}) "
                f"kernel=({kh},{kw}) stride=({sh},{sw}) padding=({ph},{pw_})"
            )
        self.out_shape = (int(c_in), int(s_out), int(p_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def _pool_out_dim(d_in: int, k: int, s: int, p: int, ceil_mode: bool) -> int:
    num = d_in + 2 * p - k
    if ceil_mode:
        return (num + s - 1) // s + 1
    return _conv_out_dim(d_in, k, s, p, d=1)
