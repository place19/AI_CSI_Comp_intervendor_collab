"""Depthwise-separable conv block (DW → BN? → act? → 1×1 → BN? → act?)."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation, to_int_pair
from .cnn import _conv_out_dim, _resolve_padding


IntOrPair = Union[int, List[int], Tuple[int, int]]


@register("block", "dw_sep_conv")
class DwSepConvBlock(Block):
    """Depthwise (groups=C_in) → optional BN → optional act → pointwise 1×1
    → optional BN → optional act.

    Conv knobs apply to the **depthwise** conv (the pointwise is by definition
    a 1×1 stride-1 conv):
        kernel:    int OR (h, w). Default 3.
        stride:    int OR (h, w). Default 1.
        padding:   int OR (h, w) OR "same". Default "same" (needs odd kernel).
        dilation:  int OR (h, w). Default 1.

    Bias convention: when a BN follows a conv, that conv's bias is redundant
    (BN re-centers), so `bias_dw` / `bias_pw` default to `not use_bn*`. An
    explicit YAML value still wins.

    The BN layers are paired with their preceding conv via `fusion_pairs`
    so the profiler accounts for inference-time fold: the BN drops out of
    the FLOP / param totals and the absorbing conv is counted as if biased.

    Like `cnn_block`, this block treats input as a fixed-shape feature map.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_channels: int,
        kernel: IntOrPair = 3,
        stride: IntOrPair = 1,
        padding: Union[IntOrPair, str] = "same",
        dilation: IntOrPair = 1,
        use_bn1: bool = True,
        use_act1: bool = True,
        use_bn2: bool = True,
        use_act2: bool = True,
        activation: str = "relu",
        bias_dw: Optional[bool] = None,
        bias_pw: Optional[bool] = None,
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 3:
            raise ValueError(f"dw_sep_conv expects (C, S, P), got {self.in_shape}")
        c_in, S, P = self.in_shape

        kh, kw = to_int_pair(kernel, name="kernel")
        sh, sw = to_int_pair(stride, name="stride")
        dh, dw_ = to_int_pair(dilation, name="dilation")
        ph, pw_ = _resolve_padding(padding, (kh, kw), (sh, sw), (dh, dw_))

        bias_dw_eff = (not use_bn1) if bias_dw is None else bool(bias_dw)
        bias_pw_eff = (not use_bn2) if bias_pw is None else bool(bias_pw)

        self.dw = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw_),
            dilation=(dh, dw_),
            groups=c_in,
            bias=bias_dw_eff,
        )
        self.bn1: nn.Module = nn.BatchNorm2d(c_in) if use_bn1 else nn.Identity()
        self.act1: nn.Module = make_activation(activation) if use_act1 else nn.Identity()

        self.pw = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(out_channels),
            kernel_size=1,
            bias=bias_pw_eff,
        )
        self.bn2: nn.Module = nn.BatchNorm2d(int(out_channels)) if use_bn2 else nn.Identity()
        self.act2: nn.Module = make_activation(activation) if use_act2 else nn.Identity()

        s_out = _conv_out_dim(S, kh, sh, ph, dh)
        p_out = _conv_out_dim(P, kw, sw, pw_, dw_)
        if s_out <= 0 or p_out <= 0:
            raise ValueError(
                f"dw_sep_conv: non-positive output spatial dims "
                f"({s_out}, {p_out}) from in=({S},{P}) kernel=({kh},{kw}) "
                f"stride=({sh},{sw}) padding=({ph},{pw_}) dilation=({dh},{dw_})"
            )
        self.out_shape = (int(out_channels), int(s_out), int(p_out))

        pairs: list[tuple[nn.Module, nn.Module]] = []
        if isinstance(self.bn1, nn.BatchNorm2d):
            pairs.append((self.dw, self.bn1))
        if isinstance(self.bn2, nn.BatchNorm2d):
            pairs.append((self.pw, self.bn2))
        self.fusion_pairs = pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
