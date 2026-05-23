"""2D CNN block: configurable Conv2d wrapper. Mask is passed through unchanged."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation, to_int_pair


IntOrPair = Union[int, List[int], Tuple[int, int]]


@register("block", "cnn_block")
class CnnBlock(Block):
    """Conv2d → (optional norm) → activation.

    All standard `nn.Conv2d` knobs are exposed:
        channels:  out channels.
        kernel:    int OR (h, w). Default 3.
        stride:    int OR (h, w). Default 1.
        padding:   int OR (h, w) OR "same". "same" only valid for stride=1.
        dilation:  int OR (h, w). Default 1.
        groups:    int. Default 1.
        bias:      bool. Default depends on `norm`:
                     - norm == "batchnorm" → False (BN re-centers, bias is redundant)
                     - otherwise           → True
                   An explicit `bias: true|false` in YAML always wins.
        activation, norm: as before.

    Treats input as a fixed-shape feature map (image-classification style).
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        channels: int,
        kernel: IntOrPair = 3,
        stride: IntOrPair = 1,
        padding: Union[IntOrPair, str] = "same",
        dilation: IntOrPair = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        activation: str = "relu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 3:
            raise ValueError(f"cnn_block expects (C, S, P), got {self.in_shape}")
        c_in, S, P = self.in_shape

        kh, kw = to_int_pair(kernel, name="kernel")
        sh, sw = to_int_pair(stride, name="stride")
        dh, dw_ = to_int_pair(dilation, name="dilation")
        ph, pw_ = _resolve_padding(padding, (kh, kw), (sh, sw), (dh, dw_))

        bias_eff = (norm != "batchnorm") if bias is None else bool(bias)
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(channels),
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw_),
            dilation=(dh, dw_),
            groups=int(groups),
            bias=bias_eff,
        )

        if norm in (None, "none"):
            self.norm: nn.Module = nn.Identity()
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm2d(int(channels))
        elif norm == "groupnorm":
            self.norm = nn.GroupNorm(min(8, int(channels)), int(channels))
        else:
            raise ValueError(f"unknown norm: {norm!r}")
        self.act = make_activation(activation)

        if isinstance(self.norm, nn.BatchNorm2d):
            self.fusion_pairs = [(self.conv, self.norm)]

        s_out = _conv_out_dim(S, kh, sh, ph, dh)
        p_out = _conv_out_dim(P, kw, sw, pw_, dw_)
        if s_out <= 0 or p_out <= 0:
            raise ValueError(
                f"cnn_block: computed non-positive output spatial dims "
                f"({s_out}, {p_out}) from in=({S},{P}) "
                f"kernel=({kh},{kw}) stride=({sh},{sw}) "
                f"padding=({ph},{pw_}) dilation=({dh},{dw_})"
            )
        self.out_shape = (int(channels), s_out, p_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _resolve_padding(
    padding: Union[int, List[int], Tuple[int, int], str],
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"unknown padding string {padding!r}; use 'same' or an int/pair")
        if stride != (1, 1):
            raise ValueError("padding='same' requires stride=1; use explicit ints for strided convs")
        # nn.Conv2d's own padding="same" exists, but we need an int to compute out_shape.
        # For "same" we set padding so that effective kernel reach (dilation * (k-1) + 1)
        # extends symmetrically. This requires (dilation*(k-1)) to be even, i.e. odd k for d=1.
        ph = (dilation[0] * (kernel[0] - 1)) // 2
        pw = (dilation[1] * (kernel[1] - 1)) // 2
        if dilation[0] * (kernel[0] - 1) % 2 != 0 or dilation[1] * (kernel[1] - 1) % 2 != 0:
            raise ValueError(
                "padding='same' needs odd `kernel` (per dim) at dilation=1, or an even "
                "`dilation*(kernel-1)`. Pass an explicit padding int/pair instead."
            )
        return (int(ph), int(pw))
    return to_int_pair(padding, name="padding")


def _conv_out_dim(d_in: int, k: int, s: int, p: int, d: int) -> int:
    return (d_in + 2 * p - d * (k - 1) - 1) // s + 1
