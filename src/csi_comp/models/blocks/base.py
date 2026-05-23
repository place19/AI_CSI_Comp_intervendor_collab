"""Block contract: shape-declaring nn.Module with forward(x) -> x."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Block(nn.Module):
    """Base block. Subclasses MUST set self.out_shape during __init__.

    Forward contract:
        forward(x) -> x_out
    The framework assumes fixed-shape inputs and does not thread a
    padding mask through the encoder/decoder. If variable-length data
    is needed in the future, mask handling can be reintroduced as a
    separate concern at the model API boundary.

    Fusion declarations (`fusion_pairs`):
        List of `(absorber, absorbee)` `nn.Module` tuples that will be fused
        into a single op at inference. Canonical case: `(Conv2d, BatchNorm2d)`
        where the BN folds into the preceding Conv. Semantics consumed by the
        profiler:
            - absorbee contributes 0 FLOPs and 0 params.
            - absorber is counted as if `bias=True` (FLOPs gain a bias add per
              output; params gain `out_channels`/`out_features` bias terms)
              regardless of its actual PyTorch `bias` flag — because the fold
              produces an effective bias term.
        Block subclasses populate this list in `__init__`. The default empty
        list means "no fusion intent".

    Optional profiling hooks (`count_flops`, `count_params`):
        Default implementations walk the block's leaf modules, apply per-op
        FLOP formulas, and honour any `fusion_pairs` declared by this block
        or any descendant block. Subclasses override when the forward contains
        functional ops that aren't `nn.Module`s (e.g. `transformer_block`'s
        attention matmuls + softmax).
    """

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    fusion_pairs: List[Tuple[nn.Module, nn.Module]]

    def __init__(self, in_shape: Tuple[int, ...]):
        super().__init__()
        self.in_shape = tuple(int(x) for x in in_shape)
        self.fusion_pairs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def count_flops(self, in_shape: Tuple[int, ...]) -> int:
        """Strict mul/add FLOPs of this block on `(N=1, *in_shape)` input.

        Default impl walks leaf `nn.Module`s and applies known per-op formulas;
        respects `fusion_pairs` declared anywhere in the block subtree.
        """
        # Lazy import to avoid a top-level cycle (blocks → analysis → blocks).
        from ...analysis.profiler import default_block_flops
        return default_block_flops(self, in_shape)

    def count_params(self) -> int:
        """Trainable param count on the fused-model view.

        Default impl: sum of `numel()` over all params, **minus** params of any
        absorbee in this subtree's `fusion_pairs`, **plus** an `out_channels`
        (Conv) or `out_features` (Linear) bias term for each absorber that
        wasn't already biased.
        """
        from ...analysis.profiler import default_block_params
        return default_block_params(self)


def to_int_pair(x, *, name: str = "value") -> Tuple[int, int]:
    """Accept an int or a length-2 list/tuple and return a (h, w) int pair.

    Mirrors PyTorch's convention for Conv2d's kernel_size / stride / padding /
    dilation kwargs, so users can write `kernel: 3` for square or
    `kernel: [3, 5]` for asymmetric in YAML.
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return (int(x), int(x))
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError(f"{name} must be int or length-2 list/tuple, got {x!r}")


def make_activation(name: Optional[str]) -> nn.Module:
    if name in (None, "identity"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unknown activation: {name!r}")
