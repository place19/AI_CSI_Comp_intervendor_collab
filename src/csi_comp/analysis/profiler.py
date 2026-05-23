"""Per-block FLOPs / parameter / output-shape profiler.

Strict op count (each multiply or add is 1 op) on the **fused inference model**:
conv/linear ↔ BN pairs declared via `Block.fusion_pairs` are accounted as fused,
so the BN contributes 0 FLOPs / 0 params and the absorbing conv/linear gains an
effective bias even if `bias=False` was set.

Activations, softmax, LayerNorm, unfused BN, AvgPool are counted with strict
mul/add formulas (approximate where exp/erf are involved). See `op_flops.py`
for the per-op formulas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn

from ..models import BlockTraceEntry
from . import op_flops as F


@dataclass
class BlockProfile:
    idx: int
    name: str
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    num_params: int
    flops: int


@dataclass
class ProfileResult:
    encoder: List[BlockProfile] = field(default_factory=list)
    decoder: List[BlockProfile] = field(default_factory=list)
    encoder_total_params: int = 0
    encoder_total_flops: int = 0
    decoder_total_params: int = 0
    decoder_total_flops: int = 0


# ----- Fusion bookkeeping -----


def _collect_fusion(block: nn.Module) -> Tuple[Set[int], Set[int]]:
    """Return `(absorbee_ids, absorber_ids)` from every `Block` descendant
    (including `block` itself if it's a Block).
    """
    from ..models.blocks.base import Block
    absorbees: Set[int] = set()
    absorbers: Set[int] = set()
    for m in block.modules():
        if isinstance(m, Block):
            for absorber, absorbee in m.fusion_pairs:
                absorbers.add(id(absorber))
                absorbees.add(id(absorbee))
    return absorbees, absorbers


# ----- Sub-block & leaf identification -----


def _shallow_sub_blocks(block: nn.Module) -> List[nn.Module]:
    """Block descendants reachable without crossing another Block boundary."""
    from ..models.blocks.base import Block
    out: List[nn.Module] = []

    def visit(m: nn.Module) -> None:
        for child in m.children():
            if isinstance(child, Block):
                out.append(child)
            else:
                visit(child)

    visit(block)
    return out


_COUNTABLE_LEAF_TYPES: Tuple[type, ...] = (
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid,
    nn.Softmax,
    nn.Dropout, nn.Identity,
)


def _is_countable_leaf(m: nn.Module) -> bool:
    return isinstance(m, _COUNTABLE_LEAF_TYPES)


def _direct_leaves(block: nn.Module, sub_blocks: List[nn.Module]) -> List[nn.Module]:
    """Countable leaves under `block` that are NOT inside any sub-Block."""
    excluded: Set[int] = set()
    for sb in sub_blocks:
        for m in sb.modules():
            excluded.add(id(m))
    leaves: List[nn.Module] = []
    for m in block.modules():
        if m is block:
            continue
        if id(m) in excluded:
            continue
        if _is_countable_leaf(m):
            leaves.append(m)
    return leaves


# ----- Forward-pre hook based shape tracing -----


def _trace_shapes(
    block: nn.Module, in_shape: Tuple[int, ...], targets: List[nn.Module]
) -> Dict[int, Tuple[int, ...]]:
    """Record the input shape (batch-inclusive) of each target's FIRST call."""
    shapes: Dict[int, Tuple[int, ...]] = {}
    hooks = []
    for m in targets:
        def make_hook(_m: nn.Module):
            def hook(mod: nn.Module, args: Tuple[Any, ...]) -> None:
                if id(mod) in shapes:
                    return
                if args and hasattr(args[0], "shape"):
                    shapes[id(mod)] = tuple(int(s) for s in args[0].shape)
            return hook
        hooks.append(m.register_forward_pre_hook(make_hook(m)))

    was_training = block.training
    block.eval()
    try:
        with torch.no_grad():
            block(torch.zeros((1, *in_shape)))
    finally:
        block.train(was_training)
        for h in hooks:
            h.remove()
    return shapes


# ----- Per-leaf FLOP computation -----


def _flops_for_leaf(
    m: nn.Module, in_shape: Tuple[int, ...], force_bias: bool,
) -> int:
    """Apply the appropriate per-op formula to a single leaf module.
    `in_shape` is batch-inclusive (i.e. starts with the dummy batch dim).
    """
    if isinstance(m, nn.Linear):
        # in_shape like (N, ..., in_F); N = prod(all leading).
        N = int(1)
        for d in in_shape[:-1]:
            N *= int(d)
        in_F = int(in_shape[-1])
        out_F = int(m.out_features)
        bias = bool(m.bias is not None) or force_bias
        return F.linear_flops(N=N, in_F=in_F, out_F=out_F, bias=bias)

    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # in_shape: (N, C_in, *spatial). Output spatial recomputed from m's attrs.
        N = int(in_shape[0])
        C_in = int(m.in_channels)
        groups = int(getattr(m, "groups", 1))
        k_C = C_in // groups
        kernel = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        stride = m.stride if isinstance(m.stride, tuple) else (m.stride,)
        padding = m.padding if isinstance(m.padding, tuple) else (m.padding,)
        dilation = m.dilation if isinstance(m.dilation, tuple) else (m.dilation,)
        spatial_in = in_shape[2:]
        spatial_out: List[int] = []
        is_transpose = isinstance(
            m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        )
        for i, d_in in enumerate(spatial_in):
            k = int(kernel[i]); s = int(stride[i])
            p = int(padding[i]); dl = int(dilation[i])
            if is_transpose:
                op = int(getattr(m, "output_padding", (0,) * len(spatial_in))[i])
                d_out = (int(d_in) - 1) * s - 2 * p + dl * (k - 1) + op + 1
            else:
                d_out = (int(d_in) + 2 * p - dl * (k - 1) - 1) // s + 1
            spatial_out.append(int(d_out))
        # 2D-style formula generalises: k_h*k_w → product of kernel sizes.
        k_prod = 1
        for k in kernel:
            k_prod *= int(k)
        sp_prod = 1
        for d in spatial_out:
            sp_prod *= int(d)
        # Reuse the 2D helper by treating k_w=1, H=sp_prod, W=1 (math is identical).
        C_out = int(m.out_channels)
        bias = bool(m.bias is not None) or force_bias
        return F.conv2d_flops(
            N=N, C_out=C_out, H_out=sp_prod, W_out=1,
            k_h=k_prod, k_w=1, k_C=k_C, bias=bias,
        )

    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        n_elem = 1
        for d in in_shape:
            n_elem *= int(d)
        return F.batchnorm_flops(N_elements=n_elem)

    if isinstance(m, nn.LayerNorm):
        normalized = m.normalized_shape
        if isinstance(normalized, int):
            d = int(normalized)
        else:
            d = 1
            for x in normalized:
                d *= int(x)
        n_elem = 1
        for x in in_shape:
            n_elem *= int(x)
        N_tokens = n_elem // d if d > 0 else 0
        return F.layernorm_flops(N_tokens=N_tokens, d=d)

    if isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        N = int(in_shape[0])
        C = int(in_shape[1])
        spatial_in = in_shape[2:]
        kernel = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        stride = m.stride if isinstance(m.stride, tuple) else (m.stride,)
        padding = m.padding if isinstance(m.padding, tuple) else (m.padding,)
        ceil_mode = bool(getattr(m, "ceil_mode", False))
        # Normalize: pad/kernel/stride may be ints if input rank matches; broadcast.
        rank = len(spatial_in)
        if len(kernel) == 1: kernel = kernel * rank
        if len(stride) == 1: stride = stride * rank
        if len(padding) == 1: padding = padding * rank
        sp_out: List[int] = []
        for i, d_in in enumerate(spatial_in):
            k = int(kernel[i]); s = int(stride[i]); p = int(padding[i])
            num = int(d_in) + 2 * p - k
            if ceil_mode:
                sp_out.append((num + s - 1) // s + 1)
            else:
                sp_out.append(num // s + 1)
        k_prod = 1
        for k in kernel:
            k_prod *= int(k)
        H_out = sp_out[0] if sp_out else 1
        W_out = sp_out[1] if len(sp_out) > 1 else 1
        # Generalised to higher ranks by flattening the rest into H_out:
        for extra in sp_out[2:]:
            H_out *= int(extra)
        return F.avgpool_flops(N=N, C=C, H_out=H_out, W_out=W_out, k_h=k_prod, k_w=1)

    if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        # Approximate: each output is mean of `numel_in / numel_out` elements.
        N = int(in_shape[0])
        C = int(in_shape[1])
        spatial_in = in_shape[2:]
        sp_in_prod = 1
        for d in spatial_in:
            sp_in_prod *= int(d)
        out_size = m.output_size
        if isinstance(out_size, int):
            out_size = (out_size,) * len(spatial_in)
        sp_out_prod = 1
        for d in out_size:
            sp_out_prod *= int(d)
        k_avg = sp_in_prod // max(sp_out_prod, 1)
        return F.avgpool_flops(
            N=N, C=C, H_out=sp_out_prod, W_out=1, k_h=k_avg, k_w=1,
        )

    if isinstance(m, nn.ReLU):
        n = 1
        for d in in_shape: n *= int(d)
        return F.activation_flops("relu", n)
    if isinstance(m, nn.GELU):
        n = 1
        for d in in_shape: n *= int(d)
        return F.activation_flops("gelu", n)
    if isinstance(m, nn.Tanh):
        n = 1
        for d in in_shape: n *= int(d)
        return F.activation_flops("tanh", n)
    if isinstance(m, nn.Sigmoid):
        n = 1
        for d in in_shape: n *= int(d)
        return F.activation_flops("sigmoid", n)

    if isinstance(m, nn.Softmax):
        # Softmax over dim `m.dim` (default -1).
        dim = m.dim if m.dim is not None else -1
        if dim < 0:
            dim += len(in_shape)
        D = int(in_shape[dim])
        N = 1
        for i, d in enumerate(in_shape):
            if i != dim:
                N *= int(d)
        return F.softmax_flops(N=N, D=D)

    if isinstance(m, (nn.Dropout, nn.Identity)):
        return 0

    return 0


# ----- Default block hooks (invoked by Block.count_flops / count_params) -----


def default_block_flops(block: nn.Module, in_shape: Tuple[int, ...]) -> int:
    sub_blocks = _shallow_sub_blocks(block)
    leaves = _direct_leaves(block, sub_blocks)
    targets = list(sub_blocks) + list(leaves)
    if not targets:
        return 0
    shapes = _trace_shapes(block, in_shape, targets)
    absorbees, absorbers = _collect_fusion(block)

    total = 0
    for sb in sub_blocks:
        sh = shapes.get(id(sb))
        if sh is None:
            continue
        # Strip the batch dim — sub-Block's count_flops convention is excl-batch.
        sb_in = tuple(sh[1:])
        total += int(sb.count_flops(sb_in))

    for leaf in leaves:
        if id(leaf) in absorbees:
            continue
        sh = shapes.get(id(leaf))
        if sh is None:
            continue
        force_bias = id(leaf) in absorbers
        total += int(_flops_for_leaf(leaf, sh, force_bias))
    return total


def default_block_params(block: nn.Module) -> int:
    """Total params on the fused-model view: absorbee params dropped, absorber
    gains a bias term if not already biased."""
    absorbees, absorbers = _collect_fusion(block)

    total = 0
    for m in block.modules():
        # Sum parameters owned directly by this module (recurse=False), so that
        # nested submodules aren't double-counted via the outer module's recursion.
        own_params = sum(p.numel() for p in m.parameters(recurse=False))
        if own_params == 0:
            continue
        if id(m) in absorbees:
            continue
        total += own_params

    # Add the fused-in bias for absorbers that weren't already biased.
    for m in block.modules():
        if id(m) in absorbers and isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                                 nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if m.bias is None:
                total += int(m.out_channels)
        elif id(m) in absorbers and isinstance(m, nn.Linear):
            if m.bias is None:
                total += int(m.out_features)
    return int(total)


# ----- Public profiling entrypoint -----


def _profile_side(
    blocks: List[nn.Module], trace: List[BlockTraceEntry]
) -> Tuple[List[BlockProfile], int, int]:
    out: List[BlockProfile] = []
    total_params = 0
    total_flops = 0
    for block, entry in zip(blocks, trace):
        flops = int(block.count_flops(entry.in_shape)) if hasattr(block, "count_flops") \
            else int(default_block_flops(block, entry.in_shape))
        num_params = int(block.count_params()) if hasattr(block, "count_params") \
            else int(default_block_params(block))
        out.append(
            BlockProfile(
                idx=entry.idx,
                name=entry.name,
                in_shape=entry.in_shape,
                out_shape=entry.out_shape,
                num_params=num_params,
                flops=flops,
            )
        )
        total_params += num_params
        total_flops += flops
    return out, total_params, total_flops


def profile_model(
    autoencoder,
    encoder_trace: List[BlockTraceEntry],
    decoder_trace: List[BlockTraceEntry],
) -> ProfileResult:
    res = ProfileResult()
    if autoencoder.encoder is not None and encoder_trace:
        res.encoder, res.encoder_total_params, res.encoder_total_flops = _profile_side(
            list(autoencoder.encoder.blocks), encoder_trace
        )
    if autoencoder.decoder is not None and decoder_trace:
        res.decoder, res.decoder_total_params, res.decoder_total_flops = _profile_side(
            list(autoencoder.decoder.blocks), decoder_trace
        )
    return res
