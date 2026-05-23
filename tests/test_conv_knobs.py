"""cnn_block / dw_sep_conv now expose the full Conv2d knob set.

Covers:
- int and (h, w) tuple/list accepted for kernel / stride / padding / dilation
- correct out_shape for strided / dilated convs
- groups parameter
- bias default depends on `norm` (False when norm='batchnorm')
- explicit `bias: true` wins over the default
- single-arg forward(x) -> x contract (no mask)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from csi_comp.models.blocks.cnn import CnnBlock
from csi_comp.models.blocks.dw_sep_conv import DwSepConvBlock


# ---------- cnn_block ----------

def test_cnn_default_bias_false_with_batchnorm():
    blk = CnnBlock(in_shape=(2, 8, 8), channels=8)            # norm=batchnorm default
    assert blk.conv.bias is None


def test_cnn_default_bias_true_without_norm():
    blk = CnnBlock(in_shape=(2, 8, 8), channels=8, norm="none")
    assert blk.conv.bias is not None


def test_cnn_explicit_bias_true_overrides_bn_default():
    blk = CnnBlock(in_shape=(2, 8, 8), channels=8, norm="batchnorm", bias=True)
    assert blk.conv.bias is not None


def test_cnn_tuple_kernel_and_padding_keep_shape():
    blk = CnnBlock(in_shape=(2, 8, 16), channels=4, kernel=[3, 5], padding=[1, 2])
    assert blk.out_shape == (4, 8, 16)
    x = torch.randn(1, 2, 8, 16)
    y = blk(x)
    assert y.shape == (1, 4, 8, 16)


def test_cnn_strided_shrinks_spatial():
    blk = CnnBlock(in_shape=(2, 8, 8), channels=4, kernel=3, stride=2, padding=1, norm="none")
    # (8 + 2*1 - 3) // 2 + 1 = 4
    assert blk.out_shape == (4, 4, 4)
    x = torch.randn(1, 2, 8, 8)
    y = blk(x)
    assert y.shape == (1, 4, 4, 4)


def test_cnn_dilation_widens_receptive_field_and_shape():
    blk = CnnBlock(in_shape=(2, 9, 9), channels=4, kernel=3, dilation=2, padding=2, norm="none")
    # effective k = (k-1)*d + 1 = 5; (9 + 4 - 5) + 1 = 9
    assert blk.out_shape == (4, 9, 9)


def test_cnn_groups_partitions_channels():
    blk = CnnBlock(in_shape=(8, 6, 6), channels=8, kernel=3, groups=4, norm="none")
    # Per-group weights: in_per_group=2, out_per_group=2, kernel 3x3 ⇒ 4*(2*2*3*3) = 144
    weight_count = blk.conv.weight.numel()
    assert weight_count == 4 * 2 * 2 * 3 * 3


def test_cnn_padding_same_requires_odd_kernel():
    with pytest.raises(ValueError):
        CnnBlock(in_shape=(2, 8, 8), channels=4, kernel=4)   # default padding='same'


def test_cnn_padding_same_rejects_stride_gt_1():
    with pytest.raises(ValueError):
        CnnBlock(in_shape=(2, 8, 8), channels=4, kernel=3, stride=2)


def test_cnn_forward_is_single_arg():
    """Blocks have a single forward(x) -> x contract — calling with a mask
    kwarg is no longer supported."""
    blk = CnnBlock(in_shape=(2, 8, 8), channels=4, kernel=3, norm="none")
    x = torch.randn(2, 2, 8, 8)
    y = blk(x)
    assert y.shape == (2, 4, 8, 8)


# ---------- dw_sep_conv ----------

def test_dwsep_tuple_kernel_and_padding():
    blk = DwSepConvBlock(in_shape=(4, 8, 16), out_channels=8, kernel=[3, 5], padding=[1, 2])
    assert blk.out_shape == (8, 8, 16)


def test_dwsep_strided_shrinks_spatial():
    blk = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, stride=2, padding=1,
    )
    assert blk.out_shape == (8, 4, 4)
    x = torch.randn(2, 4, 8, 8)
    y = blk(x)
    assert y.shape == (2, 8, 4, 4)


def test_dwsep_pointwise_is_always_1x1_stride1():
    """Even when DW is strided, PW stays a plain 1x1."""
    blk = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, stride=2, padding=1,
    )
    assert blk.pw.kernel_size == (1, 1)
    assert blk.pw.stride == (1, 1)
    assert blk.pw.padding == (0, 0)


def test_dwsep_dilation():
    blk = DwSepConvBlock(
        in_shape=(4, 9, 9), out_channels=4, kernel=3, dilation=2, padding=2,
    )
    # (9 + 4 - 4 - 1) // 1 + 1 = 9 — effective kernel covered by padding.
    assert blk.out_shape == (4, 9, 9)


def test_dwsep_bias_dw_explicit_true_with_bn():
    blk = DwSepConvBlock(in_shape=(4, 8, 8), out_channels=8, bias_dw=True)
    assert blk.dw.bias is not None  # explicit override wins
    assert blk.pw.bias is None      # PW still defaults to bias=False (BN follows)
