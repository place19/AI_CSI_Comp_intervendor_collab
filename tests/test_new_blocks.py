"""Tests for the reshape, dw_sep_conv, and residual blocks (+ profiler fusion)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from csi_comp.models.blocks.dw_sep_conv import DwSepConvBlock
from csi_comp.models.blocks.reshape import ReshapeBlock
from csi_comp.models.blocks.residual import ResidualBlock


# ---------- reshape ----------

def test_reshape_infers_minus_one():
    blk = ReshapeBlock(in_shape=(2, 8, 8), out_shape=[-1, 8])
    assert blk.out_shape == (16, 8)


def test_reshape_explicit_full_shape():
    blk = ReshapeBlock(in_shape=(2, 13, 32), out_shape=[2, 13, 32])
    assert blk.out_shape == (2, 13, 32)


def test_reshape_non_divisible_raises():
    with pytest.raises(ValueError):
        ReshapeBlock(in_shape=(2, 8, 8), out_shape=[5, -1])  # 128 % 5 != 0


def test_reshape_two_negatives_raise():
    with pytest.raises(ValueError):
        ReshapeBlock(in_shape=(2, 8, 8), out_shape=[-1, -1])


def test_reshape_forward_is_pure_view():
    blk = ReshapeBlock(in_shape=(2, 4, 4), out_shape=[-1])
    x = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
    y = blk(x)
    assert y.shape == (2, 32)
    assert torch.equal(y.flatten(), x.flatten())


def test_reshape_has_zero_params():
    blk = ReshapeBlock(in_shape=(2, 8, 8), out_shape=[-1])
    assert sum(p.numel() for p in blk.parameters()) == 0


# ---------- dw_sep_conv ----------

def test_dw_sep_default_bias_is_false_when_bn_on():
    blk = DwSepConvBlock(in_shape=(4, 8, 8), out_channels=8, kernel=3)
    assert blk.dw.bias is None
    assert blk.pw.bias is None
    assert isinstance(blk.bn1, nn.BatchNorm2d)
    assert isinstance(blk.bn2, nn.BatchNorm2d)


def test_dw_sep_bias_true_override_wins():
    blk = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, bias_dw=True, bias_pw=True,
    )
    assert blk.dw.bias is not None
    assert blk.pw.bias is not None


def test_dw_sep_no_bn_uses_identity_and_default_bias_true():
    blk = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, use_bn1=False, use_bn2=False,
    )
    assert isinstance(blk.bn1, nn.Identity)
    assert isinstance(blk.bn2, nn.Identity)
    # default bias = NOT use_bn → True when BN is off
    assert blk.dw.bias is not None
    assert blk.pw.bias is not None
    assert blk.fusion_pairs == []


def test_dw_sep_forward_shape():
    blk = DwSepConvBlock(in_shape=(2, 6, 6), out_channels=8, kernel=3).eval()
    x = torch.randn(3, 2, 6, 6)
    y = blk(x)
    assert y.shape == (3, 8, 6, 6)


def test_dw_sep_profiler_fuses_bn_into_convs():
    """Fused-model accounting: BN params drop out, the absorbing conv gains
    `out_channels` extra bias params (effective bias from the fold)."""
    blk = DwSepConvBlock(in_shape=(4, 8, 8), out_channels=8, kernel=3)
    raw = sum(p.numel() for p in blk.parameters())
    adj = blk.count_params()
    bn_params = sum(p.numel() for p in blk.bn1.parameters()) + sum(
        p.numel() for p in blk.bn2.parameters()
    )
    # DW absorber has in_channels=4 → out_channels=4 bias; PW out_channels=8 → 8 bias.
    fused_bias = 4 + 8
    assert adj == raw - bn_params + fused_bias
    assert bn_params > 0


def test_dw_sep_profiler_no_fusion_without_bn():
    blk = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, use_bn1=False, use_bn2=False,
    )
    raw = sum(p.numel() for p in blk.parameters())
    assert blk.count_params() == raw
    assert blk.fusion_pairs == []


def test_dw_sep_profiler_flops_match_fused_view():
    """With BN fused, the BN ops are gone but the conv-bias adds are counted."""
    blk_bn = DwSepConvBlock(in_shape=(4, 8, 8), out_channels=8, kernel=3)
    blk_no_bn = DwSepConvBlock(
        in_shape=(4, 8, 8), out_channels=8, kernel=3, use_bn1=False, use_bn2=False,
    )
    f_bn = blk_bn.count_flops((4, 8, 8))
    f_no = blk_no_bn.count_flops((4, 8, 8))
    # The conv ops with bias are identical between the two (no_bn has actual bias=True;
    # bn variant has bias=False but is forced to True via fusion). The bn variant adds
    # activation ops on bn1/bn2 output (same shape as no_bn's act1/act2 path).
    assert f_bn == f_no


# ---------- residual ----------

def _cnn_spec(channels: int = 4, kernel: int = 3, norm: str = "none"):
    return {
        "name": "cnn_block",
        "channels": channels,
        "kernel": kernel,
        "norm": norm,
        "activation": "identity",
    }


def test_residual_identity_skip_matches_manual_sum():
    torch.manual_seed(0)
    blk = ResidualBlock(
        in_shape=(4, 6, 6),
        main_blocks=[_cnn_spec(channels=4)],
        skip_blocks=[],
        post_activation="identity",
    ).eval()
    x = torch.randn(2, 4, 6, 6)
    y = blk(x)
    main_only = blk.main_blocks[0](x)
    assert torch.allclose(y, main_only + x, atol=1e-6)


def test_residual_shape_mismatch_with_empty_skip_raises():
    with pytest.raises(ValueError):
        ResidualBlock(
            in_shape=(4, 6, 6),
            main_blocks=[_cnn_spec(channels=8)],  # changes channels
            skip_blocks=[],
        )


def test_residual_explicit_projection_skip():
    blk = ResidualBlock(
        in_shape=(4, 6, 6),
        main_blocks=[_cnn_spec(channels=8)],
        skip_blocks=[_cnn_spec(channels=8, kernel=1)],
    )
    x = torch.randn(2, 4, 6, 6)
    y = blk(x)
    assert y.shape == (2, 8, 6, 6)


def test_residual_post_activation_clamps_negatives():
    blk = ResidualBlock(
        in_shape=(4, 4, 4),
        main_blocks=[_cnn_spec(channels=4)],
        skip_blocks=[],
        post_activation="relu",
    )
    x = torch.randn(2, 4, 4, 4)
    y = blk(x)
    assert (y >= 0).all()


def test_residual_nested_dw_sep_bn_fused():
    """Profiler walks recursively: nested dw_sep_conv's BNs fold into its convs."""
    blk = ResidualBlock(
        in_shape=(4, 6, 6),
        main_blocks=[
            {"name": "dw_sep_conv", "out_channels": 4, "kernel": 3},
        ],
        skip_blocks=[],
    )
    raw = sum(p.numel() for p in blk.parameters())
    adj = blk.count_params()
    inner = blk.main_blocks[0]
    bn_params = sum(p.numel() for p in inner.bn1.parameters()) + sum(
        p.numel() for p in inner.bn2.parameters()
    )
    # DW absorber gains in_channels=4 bias; PW absorber gains out_channels=4 bias.
    fused_bias = 4 + 4
    assert adj == raw - bn_params + fused_bias
    assert bn_params > 0
