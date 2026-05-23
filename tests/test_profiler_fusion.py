"""Tests for the fusion-aware profiler default impls (Block.count_flops/params)."""
from __future__ import annotations

import torch.nn as nn

from csi_comp.analysis import op_flops as F
from csi_comp.analysis.profiler import default_block_flops, default_block_params
from csi_comp.models.blocks.base import Block
from csi_comp.models.blocks.cnn import CnnBlock
from csi_comp.models.blocks.dw_sep_conv import DwSepConvBlock


# A minimal "Conv + BN" block to drive fusion testing in isolation.
class ConvBnBlock(Block):
    def __init__(self, in_shape, channels, kernel=3, padding=1, bias=False, with_bn=True):
        super().__init__(in_shape)
        c_in, S, P = self.in_shape
        self.conv = nn.Conv2d(c_in, channels, kernel_size=kernel, padding=padding, bias=bias)
        if with_bn:
            self.bn: nn.Module = nn.BatchNorm2d(channels)
            self.fusion_pairs = [(self.conv, self.bn)]
        else:
            self.bn = nn.Identity()
        self.out_shape = (channels, S, P)

    def forward(self, x):
        return self.bn(self.conv(x))


# ---------- Params under fusion ----------

def test_params_drop_bn_and_add_bias_when_conv_unbiased():
    blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=False, with_bn=True)
    raw = sum(p.numel() for p in blk.parameters())
    bn_params = sum(p.numel() for p in blk.bn.parameters())
    fused = default_block_params(blk)
    assert fused == raw - bn_params + 8  # +out_channels for the fused bias
    assert bn_params == 16  # γ + β


def test_params_no_double_bias_when_conv_already_biased():
    blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=True, with_bn=True)
    raw = sum(p.numel() for p in blk.parameters())
    bn_params = sum(p.numel() for p in blk.bn.parameters())
    fused = default_block_params(blk)
    # Bias was already in `raw`; only the BN is removed.
    assert fused == raw - bn_params


def test_params_no_change_when_no_fusion_declared():
    blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=False, with_bn=False)
    raw = sum(p.numel() for p in blk.parameters())
    assert default_block_params(blk) == raw


# ---------- FLOPs under fusion ----------

def test_flops_fused_equal_biased_conv_alone():
    """Fused Conv(no bias) + BN should report the same FLOPs as a biased Conv alone."""
    fused_blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=False, with_bn=True)
    plain_blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=True, with_bn=False)
    f_fused = default_block_flops(fused_blk, (4, 6, 6))
    f_plain = default_block_flops(plain_blk, (4, 6, 6))
    assert f_fused == f_plain


def test_flops_fused_matches_formula():
    blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=False, with_bn=True)
    f = default_block_flops(blk, (4, 6, 6))
    # Conv2d 4→8, k=3×3, padding=1 → H_out=W_out=6, k_C=4
    # Forced bias=True → 2 * 1 * 8 * 6 * 6 * 3 * 3 * 4
    expected = F.conv2d_flops(N=1, C_out=8, H_out=6, W_out=6, k_h=3, k_w=3, k_C=4, bias=True)
    assert f == expected


def test_flops_unfused_bn_is_counted():
    blk = ConvBnBlock(in_shape=(4, 6, 6), channels=8, bias=True, with_bn=True)
    # Forge "no fusion declared" by clearing pairs after construction.
    blk.fusion_pairs = []
    f = default_block_flops(blk, (4, 6, 6))
    conv = F.conv2d_flops(N=1, C_out=8, H_out=6, W_out=6, k_h=3, k_w=3, k_C=4, bias=True)
    bn = F.batchnorm_flops(N_elements=1 * 8 * 6 * 6)
    assert f == conv + bn


# ---------- Block migration regressions ----------

def test_cnn_block_now_fuses_its_bn():
    """cnn_block previously didn't declare BN as fusable — this is now fixed."""
    blk = CnnBlock(in_shape=(4, 8, 8), channels=16, kernel=3, norm="batchnorm")
    assert len(blk.fusion_pairs) == 1
    absorber, absorbee = blk.fusion_pairs[0]
    assert absorber is blk.conv
    assert absorbee is blk.norm


def test_cnn_block_no_fusion_for_groupnorm_or_none():
    blk_gn = CnnBlock(in_shape=(4, 8, 8), channels=16, kernel=3, norm="groupnorm")
    blk_none = CnnBlock(in_shape=(4, 8, 8), channels=16, kernel=3, norm=None)
    assert blk_gn.fusion_pairs == []
    assert blk_none.fusion_pairs == []


def test_dw_sep_pairs_each_bn_with_its_conv():
    blk = DwSepConvBlock(in_shape=(4, 8, 8), out_channels=8, kernel=3)
    assert len(blk.fusion_pairs) == 2
    assert blk.fusion_pairs[0][0] is blk.dw
    assert blk.fusion_pairs[0][1] is blk.bn1
    assert blk.fusion_pairs[1][0] is blk.pw
    assert blk.fusion_pairs[1][1] is blk.bn2
