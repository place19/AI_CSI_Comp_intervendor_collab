"""Tests for the avg_pool and positional_encoding blocks."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from csi_comp.models.blocks.avg_pool import AvgPoolBlock
from csi_comp.models.blocks.positional_encoding import (
    PositionalEncodingBlock,
    _sinusoidal_table,
)
from csi_comp.registry import REGISTRY


# ---------- registry ----------

def test_new_blocks_registered():
    assert "avg_pool" in REGISTRY["block"]
    assert "positional_encoding" in REGISTRY["block"]


# ---------- avg_pool ----------

def test_avg_pool_default_stride_is_one():
    """Default stride is 1 (matching cnn_block), not kernel (PyTorch default)."""
    blk = AvgPoolBlock(in_shape=(2, 8, 8), kernel=2)
    assert blk.out_shape == (2, 7, 7)


def test_avg_pool_strided_downsamples():
    blk = AvgPoolBlock(in_shape=(2, 8, 8), kernel=2, stride=2)
    assert blk.out_shape == (2, 4, 4)
    x = torch.randn(3, 2, 8, 8)
    y = blk(x)
    assert y.shape == (3, 2, 4, 4)


def test_avg_pool_tuple_kernel_and_padding_keep_shape():
    blk = AvgPoolBlock(in_shape=(2, 8, 16), kernel=[3, 5], padding=[1, 2])
    assert blk.out_shape == (2, 8, 16)
    x = torch.randn(1, 2, 8, 16)
    y = blk(x)
    assert y.shape == (1, 2, 8, 16)


def test_avg_pool_padding_same_works_with_stride1():
    blk = AvgPoolBlock(in_shape=(2, 8, 8), kernel=3, padding="same")
    assert blk.out_shape == (2, 8, 8)


def test_avg_pool_padding_same_rejects_strided():
    with pytest.raises(ValueError):
        AvgPoolBlock(in_shape=(2, 8, 8), kernel=3, padding="same", stride=2)


def test_avg_pool_has_zero_params():
    blk = AvgPoolBlock(in_shape=(2, 8, 8), kernel=2)
    assert sum(p.numel() for p in blk.parameters()) == 0


def test_avg_pool_forward_matches_nn_avgpool2d():
    blk = AvgPoolBlock(in_shape=(4, 8, 8), kernel=2, stride=2)
    ref = nn.AvgPool2d(kernel_size=2, stride=2)
    x = torch.randn(2, 4, 8, 8)
    assert torch.allclose(blk(x), ref(x), atol=1e-6)


def test_avg_pool_non_positive_output_raises():
    with pytest.raises(ValueError):
        AvgPoolBlock(in_shape=(2, 8, 8), kernel=10)  # too big with padding=0


def test_avg_pool_wrong_rank_raises():
    with pytest.raises(ValueError):
        AvgPoolBlock(in_shape=(8, 8), kernel=2)


# ---------- positional_encoding ----------

@pytest.mark.parametrize("mode", ["fixed_sincos", "learnable_random", "learnable_sincos"])
def test_pe_forward_preserves_shape(mode):
    blk = PositionalEncodingBlock(in_shape=(13, 64), mode=mode, seq_len=13, dim=64)
    x = torch.randn(4, 13, 64)
    y = blk(x)
    assert y.shape == (4, 13, 64)


def test_pe_fixed_sincos_has_zero_trainable_params():
    blk = PositionalEncodingBlock(in_shape=(13, 64), mode="fixed_sincos", seq_len=13, dim=64)
    trainable = sum(p.numel() for p in blk.parameters() if p.requires_grad)
    assert trainable == 0
    # but the table should still ship in state_dict (so checkpoint round-trips it)
    assert "pe" in blk.state_dict()


def test_pe_learnable_random_has_expected_params_and_nonzero_std():
    torch.manual_seed(0)
    blk = PositionalEncodingBlock(
        in_shape=(13, 64), mode="learnable_random", seq_len=13, dim=64, init_std=0.02,
    )
    trainable = sum(p.numel() for p in blk.parameters() if p.requires_grad)
    assert trainable == 13 * 64
    assert blk.pe.std().item() > 0


def test_pe_learnable_sincos_initialised_to_sincos_table():
    blk = PositionalEncodingBlock(
        in_shape=(13, 64), mode="learnable_sincos", seq_len=13, dim=64,
    )
    expected = _sinusoidal_table(13, 64)
    assert torch.equal(blk.pe.detach(), expected)
    # and it must be a real Parameter — a gradient step should change it
    assert isinstance(blk.pe, nn.Parameter)
    blk.pe.sum().backward()
    assert blk.pe.grad is not None
    assert blk.pe.grad.abs().sum().item() > 0


def test_pe_seq_len_or_dim_mismatch_raises():
    with pytest.raises(ValueError, match="in_shape"):
        PositionalEncodingBlock(in_shape=(13, 64), mode="fixed_sincos", seq_len=14, dim=64)
    with pytest.raises(ValueError, match="in_shape"):
        PositionalEncodingBlock(in_shape=(13, 64), mode="fixed_sincos", seq_len=13, dim=32)


def test_pe_wrong_rank_raises():
    with pytest.raises(ValueError, match=r"\(S, F\)"):
        PositionalEncodingBlock(in_shape=(2, 13, 64), mode="fixed_sincos", seq_len=13, dim=64)


def test_pe_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        PositionalEncodingBlock(in_shape=(13, 64), mode="rope", seq_len=13, dim=64)


def test_pe_odd_dim_rejected_for_sincos_modes():
    with pytest.raises(ValueError, match="even"):
        PositionalEncodingBlock(in_shape=(13, 7), mode="fixed_sincos", seq_len=13, dim=7)
    with pytest.raises(ValueError, match="even"):
        PositionalEncodingBlock(in_shape=(13, 7), mode="learnable_sincos", seq_len=13, dim=7)


def test_pe_odd_dim_ok_for_learnable_random():
    # No sin/cos involved, so odd dim is fine.
    blk = PositionalEncodingBlock(in_shape=(13, 7), mode="learnable_random", seq_len=13, dim=7)
    assert blk.out_shape == (13, 7)


def test_pe_fixed_sincos_adds_table_to_zero_input():
    blk = PositionalEncodingBlock(in_shape=(13, 64), mode="fixed_sincos", seq_len=13, dim=64)
    x = torch.zeros(2, 13, 64)
    y = blk(x)
    expected = _sinusoidal_table(13, 64)
    # Both batch elements should equal the PE table.
    assert torch.allclose(y[0], expected, atol=1e-6)
    assert torch.allclose(y[1], expected, atol=1e-6)


def test_pe_dropout_applied():
    """With dropout=1.0 (drop everything) in train mode, output is zero."""
    blk = PositionalEncodingBlock(
        in_shape=(13, 64), mode="fixed_sincos", seq_len=13, dim=64, dropout=1.0,
    ).train()
    x = torch.randn(2, 13, 64)
    y = blk(x)
    assert torch.all(y == 0)
