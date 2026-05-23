"""Tests for the complex_ffn_head block."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from csi_comp.models.blocks.complex_ffn_head import ComplexFFNHead
from csi_comp.registry import REGISTRY


def test_block_registered():
    assert "complex_ffn_head" in REGISTRY["block"]


def test_default_stack_axis_last():
    blk = ComplexFFNHead(in_shape=(13, 64), max_port=32)
    assert blk.out_shape == (13, 32, 2)
    x = torch.randn(4, 13, 64)
    y = blk(x)
    assert y.shape == (4, 13, 32, 2)


def test_stack_axis_minus_two():
    blk = ComplexFFNHead(in_shape=(13, 64), max_port=32, stack_axis=-2)
    assert blk.out_shape == (13, 2, 32)
    x = torch.randn(2, 13, 64)
    y = blk(x)
    assert y.shape == (2, 13, 2, 32)


def test_stack_axis_minus_three():
    blk = ComplexFFNHead(in_shape=(13, 64), max_port=32, stack_axis=-3)
    assert blk.out_shape == (2, 13, 32)
    x = torch.randn(2, 13, 64)
    y = blk(x)
    assert y.shape == (2, 2, 13, 32)


def test_stack_axis_positive_one_matches_minus_three():
    blk_neg = ComplexFFNHead(in_shape=(8, 16), max_port=4, stack_axis=-3)
    blk_pos = ComplexFFNHead(in_shape=(8, 16), max_port=4, stack_axis=1)
    assert blk_neg.out_shape == blk_pos.out_shape == (2, 8, 4)


def test_ff_dim_defaults_to_4F():
    blk = ComplexFFNHead(in_shape=(13, 64), max_port=32)
    assert blk.ff_dim == 256
    first_linear = blk.real_ffn[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.out_features == 256


def test_ff_dim_explicit_honoured():
    blk = ComplexFFNHead(in_shape=(13, 64), max_port=32, ff_dim=128)
    assert blk.ff_dim == 128
    assert blk.real_ffn[0].out_features == 128


def test_activation_kwarg_affects_output():
    """gelu vs relu produce different outputs when the weights are identical."""
    torch.manual_seed(0)
    blk_gelu = ComplexFFNHead(in_shape=(8, 16), max_port=4, activation="gelu")
    blk_relu = ComplexFFNHead(in_shape=(8, 16), max_port=4, activation="relu")
    # Copy weights so only the activation differs.
    blk_relu.load_state_dict(blk_gelu.state_dict())
    x = torch.randn(2, 8, 16)
    y_gelu = blk_gelu(x)
    y_relu = blk_relu(x)
    assert y_gelu.shape == y_relu.shape
    assert not torch.allclose(y_gelu, y_relu, atol=1e-4)


def test_real_imag_branches_are_independent():
    """Zeroing real_ffn weights should change only the [...,0] slice (default axis)."""
    torch.manual_seed(0)
    blk = ComplexFFNHead(in_shape=(8, 16), max_port=4)
    x = torch.randn(2, 8, 16)
    y_before = blk(x)
    with torch.no_grad():
        for p in blk.real_ffn.parameters():
            p.zero_()
    y_after = blk(x)
    # The imag slice (..., 1) must be unchanged; the real slice (..., 0) must change.
    assert torch.allclose(y_before[..., 1], y_after[..., 1])
    assert not torch.allclose(y_before[..., 0], y_after[..., 0])


def test_output_slices_match_branch_outputs():
    """Default axis: y[..., 0] == real_ffn(x); y[..., 1] == imag_ffn(x)."""
    blk = ComplexFFNHead(in_shape=(8, 16), max_port=4)
    x = torch.randn(2, 8, 16)
    y = blk(x)
    assert torch.allclose(y[..., 0], blk.real_ffn(x))
    assert torch.allclose(y[..., 1], blk.imag_ffn(x))


def test_gradients_flow_into_both_branches():
    blk = ComplexFFNHead(in_shape=(8, 16), max_port=4)
    x = torch.randn(2, 8, 16)
    blk(x).sum().backward()
    real_grad_sum = sum(p.grad.abs().sum().item() for p in blk.real_ffn.parameters())
    imag_grad_sum = sum(p.grad.abs().sum().item() for p in blk.imag_ffn.parameters())
    assert real_grad_sum > 0
    assert imag_grad_sum > 0


def test_wrong_rank_input_raises():
    with pytest.raises(ValueError, match=r"\(S, F\)"):
        ComplexFFNHead(in_shape=(2, 8, 16), max_port=4)


def test_stack_axis_zero_rejected():
    with pytest.raises(ValueError, match="batch"):
        ComplexFFNHead(in_shape=(8, 16), max_port=4, stack_axis=0)


def test_stack_axis_out_of_range_rejected():
    with pytest.raises(ValueError, match="out of range"):
        ComplexFFNHead(in_shape=(8, 16), max_port=4, stack_axis=5)


def test_max_port_must_be_positive():
    with pytest.raises(ValueError, match="max_port"):
        ComplexFFNHead(in_shape=(8, 16), max_port=0)


def test_dropout_between_linears():
    """With dropout=1.0 in train mode, the middle dropout zeros everything →
    both branch outputs are just the second Linear's bias."""
    blk = ComplexFFNHead(in_shape=(8, 16), max_port=4, dropout=1.0).train()
    x = torch.randn(2, 8, 16)
    y = blk(x)
    # Each branch reduces to bias only, so output is constant over the batch + S dims.
    real_bias = blk.real_ffn[-1].bias  # (max_port,)
    imag_bias = blk.imag_ffn[-1].bias
    expected_real = real_bias.expand(2, 8, 4)
    expected_imag = imag_bias.expand(2, 8, 4)
    assert torch.allclose(y[..., 0], expected_real)
    assert torch.allclose(y[..., 1], expected_imag)
