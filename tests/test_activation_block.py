"""Tests for the standalone activation block."""
from __future__ import annotations

import pytest
import torch

from csi_comp.models.blocks.activation import ActivationBlock
from csi_comp.registry import REGISTRY


def test_block_registered():
    assert "activation" in REGISTRY["block"]


def test_default_relu_preserves_shape():
    blk = ActivationBlock(in_shape=(4, 8, 12))
    assert blk.out_shape == (4, 8, 12)
    x = torch.randn(2, 4, 8, 12)
    y = blk(x)
    assert y.shape == x.shape


def test_relu_matches_torch_relu():
    blk = ActivationBlock(in_shape=(8,), activation="relu")
    x = torch.randn(3, 8)
    assert torch.equal(blk(x), torch.relu(x))


def test_gelu_matches_torch_gelu():
    blk = ActivationBlock(in_shape=(8,), activation="gelu")
    x = torch.randn(3, 8)
    assert torch.allclose(blk(x), torch.nn.functional.gelu(x))


def test_identity_passes_through():
    blk = ActivationBlock(in_shape=(8,), activation="identity")
    x = torch.randn(3, 8)
    assert torch.equal(blk(x), x)


def test_zero_params():
    blk = ActivationBlock(in_shape=(4, 8, 12), activation="gelu")
    assert sum(p.numel() for p in blk.parameters()) == 0


def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="unknown activation"):
        ActivationBlock(in_shape=(8,), activation="softmax")


def test_works_in_2d_and_3d_and_4d():
    """Shape-agnostic: 1-D feature, (S, F), (C, H, W) all pass through."""
    for in_shape in [(16,), (8, 16), (4, 8, 12)]:
        blk = ActivationBlock(in_shape=in_shape, activation="tanh")
        x = torch.randn(2, *in_shape)
        y = blk(x)
        assert y.shape == x.shape
        assert blk.out_shape == in_shape
