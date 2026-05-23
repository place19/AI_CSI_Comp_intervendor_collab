"""Sanity checks for the explicit-Q/K/V/O multi-head self-attention."""
import pytest
import torch

from csi_comp.models.blocks.transformer import MultiHeadSelfAttention, TransformerBlock


def test_mha_shapes():
    mha = MultiHeadSelfAttention(d_model=24, nhead=4)
    x = torch.randn(2, 7, 24)
    y = mha(x)
    assert y.shape == (2, 7, 24)


def test_mha_has_four_linear_weights():
    mha = MultiHeadSelfAttention(d_model=24, nhead=4)
    names = {n for n, _ in mha.named_parameters()}
    # Q, K, V, O each contribute weight + bias
    for prefix in ("W_Q", "W_K", "W_V", "W_O"):
        assert f"{prefix}.weight" in names
        assert f"{prefix}.bias" in names


def test_mha_invalid_dim_raises():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model=10, nhead=3)


def test_mha_gradients_flow_through_all_four_projections():
    mha = MultiHeadSelfAttention(d_model=24, nhead=4)
    x = torch.randn(2, 5, 24, requires_grad=False)
    y = mha(x)
    y.sum().backward()
    for proj in (mha.W_Q, mha.W_K, mha.W_V, mha.W_O):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum().item() > 0


def test_transformer_block_forward_shape():
    """TransformerBlock has the single forward(x) -> x contract (fixed-shape
    inputs, no padding mask threading). Feature dim is preserved — d_model
    must equal F."""
    blk = TransformerBlock(in_shape=(8, 16), d_model=16, nhead=4)
    x = torch.randn(4, 8, 16)
    y = blk(x)
    assert y.shape == (4, 8, 16)


def test_transformer_block_d_model_mismatch_raises():
    """No in-block projection: d_model != F must raise instead of silently projecting."""
    with pytest.raises(ValueError, match="d_model"):
        TransformerBlock(in_shape=(8, 16), d_model=24, nhead=4)


def test_transformer_block_d_model_defaults_to_F():
    """Omitting d_model picks up F from in_shape."""
    blk = TransformerBlock(in_shape=(8, 16), nhead=4)
    assert blk.out_shape == (8, 16)


def test_transformer_block_default_norm_position_is_pre():
    blk = TransformerBlock(in_shape=(8, 16), nhead=4)
    assert blk.norm_position == "pre"


def test_transformer_block_pre_post_norm_differ():
    """Pre-norm and post-norm compute different functions for the same weights."""
    torch.manual_seed(0)
    blk_pre = TransformerBlock(in_shape=(8, 16), nhead=4, norm_position="pre")
    blk_post = TransformerBlock(in_shape=(8, 16), nhead=4, norm_position="post")
    # Copy weights so the only thing that differs is the LN ordering.
    blk_post.load_state_dict(blk_pre.state_dict())
    x = torch.randn(2, 8, 16)
    y_pre = blk_pre(x)
    y_post = blk_post(x)
    assert y_pre.shape == y_post.shape
    assert not torch.allclose(y_pre, y_post, atol=1e-4)


def test_transformer_block_invalid_norm_position_raises():
    with pytest.raises(ValueError, match="norm_position"):
        TransformerBlock(in_shape=(8, 16), nhead=4, norm_position="middle")
