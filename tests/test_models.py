import pytest
import torch
from torch.utils.data import DataLoader

from csi_comp.data import NpzDataset, make_collate_fn
from csi_comp.models import (
    Autoencoder,
    BlockTraceEntry,
    build_decoder,
    build_encoder,
)
from csi_comp.models.blocks.cnn import CnnBlock
from csi_comp.models.blocks.heads import BoundingHead, ReshapeHead
from csi_comp.models.blocks.mlp import LinearProj
from csi_comp.models.blocks.transformer import TransformerBlock
from csi_comp.registry import REGISTRY


def test_blocks_registered():
    for name in ("cnn_block", "transformer_block", "linear_proj", "bounding_head", "reshape_head"):
        assert name in REGISTRY["block"], f"missing block: {name}"


def test_cnn_block_shapes():
    """CNN block treats input as a fixed-shape feature map and has a
    single forward(x) -> x contract — no mask threading."""
    blk = CnnBlock(in_shape=(2, 8, 12), channels=16, kernel=3)
    assert blk.out_shape == (16, 8, 12)
    x = torch.randn(4, 2, 8, 12)
    y = blk(x)
    assert y.shape == (4, 16, 8, 12)


def test_linear_proj_shapes():
    blk = LinearProj(in_shape=(16, 8, 12), out_dim=64)
    assert blk.out_shape == (64,)
    x = torch.randn(4, 16, 8, 12)
    y = blk(x)
    assert y.shape == (4, 64)


def test_transformer_block_shapes():
    blk = TransformerBlock(in_shape=(8, 16), d_model=16, nhead=4)
    assert blk.out_shape == (8, 16)
    x = torch.randn(4, 8, 16)
    y = blk(x)
    assert y.shape == (4, 8, 16)


def test_bounding_head_tanh_range():
    head = BoundingHead(in_shape=(64,), activation="tanh", value_range=(-1.0, 1.0))
    x = torch.randn(4, 64) * 100  # extreme
    y = head(x)
    assert y.shape == (4, 64)
    assert y.min().item() >= -1.0 - 1e-6
    assert y.max().item() <= 1.0 + 1e-6


def test_bounding_head_sigmoid_custom_range():
    head = BoundingHead(in_shape=(8,), activation="sigmoid", value_range=(2.0, 5.0))
    x = torch.randn(2, 8) * 50
    y = head(x)
    assert y.min().item() >= 2.0 - 1e-6
    assert y.max().item() <= 5.0 + 1e-6


def test_bounding_head_invalid():
    with pytest.raises(ValueError):
        BoundingHead(in_shape=(8,), activation="softmax")
    with pytest.raises(ValueError):
        BoundingHead(in_shape=(8,), value_range=(1.0, 1.0))


def test_reshape_head_shapes():
    head = ReshapeHead(in_shape=(128,), max_subband=8, max_port=12)
    assert head.out_shape == (8, 12, 2)
    x = torch.randn(4, 128)
    y = head(x)
    assert y.shape == (4, 8, 12, 2)


def _make_cfg(layout: str, max_S: int = 8, max_P: int = 12):
    return (
        {
            "encoder": {
                "blocks": (
                    [
                        {"name": "cnn_block", "channels": 8, "kernel": 3},
                        {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                        {"name": "activation", "activation": "tanh"},
                    ]
                    if layout == "cnn"
                    else [
                        {"name": "transformer_block", "d_model": max_P * 2, "nhead": 4},
                        {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                        {"name": "activation", "activation": "tanh"},
                    ]
                ),
            },
            "decoder": {
                "blocks": [
                    {"name": "linear_proj", "out_dim": 64, "activation": "relu"},
                    {"name": "reshape_head", "max_subband": max_S, "max_port": max_P},
                ],
            },
        },
        {"layout": layout, "max_subband": max_S, "max_port": max_P},
    )


def test_build_encoder_cnn_trace():
    model_cfg, data_cfg = _make_cfg("cnn")
    enc, trace = build_encoder(model_cfg, data_cfg)
    # cnn_block + linear_proj + activation (terminal, explicit)
    assert [t.name for t in trace] == ["cnn_block", "linear_proj", "activation"]
    assert trace[0].in_shape == (2, 8, 12)
    assert trace[0].out_shape == (8, 8, 12)
    assert trace[1].in_shape == (8, 8, 12)
    assert trace[1].out_shape == (32,)
    assert trace[2].in_shape == (32,)
    assert trace[2].out_shape == (32,)
    assert all(isinstance(t, BlockTraceEntry) for t in trace)


def test_build_decoder_trace():
    model_cfg, data_cfg = _make_cfg("cnn")
    dec, trace = build_decoder(model_cfg, data_cfg, latent_shape=(32,))
    # User config now ends in an explicit reshape_head; the builder no longer
    # auto-appends one.
    assert [t.name for t in trace] == ["linear_proj", "reshape_head"]
    assert trace[0].in_shape == (32,)
    assert trace[0].out_shape == (64,)
    assert trace[1].in_shape == (64,)
    assert trace[1].out_shape == (8, 12, 2)


def test_build_decoder_raises_when_terminal_shape_mismatches():
    """If user's decoder doesn't terminate in (max_S, max_P, 2), build_decoder
    raises a clear error pointing at the head options."""
    model_cfg, data_cfg = _make_cfg("cnn")
    # Drop the terminating reshape_head to force a shape mismatch.
    model_cfg["decoder"]["blocks"] = [
        {"name": "linear_proj", "out_dim": 64, "activation": "relu"},
    ]
    with pytest.raises(ValueError, match="reshape_head|complex_ffn_head"):
        build_decoder(model_cfg, data_cfg, latent_shape=(32,))


def test_autoencoder_cnn_forward(npz_root):
    model_cfg, data_cfg = _make_cfg("cnn")
    enc, _ = build_encoder(model_cfg, data_cfg)
    dec, _ = build_decoder(model_cfg, data_cfg, latent_shape=enc.blocks[-1].out_shape)
    ae = Autoencoder(enc, quantizer=None, decoder=dec)

    ds = NpzDataset(npz_root / "train.npz")
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=make_collate_fn(8, 12))
    batch = next(iter(loader))

    out = ae(batch["real"], batch["imag"])
    assert out["recon"].shape == (4, 8, 12, 2)
    # Latent stays in the encoder's bounded range
    assert out["latent"].min().item() >= -1.0 - 1e-6
    assert out["latent"].max().item() <= 1.0 + 1e-6


def test_autoencoder_transformer_forward(npz_root):
    model_cfg, data_cfg = _make_cfg("transformer")
    enc, _ = build_encoder(model_cfg, data_cfg)
    dec, _ = build_decoder(model_cfg, data_cfg, latent_shape=enc.blocks[-1].out_shape)
    ae = Autoencoder(enc, quantizer=None, decoder=dec)

    ds = NpzDataset(npz_root / "train.npz")
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=make_collate_fn(8, 12))
    batch = next(iter(loader))

    out = ae(batch["real"], batch["imag"])
    assert out["recon"].shape == (4, 8, 12, 2)
    assert out["latent"].shape[0] == 4


def test_autoencoder_backward_runs():
    """Verify gradients flow end-to-end through encoder + decoder."""
    model_cfg, data_cfg = _make_cfg("cnn", max_S=6, max_P=8)
    enc, _ = build_encoder(model_cfg, data_cfg)
    dec, _ = build_decoder(model_cfg, data_cfg, latent_shape=enc.blocks[-1].out_shape)
    ae = Autoencoder(enc, decoder=dec)

    real = torch.randn(2, 6, 8)
    imag = torch.randn(2, 6, 8)

    out = ae(real, imag)
    out["recon"].sum().backward()
    # at least one parameter should have a non-trivial gradient
    grads = [p.grad for p in ae.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads)
