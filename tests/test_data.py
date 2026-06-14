import struct

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from csi_comp.data import (
    LayoutAdapter,
    NpzDataset,
    cnn_mask,
    make_collate_fn,
    pad_and_collate,
    transformer_seq_mask,
)
from csi_comp.registry import get as reg_get
from csi_comp.utils import parse_scale


# ----- parse_scale (number | little-endian float64 hex bit pattern) -----

def test_parse_scale_number_passthrough():
    assert parse_scale(0.25) == 0.25
    assert parse_scale(2) == 2.0


def test_parse_scale_hex_little_endian_roundtrip():
    for x in (1.0 / 128.0, 0.5, 2.0, -0.375):
        h = struct.pack("<d", x).hex()  # little-endian bit pattern
        assert parse_scale(h) == x
        assert parse_scale("0x" + h) == x       # 0x prefix tolerated
        assert parse_scale("  " + h + " ") == x  # surrounding whitespace tolerated


def test_parse_scale_hex_known_value():
    # 1/128 == 0.0078125 == 2^-7 → little-endian "000000000000803f".
    assert parse_scale("0x000000000000803f") == 1.0 / 128.0


def test_parse_scale_bad_hex_length_raises():
    with pytest.raises(ValueError, match="8 bytes"):
        parse_scale("0x3f80")  # 2 bytes, not a float64


def test_parse_scale_bad_hex_chars_raises():
    with pytest.raises(ValueError, match="invalid hex"):
        parse_scale("0xnothex")


# ----- per-component encoder-input scale -----

def test_npz_scale_real_imag_override(npz_root):
    ds = NpzDataset(npz_root / "train.npz", scale=1.0,
                    scale_real=2.0, scale_imag=3.0, target_offset=0.0)
    base = NpzDataset(npz_root / "train.npz", scale=1.0, target_offset=0.0)
    s, b = ds[0], base[0]
    assert torch.allclose(s["real"], b["real"] * 2.0)
    assert torch.allclose(s["imag"], b["imag"] * 3.0)
    # target follows the (per-component) scale too — only the factory scopes these
    # to the aug input; the dataset itself stays internally consistent.
    assert torch.allclose(s["real_target"], b["real"] * 2.0)


def test_npz_scale_real_only_falls_back_to_scale_for_imag(npz_root):
    ds = NpzDataset(npz_root / "train.npz", scale=0.01, scale_real=2.0, target_offset=0.0)
    base = NpzDataset(npz_root / "train.npz", scale=0.01, target_offset=0.0)
    s, b = ds[0], base[0]
    assert torch.allclose(s["real"], b["real"] * 200.0)  # 2.0 vs 0.01
    assert torch.allclose(s["imag"], b["imag"])           # imag still uses scale=0.01


def test_npz_scale_hex_string(npz_root):
    h = struct.pack("<d", 2.0).hex()
    ds = NpzDataset(npz_root / "train.npz", scale=1.0, scale_real=h, target_offset=0.0)
    base = NpzDataset(npz_root / "train.npz", scale=1.0, target_offset=0.0)
    assert torch.allclose(ds[0]["real"], base[0]["real"] * 2.0)


def test_npz_dataset_basic(npz_root):
    ds = NpzDataset(npz_root / "train.npz")
    assert len(ds) == 8
    sample = ds[0]
    assert set(sample.keys()) >= {"real", "imag", "real_target", "imag_target", "true_shape"}
    S, P = sample["true_shape"]
    assert sample["real"].shape == (S, P)
    assert sample["imag"].shape == (S, P)
    assert sample["real"].dtype == torch.float32


def test_npz_dataset_default_target_offset_is_one_over_256(npz_root):
    """Default target_offset = 1/256: real_target == real + 1/256 (bin midpoint)."""
    ds = NpzDataset(npz_root / "train.npz")
    sample = ds[0]
    assert torch.allclose(sample["real_target"], sample["real"] + 1.0 / 256.0)
    assert torch.allclose(sample["imag_target"], sample["imag"] + 1.0 / 256.0)


def test_npz_dataset_target_offset_zero_matches_input(npz_root):
    """target_offset=0 makes target identical to encoder input (back-compat)."""
    ds = NpzDataset(npz_root / "train.npz", target_offset=0.0)
    sample = ds[0]
    assert torch.equal(sample["real_target"], sample["real"])
    assert torch.equal(sample["imag_target"], sample["imag"])


def test_npz_dataset_custom_target_offset(npz_root):
    ds = NpzDataset(npz_root / "train.npz", target_offset=0.01)
    sample = ds[0]
    assert torch.allclose(sample["real_target"], sample["real"] + 0.01)


def test_npz_dataset_default_has_no_latent(npz_root):
    ds = NpzDataset(npz_root / "train.npz")
    sample = ds[0]
    assert "latent_target" not in sample


def test_npz_dataset_latent_key_z(npz_root):
    ds = NpzDataset(npz_root / "train.npz", latent_key="Z")
    sample = ds[0]
    # latent_target now equals what latent_target_z also is (both come from Z)
    assert torch.equal(sample["latent_target"], sample["latent_target_z"])


def test_npz_dataset_latent_key_none(npz_root):
    ds = NpzDataset(npz_root / "train.npz", latent_key=None, expose_z=False)
    sample = ds[0]
    assert "latent_target" not in sample
    assert "latent_target_z" not in sample
    assert "latent_target_zq" not in sample


def test_npz_dataset_expose_zq(npz_root):
    """expose_zq surfaces Zq as latent_target_zq, independent of latent_key."""
    ds = NpzDataset(npz_root / "train.npz", latent_key=None,
                    expose_z=False, expose_zq=True)
    sample = ds[0]
    assert "latent_target" not in sample
    assert "latent_target_z" not in sample
    assert sample["latent_target_zq"].shape == (16,)


def test_npz_dataset_expose_both_z_and_zq(npz_root):
    """Z and Zq are exposed under distinct keys (different arrays)."""
    ds = NpzDataset(npz_root / "train.npz", latent_key=None,
                    expose_z=True, expose_zq=True)
    sample = ds[0]
    assert sample["latent_target_z"].shape == (16,)
    assert sample["latent_target_zq"].shape == (16,)
    assert not torch.equal(sample["latent_target_z"], sample["latent_target_zq"])


def test_npz_dataset_bad_latent_key(npz_root):
    with pytest.raises(ValueError):
        NpzDataset(npz_root / "train.npz", latent_key="banana")


def test_npz_dataset_registered():
    assert reg_get("dataset", "npz") is NpzDataset


def test_pad_and_collate_shapes_and_mask():
    a = {"real": torch.ones(5, 8), "imag": torch.full((5, 8), 2.0), "true_shape": (5, 8)}
    b = {"real": torch.ones(7, 10) * 3, "imag": torch.ones(7, 10) * 4, "true_shape": (7, 10)}
    out = pad_and_collate([a, b], max_subband=8, max_port=12)
    assert out["real"].shape == (2, 8, 12)
    assert out["imag"].shape == (2, 8, 12)
    assert out["mask"].shape == (2, 8, 12)
    assert out["true_shapes"] == [(5, 8), (7, 10)]
    assert out["mask"][0, :5, :8].all() and not out["mask"][0, 5:, :].any()
    assert out["mask"][1, :7, :10].all() and not out["mask"][1, 7:, :].any()
    assert (out["real"][0, 5:, :] == 0).all()
    assert (out["real"][0, :, 8:] == 0).all()
    assert torch.equal(out["real"][0, :5, :8], torch.ones(5, 8))
    assert torch.equal(out["imag"][1, :7, :10], torch.ones(7, 10) * 4)


def test_collate_rejects_oversize_sample():
    a = {"real": torch.zeros(20, 8), "imag": torch.zeros(20, 8), "true_shape": (20, 8)}
    with pytest.raises(ValueError):
        pad_and_collate([a], max_subband=8, max_port=8)


def test_collate_latent_all_or_none():
    a = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4)}
    b = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4),
         "latent_target": torch.ones(5)}
    with pytest.raises(ValueError):
        pad_and_collate([a, b], max_subband=8, max_port=8)


def test_collate_latent_z_stacked():
    """latent_target_z stacks on the batch dim alongside latent_target."""
    s = lambda i: {
        "real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4),
        "latent_target": torch.full((5,), float(i)),
        "latent_target_z": torch.full((5,), float(10 + i)),
    }
    out = pad_and_collate([s(0), s(1)], max_subband=8, max_port=8)
    assert out["latent_target"].shape == (2, 5)
    assert out["latent_target_z"].shape == (2, 5)
    assert torch.equal(out["latent_target_z"][0], torch.full((5,), 10.0))
    assert torch.equal(out["latent_target_z"][1], torch.full((5,), 11.0))


def test_collate_latent_zq_stacked():
    """latent_target_zq stacks like the other latent targets."""
    s = lambda i: {
        "real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4),
        "latent_target_zq": torch.full((5,), float(20 + i)),
    }
    out = pad_and_collate([s(0), s(1)], max_subband=8, max_port=8)
    assert out["latent_target_zq"].shape == (2, 5)
    assert torch.equal(out["latent_target_zq"][1], torch.full((5,), 21.0))
    assert "latent_target" not in out


def test_collate_latent_zq_all_or_none():
    a = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4)}
    b = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4),
         "latent_target_zq": torch.ones(5)}
    with pytest.raises(ValueError):
        pad_and_collate([a, b], max_subband=8, max_port=8)


def test_collate_pads_real_imag_target():
    """real_target/imag_target are padded with zeros outside the valid region."""
    a = {
        "real": torch.ones(3, 4), "imag": torch.full((3, 4), 2.0),
        "real_target": torch.full((3, 4), 1.5), "imag_target": torch.full((3, 4), 2.5),
        "true_shape": (3, 4),
    }
    out = pad_and_collate([a], max_subband=5, max_port=6)
    assert "real_target" in out and "imag_target" in out
    assert out["real_target"].shape == (1, 5, 6)
    assert torch.equal(out["real_target"][0, :3, :4], torch.full((3, 4), 1.5))
    assert (out["real_target"][0, 3:, :] == 0).all()
    assert (out["real_target"][0, :, 4:] == 0).all()


def test_collate_real_target_all_or_none():
    a = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4)}
    b = {
        "real": torch.zeros(3, 4), "imag": torch.zeros(3, 4),
        "real_target": torch.zeros(3, 4), "imag_target": torch.zeros(3, 4),
        "true_shape": (3, 4),
    }
    with pytest.raises(ValueError):
        pad_and_collate([a, b], max_subband=8, max_port=8)


def test_collate_real_target_requires_both_real_and_imag():
    a = {
        "real": torch.zeros(3, 4), "imag": torch.zeros(3, 4),
        "real_target": torch.zeros(3, 4),  # missing imag_target
        "true_shape": (3, 4),
    }
    with pytest.raises(ValueError):
        pad_and_collate([a], max_subband=8, max_port=8)


def test_collate_latent_z_all_or_none():
    a = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4)}
    b = {"real": torch.zeros(3, 4), "imag": torch.zeros(3, 4), "true_shape": (3, 4),
         "latent_target_z": torch.ones(5)}
    with pytest.raises(ValueError):
        pad_and_collate([a, b], max_subband=8, max_port=8)


def test_dataloader_end_to_end(npz_root):
    ds = NpzDataset(npz_root / "train.npz", latent_key="Zq")
    loader = DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(8, 16)
    )
    batch = next(iter(loader))
    assert batch["real"].shape == (4, 8, 16)
    assert batch["mask"].dtype == torch.bool
    for i, (S, P) in enumerate(batch["true_shapes"]):
        assert (batch["real"][i, S:, :] == 0).all()
        assert (batch["real"][i, :, P:] == 0).all()
        assert batch["mask"][i, :S, :P].all()
    # latents flow through the collate
    assert "latent_target" in batch and batch["latent_target"].shape == (4, 16)
    assert "latent_target_z" in batch and batch["latent_target_z"].shape == (4, 16)


def test_layout_adapter_cnn():
    real = torch.randn(2, 5, 8)
    imag = torch.randn(2, 5, 8)
    out = LayoutAdapter("cnn")(real, imag)
    assert out.shape == (2, 2, 5, 8)
    assert torch.equal(out[:, 0], imag)
    assert torch.equal(out[:, 1], real)


def test_layout_adapter_transformer_interleaved():
    real = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    imag = real + 100
    out = LayoutAdapter("transformer")(real, imag)
    assert out.shape == (2, 3, 8)
    for b in range(2):
        for s in range(3):
            for p in range(4):
                assert out[b, s, 2 * p] == imag[b, s, p]
                assert out[b, s, 2 * p + 1] == real[b, s, p]


def test_layout_adapter_bad_layout():
    with pytest.raises(ValueError):
        LayoutAdapter("rnn")


def test_mask_helpers():
    mask = torch.tensor(
        [
            [[True, True, False], [True, False, False], [False, False, False]],
            [[True, True, True],  [False, False, False], [False, False, False]],
        ]
    )
    cm = cnn_mask(mask)
    assert cm.shape == (2, 1, 3, 3)
    sm = transformer_seq_mask(mask)
    assert sm.shape == (2, 3)
    assert sm[0, 0].item() is True
    assert sm[0, 2].item() is False
    assert sm[1, 0].item() is True
    assert sm[1, 1].item() is False
