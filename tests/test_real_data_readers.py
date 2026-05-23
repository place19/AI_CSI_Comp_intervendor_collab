"""Sanity tests against the real CDL data sitting in ../make_lmdb.

These tests are skipped automatically when that directory is absent so the suite
stays portable.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

REAL_ROOT = Path(__file__).resolve().parents[1] / ".." / "make_lmdb"
REAL_ROOT = REAL_ROOT.resolve()


requires_lmdb_raw = pytest.mark.skipif(
    not (REAL_ROOT / "train").exists(),
    reason="real CDL lmdb_raw data not present in ../make_lmdb",
)

requires_npz = pytest.mark.skipif(
    not (REAL_ROOT / "train_dataset.npz").exists(),
    reason="real CDL npz not present in ../make_lmdb",
)


@requires_lmdb_raw
def test_lmdb_raw_shapes():
    from csi_comp.data import LmdbRawDataset

    ds = LmdbRawDataset(REAL_ROOT / "train", subband=13, port=32)
    s = ds[0]
    assert s["true_shape"] == (13, 32)
    assert s["real"].shape == (13, 32)
    assert s["imag"].shape == (13, 32)
    assert s["real"].dtype == torch.float32
    assert s["real"].abs().max().item() < 1.0


@requires_lmdb_raw
def test_dataloader_yields_expected_batch():
    from torch.utils.data import DataLoader
    from csi_comp.data import LmdbRawDataset, make_collate_fn

    ds = LmdbRawDataset(REAL_ROOT / "valid", subband=13, port=32)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate_fn(13, 32))
    batch = next(iter(loader))
    assert batch["real"].shape == (8, 13, 32)
    assert batch["mask"].all()


@requires_npz
def test_npz_shapes_and_latents():
    from csi_comp.data import NpzDataset

    ds = NpzDataset(REAL_ROOT / "train_dataset.npz", latent_key="Zq")
    s = ds[0]
    assert s["true_shape"] == (13, 32)
    assert s["real"].shape == (13, 32)
    assert s["real"].dtype == torch.float32
    assert s["real"].abs().max().item() < 1.0
    assert "latent_target" in s
    assert "latent_target_z" in s
    assert s["latent_target"].dtype == torch.float32
    assert s["latent_target_z"].dtype == torch.float32
