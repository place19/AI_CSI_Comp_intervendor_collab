"""LmdbRawDataset with with_latent=True reads both D and Zq keys correctly."""
from __future__ import annotations

from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch

from csi_comp.data.lmdb_raw import LmdbRawDataset


def _make_lmdb(path: Path, n: int, S: int, P: int, latent_dim: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i in range(n):
            d = rng.integers(-128, 128, size=(S, P, 2), dtype=np.int8)
            zq = rng.standard_normal(latent_dim).astype(np.float32)
            txn.put(f"D{i:06d}".encode("ascii"), d.tobytes())
            txn.put(f"Zq{i:06d}".encode("ascii"), zq.tobytes())
    env.close()


def test_lmdb_raw_loads_paired_latent(tmp_path):
    n, S, P, latent_dim = 6, 4, 8, 16
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim)
    ds = LmdbRawDataset(
        root=tmp_path / "db",
        subband=S, port=P,
        with_latent=True,
    )
    assert len(ds) == n
    sample = ds[3]
    assert sample["real"].shape == (S, P)
    assert sample["imag"].shape == (S, P)
    assert sample["latent_target"].shape == (latent_dim,)
    assert sample["latent_target"].dtype == torch.float32


def test_lmdb_raw_latent_values_round_trip(tmp_path):
    """Bytes go in → tensor comes out matching the original float32 contents."""
    n, S, P, latent_dim = 2, 3, 4, 8
    path = tmp_path / "db"
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=8 * 1024 * 1024, subdir=True)
    expected = np.arange(n * latent_dim, dtype=np.float32).reshape(n, latent_dim)
    with env.begin(write=True) as txn:
        for i in range(n):
            d = np.zeros((S, P, 2), dtype=np.int8)
            txn.put(f"D{i:06d}".encode("ascii"), d.tobytes())
            txn.put(f"Zq{i:06d}".encode("ascii"), expected[i].tobytes())
    env.close()

    ds = LmdbRawDataset(root=path, subband=S, port=P, with_latent=True)
    for i in range(n):
        np.testing.assert_array_equal(ds[i]["latent_target"].numpy(), expected[i])


def test_lmdb_raw_with_latent_counts_only_primary_prefix(tmp_path):
    """The real CDL lmdb holds D, Z, *and* Zq keys per sample. with_latent=True
    must count by the primary `key_prefix` ('D' by default) so extra key families
    (here 'Z' for pre-quant latents) don't inflate the dataset length."""
    n, S, P, latent_dim = 4, 3, 4, 8
    path = tmp_path / "db"
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=8 * 1024 * 1024, subdir=True)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i in range(n):
            txn.put(
                f"D{i:06d}".encode("ascii"),
                rng.integers(-128, 128, size=(S, P, 2), dtype=np.int8).tobytes(),
            )
            txn.put(
                f"Z{i:06d}".encode("ascii"),  # extra family — must be ignored
                rng.standard_normal(latent_dim).astype(np.float32).tobytes(),
            )
            txn.put(
                f"Zq{i:06d}".encode("ascii"),
                rng.standard_normal(latent_dim).astype(np.float32).tobytes(),
            )
    env.close()

    ds = LmdbRawDataset(root=path, subband=S, port=P, with_latent=True)
    assert len(ds) == n, f"expected {n} samples, got {len(ds)} (extra 'Z' keys inflated count)"
    sample = ds[n - 1]
    assert sample["latent_target"].shape == (latent_dim,)


def test_lmdb_raw_count_ignores_d_prefixed_metadata_keys(tmp_path):
    """D-prefixed metadata keys (e.g. b'D_meta') must not inflate len(ds).

    The counter now requires exactly prefix + 6 digits, so any key that doesn't
    match that pattern (even if it starts with 'D') is silently ignored.
    """
    n, S, P = 3, 4, 6
    path = tmp_path / "db"
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=8 * 1024 * 1024, subdir=True)
    rng = np.random.default_rng(42)
    with env.begin(write=True) as txn:
        for i in range(n):
            txn.put(
                f"D{i:06d}".encode("ascii"),
                rng.integers(-128, 128, size=(S, P, 2), dtype=np.int8).tobytes(),
            )
        # Metadata key with same D prefix — must be ignored by the counter.
        txn.put(b"D_meta", b"some metadata payload")
    env.close()

    ds = LmdbRawDataset(root=path, subband=S, port=P)
    assert len(ds) == n, f"expected {n}, got {len(ds)} — D_meta inflated the count"


def test_lmdb_raw_without_latent_counts_primary_only(tmp_path):
    """Default with_latent=False still counts only the primary prefix keys.
    Auxiliary key families (Zq here) in the env are silently ignored — this
    keeps len(ds) sane when the lmdb holds paired data but the user only wants
    the precoder side."""
    n, S, P, latent_dim = 5, 4, 6, 16
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P)
    assert len(ds) == n
    sample = ds[0]
    assert "latent_target" not in sample
