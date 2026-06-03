"""LmdbRawDataset exposes Z / Zq teacher latents via expose_z / expose_zq."""
from __future__ import annotations

from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch

from csi_comp.data.lmdb_raw import LmdbRawDataset


def _make_lmdb(path: Path, n: int, S: int, P: int, latent_dim: int,
               with_z: bool = False) -> None:
    """Write D + Zq (and optionally Z) key families for each sample."""
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=64 * 1024 * 1024, subdir=True)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i in range(n):
            d = rng.integers(-128, 128, size=(S, P, 2), dtype=np.int8)
            txn.put(f"D{i:06d}".encode("ascii"), d.tobytes())
            txn.put(f"Zq{i:06d}".encode("ascii"),
                    rng.standard_normal(latent_dim).astype(np.float32).tobytes())
            if with_z:
                txn.put(f"Z{i:06d}".encode("ascii"),
                        rng.standard_normal(latent_dim).astype(np.float32).tobytes())
    env.close()


def test_lmdb_raw_expose_zq(tmp_path):
    n, S, P, latent_dim = 6, 4, 8, 16
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P, expose_zq=True)
    assert len(ds) == n
    sample = ds[3]
    assert sample["real"].shape == (S, P)
    assert sample["latent_target_zq"].shape == (latent_dim,)
    assert sample["latent_target_zq"].dtype == torch.float32
    assert "latent_target" not in sample
    assert "latent_target_z" not in sample


def test_lmdb_raw_expose_both_z_and_zq(tmp_path):
    n, S, P, latent_dim = 4, 3, 4, 8
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim, with_z=True)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P,
                        expose_z=True, expose_zq=True)
    sample = ds[2]
    assert sample["latent_target_z"].shape == (latent_dim,)
    assert sample["latent_target_zq"].shape == (latent_dim,)
    assert not torch.equal(sample["latent_target_z"], sample["latent_target_zq"])


def test_lmdb_raw_expose_missing_key_raises(tmp_path):
    """expose_z=True but no Z family on disk → clear KeyError at access."""
    n, S, P, latent_dim = 3, 4, 8, 16
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim, with_z=False)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P, expose_z=True)
    with pytest.raises(KeyError, match="Z000000"):
        _ = ds[0]


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

    ds = LmdbRawDataset(root=path, subband=S, port=P, expose_zq=True)
    for i in range(n):
        np.testing.assert_array_equal(ds[i]["latent_target_zq"].numpy(), expected[i])


def test_lmdb_raw_counts_exact_pattern_with_aux_families(tmp_path):
    """The real CDL lmdb holds D, Z, *and* Zq keys per sample. len(ds) counts only
    keys matching {key_prefix}{idx:06d} ('D000000' by default) so the extra Z/Zq
    families don't inflate the count."""
    n, S, P, latent_dim = 4, 3, 4, 8
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim, with_z=True)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P, expose_zq=True)
    assert len(ds) == n, f"expected {n} samples, got {len(ds)} (aux keys inflated count)"
    assert ds[n - 1]["latent_target_zq"].shape == (latent_dim,)


def test_lmdb_raw_count_ignores_d_prefixed_metadata_keys(tmp_path):
    """D-prefixed metadata keys (e.g. b'D_meta') must not inflate len(ds).

    The counter requires exactly prefix + 6 digits, so any key that doesn't match
    that pattern (even if it starts with 'D') is silently ignored.
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


def test_lmdb_raw_default_exposes_no_latent(tmp_path):
    """Default (no expose_*) counts only {key_prefix}{idx:06d} and emits no latent.
    Auxiliary key families (Zq here) are silently ignored — keeps len(ds) sane when
    the lmdb holds paired data but the user only wants the precoder side."""
    n, S, P, latent_dim = 5, 4, 6, 16
    _make_lmdb(tmp_path / "db", n, S, P, latent_dim)
    ds = LmdbRawDataset(root=tmp_path / "db", subband=S, port=P)
    assert len(ds) == n
    sample = ds[0]
    assert "latent_target" not in sample
    assert "latent_target_z" not in sample
    assert "latent_target_zq" not in sample
