"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_npz(path: Path, n: int, S: int, P: int, latent_dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    D = rng.integers(-128, 128, size=(n, S, P, 2), dtype=np.int8)
    Z = rng.standard_normal((n, latent_dim)).astype(np.float32)
    Zq = rng.standard_normal((n, latent_dim)).astype(np.float32)
    np.savez(path, D=D, Z=Z, Zq=Zq)


@pytest.fixture(scope="session")
def npz_root(tmp_path_factory) -> Path:
    """A small npz dataset (train + val) shared across the session.

    Files at <root>/train.npz and <root>/val.npz, each with keys D/Z/Zq.
    Uniform shape S=6, P=10, latent_dim=16.
    """
    root = tmp_path_factory.mktemp("npz")
    _write_npz(root / "train.npz", n=8, S=6, P=10, latent_dim=16, seed=0)
    _write_npz(root / "val.npz", n=4, S=6, P=10, latent_dim=16, seed=1)
    return root


@pytest.fixture
def make_npz():
    """Factory fixture: returns a function that writes an npz at the given path."""
    def _factory(path: Path, n: int, S: int = 6, P: int = 10,
                 latent_dim: int = 16, seed: int = 0) -> Path:
        _write_npz(Path(path), n=n, S=S, P=P, latent_dim=latent_dim, seed=seed)
        return Path(path)
    return _factory
