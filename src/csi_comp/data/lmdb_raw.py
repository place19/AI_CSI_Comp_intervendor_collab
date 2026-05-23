"""Raw-bytes LMDB dataset.

Matches the data produced by `../make_lmdb/make_lmdb.py`:
    key:    f"D{idx:06d}".encode("ascii")    e.g. b"D000000"
    value:  raw int8 bytes of shape (S, P, 2), little-endian, C-order

No __index__ key, no msgpack envelope. The sample count is derived by counting
keys that match the exact pattern `{prefix}{idx:06d}` (prefix + 6 decimal digits),
so D-prefixed metadata keys (e.g. "D_meta") don't inflate the count.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import register

_DEFAULT_SCALE = 1.0 / 128.0
_DEFAULT_TARGET_OFFSET = 1.0 / 256.0


def _np_dtype(spec: str) -> np.dtype:
    return np.dtype(spec) if spec.startswith(("<", ">", "=", "|")) else np.dtype("<" + spec)


@register("dataset", "lmdb_raw")
class LmdbRawDataset(Dataset):
    """Args:
        root:                path to the lmdb env (directory containing data.mdb / lock.mdb).
        subband:             S, the subband count.
        port:                P, the port count.
        scale:               multiplicative scale applied after int8 → float32 cast.
        key_prefix:          default "D" — keys look like "D000000".
        dtype:               on-disk integer dtype ("i1" for int8, "i2" for int16, etc.).
        with_latent:         if True, also load a paired latent target.
        latent_key_prefix:   default "Zq" — keys look like "Zq000000".
        latent_dtype:        on-disk float dtype for the latent ("f4" = float32).

    When `with_latent=True` the lmdb is expected to hold paired latent keys
    (e.g. b"Zq000000" alongside b"D000000"). `len(self)` counts only keys with
    the primary prefix so auxiliary key families do not inflate the count.
    """

    def __init__(
        self,
        root: str | Path,
        subband: int,
        port: int,
        scale: float = _DEFAULT_SCALE,
        target_offset: float = _DEFAULT_TARGET_OFFSET,
        key_prefix: str = "D",
        dtype: str = "i1",
        with_latent: bool = False,
        latent_key_prefix: str = "Zq",
        latent_dtype: str = "f4",
    ):
        self.root = str(Path(root))
        self.S = int(subband)
        self.P = int(port)
        self.scale = float(scale)
        self.target_offset = float(target_offset)
        self.key_prefix = key_prefix
        self.np_dtype = _np_dtype(dtype)
        self.with_latent = bool(with_latent)
        self.latent_key_prefix = latent_key_prefix
        self.latent_np_dtype = _np_dtype(latent_dtype)
        self._env = None
        self._txn = None
        with self._open_env() as env:
            # Always count only keys with the primary prefix — the env may hold
            # auxiliary key families (e.g. Z, Zq for paired latents) and we
            # don't want to inflate `len(self)` by counting them.
            self._n = self._count_keys_with_prefix(env, self.key_prefix)

    def _open_env(self) -> lmdb.Environment:
        return lmdb.open(
            self.root, readonly=True, lock=False, readahead=False, subdir=True,
        )

    @staticmethod
    def _count_keys_with_prefix(env: lmdb.Environment, prefix: str) -> int:
        """Walk the env once at init time and count keys matching `prefix` + 6 digits.

        Using the exact `{prefix}{idx:06d}` pattern (rather than startswith) prevents
        D-prefixed metadata keys (e.g. b"D_meta") from inflating the count.
        """
        pbytes = prefix.encode("ascii")
        key_len = len(pbytes) + 6  # e.g. b"D000000" is 7 bytes
        n = 0
        with env.begin(write=False) as txn:
            cur = txn.cursor()
            if not cur.set_range(pbytes):
                return 0
            while True:
                k = bytes(cur.key())
                if not k.startswith(pbytes):
                    break
                # Only count keys that match exactly prefix + 6 ASCII digits.
                if len(k) == key_len and k[len(pbytes):].isdigit():
                    n += 1
                if not cur.next():
                    break
        return n

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = self._open_env()
        return self._env

    def _ensure_txn(self):
        """Open one long-lived read txn per worker process and reuse it.
        LMDB is read-only during training, so the MVCC snapshot is fine; this
        skips the per-sample txn open/close that dominated __getitem__."""
        if self._txn is None:
            self._txn = self._ensure_env().begin(write=False, buffers=True)
        return self._txn

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, Any]:
        key = f"{self.key_prefix}{i:06d}".encode("ascii")
        txn = self._ensure_txn()
        raw = txn.get(key)
        latent_raw = None
        if self.with_latent:
            lkey = f"{self.latent_key_prefix}{i:06d}".encode("ascii")
            latent_raw = txn.get(lkey)
            if latent_raw is None:
                raise KeyError(f"missing latent key {lkey!r} in lmdb at {self.root}")
        if raw is None:
            raise KeyError(f"missing key {key!r} in lmdb at {self.root}")
        a = np.frombuffer(raw, dtype=self.np_dtype).reshape(self.S, self.P, 2)
        a = a.astype(np.float32) * self.scale
        t = torch.from_numpy(a)
        tgt = t + self.target_offset                  # bin-midpoint target
        sample: dict[str, Any] = {
            "real": t[..., 0],
            "imag": t[..., 1],
            "real_target": tgt[..., 0],
            "imag_target": tgt[..., 1],
            "true_shape": (self.S, self.P),
        }
        if latent_raw is not None:
            # Stored 1D float32 (or whatever latent_dtype was set to). Cast to float32
            # so downstream code is dtype-uniform regardless of on-disk format.
            lat = np.frombuffer(latent_raw, dtype=self.latent_np_dtype).astype(np.float32, copy=False)
            # .copy() because torch.from_numpy on a read-only buffer view warns.
            sample["latent_target"] = torch.from_numpy(lat.copy())
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state
