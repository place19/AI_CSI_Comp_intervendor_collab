"""Single-file .npz dataset.

Layout: one .npz file with D required and Z/Zq optional:
    D   int (typically int8), shape (N, S, P, 2)  — CSI (real/imag last)  [required]
    Z   float, shape (N, latent_dim)              — pre-quant encoder output  [optional]
    Zq  float, shape (N, latent_dim)              — post-quant encoder output  [optional]

The whole file is loaded into RAM at construction (npz is a zip-of-npys; mmap is
not meaningful here). `D` is kept in its on-disk integer dtype to save memory
and cast to float32 ×scale per-sample in __getitem__.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import register
from ..utils import parse_scale

_DEFAULT_SCALE = 1.0 / 128.0
_DEFAULT_TARGET_OFFSET = 1.0 / 256.0


@register("dataset", "npz")
class NpzDataset(Dataset):
    """Args:
        path:           path to the .npz file.
        scale:          multiplicative scale applied after int → float32 cast on `D`.
                        Accepts a number or a hex float64 bit pattern (see
                        `csi_comp.utils.parse_scale`).
        scale_real:     OPTIONAL per-component override of `scale` for the real
                        channel (`D[..., 0]`). When None (default) `scale` is used.
        scale_imag:     OPTIONAL per-component override of `scale` for the imag
                        channel (`D[..., 1]`). When None (default) `scale` is used.
                        These exist for phase-augmentation: the data factory wires
                        them onto the **augmented encoder-input** dataset only, so a
                        run can feed the encoder real/imag scaled differently while
                        the reconstruction target keeps the plain `scale`. Each
                        accepts a number or a hex float64 bit pattern.
        target_offset:  additive offset applied on top of `scale` to produce the
                        reconstruction target (`real_target`/`imag_target`). Models
                        the bin-midpoint dequantization step required by 3GPP/HW
                        (int → float quantizes with `floor`, so the original
                        continuous value sits at `int*scale + scale/2`). Encoder
                        input keeps the raw `int*scale` convention so the model
                        sees what HW actually produces.
        latent_key:     which array becomes `latent_target` in each sample. One of
                        None (default, do not emit `latent_target`), "Zq"
                        (post-quant), or "Z" (pre-quant). This is the "primary
                        latent" slot — it doubles as the decoder input in
                        `decoder_only` mode. Set it only when `latent_target` is
                        actually consumed.
        expose_z:       if True and `Z` is present, every sample also carries
                        `latent_target_z` (float32) — the pre-quant teacher latent.
        expose_zq:      if True and `Zq` is present, every sample also carries
                        `latent_target_zq` (float32) — the post-quant teacher
                        latent. Together with `expose_z` this lets a single run
                        supervise different stages against Z and Zq independently
                        (each loss term selects its target via `target_key`).
                        Exposure is opportunistic: a missing array is silently
                        skipped here; a loss that asks for an absent target raises.
    """

    def __init__(
        self,
        path: str | Path,
        scale: float | str = _DEFAULT_SCALE,
        target_offset: float = _DEFAULT_TARGET_OFFSET,
        scale_real: float | str | None = None,
        scale_imag: float | str | None = None,
        latent_key: Optional[str] = None,
        expose_z: bool = True,
        expose_zq: bool = False,
    ):
        self.path = Path(path)
        with np.load(self.path) as f:
            files = set(f.files)
            D = np.asarray(f["D"])
            Z = np.asarray(f["Z"]) if "Z" in files else None
            Zq = np.asarray(f["Zq"]) if "Zq" in files else None

        if D.ndim != 4 or D.shape[-1] != 2:
            raise ValueError(f"D must have shape (N, S, P, 2); got {D.shape}")
        self.D = torch.from_numpy(np.ascontiguousarray(D))
        self.S, self.P = int(D.shape[1]), int(D.shape[2])
        self.scale = parse_scale(scale)
        self.target_offset = float(target_offset)
        # Per-component scales fall back to the single `scale` when unset.
        self.scale_real = self.scale if scale_real is None else parse_scale(scale_real)
        self.scale_imag = self.scale if scale_imag is None else parse_scale(scale_imag)

        latents = {"Zq": Zq, "Z": Z, None: None}
        if latent_key not in latents:
            raise ValueError(
                f"latent_key must be 'Zq', 'Z', or None; got {latent_key!r}"
            )
        chosen = latents[latent_key]
        if latent_key is not None and chosen is None:
            raise ValueError(
                f"latent_key={latent_key!r} but array {latent_key!r} is absent "
                f"from {self.path}; available arrays: {sorted(files)}"
            )
        for arr_name, arr in (("Z", Z), ("Zq", Zq)):
            if arr is not None and arr.shape[0] != D.shape[0]:
                raise ValueError(
                    f"{arr_name}.shape[0]={arr.shape[0]} != D.shape[0]={D.shape[0]} "
                    f"in {self.path}"
                )
        self.latent_target = (
            torch.from_numpy(np.ascontiguousarray(chosen).astype(np.float32, copy=False))
            if chosen is not None else None
        )
        self.latent_z = (
            torch.from_numpy(np.ascontiguousarray(Z).astype(np.float32, copy=False))
            if (expose_z and Z is not None) else None
        )
        self.latent_zq = (
            torch.from_numpy(np.ascontiguousarray(Zq).astype(np.float32, copy=False))
            if (expose_zq and Zq is not None) else None
        )

    def __len__(self) -> int:
        return int(self.D.shape[0])

    def __getitem__(self, i: int) -> dict[str, Any]:
        d = self.D[i].to(torch.float32)               # (S, P, 2)
        real = d[..., 0] * self.scale_real
        imag = d[..., 1] * self.scale_imag
        out: dict[str, Any] = {
            "real": real,
            "imag": imag,
            "real_target": real + self.target_offset,  # bin-midpoint target
            "imag_target": imag + self.target_offset,
            "true_shape": (self.S, self.P),
        }
        if self.latent_target is not None:
            out["latent_target"] = self.latent_target[i]
        if self.latent_z is not None:
            out["latent_target_z"] = self.latent_z[i]
        if self.latent_zq is not None:
            out["latent_target_zq"] = self.latent_zq[i]
        return out
