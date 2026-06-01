"""Paired-input dataset: augmented encoder input, clean reconstruction target.

By default the framework trains a `target CSI -> target CSI` autoencoder: the
encoder input (`real`/`imag`) and the reconstruction target
(`real_target`/`imag_target`) both come from the same on-disk `D`.

`PairedInputDataset` enables `augmented CSI -> target CSI` instead. It wraps two
datasets aligned index-by-index:

    target_ds:  supplies the reconstruction target (and any latent/aux fields).
    input_ds:   supplies the encoder input (its `real`/`imag` overwrite the
                target dataset's).

This models real UE conditions, where the CSI seen at the UE (augmented/degraded)
differs from the ideal precoder the decoder should reconstruct. Only meaningful
when the encoder is trained (`joint` / `encoder_only` / `encoder_only_frozen_decoder`);
in `decoder_only` mode the encoder input is unused, so wrapping is a harmless no-op.

The input dataset's own `real_target`/offset/latent fields are ignored — the
augmented file only needs the raw `D`.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class PairedInputDataset(Dataset):
    """Swap the encoder input (`real`/`imag`) with samples from a second dataset.

    Args:
        target_ds:  dataset providing the reconstruction target and aux fields.
        input_ds:   dataset providing the encoder input (its `real`/`imag`).

    Both datasets must have the same length and per-sample `(S, P)` shape; samples
    are paired by index, so `input_ds[i]` must correspond to `target_ds[i]`.
    """

    def __init__(self, target_ds: Dataset, input_ds: Dataset):
        if len(target_ds) != len(input_ds):  # type: ignore[arg-type]
            raise ValueError(
                f"paired datasets length mismatch: target has {len(target_ds)} "  # type: ignore[arg-type]
                f"samples, input has {len(input_ds)}"  # type: ignore[arg-type]
            )
        self.target_ds = target_ds
        self.input_ds = input_ds

    def __len__(self) -> int:
        return len(self.target_ds)  # type: ignore[arg-type]

    def __getitem__(self, i: int) -> dict[str, Any]:
        out = dict(self.target_ds[i])  # keep target/true_shape/latent* fields
        a = self.input_ds[i]
        if a["real"].shape != out["real"].shape:
            raise ValueError(
                f"paired sample {i} shape mismatch: input {tuple(a['real'].shape)} "
                f"!= target {tuple(out['real'].shape)}"
            )
        out["real"] = a["real"]  # encoder input only
        out["imag"] = a["imag"]
        return out
