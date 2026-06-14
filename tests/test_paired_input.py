"""Augmented-input training: PairedInputDataset + data_factory wiring.

The encoder input comes from an augmented dataset while the reconstruction
target stays the clean target CSI ("augmented CSI -> target CSI").
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from csi_comp.data import NpzDataset, PairedInputDataset
from csi_comp.training import build_dataloaders, build_val_loader


def _base_cfg(npz_root):
    return {
        "format": "npz",
        "train_path": str(npz_root / "train.npz"),
        "val_path": str(npz_root / "val.npz"),
        "max_subband": 8,
        "max_port": 16,
        "batch_size": 4,
    }


# ----- PairedInputDataset unit behaviour -----

def test_paired_swaps_input_keeps_target(npz_root, tmp_path, make_npz):
    """Encoder input (real/imag) comes from input_ds; target stays from target_ds."""
    aug = make_npz(tmp_path / "aug_train.npz", n=8, seed=99)  # different D, same shape
    target_ds = NpzDataset(npz_root / "train.npz")
    input_ds = NpzDataset(aug)
    paired = PairedInputDataset(target_ds, input_ds)

    assert len(paired) == len(target_ds)
    s = paired[0]
    # real/imag taken from the augmented input dataset
    assert torch.equal(s["real"], input_ds[0]["real"])
    assert torch.equal(s["imag"], input_ds[0]["imag"])
    # reconstruction target taken from the clean target dataset
    assert torch.equal(s["real_target"], target_ds[0]["real_target"])
    assert torch.equal(s["imag_target"], target_ds[0]["imag_target"])
    # the two datasets actually differ (sanity: swap was meaningful)
    assert not torch.equal(s["real"], target_ds[0]["real"])


def test_paired_length_mismatch_raises(npz_root, tmp_path, make_npz):
    short = make_npz(tmp_path / "aug_short.npz", n=4, seed=7)
    with pytest.raises(ValueError, match="length mismatch"):
        PairedInputDataset(NpzDataset(npz_root / "train.npz"), NpzDataset(short))


def test_paired_shape_mismatch_raises(npz_root, tmp_path, make_npz):
    odd = make_npz(tmp_path / "aug_odd.npz", n=8, S=5, P=7, seed=3)
    paired = PairedInputDataset(NpzDataset(npz_root / "train.npz"), NpzDataset(odd))
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = paired[0]


# ----- data_factory wiring -----

def test_build_dataloaders_wraps_when_aug_paths_set(npz_root, tmp_path, make_npz):
    aug_train = make_npz(tmp_path / "aug_train.npz", n=8, seed=11)
    aug_val = make_npz(tmp_path / "aug_val.npz", n=4, seed=12)
    cfg = _base_cfg(npz_root)
    cfg["aug_train_path"] = str(aug_train)
    cfg["aug_val_path"] = str(aug_val)
    train, val = build_dataloaders(cfg)
    assert isinstance(train.dataset, PairedInputDataset)
    assert isinstance(val.dataset, PairedInputDataset)


def test_build_dataloaders_no_wrap_without_aug(npz_root):
    train, val = build_dataloaders(_base_cfg(npz_root))
    assert not isinstance(train.dataset, PairedInputDataset)
    assert not isinstance(val.dataset, PairedInputDataset)
    assert isinstance(train.dataset, NpzDataset)


def test_build_dataloaders_only_train_aug(npz_root, tmp_path, make_npz):
    """aug_train_path / aug_val_path are independent — train-only is allowed."""
    aug_train = make_npz(tmp_path / "aug_train.npz", n=8, seed=21)
    cfg = _base_cfg(npz_root)
    cfg["aug_train_path"] = str(aug_train)
    train, val = build_dataloaders(cfg)
    assert isinstance(train.dataset, PairedInputDataset)
    assert not isinstance(val.dataset, PairedInputDataset)


def test_build_val_loader_wraps_with_aug_val(npz_root, tmp_path, make_npz):
    aug_val = make_npz(tmp_path / "aug_val.npz", n=4, seed=31)
    cfg = {
        "format": "npz",
        "val_path": str(npz_root / "val.npz"),
        "max_subband": 8,
        "max_port": 16,
        "batch_size": 4,
        "aug_val_path": str(aug_val),
    }
    val = build_val_loader(cfg)
    assert isinstance(val.dataset, PairedInputDataset)


# ----- per-component encoder-input scale (phase augmentation) -----

def test_aug_dataset_args_scales_input_only(npz_root, tmp_path, make_npz):
    """aug_dataset_args.scale_real/imag scale the encoder input; the clean target
    keeps the base `scale`."""
    aug_train = make_npz(tmp_path / "aug_train.npz", n=8, seed=11)
    cfg = _base_cfg(npz_root)
    cfg["aug_train_path"] = str(aug_train)
    cfg["dataset_args"] = {"scale": 1.0, "target_offset": 0.0}
    cfg["aug_dataset_args"] = {"scale_real": 2.0, "scale_imag": 3.0}
    cfg["train_loader"] = {"shuffle": False}
    train, _ = build_dataloaders(cfg)

    batch = next(iter(train))
    aug_ds = NpzDataset(aug_train, scale=1.0, target_offset=0.0)
    tgt_ds = NpzDataset(npz_root / "train.npz", scale=1.0, target_offset=0.0)
    S, P = batch["true_shapes"][0]
    # encoder input = augmented D × per-component scale
    assert torch.allclose(batch["real"][0, :S, :P], aug_ds[0]["real"] * 2.0)
    assert torch.allclose(batch["imag"][0, :S, :P], aug_ds[0]["imag"] * 3.0)
    # reconstruction target = clean D × base scale (NOT per-component)
    assert torch.allclose(batch["real_target"][0, :S, :P], tgt_ds[0]["real_target"])
    assert torch.allclose(batch["imag_target"][0, :S, :P], tgt_ds[0]["imag_target"])


def test_scale_real_in_dataset_args_routes_to_aug_only(npz_root, tmp_path, make_npz):
    """scale_real may sit in dataset_args: it still only scales the aug input,
    never the target (the base build strips it)."""
    aug_train = make_npz(tmp_path / "aug_train.npz", n=8, seed=13)
    cfg = _base_cfg(npz_root)
    cfg["aug_train_path"] = str(aug_train)
    cfg["dataset_args"] = {"scale": 1.0, "target_offset": 0.0, "scale_real": 2.0}
    cfg["train_loader"] = {"shuffle": False}
    train, _ = build_dataloaders(cfg)

    batch = next(iter(train))
    aug_ds = NpzDataset(aug_train, scale=1.0, target_offset=0.0)
    tgt_ds = NpzDataset(npz_root / "train.npz", scale=1.0, target_offset=0.0)
    S, P = batch["true_shapes"][0]
    assert torch.allclose(batch["real"][0, :S, :P], aug_ds[0]["real"] * 2.0)
    assert torch.allclose(batch["real_target"][0, :S, :P], tgt_ds[0]["real_target"])


def test_scale_real_ignored_without_aug(npz_root):
    """Without aug paths the per-component scale is a no-op (aug data가 있을 때만)."""
    cfg = _base_cfg(npz_root)
    cfg["dataset_args"] = {"scale": 1.0, "target_offset": 0.0, "scale_real": 2.0}
    cfg["train_loader"] = {"shuffle": False}
    train, _ = build_dataloaders(cfg)

    batch = next(iter(train))
    base = NpzDataset(npz_root / "train.npz", scale=1.0, target_offset=0.0)
    S, P = batch["true_shapes"][0]
    # scale_real NOT applied — encoder input == base scale
    assert torch.allclose(batch["real"][0, :S, :P], base[0]["real"])


def test_paired_batch_flows_through_collate(npz_root, tmp_path, make_npz):
    """End-to-end: a paired loader yields a batch with swapped encoder input
    and clean target, padded correctly."""
    aug_train = make_npz(tmp_path / "aug_train.npz", n=8, seed=41)
    cfg = _base_cfg(npz_root)
    cfg["aug_train_path"] = str(aug_train)
    cfg["train_loader"] = {"shuffle": False}
    train, _ = build_dataloaders(cfg)
    batch = next(iter(train))
    assert batch["real"].shape == (4, 8, 16)
    assert "real_target" in batch and batch["real_target"].shape == (4, 8, 16)

    # First sample's valid region: input from aug, target from clean train set.
    aug_ds = NpzDataset(aug_train)
    tgt_ds = NpzDataset(npz_root / "train.npz")
    S, P = batch["true_shapes"][0]
    assert torch.allclose(batch["real"][0, :S, :P], aug_ds[0]["real"])
    assert torch.allclose(batch["real_target"][0, :S, :P], tgt_ds[0]["real_target"])
