"""DataLoader knobs flow from yaml through `build_dataloaders` to `DataLoader`."""
from __future__ import annotations

import pytest

from csi_comp.training import build_dataloaders


def _base_cfg(npz_root):
    return {
        "format": "npz",
        "train_path": str(npz_root / "train.npz"),
        "val_path": str(npz_root / "val.npz"),
        "max_subband": 8,
        "max_port": 16,
        "batch_size": 4,
    }


def test_top_level_defaults_apply_to_both_loaders(npz_root):
    cfg = _base_cfg(npz_root)
    train, val = build_dataloaders(cfg)
    assert train.batch_size == 4
    assert val.batch_size == 4
    # Defaults: train shuffles, val does not.
    # DataLoader exposes the sampler type, not the bool directly.
    from torch.utils.data import RandomSampler, SequentialSampler
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert train.drop_last is False
    assert val.drop_last is False
    assert train.num_workers == 0


def test_per_loader_override_wins(npz_root):
    cfg = _base_cfg(npz_root)
    cfg["train_loader"] = {"batch_size": 2, "drop_last": True}
    cfg["val_loader"] = {"batch_size": 8}
    train, val = build_dataloaders(cfg)
    assert train.batch_size == 2
    assert train.drop_last is True
    assert val.batch_size == 8
    assert val.drop_last is False  # default for val


def test_pin_memory_and_drop_last_top_level(npz_root):
    cfg = _base_cfg(npz_root)
    cfg["pin_memory"] = True
    cfg["drop_last"] = True
    train, val = build_dataloaders(cfg)
    assert train.pin_memory is True
    assert val.pin_memory is True
    assert train.drop_last is True
    assert val.drop_last is True


def test_prefetch_factor_only_with_workers(npz_root):
    # prefetch_factor must not be set on a num_workers=0 DataLoader.
    cfg = _base_cfg(npz_root)
    cfg["prefetch_factor"] = 4   # should be silently dropped (num_workers=0)
    train, _ = build_dataloaders(cfg)
    assert train.num_workers == 0
    # DataLoader normalises prefetch_factor to None when num_workers==0.
    assert train.prefetch_factor is None


def test_persistent_workers_only_with_workers(npz_root):
    cfg = _base_cfg(npz_root)
    cfg["persistent_workers"] = True  # silently dropped at num_workers=0
    train, _ = build_dataloaders(cfg)
    assert train.num_workers == 0
    assert train.persistent_workers is False


def test_train_loader_shuffle_override_to_false(npz_root):
    cfg = _base_cfg(npz_root)
    cfg["train_loader"] = {"shuffle": False}
    train, _ = build_dataloaders(cfg)
    from torch.utils.data import SequentialSampler
    assert isinstance(train.sampler, SequentialSampler)


def test_batch_size_required(npz_root):
    cfg = _base_cfg(npz_root)
    del cfg["batch_size"]
    cfg["train_loader"] = {"batch_size": 4}
    # val_loader gets no batch_size → should fail clearly
    with pytest.raises(ValueError):
        build_dataloaders(cfg)
