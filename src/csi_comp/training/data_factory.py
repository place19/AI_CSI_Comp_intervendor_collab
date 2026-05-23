"""Build train/val DataLoaders from the data section of a config.

YAML schema (all keys optional unless marked required):

    data:
      format: npz | lmdb_raw            # required
      train_path: ...                   # required for build_dataloaders; not used by build_val_loader
      val_path: ...                     # required
      max_subband: int                  # required (for collate)
      max_port: int                     # required (for collate)
      batch_size: int                   # required (used as default for both loaders)

      # Optional defaults applied to both train and val loaders:
      num_workers: int                  # default 0
      pin_memory: bool                  # default False
      prefetch_factor: int              # forwarded only when num_workers > 0
      persistent_workers: bool          # forwarded only when num_workers > 0
      drop_last: bool                   # default False

      # Optional per-split overrides — merged on top of the defaults above:
      train_loader: { batch_size?, shuffle?, num_workers?, prefetch_factor?,
                      pin_memory?, persistent_workers?, drop_last? }
      val_loader:   { batch_size?, shuffle?, num_workers?, ... }

Default `shuffle`: True for train, False for val.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from ..data import make_collate_fn
from ..registry import get as reg_get
from ..utils import filter_kwargs


_LOADER_KEYS = (
    "batch_size",
    "num_workers",
    "pin_memory",
    "prefetch_factor",
    "persistent_workers",
    "drop_last",
)


def _loader_kwargs(loader_cfg: dict[str, Any] | None, *, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge `defaults` with `loader_cfg` and produce kwargs for `DataLoader`."""
    cfg = {**defaults, **(loader_cfg or {})}
    if "batch_size" not in cfg:
        raise ValueError("data: batch_size is required (either top-level or per-loader)")
    if "shuffle" not in cfg:
        raise ValueError("internal: defaults must always supply `shuffle`")
    nw = int(cfg.get("num_workers", 0))
    kw: dict[str, Any] = dict(
        batch_size=int(cfg["batch_size"]),
        shuffle=bool(cfg["shuffle"]),
        num_workers=nw,
        drop_last=bool(cfg.get("drop_last", False)),
        pin_memory=bool(cfg.get("pin_memory", False)),
    )
    if nw > 0:
        if "prefetch_factor" in cfg and cfg["prefetch_factor"] is not None:
            kw["prefetch_factor"] = int(cfg["prefetch_factor"])
        if "persistent_workers" in cfg and cfg["persistent_workers"] is not None:
            kw["persistent_workers"] = bool(cfg["persistent_workers"])
    return kw


def build_val_loader(data_cfg: dict[str, Any]) -> DataLoader:
    """Build only the val DataLoader — use in test/infer scripts where train data is not needed.

    Top-level `drop_last` is intentionally ignored: it exists for training convenience
    (e.g. keeping batch sizes uniform) but evaluation must see every sample. Only an
    explicit `val_loader.drop_last` override is honoured.
    """
    fmt = data_cfg["format"]
    cls = reg_get("dataset", fmt)
    extra: dict[str, Any] = filter_kwargs(cls.__init__, dict(data_cfg.get("dataset_args", {}) or {}))
    val_ds = cls(data_cfg["val_path"], **extra)
    coll = make_collate_fn(int(data_cfg["max_subband"]), int(data_cfg["max_port"]))
    # Exclude drop_last from common so training-convenience settings don't silently
    # drop the last partial batch during evaluation.
    _VAL_KEYS = tuple(k for k in _LOADER_KEYS if k != "drop_last")
    common: dict[str, Any] = {k: data_cfg[k] for k in _VAL_KEYS if k in data_cfg}
    val_kw = _loader_kwargs(
        data_cfg.get("val_loader"),
        defaults={**common, "shuffle": False, "drop_last": False},
    )
    return DataLoader(val_ds, collate_fn=coll, **val_kw)


def build_dataloaders(data_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    fmt = data_cfg["format"]
    cls = reg_get("dataset", fmt)
    extra: dict[str, Any] = filter_kwargs(cls.__init__, dict(data_cfg.get("dataset_args", {}) or {}))
    train_ds = cls(data_cfg["train_path"], **extra)
    val_ds = cls(data_cfg["val_path"], **extra)
    coll = make_collate_fn(int(data_cfg["max_subband"]), int(data_cfg["max_port"]))

    # Top-level fields act as defaults for both loaders.
    common: dict[str, Any] = {k: data_cfg[k] for k in _LOADER_KEYS if k in data_cfg}

    train_kw = _loader_kwargs(
        data_cfg.get("train_loader"),
        defaults={**common, "shuffle": True},
    )
    val_kw = _loader_kwargs(
        data_cfg.get("val_loader"),
        defaults={**common, "shuffle": False},
    )

    train_loader = DataLoader(train_ds, collate_fn=coll, **train_kw)
    val_loader = DataLoader(val_ds, collate_fn=coll, **val_kw)
    return train_loader, val_loader
