"""Evaluate a checkpoint on the val split.

    python scripts/test.py --checkpoint outputs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import apply_cli_device, set_cuda_visible_early, set_cuda_visible_from_args


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (repeatable)",
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default=None)
    ap.add_argument("--gpu-index", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    set_cuda_visible_from_args(args)  # must precede torch import
    import torch
    from csi_comp.config import apply_overrides, resolve
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.models.latent_mask import parse_latent_mask_spec
    from csi_comp.training import (
        Trainer, build_dataloaders, build_model,
        compile_autoencoder_inplace, configure_device, get_mode_spec,
        resolve_amp_cfg, seed_everything,
    )
    from csi_comp.training.checkpoint import load_checkpoint

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = sd.get("config")
    if not cfg:
        print("checkpoint has no embedded config", file=sys.stderr)
        return 2
    apply_overrides(cfg, args.overrides)
    cfg = resolve(cfg)
    apply_cli_device(cfg["experiment"], args)
    set_cuda_visible_early(cfg["experiment"])

    seed_everything(cfg["experiment"].get("seed", 0))
    device = configure_device(cfg["experiment"])

    mode = cfg["training"]["mode"]
    spec = get_mode_spec(mode)
    ae, _, _ = build_model(cfg, spec)
    load_checkpoint(args.checkpoint, ae, optimizer=None, scheduler=None, strict=False)
    compile_autoencoder_inplace(ae, cfg["training"].get("compile"))
    amp_spec = resolve_amp_cfg(cfg["training"].get("amp"), device)
    mask_spec = parse_latent_mask_spec(cfg["model"].get("latent_mask"))

    _, val_loader = build_dataloaders(cfg["data"])
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode=mode)
    # No-op optimizer to satisfy the Trainer constructor.
    optimizer = torch.optim.SGD(
        [p for p in ae.parameters() if p.requires_grad]
        or [torch.zeros(1, requires_grad=True)],
        lr=0.0,
    )

    trainer = Trainer(
        model=ae, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=val_loader, val_loader=val_loader,
        mode_spec=spec, device=device,
        epochs=0,
        best_metric=cfg["training"].get("best_metric", {"name": "sgcs", "mode": "max"}),
        amp_spec=amp_spec,
        mask_spec=mask_spec,
    )
    val_metrics = trainer.validate()
    for k, v in sorted(val_metrics.items()):
        print(f"{k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
