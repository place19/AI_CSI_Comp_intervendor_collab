"""Evaluate a checkpoint on the val split.

    python scripts/test.py --config <cfg> --checkpoint outputs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_common_args, apply_cli_device, load_resolved_config, set_cuda_visible_early,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--checkpoint", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.config is None:
        print("--config is required", file=sys.stderr)
        return 2
    cfg = load_resolved_config(args.config, args.overrides)
    apply_cli_device(cfg["experiment"], args)
    set_cuda_visible_early(cfg["experiment"])

    import torch
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.models.latent_mask import parse_latent_mask_spec
    from csi_comp.training import (
        Trainer, build_dataloaders, build_model, build_optimizer,
        compile_autoencoder_inplace, configure_device, get_mode_spec,
        resolve_amp_cfg, seed_everything,
    )
    from csi_comp.training.checkpoint import load_checkpoint

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
    optimizer = torch.optim.SGD([p for p in ae.parameters() if p.requires_grad] or [torch.zeros(1, requires_grad=True)], lr=0.0)

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
