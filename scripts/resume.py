"""Resume training from a checkpoint. Uses the config embedded in the
checkpoint; CLI --set overrides can adjust it (e.g. extend epochs).

    python scripts/resume.py --checkpoint outputs/<run>/latest.pt
    python scripts/resume.py --checkpoint outputs/<run>/latest.pt --set training.epochs=20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _common import (
    add_common_args, apply_cli_device, dump_yaml, resolve_run_name,
    set_cuda_visible_early,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    import torch
    from csi_comp.config import apply_overrides, resolve

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = sd.get("config")
    if not cfg:
        print("checkpoint has no embedded config", file=sys.stderr)
        return 2
    apply_overrides(cfg, args.overrides)
    cfg = resolve(cfg)
    apply_cli_device(cfg["experiment"], args)
    set_cuda_visible_early(cfg["experiment"])

    import contextlib
    from csi_comp.analysis import build_note, profile_model
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.models.latent_mask import parse_latent_mask_spec
    from csi_comp.training import (
        ConsoleCallback, Trainer, build_dataloaders, build_model,
        build_optimizer, build_scheduler, compile_autoencoder_inplace,
        configure_device, get_mode_spec, resolve_amp_cfg, seed_everything,
    )
    from csi_comp.training.checkpoint import CheckpointCallback, load_checkpoint
    from csi_comp.training.mlflow_logger import MLflowCallback, MLflowLogger

    seed_everything(cfg["experiment"].get("seed", 0))
    device = configure_device(cfg["experiment"])

    mode = cfg["training"]["mode"]
    spec = get_mode_spec(mode)
    ae, enc_trace, dec_trace = build_model(cfg, spec)
    train_loader, val_loader = build_dataloaders(cfg["data"])
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode=mode)
    optimizer = build_optimizer(ae, cfg["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        cfg["training"].get("scheduler"),
        epochs=int(cfg["training"]["epochs"]),
        steps_per_epoch=len(train_loader),
    )

    mask_spec = parse_latent_mask_spec(cfg["model"].get("latent_mask"))
    # Load checkpoint into the uncompiled model first, then optionally compile.
    # Saved state_dict never carries `_orig_mod.` prefixes (see compile_utils),
    # so this ordering keeps load semantics simple.
    restored = load_checkpoint(args.checkpoint, ae, optimizer, scheduler, strict=True)
    compile_autoencoder_inplace(ae, cfg["training"].get("compile"))
    amp_spec = resolve_amp_cfg(cfg["training"].get("amp"), device)

    # Resume into a new timestamped folder (suffix `_resume`) so the original
    # run's best.pt / latest.pt are preserved. Use --no-timestamp to opt out
    # (in which case the resumed run writes into `outputs/<name>_resume/`).
    base_name = cfg["experiment"].get("name", f"resume_{int(time.time())}")
    run_name = resolve_run_name(base_name, timestamp=not args.no_timestamp, suffix="_resume")
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, out_dir / "config.resolved.yaml")

    exp_cfg = cfg["experiment"]
    mlf_cfg = exp_cfg.get("mlflow") or {}
    mlflow_enabled = bool(mlf_cfg) and bool(mlf_cfg.get("enabled", True))
    log_every = int(
        exp_cfg.get("log_every_n_iters", mlf_cfg.get("log_every_n_iters", 50))
    )

    logger = None
    if mlflow_enabled:
        logger = MLflowLogger(
            tracking_uri=mlf_cfg["tracking_uri"],
            experiment_name=mlf_cfg["experiment_name"],
            log_every_n_iters=log_every,
            run_name=run_name,
        )

    with (logger if logger is not None else contextlib.nullcontext()):
        prof = profile_model(ae, enc_trace, dec_trace)
        if logger is not None:
            logger.set_note(build_note(cfg, prof))
            logger.log_artifact(out_dir / "config.resolved.yaml")

        callbacks = [ConsoleCallback(log_every_n_iters=log_every)]
        if logger is not None:
            callbacks.append(MLflowCallback(logger))
        callbacks.append(CheckpointCallback(out_dir=out_dir, config=cfg))
        trainer = Trainer(
            model=ae, optimizer=optimizer, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            mode_spec=spec, device=device,
            epochs=int(cfg["training"]["epochs"]),
            val_every_n_epochs=int(cfg["training"].get("val_every_n_epochs", 1)),
            scheduler=scheduler, callbacks=callbacks,
            best_metric=cfg["training"].get("best_metric", {"name": "sgcs", "mode": "max"}),
            amp_spec=amp_spec,
            mask_spec=mask_spec,
        )
        trainer.epoch = restored.epoch + 1
        trainer.global_step = restored.global_step
        trainer.fit()

    print(f"done. outputs at {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
