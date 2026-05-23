"""Train an AI-CSI compression model from a YAML config.

    python scripts/train.py --config configs/examples/joint_cnn.yaml
    python scripts/train.py --config <cfg> --set training.epochs=5 --set data.batch_size=8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _common import (
    add_common_args, apply_cli_device, dump_yaml,
    load_resolved_config, resolve_run_name, set_cuda_visible_early,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.config is None:
        print("--config is required", file=sys.stderr)
        return 2
    cfg = load_resolved_config(args.config, args.overrides)
    apply_cli_device(cfg["experiment"], args)
    set_cuda_visible_early(cfg["experiment"])

    # Heavy imports go after CUDA_VISIBLE_DEVICES is set
    import contextlib
    import torch
    from csi_comp.analysis import build_note, profile_model
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.models.latent_mask import parse_latent_mask_spec
    from csi_comp.training import (
        ConsoleCallback, Trainer, build_dataloaders, build_model,
        build_optimizer, build_scheduler, compile_autoencoder_inplace,
        configure_device, get_mode_spec, resolve_amp_cfg, seed_everything,
    )
    from csi_comp.training.checkpoint import CheckpointCallback
    from csi_comp.training.mlflow_logger import MLflowCallback, MLflowLogger

    seed_everything(cfg["experiment"].get("seed", 0))
    device = configure_device(cfg["experiment"])

    mode = cfg["training"]["mode"]
    spec = get_mode_spec(mode)
    ae, enc_trace, dec_trace = build_model(cfg, spec)
    # Compile encoder/decoder *after* build (and before optimizer construction
    # so the optimizer sees the compiled-module parameters — they share storage
    # with `_orig_mod` so this is a no-op for state but keeps everything tidy).
    compile_autoencoder_inplace(ae, cfg["training"].get("compile"))
    train_loader, val_loader = build_dataloaders(cfg["data"])
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode=mode)
    optimizer = build_optimizer(ae, cfg["training"]["optimizer"])
    amp_spec = resolve_amp_cfg(cfg["training"].get("amp"), device)
    mask_spec = parse_latent_mask_spec(cfg["model"].get("latent_mask"))
    scheduler = build_scheduler(
        optimizer,
        cfg["training"].get("scheduler"),
        epochs=int(cfg["training"]["epochs"]),
        steps_per_epoch=len(train_loader),
    )

    # Append a YYYYMMDD_HHMMSS suffix so each invocation gets its own folder
    # and MLflow run name (preventing overwrites of prior best.pt/latest.pt).
    # Use --no-timestamp to opt out.
    base_name = cfg["experiment"].get("name", f"run_{int(time.time())}")
    run_name = resolve_run_name(base_name, timestamp=not args.no_timestamp)
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = dump_yaml(cfg, out_dir / "config.resolved.yaml")

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
            logger.log_artifact(resolved_cfg_path)
            if args.config is not None:
                logger.log_artifact(args.config)

        callbacks = [ConsoleCallback(log_every_n_iters=log_every)]
        if logger is not None:
            callbacks.append(MLflowCallback(logger))
        callbacks.append(CheckpointCallback(out_dir=out_dir, config=cfg))
        trainer = Trainer(
            model=ae,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            mode_spec=spec,
            device=device,
            epochs=int(cfg["training"]["epochs"]),
            val_every_n_epochs=int(cfg["training"].get("val_every_n_epochs", 1)),
            scheduler=scheduler,
            callbacks=callbacks,
            best_metric=cfg["training"].get("best_metric", {"name": "sgcs", "mode": "max"}),
            amp_spec=amp_spec,
            mask_spec=mask_spec,
        )
        trainer.fit()

    print(f"done. outputs at {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
