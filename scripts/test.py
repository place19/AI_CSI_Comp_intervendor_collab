"""Evaluate a checkpoint on the val split or a supplied data path.

    # single checkpoint
    python scripts/test.py --checkpoint outputs/<run>/best.pt

    # single checkpoint with a specific test dataset
    python scripts/test.py --checkpoint outputs/<run>/best.pt \\
        --data-path /path/to/test_data.npz

    # cross-checkpoint: encoder and decoder from separate checkpoints
    python scripts/test.py \\
        --encoder-checkpoint outputs/<enc_run>/best.pt \\
        --decoder-checkpoint outputs/<dec_run>/best.pt

    # cross-checkpoint with a specific test dataset
    python scripts/test.py \\
        --encoder-checkpoint outputs/<enc_run>/best.pt \\
        --decoder-checkpoint outputs/<dec_run>/best.pt \\
        --data-path /path/to/test_data.npz
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from _common import (
    apply_cli_device, check_quantizer_compat,
    set_cuda_visible_early, set_cuda_visible_from_args,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="single checkpoint (encoder + decoder in one file)")
    ap.add_argument("--encoder-checkpoint", type=Path, default=None,
                    help="checkpoint to load encoder (and quantizer) from")
    ap.add_argument("--decoder-checkpoint", type=Path, default=None,
                    help="checkpoint to load decoder from")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (repeatable)",
    )
    ap.add_argument("--data-path", type=Path, default=None,
                    help="override the validation data path from the checkpoint config")
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    ap.add_argument("--gpu-index", type=int, default=None)
    args = ap.parse_args()

    cross = (args.encoder_checkpoint, args.decoder_checkpoint)
    if args.checkpoint is None and not all(cross):
        ap.error(
            "specify either --checkpoint or both "
            "--encoder-checkpoint and --decoder-checkpoint"
        )
    if args.checkpoint is not None and any(cross):
        ap.error(
            "--checkpoint cannot be combined with "
            "--encoder-checkpoint / --decoder-checkpoint"
        )
    if any(cross) and not all(cross):
        ap.error(
            "--encoder-checkpoint and --decoder-checkpoint "
            "must be specified together"
        )
    return args


def _merge_cross_cfg(enc_sd: dict, dec_sd: dict) -> dict:
    enc_cfg = enc_sd.get("config") or {}
    dec_cfg = dec_sd.get("config") or {}
    if not enc_cfg:
        raise ValueError("encoder checkpoint has no embedded config")
    if not dec_cfg:
        raise ValueError("decoder checkpoint has no embedded config")
    cfg = copy.deepcopy(enc_cfg)
    cfg["model"]["decoder"] = copy.deepcopy(dec_cfg["model"]["decoder"])
    cfg["training"]["mode"] = "joint"
    # Default loss for cross-mode evaluation; user can override via --set.
    cfg["loss"] = {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]}
    return cfg


def main() -> int:
    args = _parse_args()
    set_cuda_visible_from_args(args)  # must precede torch import
    import torch
    from csi_comp.config import apply_overrides, resolve
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.models.latent_mask import parse_latent_mask_spec
    from csi_comp.training import (
        Trainer, build_val_loader, build_model,
        compile_autoencoder_inplace, configure_device, get_mode_spec,
        resolve_amp_cfg, seed_everything, uses_cuda_graphs,
    )
    from csi_comp.training.checkpoint import load_checkpoint

    # --data-path is a shortcut for --set data.val_path=...; applied before
    # user --set overrides so explicit --set data.val_path=... wins.
    if args.data_path is not None:
        args.overrides = [f"data.val_path={args.data_path}"] + args.overrides

    if args.encoder_checkpoint and args.decoder_checkpoint:
        # --- cross-checkpoint mode ---
        enc_sd = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
        dec_sd = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
        check_quantizer_compat(
            enc_sd.get("config", {}).get("quantizer", {}),
            dec_sd.get("config", {}).get("quantizer", {}),
        )
        cfg = _merge_cross_cfg(enc_sd, dec_sd)
        apply_overrides(cfg, args.overrides)
        cfg = resolve(cfg)
        apply_cli_device(cfg["experiment"], args)
        set_cuda_visible_early(cfg["experiment"])

        seed_everything(cfg["experiment"].get("seed", 0))
        device = configure_device(cfg["experiment"])

        spec = get_mode_spec("joint")
        ae, _, _ = build_model(cfg, spec)
        load_checkpoint(args.encoder_checkpoint, ae, optimizer=None, scheduler=None, strict=False,
                        components=("encoder", "quantizer"))
        load_checkpoint(args.decoder_checkpoint, ae, optimizer=None, scheduler=None, strict=False,
                        components=("decoder",))
        compile_autoencoder_inplace(ae, cfg["training"].get("compile"))
        amp_spec = resolve_amp_cfg(cfg["training"].get("amp"), device)
        mask_spec = parse_latent_mask_spec(cfg["model"].get("latent_mask"))
        mode = "joint"
    else:
        # --- single checkpoint mode (existing behaviour) ---
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

    val_loader = build_val_loader(cfg["data"])
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
        use_cuda_graphs=uses_cuda_graphs(cfg["training"].get("compile")),
    )
    prefix = "test" if args.data_path is not None else "val"
    val_metrics = trainer.validate(prefix=prefix)
    for k, v in sorted(val_metrics.items()):
        print(f"{k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
