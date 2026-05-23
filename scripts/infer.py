"""Dump per-sample inference outputs from a trained checkpoint.

    python scripts/infer.py \\
        --checkpoint outputs/<run>/best.pt \\
        --data-path ../make_lmdb/test \\
        --out outputs/<run>/infer_test

Outputs in `--out` (one `.npy` per item; load directly with `np.load(path)`):
    recon.npy            (N, max_subband, max_port, 2)   reconstructed precoder
    latent.npy           (N, latent_dim)                 encoder output (pre-quant)
    quant_latent.npy     (N, latent_dim)                 post-quant latent (decoder input)
    mask.npy             (N, max_subband, max_port)      valid-cell mask (bool)
    sgcs_per_sample.npy  (N,)                            per-sample SGCS (mean over valid SBs)
    original.npy         (N, max_subband, max_port, 2)   input precoder (only with --save all)
    meta.json                                            run metadata + saved items

`--save` selects which of those to write. Default = everything except `original`
(originals are large and usually shared across inference runs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from _common import apply_cli_device, set_cuda_visible_early, set_cuda_visible_from_args


SAVE_ITEMS = ["recon", "latent", "quant_latent", "original", "mask", "sgcs_per_sample"]
DEFAULT_SAVE = [k for k in SAVE_ITEMS if k != "original"]


def _parse_save_arg(raw: str | None) -> List[str]:
    if raw is None:
        return list(DEFAULT_SAVE)
    raw = raw.strip()
    if raw == "all":
        return list(SAVE_ITEMS)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in SAVE_ITEMS]
    if bad:
        raise SystemExit(
            f"unknown --save items {bad}; valid: {SAVE_ITEMS} or 'all'"
        )
    return items


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    ap.add_argument(
        "--data-path", type=Path, default=None,
        help="dataset to run inference on; defaults to data.val_path from the config",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output directory; defaults to <ckpt parent>/infer_<timestamp>",
    )
    ap.add_argument(
        "--save", type=str, default=None,
        help=(
            "comma-separated list from " + str(SAVE_ITEMS) + " or 'all'. "
            "default: everything except 'original'."
        ),
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="optional cap on number of samples (handy for smoke tests)",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    set_cuda_visible_from_args(args)  # must precede torch import
    import torch
    from csi_comp.config import apply_overrides, resolve

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = sd.get("config")
    if not cfg:
        print("checkpoint has no embedded config", file=sys.stderr)
        return 2

    # --data-path is a shortcut for --set data.val_path=...; applied before
    # user --set overrides so explicit --set data.val_path=... wins.
    overrides = []
    if args.data_path is not None:
        overrides.append(f"data.val_path={args.data_path}")
    overrides.extend(args.overrides)
    apply_overrides(cfg, overrides)
    cfg = resolve(cfg)
    apply_cli_device(cfg["experiment"], args)
    set_cuda_visible_early(cfg["experiment"])

    save = _parse_save_arg(args.save)

    # Remaining heavy imports after CUDA_VISIBLE_DEVICES is set.
    import numpy as np
    from csi_comp.losses.sgcs import sgcs_per_subband
    from csi_comp.models.latent_mask import (
        apply_latent_mask, apply_random_latent_mask, parse_latent_mask_spec,
    )
    from csi_comp.training import (
        build_dataloaders, build_model, compile_autoencoder_inplace,
        configure_device, get_mode_spec, seed_everything,
    )
    from csi_comp.training.checkpoint import load_checkpoint

    seed_everything(cfg["experiment"].get("seed", 0))
    device = configure_device(cfg["experiment"])

    mode = cfg["training"]["mode"]
    spec = get_mode_spec(mode)
    ae, _, _ = build_model(cfg, spec)
    load_checkpoint(args.checkpoint, ae, optimizer=None, scheduler=None, strict=False)
    compile_autoencoder_inplace(ae, cfg["training"].get("compile"))
    ae.to(device).eval()
    mask_spec = parse_latent_mask_spec(cfg["model"].get("latent_mask"))

    _, val_loader = build_dataloaders(cfg["data"])

    out_dir = args.out or (args.checkpoint.parent / f"infer_{int(time.time())}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list] = {k: [] for k in save}
    n_done = 0

    with torch.no_grad():
        for batch in val_loader:
            if args.limit is not None and n_done >= args.limit:
                break
            real = batch["real"].to(device)
            imag = batch["imag"].to(device)
            mask = batch["mask"].to(device)

            if spec.needs_encoder:
                latent = ae.encoder(real, imag)
                q_latent = ae.quantizer(latent) if ae.quantizer is not None else latent
                if mask_spec is not None and ae.decoder is not None:
                    if mask_spec.mode == "half":
                        decoder_input = apply_latent_mask(q_latent, mask_spec.mask_ratio)
                    elif mask_spec.mode == "random":
                        decoder_input = apply_random_latent_mask(q_latent, mask_spec.mask_ratio)
                    else:  # dual: full path (matches validate() which uses recon_full)
                        decoder_input = q_latent
                else:
                    decoder_input = q_latent
            else:
                # decoder_only: latent is provided by the dataset
                raw_latent = batch.get("latent_target")
                if raw_latent is None:
                    raise RuntimeError(
                        "decoder_only mode requires 'latent_target' in the batch; "
                        "set dataset_args.latent_key in the config"
                    )
                latent = raw_latent.to(device)
                q_latent = latent
                decoder_input = latent

            recon = ae.decoder(decoder_input) if ae.decoder is not None else None
            out = {"latent": latent, "quantized_latent": q_latent, "recon": recon}

            B = real.shape[0]
            take = B if args.limit is None else min(B, args.limit - n_done)

            recon = out.get("recon")
            if "recon" in save and recon is not None:
                buffers["recon"].append(recon[:take].cpu().numpy())
            if "latent" in save:
                buffers["latent"].append(out["latent"][:take].cpu().numpy())
            if "quant_latent" in save:
                buffers["quant_latent"].append(out["quantized_latent"][:take].cpu().numpy())
            if "original" in save:
                precoder = torch.stack([real, imag], dim=-1)
                buffers["original"].append(precoder[:take].cpu().numpy())
            if "mask" in save:
                buffers["mask"].append(mask[:take].cpu().numpy())
            if "sgcs_per_sample" in save and recon is not None:
                real_t = batch.get("real_target", batch["real"]).to(device)
                imag_t = batch.get("imag_target", batch["imag"]).to(device)
                precoder_t = torch.stack([real_t, imag_t], dim=-1)
                sgcs_sb = sgcs_per_subband(precoder_t, recon)       # (B, S)
                sb_valid = mask.any(dim=-1).to(sgcs_sb.dtype)        # (B, S)
                denom = sb_valid.sum(dim=1).clamp(min=1.0)
                sample_sgcs = (sgcs_sb * sb_valid).sum(dim=1) / denom
                buffers["sgcs_per_sample"].append(sample_sgcs[:take].cpu().numpy())

            n_done += take

    meta: dict = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_path": cfg["data"].get("val_path"),
        "n_samples": int(n_done),
        "device": str(device),
        "saved": [],
    }
    summary = [f"wrote {n_done} samples to {out_dir.resolve()}"]
    for k in save:
        if not buffers.get(k):
            summary.append(f"  skip {k} (decoder/recon unavailable)")
            continue
        arr = np.concatenate(buffers[k], axis=0)
        path = out_dir / f"{k}.npy"
        np.save(path, arr)
        if k == "sgcs_per_sample":
            meta["sgcs_mean"] = float(arr.mean())
            meta["sgcs_std"] = float(arr.std())
            summary.append(
                f"  {k}: shape={tuple(arr.shape)} mean={arr.mean():.4f} "
                f"std={arr.std():.4f} → {path.name}"
            )
        else:
            summary.append(f"  {k}: shape={tuple(arr.shape)} → {path.name}")
        meta["saved"].append(k)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
