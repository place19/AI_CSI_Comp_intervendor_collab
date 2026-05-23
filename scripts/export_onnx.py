"""Export a trained checkpoint to ONNX.

    python scripts/export_onnx.py --checkpoint outputs/<run>/best.pt \\
        --scope encoder,encoder_quant,decoder,full --out outputs/<run>/onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401 — side-effect: makes csi_comp importable without `pip install -e .`

# Scopes available per training mode. Requesting a scope that requires a module
# not trained in that mode (e.g. encoder from a decoder_only checkpoint) would
# export random/unloaded weights — so we reject it early.
_MODE_ALLOWED_SCOPES: dict[str, frozenset[str]] = {
    "encoder_only":               frozenset(("encoder", "encoder_quant")),
    "decoder_only":               frozenset(("decoder",)),
    "joint":                      frozenset(("encoder", "encoder_quant", "decoder", "full")),
    "encoder_only_frozen_decoder": frozenset(("encoder", "encoder_quant", "decoder", "full")),
}
_MODE_DEFAULT_SCOPE: dict[str, str] = {
    "encoder_only": "encoder,encoder_quant",
    "decoder_only": "decoder",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--scope",
        default=None,
        help=(
            "comma-separated subset of encoder,encoder_quant,decoder,full. "
            "Defaults to all scopes valid for the checkpoint's training mode."
        ),
    )
    ap.add_argument("--dynamic-shape", action="store_true")
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--atol", type=float, default=1e-3)
    # Fusion: fold Conv↔BN (and Linear↔BN1d) absorbers declared by each block's
    # `fusion_pairs` metadata before export. Default on — disables only if you
    # want the un-fused graph for debugging or external optimisers.
    ap.add_argument("--fuse", action="store_true", default=True)
    ap.add_argument("--no-fuse", dest="fuse", action="store_false")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    import torch
    from csi_comp.export import VALID_SCOPES, export_to_onnx, verify_onnx_parity
    from csi_comp.training import build_model, get_mode_spec
    from csi_comp.training.checkpoint import load_checkpoint

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = sd.get("config")
    if not cfg:
        print("checkpoint missing embedded config", file=sys.stderr)
        return 2

    original_mode = cfg.get("training", {}).get("mode", "joint")

    # Resolve scope: use explicit --scope, or the mode-appropriate default.
    scope_str = args.scope or _MODE_DEFAULT_SCOPE.get(original_mode, "encoder,encoder_quant,decoder,full")
    scopes = [s.strip() for s in scope_str.split(",") if s.strip()]

    for s in scopes:
        if s not in VALID_SCOPES:
            print(f"unknown scope: {s!r}; valid: {VALID_SCOPES}", file=sys.stderr)
            return 2

    # Reject scopes that require modules not trained in this mode (would export
    # random/unloaded weights).
    allowed = _MODE_ALLOWED_SCOPES.get(original_mode, frozenset(VALID_SCOPES))
    bad = [s for s in scopes if s not in allowed]
    if bad:
        print(
            f"scope(s) {bad} not valid for a {original_mode!r} checkpoint "
            f"(allowed: {sorted(allowed)})",
            file=sys.stderr,
        )
        return 2

    # Build the minimum model for the requested scopes.
    if original_mode == "decoder_only":
        build_mode = "decoder_only"
    elif original_mode == "encoder_only":
        build_mode = "encoder_only"
    else:
        build_mode = "joint"
    cfg = dict(cfg)
    cfg.setdefault("training", {})["mode"] = build_mode
    spec = get_mode_spec(build_mode)
    ae, _, _ = build_model(cfg, spec)
    load_checkpoint(args.checkpoint, ae, optimizer=None, scheduler=None, strict=False)

    args.out.mkdir(parents=True, exist_ok=True)
    for scope in scopes:
        out_path = args.out / f"{scope}.onnx"
        export_to_onnx(ae, cfg, scope=scope, out_path=out_path,
                       dynamic_shape=args.dynamic_shape, fuse=args.fuse)
        print(f"  wrote {out_path}")
        if args.verify:
            diff = verify_onnx_parity(out_path, ae, cfg, scope=scope, atol=args.atol)
            print(f"  parity diff = {diff:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
