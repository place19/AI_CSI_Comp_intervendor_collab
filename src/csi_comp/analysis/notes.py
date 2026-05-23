"""Build the markdown blob that lands in `mlflow.note.content`."""
from __future__ import annotations

from typing import Any

import yaml

from .profiler import BlockProfile, ProfileResult


def _fmt_block_row(b: BlockProfile) -> str:
    return (
        f"| {b.idx} | `{b.name}` | `{tuple(b.in_shape)}` "
        f"| `{tuple(b.out_shape)}` | {b.num_params:,} | {b.flops:,} |"
    )


def _fmt_side(name: str, blocks: list[BlockProfile], total_params: int, total_flops: int) -> str:
    if not blocks:
        return f"### {name}\n_not built for this mode_\n"
    rows = [
        f"### {name}",
        "",
        "| # | block | in_shape | out_shape | params | FLOPs |",
        "|---|-------|----------|-----------|-------:|------:|",
    ]
    rows.extend(_fmt_block_row(b) for b in blocks)
    rows.append(f"| | **TOTAL** | | | **{total_params:,}** | **{total_flops:,}** |")
    return "\n".join(rows) + "\n"


def build_note(config: dict[str, Any], profile: ProfileResult) -> str:
    parts = [
        "## Configuration",
        "",
        "```yaml",
        yaml.safe_dump(config, sort_keys=False).rstrip(),
        "```",
        "",
        "## Model profile",
        "",
        _fmt_side("Encoder", profile.encoder, profile.encoder_total_params, profile.encoder_total_flops),
        _fmt_side("Decoder", profile.decoder, profile.decoder_total_params, profile.decoder_total_flops),
    ]
    return "\n".join(parts)
