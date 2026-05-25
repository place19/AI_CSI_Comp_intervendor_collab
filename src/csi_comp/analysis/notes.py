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


def _fmt_summary(profile: ProfileResult) -> str:
    total_params = profile.encoder_total_params + profile.decoder_total_params
    total_flops = profile.encoder_total_flops + profile.decoder_total_flops
    rows = [
        "### Summary",
        "",
        "| | params | FLOPs |",
        "|---|---:|---:|",
        f"| Encoder | {profile.encoder_total_params:,} | {profile.encoder_total_flops:,} |",
        f"| Decoder | {profile.decoder_total_params:,} | {profile.decoder_total_flops:,} |",
        f"| **Total** | **{total_params:,}** | **{total_flops:,}** |",
    ]
    return "\n".join(rows) + "\n"


class _InlineListDumper(yaml.SafeDumper):
    """YAML dumper that renders scalar-only lists in flow style: [a, b, c]."""


def _represent_list(dumper: yaml.Dumper, data: list) -> yaml.Node:
    all_scalar = all(isinstance(x, (int, float, str, bool, type(None))) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=all_scalar)


_InlineListDumper.add_representer(list, _represent_list)


def _yaml_code_block(cfg: dict[str, Any]) -> str:
    """Render config as a 4-space indented code block (CommonMark-safe).

    Fenced code blocks (```) have inconsistent rendering across MLflow viewers;
    the 4-space indent form is part of the original CommonMark spec and more robust.
    """
    text = yaml.dump(cfg, Dumper=_InlineListDumper, sort_keys=False, allow_unicode=True).rstrip()
    return "\n".join("    " + line for line in text.split("\n"))


def build_note(config: dict[str, Any], profile: ProfileResult) -> str:
    parts = [
        "## Model profile",
        "",
        _fmt_summary(profile),
        _fmt_side("Encoder", profile.encoder, profile.encoder_total_params, profile.encoder_total_flops),
        _fmt_side("Decoder", profile.decoder, profile.decoder_total_params, profile.decoder_total_flops),
        "## Configuration",
        "",
        _yaml_code_block(config),
    ]
    return "\n".join(parts)
