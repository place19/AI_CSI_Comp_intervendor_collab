"""resolve_run_name composes outputs/<name>/ and MLflow run name."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _common import resolve_run_name  # noqa: E402


def test_no_timestamp_returns_base():
    assert resolve_run_name("CNN_Res_TF_Dec", timestamp=False) == "CNN_Res_TF_Dec"


def test_with_timestamp_format():
    out = resolve_run_name("my_run", timestamp=True, now=0.0)
    assert out.startswith("my_run_")
    # _YYYYMMDD_HHMMSS — 15 chars after the underscore.
    tail = out.split("_", 1)[1].split("_", 1)[1]  # drop "my_run_"
    assert re.fullmatch(r"\d{8}_\d{6}", tail), f"unexpected ts: {tail!r}"


def test_suffix_appended_after_timestamp():
    out = resolve_run_name("base", timestamp=True, suffix="_resume", now=1717000000.0)
    # Pattern: base_<YYYYMMDD_HHMMSS>_resume
    assert out.startswith("base_")
    assert out.endswith("_resume")
    middle = out[len("base_") : -len("_resume")]
    assert re.fullmatch(r"\d{8}_\d{6}", middle)


def test_suffix_without_timestamp():
    assert resolve_run_name("base", timestamp=False, suffix="_resume") == "base_resume"


def test_no_double_underscore_when_suffix_starts_with_underscore():
    out = resolve_run_name("base", timestamp=False, suffix="_resume")
    assert "__" not in out
