"""Shared CLI plumbing for the entry scripts.

The whole argument surface is deliberately tiny:

    -c / --config PATH   (repeatable — fragments compose left to right)
    KEY=VALUE ...         positional OmegaConf dot-list overrides

Everything else lives in the YAML. This keeps every script ~10 lines and means a run
is fully described by (config files + override list), which is exactly what the sweeper
serialises per job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_path() -> None:
    """Make ``crossec_forecast`` importable when running the script in-place."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-c",
        "--config",
        action="append",
        required=True,
        metavar="PATH",
        help="Experiment YAML. Repeat to compose fragments left to right.",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="OmegaConf dot-list overrides, e.g. train.lr=0.0005 model.name=lstm",
    )
    return p


def split_known_overrides(items: list[str]) -> list[str]:
    """Guard against a mistyped config path landing in the override list."""
    bad = [x for x in items if "=" not in x]
    if bad:
        raise SystemExit(f"Not valid KEY=VALUE overrides (missing '='): {bad}")
    return items
