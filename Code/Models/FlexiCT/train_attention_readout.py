"""CLI entry point for the frozen FlexiCT single-head attention readout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from .attention_cache import DEFAULT_ATTENTION_CACHE_NAMESPACE
    from .attention_readout import FlexiCTAttentionAdapter
except ImportError:  # pragma: no cover
    from attention_cache import DEFAULT_ATTENTION_CACHE_NAMESPACE
    from attention_readout import FlexiCTAttentionAdapter

from _supervised_adaptation.cli import main_for


def _add_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument("--cache_namespace", default=DEFAULT_ATTENTION_CACHE_NAMESPACE)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Cache" / "AttentionTokens" / DEFAULT_ATTENTION_CACHE_NAMESPACE,
    )
    parser.add_argument("--validate_cache_shards", action="store_true")


def _validate(_parser: argparse.ArgumentParser, _args: argparse.Namespace) -> None:
    return None


def main(argv: list[str] | None = None) -> int:
    return main_for(
        argv,
        adapter_factory=FlexiCTAttentionAdapter,
        add_model_arguments=_add_arguments,
        validate_model_arguments=_validate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
