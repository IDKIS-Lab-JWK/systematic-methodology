"""CLI entry point for FlexiCT final-block fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import UPSTREAM_REVISION
    from .final_block_finetune import FlexiCTFinalBlockAdapter
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REVISION
    from final_block_finetune import FlexiCTFinalBlockAdapter

from _supervised_adaptation.cli import main_for


def _add_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument("--upstream_root", type=Path, default=root / "tmp" / "upstream_flexict")
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_trusted_legacy_checkpoint", action="store_true")
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Setup" / "asset_provisioning.jsonl",
    )


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT revision is immutable and must be {UPSTREAM_REVISION}")
    if args.allow_trusted_legacy_checkpoint and not args.checkpoint_sha256:
        parser.error("--allow_trusted_legacy_checkpoint requires --checkpoint_sha256")


def main(argv: list[str] | None = None) -> int:
    return main_for(
        argv,
        adapter_factory=FlexiCTFinalBlockAdapter,
        add_model_arguments=_add_arguments,
        validate_model_arguments=_validate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
