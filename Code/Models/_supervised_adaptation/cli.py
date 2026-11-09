"""Shared command-line surface for supervised adaptation methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from _linear_probe.common import MODEL_SOURCES, PERCENTAGES, methodology_root

from .compatibility import CompatibleAdaptationBackend
from .workflow import run_pipeline


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser(
    adapter_factory: Callable[[], Any],
    add_model_arguments: Callable[[argparse.ArgumentParser, Path], None],
) -> argparse.ArgumentParser:
    adapter = adapter_factory()
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description=f"{adapter.name} {adapter.method_label} five-class workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("grid", "cv", "test", "all"), default="all")
    parser.add_argument("--percentage", choices=PERCENTAGES, type=int, default=40)
    parser.add_argument(
        "--parameter_source",
        choices=("publication", "grid", "default"),
        default="publication",
    )
    parser.add_argument("--parameter_config", type=Path)
    parser.add_argument("--grid_config", type=Path)
    parser.add_argument("--max_grid_candidates", type=_positive)
    parser.add_argument(
        "--model_source", choices=MODEL_SOURCES, default="refit_full_development"
    )
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument(
        "--results_root", type=Path, default=root / "Results" / adapter.name
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--training_precision", choices=("auto", "bf16", "fp32"), default="auto"
    )
    parser.add_argument("--micro_batch_size", type=_positive, default=1)
    parser.add_argument("--checkpoint_interval", type=_positive, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_model_arguments(parser, root)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.stage in ('test', 'all') and not args.dry_run:
        parser.error('Legacy held-out testing is disabled because it uses persistent test caches. Use python run.py test with source images and explicit selected checkpoints; use --stage cv/grid for legacy training.')
    if args.parameter_source == "publication":
        if (
            args.parameter_config is not None
            or args.grid_config is not None
            or args.max_grid_candidates is not None
        ):
            parser.error(
                "publication uses the packaged complete profile and accepts no profile override/truncation"
            )
    elif args.parameter_source == "grid":
        if args.parameter_config is not None:
            parser.error("grid parameter source does not accept --parameter_config")
    elif args.parameter_source == "default":
        if args.grid_config is not None or args.max_grid_candidates is not None:
            parser.error("default parameter source does not accept grid controls")
    if args.stage == "grid" and args.parameter_source not in (
        "publication",
        "grid",
    ):
        parser.error("grid stage requires publication or grid parameter source")


def main_for(
    argv: list[str] | None,
    *,
    adapter_factory: Callable[[], Any],
    add_model_arguments: Callable[[argparse.ArgumentParser, Path], None],
    validate_model_arguments: Callable[[argparse.ArgumentParser, argparse.Namespace], None],
) -> int:
    parser = build_parser(adapter_factory, add_model_arguments)
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    validate_args(parser, args)
    validate_model_arguments(parser, args)
    try:
        result = run_pipeline(args, CompatibleAdaptationBackend(adapter_factory()))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


__all__ = ["build_parser", "main_for", "validate_args"]

