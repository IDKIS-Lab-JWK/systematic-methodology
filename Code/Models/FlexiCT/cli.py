"""CLI for the isolated FlexiCT linear-probe workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import DEFAULT_CACHE_NAMESPACE, UPSTREAM_REVISION
    from .backend import FlexiCTBackend
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE, UPSTREAM_REVISION
    from backend import FlexiCTBackend

from _linear_probe.common import MODEL_SOURCES, PERCENTAGES, methodology_root
from _linear_probe.workflow import run_pipeline


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    results = root / "Results" / "FlexiCT"
    parser = argparse.ArgumentParser(
        description="FlexiCT publication-backed frozen linear probing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("grid", "cv", "test", "all"), default="all")
    parser.add_argument("--percentage", choices=PERCENTAGES, type=int, default=40)
    parser.add_argument(
        "--parameter_source",
        choices=("publication", "grid", "default", "manual"),
        default="publication",
    )
    parser.add_argument("--parameter_config", type=Path)
    parser.add_argument("--grid_config", type=Path)
    parser.add_argument("--max_grid_candidates", type=_positive)
    parser.add_argument("--model_source", choices=MODEL_SOURCES, default="refit_full_development")
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--results_root", type=Path, default=results)
    parser.add_argument(
        "--cache_root",
        type=Path,
        help="feature-cache root; defaults to <results_root>/Cache",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument("--upstream_root", type=Path, default=root / "tmp" / "upstream_flexict")
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument(
        "--allow_trusted_legacy_checkpoint",
        action="store_true",
        help=(
            "opt in to weights_only=False only if safe loading fails; requires "
            "--checkpoint_sha256 for a trusted official checkpoint"
        ),
    )
    parser.add_argument(
        "--provision_log",
        type=Path,
        help="asset audit log; defaults to <results_root>/Setup/asset_provisioning.jsonl",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--source_fingerprint",
        choices=("sha256", "stat"),
        default="sha256",
        help="sha256 is the scientific default; stat is an explicitly weaker diagnostic mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=_nonnegative, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    if args.cache_root is None:
        args.cache_root = args.results_root / "Cache"
    if args.provision_log is None:
        args.provision_log = args.results_root / "Setup" / "asset_provisioning.jsonl"
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT requires pinned upstream revision {UPSTREAM_REVISION}")
    if args.allow_trusted_legacy_checkpoint and not args.checkpoint_sha256:
        parser.error("--allow_trusted_legacy_checkpoint requires --checkpoint_sha256")
    if args.num_workers != 0:
        parser.error(
            "FlexiCT linear probing consumes a materialized feature matrix; --num_workers must be 0"
        )
    if not args.dry_run and args.source_fingerprint != "sha256":
        parser.error(
            "FlexiCT LinearProbe execution requires --source_fingerprint sha256; "
            "stat mode is cache diagnostics only"
        )
    if args.parameter_source == "publication":
        if (
            args.parameter_config is not None
            or args.grid_config is not None
            or args.max_grid_candidates is not None
        ):
            parser.error(
                "publication source uses its packaged complete immutable seven-C grid"
            )
    elif args.parameter_source == "grid" and args.parameter_config is not None:
        parser.error("grid parameter source does not accept --parameter_config")
    elif args.parameter_source in ("default", "manual") and (
        args.grid_config is not None or args.max_grid_candidates is not None
    ):
        parser.error(
            "fixed default/manual parameter sources do not accept grid controls"
        )
    if args.parameter_source == "manual" and args.parameter_config is None:
        parser.error("manual parameter source requires --parameter_config")
    if args.stage in ("grid", "all") and args.parameter_source not in (
        "publication",
        "grid",
    ):
        parser.error("grid/all stages require publication or grid parameter source")
    try:
        result = run_pipeline(args, FlexiCTBackend())
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


__all__ = ["build_parser", "main"]

