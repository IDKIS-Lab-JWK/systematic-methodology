"""CLI for planning, preparing, or validating the isolated 3DINO cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:  # Package imports.
    from . import DEFAULT_CACHE_NAMESPACE
    from .cache import (
        cache_directory,
        create_cache_plan,
        prepare_cache as execute_cache_plan,
        validate_cache,
    )
    from .data import split_manifest_path
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import DEFAULT_CACHE_NAMESPACE
    from cache import cache_directory, create_cache_plan, prepare_cache as execute_cache_plan, validate_cache
    from data import split_manifest_path


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare only 3DINO's deterministic NIfTI preprocessing prefix. "
            "The random training crop and augmentations are never cached."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--action", choices=("plan", "prepare", "validate"), default="plan"
    )
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="Defaults to Dataset/Splits/<percentage>_percent.json.",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=root / "Results" / "3DINO" / "Cache",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Force a no-write plan even when --action prepare was supplied.",
    )
    parser.add_argument(
        "--skip_artifact_validation",
        action="store_true",
        help=(
            "For validate, perform structural-only checks without reading, "
            "hashing, or deserializing cache tensors; artifacts remain unverified."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    namespace_directory = cache_directory(args.cache_root, args.cache_namespace)
    if args.action == "validate" and not args.dry_run:
        return validate_cache(
            namespace_directory / "cache_manifest.json",
            verify_artifacts=not args.skip_artifact_validation,
        )

    manifest_path = args.split_manifest or split_manifest_path(
        args.dataset_root, args.percentage
    )
    plan = create_cache_plan(
        dataset_root=args.dataset_root,
        split_manifest_path=manifest_path,
        cache_root=args.cache_root,
        namespace=args.cache_namespace,
        require_files=True,
    )
    if args.action == "plan" or args.dry_run:
        return {
            "status": "plan",
            "filesystem_mutated": False,
            "plan": plan.to_dict(),
            "note": "Use --action prepare without --dry_run to materialize it.",
        }
    return execute_cache_plan(plan)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        # ``--dry_run`` is a strict no-write contract, including optional
        # report destinations.
        if args.output is not None and not args.dry_run:
            _write_json(args.output, result)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        print(json.dumps(result, indent=2, default=str))
    return 0


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
