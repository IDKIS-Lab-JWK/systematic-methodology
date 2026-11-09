"""CLI for the separate Pillar-0 RAVE cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:  # Package imports.
    from . import DEFAULT_CACHE_NAMESPACE
    from .cache import (
        DEFAULT_RAVE_REPOSITORY,
        DEFAULT_RAVE_REVISION,
        build_cache_plan,
        cache_directory,
        prepare_cache as execute_cache_plan,
        stratified_smoke_plan,
        validate_cache,
    )
    from .data import split_manifest_path
except ImportError:  # Direct invocation from this directory.
    from __init__ import DEFAULT_CACHE_NAMESPACE
    from cache import (
        DEFAULT_RAVE_REPOSITORY,
        DEFAULT_RAVE_REVISION,
        build_cache_plan,
        cache_directory,
        prepare_cache as execute_cache_plan,
        stratified_smoke_plan,
        validate_cache,
    )
    from data import split_manifest_path


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    model_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or validate the NIfTI-only, raw-HU RAVE LZ4 cache used by "
            "Pillar-0. Atlas partial fine-tuning separately creates all ten temporary "
            "rotation-specific frozen representations when training launches. "
            "This never reads or writes Merlin's cache."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--action", choices=("prepare", "plan", "validate"), default="prepare"
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
        "--cache_root", type=Path, default=root / "Results" / "Pillar0" / "Cache"
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--config",
        type=Path,
        default=model_root / "rave_abdomen_384_lz4.yaml",
        help="RAVE YAML; target shape and spacing are derived from this file.",
    )
    parser.add_argument(
        "--source_fingerprint",
        choices=("stat", "sha256"),
        default="stat",
        help="sha256 is strongest but reads every source volume in full.",
    )
    parser.add_argument(
        "--expected_shape_dhw",
        nargs=3,
        type=int,
        default=None,
        metavar=("D", "H", "W"),
        help="Optional extra assertion; preprocessing itself remains YAML-driven.",
    )
    parser.add_argument(
        "--expected_spacing_xyz_mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional extra assertion against the YAML target spacing.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--rave_python",
        type=Path,
        default=Path(sys.executable),
        help="Python environment containing the pinned RAVE installation.",
    )
    parser.add_argument("--rave_module", default="vision_engine")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--smoke_patients_per_class",
        type=int,
        default=None,
        help="Build an explicitly separate class-stratified smoke cache; never use it for experiments.",
    )
    parser.add_argument(
        "--skip_volume_validation",
        action="store_true",
        help="Check paths only; full shape validation is the safer default.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Ask RAVE to continue after an individual source failure.",
    )
    parser.add_argument("--rave_repository", default=DEFAULT_RAVE_REPOSITORY)
    parser.add_argument("--rave_revision", default=DEFAULT_RAVE_REVISION)
    parser.add_argument(
        "--pillar_checkpoint", default="YalaLab/Pillar0-AbdomenCT"
    )
    parser.add_argument(
        "--pillar_checkpoint_revision",
        default="d73d111bb278b3f6411f3c8e72c7e6f8a5fb1bb8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the plan/prepare/validation result.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout JSON; use with --output.")
    return parser


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.split_manifest or split_manifest_path(
        args.dataset_root, args.percentage
    )
    namespace_directory = cache_directory(args.cache_root, args.cache_namespace)
    if args.action == "validate":
        return validate_cache(
            namespace_directory / "cache_manifest.json",
            verify_volumes=not args.skip_volume_validation,
        )

    plan = build_cache_plan(
        dataset_root=args.dataset_root,
        split_manifest_path=manifest_path,
        cache_root=args.cache_root,
        namespace=args.cache_namespace,
        config_path=args.config,
        source_fingerprint_mode=args.source_fingerprint,
        expected_shape_dhw=args.expected_shape_dhw,
        expected_spacing_xyz_mm=args.expected_spacing_xyz_mm,
        provenance={
            "rave_repository": args.rave_repository,
            "rave_revision": args.rave_revision,
            "pillar_checkpoint": args.pillar_checkpoint,
            "pillar_checkpoint_revision": args.pillar_checkpoint_revision,
        },
    )
    if args.smoke_patients_per_class is not None:
        smoke_namespace = args.cache_namespace
        if smoke_namespace == DEFAULT_CACHE_NAMESPACE:
            smoke_namespace = (
                f"{DEFAULT_CACHE_NAMESPACE}_smoke_"
                f"{int(args.smoke_patients_per_class)}pc_v1"
            )
        plan = stratified_smoke_plan(
            plan,
            patients_per_class=args.smoke_patients_per_class,
            namespace=smoke_namespace,
        )
    if args.action == "plan":
        return {
            "status": "plan",
            "filesystem_mutated": False,
            "plan": plan.to_dict(),
            "note": "Use --action prepare to execute this immutable plan.",
        }
    return execute_cache_plan(
        plan,
        python_executable=args.rave_python,
        rave_module=args.rave_module,
        workers=args.workers,
        dry_run=args.dry_run,
        validate_volumes=not args.skip_volume_validation,
        stop_on_error=not args.continue_on_error,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
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
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
        os.replace(temporary, args.output)
    if not args.quiet:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
