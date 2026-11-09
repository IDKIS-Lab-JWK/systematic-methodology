"""Plan, build, or validate the shared FlexiCT embedding cache."""

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
    from .cache import cache_plan, prepare_feature_cache, validate_feature_cache
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE, UPSTREAM_REVISION
    from cache import cache_plan, prepare_feature_cache, validate_feature_cache

from _linear_probe.common import methodology_root


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    results = root / "Results" / "FlexiCT"
    parser = argparse.ArgumentParser(
        description="FlexiCT direct NIfTI-to-embedding cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--action", choices=("plan", "prepare", "validate"), default="plan")
    parser.add_argument(
        "--percentage",
        choices=(100,),
        type=int,
        default=100,
        help="shared feature cache always covers the complete 100%% patient inventory",
    )
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
            "The direct per-patient FlexiCT cache is serial and requires --num_workers 0"
        )
    try:
        if args.dry_run or args.action == "plan":
            result = cache_plan(args, require_assets=False)
        elif args.action == "prepare":
            result = prepare_feature_cache(args)
        else:
            result = validate_feature_cache(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
