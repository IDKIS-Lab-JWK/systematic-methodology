"""Plan, build, or validate COLIPRI probe features and fine-tuning CT inputs."""

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
    from . import DEFAULT_CACHE_NAMESPACE, MODEL_REVISION
    from .cache import cache_plan, prepare_feature_cache, validate_feature_cache
    from .volume_cache import volume_cache_plan, prepare_volume_cache, validate_volume_cache
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE, MODEL_REVISION
    from cache import cache_plan, prepare_feature_cache, validate_feature_cache
    from COLIPRI.volume_cache import volume_cache_plan, prepare_volume_cache, validate_volume_cache

from _linear_probe.common import methodology_root


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    results = root / "Results" / "COLIPRI"
    parser = argparse.ArgumentParser(
        description="COLIPRI probe-feature and shared fine-tuning CT caches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--action", choices=("plan", "prepare", "validate"), default="plan")
    parser.add_argument(
        "--cache_target", choices=("probe", "finetune", "both"), default="both",
        help="finetune creates one preprocessed CT cache shared by final-block and final-stage training",
    )
    parser.add_argument(
        "--percentage",
        choices=(100,),
        type=int,
        default=100,
        help="each cache covers the complete 100%% patient inventory",
    )
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--results_root", type=Path, default=results)
    parser.add_argument(
        "--cache_root",
        type=Path,
        help="parent of Features/ and Volumes/; defaults to <results_root>/Cache",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--upstream_root", type=Path, default=root / "tmp" / "upstream_colipri"
    )
    parser.add_argument("--upstream_revision", default=MODEL_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--provision_log",
        type=Path,
        help="asset audit log; defaults to <results_root>/Setup/asset_provisioning.jsonl",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature_precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--source_fingerprint", choices=("stat", "sha256"), default="sha256")
    parser.add_argument("--num_workers", type=_nonnegative, default=0)
    parser.add_argument(
        "--skip_artifact_validation",
        action="store_true",
        help=(
            "For validate only: check the sealed manifest, source stat, and artifact "
            "sizes without rehashing/loading artifacts; this is not execution-ready validation"
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cache_root is None:
        args.cache_root = args.results_root / "Cache"
    if args.provision_log is None:
        args.provision_log = args.results_root / "Setup" / "asset_provisioning.jsonl"
    if args.upstream_revision != MODEL_REVISION:
        parser.error(f"COLIPRI source revision is immutable and must be {MODEL_REVISION}")
    if args.num_workers != 0:
        parser.error(
            "COLIPRI cache preparation is a serial one-GPU stream; --num_workers must be 0"
        )
    if args.skip_artifact_validation and args.action != "validate":
        parser.error("--skip_artifact_validation is valid only with --action validate")
    try:
        targets = ("finetune", "probe") if args.cache_target == "both" else (args.cache_target,)
        results = {}
        for target in targets:
            if args.dry_run or args.action == "plan":
                result = volume_cache_plan(args) if target == "finetune" else cache_plan(args, require_assets=False)
            elif args.action == "prepare":
                result = prepare_volume_cache(args) if target == "finetune" else prepare_feature_cache(args)
            else:
                validate = validate_volume_cache if target == "finetune" else validate_feature_cache
                result = validate(args, exhaustive=not args.skip_artifact_validation)
            results[target] = result
        if args.cache_target == "both":
            planning = args.dry_run or args.action == "plan"
            result = {"cache_target": "both", "action": args.action,
                      "status": "plan" if planning else "finished", "targets": results}
            if planning:
                result.update({"side_effects": False, "estimated_total_raw_gib":
                    results["finetune"]["estimated_raw_volume_gib"] + results["probe"]["estimated_raw_feature_gib"]})
            else:
                result["execution_ready"] = all(item.get("execution_ready", False) for item in results.values())
        else:
            result = {**results[args.cache_target], "cache_target": args.cache_target}
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
