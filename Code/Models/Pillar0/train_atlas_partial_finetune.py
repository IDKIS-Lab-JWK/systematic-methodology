"""CLI entry point for separate Pillar-0 Atlas block/stage strategies."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import DEFAULT_CACHE_NAMESPACE
    from .model import MODEL_REPO_ID, MODEL_REVISION
    from .partial_finetune import PillarAtlasAdapter
    from .frozen_partial_finetune import FrozenPillarAtlasAdapter
    from .prefetch import PrefetchedAtlasBackend
except ImportError:  # pragma: no cover
    from Pillar0 import DEFAULT_CACHE_NAMESPACE
    from Pillar0.model import MODEL_REPO_ID, MODEL_REVISION
    from Pillar0.partial_finetune import PillarAtlasAdapter
    from Pillar0.frozen_partial_finetune import FrozenPillarAtlasAdapter
    from Pillar0.prefetch import PrefetchedAtlasBackend

from _supervised_adaptation import cli as shared_cli
from _supervised_adaptation.trainer import AdaptationBackend
from _supervised_adaptation.workflow import _load_candidates


def _add_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument(
        "--adaptation_scope",
        choices=("final_block", "final_stage", "both"),
        default="final_block",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=root / "Results" / "Pillar0" / "Cache" / "Shared" / DEFAULT_CACHE_NAMESPACE,
    )
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--input_source", choices=("frozen", "cache"), default="frozen",
                        help="all ten rotation-specific frozen outputs, or original RAVE input for legacy runs")
    parser.add_argument("--frozen_cache_root", type=Path,
                        help="temporary tensors; defaults to <results_root>/Cache/FrozenRuns")
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="prepare one CPU microbatch ahead, preserving augmentation RNG")


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.model_repo_id != MODEL_REPO_ID:
        parser.error(f"Pillar-0 repo is immutable and must be {MODEL_REPO_ID}")
    if args.model_revision != MODEL_REVISION:
        parser.error(f"Pillar-0 revision is immutable and must be {MODEL_REVISION}")
    if args.frozen_cache_root is None:
        args.frozen_cache_root = args.results_root / "Cache" / "FrozenRuns"
    backend = AdaptationBackend(PillarAtlasAdapter(args.adaptation_scope))
    candidates, _, _ = _load_candidates(args, backend)
    for candidate in candidates:
        batch = int(candidate["effective_batch_size"])
        if args.micro_batch_size > batch or batch % args.micro_batch_size:
            parser.error(f"Select a microbatch size equal to or lower than {batch} that divides the effective batch evenly.")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--adaptation_scope",
        choices=("final_block", "final_stage", "both"),
        default="final_block",
    )
    bootstrap.add_argument("--input_source", choices=("frozen", "cache"), default="frozen")
    selected, _unknown = bootstrap.parse_known_args(raw_argv)
    scopes = ["final_stage", "final_block"] if selected.adaptation_scope == "both" else [selected.adaptation_scope]
    adapter = FrozenPillarAtlasAdapter if selected.input_source == "frozen" else PillarAtlasAdapter
    parser = shared_cli.build_parser(partial(adapter, scopes[0]), _add_arguments)
    parsed = parser.parse_args(raw_argv)
    shared_cli.validate_args(parser, parsed)
    arguments, stores = [], {}
    for scope in scopes:
        args = argparse.Namespace(**vars(parsed))
        args.adaptation_scope = scope
        args.frozen_scopes, args.frozen_stores = scopes, stores
        _validate(parser, args)
        arguments.append(args)
    success, failures = False, []
    try:
        outputs = [shared_cli.run_pipeline(args, PrefetchedAtlasBackend(adapter(args.adaptation_scope)))
                   for args in arguments]
        success = all(item.get("status") in ("complete", "dry_run") for item in outputs)
        result = outputs[0] if len(outputs) == 1 else {
            "status": "dry_run" if parsed.dry_run else "complete", "scopes": outputs,
            "shared_frozen_extraction": selected.input_source == "frozen"}
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    finally:
        for store in stores.values():
            try:
                store.finish(success)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            print(json.dumps({"status": "cache_cleanup_error", "errors": failures}), file=sys.stderr)
    if failures:
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
