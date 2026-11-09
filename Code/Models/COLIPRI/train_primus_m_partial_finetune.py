"""CLI entry point for separate COLIPRI Primus-M block/stage strategies."""

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
    from . import MODEL_REVISION
    from .partial_finetune import COLIPRIPartialAdapter
    from .runtime_options import resolve_micro_batch_size
    from .cached_partial_finetune import CachedCOLIPRIPartialAdapter
    from .frozen_partial_finetune import FrozenCOLIPRIPartialAdapter
    from .prefetched_backends import PrefetchedAdaptationBackend
except ImportError:  # pragma: no cover
    from COLIPRI import MODEL_REVISION
    from COLIPRI.partial_finetune import COLIPRIPartialAdapter
    from COLIPRI.runtime_options import resolve_micro_batch_size
    from COLIPRI.cached_partial_finetune import CachedCOLIPRIPartialAdapter
    from COLIPRI.frozen_partial_finetune import FrozenCOLIPRIPartialAdapter
    from COLIPRI.prefetched_backends import PrefetchedAdaptationBackend

from _linear_probe.common import ContractError
from _supervised_adaptation import cli as shared_cli
from _supervised_adaptation.trainer import AdaptationBackend
from _supervised_adaptation.workflow import _load_candidates


def _add_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.set_defaults(micro_batch_size=None)
    parser.epilog = (
        "Omitting --micro_batch_size uses the effective batch size from the selected "
        "profile. Explicit microbatches must be no larger than that batch and divide it evenly."
    )
    parser.add_argument(
        "--adaptation_scope",
        choices=("final_block", "final_stage", "both"),
        default="final_block",
    )
    parser.add_argument("--upstream_root", type=Path, default=root / "tmp" / "upstream_colipri")
    parser.add_argument("--upstream_revision", default=MODEL_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cache_root", type=Path, help="parent of Volumes/; defaults to <results_root>/Cache")
    parser.add_argument(
        "--input_source", choices=("frozen", "cache", "nifti"), default="frozen",
        help="temporary frozen outputs (default), prepared CTs, or original NIfTI for legacy resume",
    )
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="prepare one CPU microbatch ahead, without an extra GPU batch")
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "COLIPRI" / "Setup" / "asset_provisioning.jsonl",
    )


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.cache_root is None:
        args.cache_root = args.results_root / "Cache"
    if args.upstream_revision != MODEL_REVISION:
        parser.error(f"COLIPRI revision is immutable and must be {MODEL_REVISION}")
    # Resolve from the actual profile before any model/data loading or result writes.
    # Keep this COLIPRI-specific so other adapters and sealed training code are unchanged.
    try:
        backend = AdaptationBackend(COLIPRIPartialAdapter(args.adaptation_scope))
        candidates, _profile, _path = _load_candidates(args, backend)
        batch_sizes = sorted({int(item["effective_batch_size"]) for item in candidates})
        if args.micro_batch_size is None and len(batch_sizes) != 1:
            raise ValueError(
                f"The selected grid has multiple effective batch sizes: {batch_sizes}. "
                "Specify --micro_batch_size explicitly; it must be no larger than "
                f"{batch_sizes[0]} and divide every candidate's batch size evenly."
            )
        args.micro_batch_size = resolve_micro_batch_size(args.micro_batch_size, batch_sizes[0])
        for batch_size in batch_sizes[1:]:
            resolve_micro_batch_size(args.micro_batch_size, batch_size)
    except (ContractError, ValueError) as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--adaptation_scope",
        choices=("final_block", "final_stage", "both"),
        default="final_block",
    )
    bootstrap.add_argument("--input_source", choices=("frozen", "cache", "nifti"), default="frozen")
    selected, _unknown = bootstrap.parse_known_args(raw_argv)
    scopes = ["final_stage", "final_block"] if selected.adaptation_scope == "both" else [selected.adaptation_scope]
    adapter = {"frozen": FrozenCOLIPRIPartialAdapter, "cache": CachedCOLIPRIPartialAdapter,
               "nifti": COLIPRIPartialAdapter}[selected.input_source]
    parser = shared_cli.build_parser(partial(adapter, scopes[0]), _add_arguments)
    parsed = parser.parse_args(raw_argv)
    shared_cli.validate_args(parser, parsed)
    # Validate every scope before any cache or result mutation.
    arguments = []
    stores = {}
    for scope in scopes:
        args = argparse.Namespace(**vars(parsed))
        args.adaptation_scope = scope
        args.frozen_scopes, args.frozen_stores = scopes, stores
        _validate(parser, args)
        arguments.append(args)
    success = False
    try:
        outputs = [shared_cli.run_pipeline(args, PrefetchedAdaptationBackend(adapter(args.adaptation_scope)))
                   for args in arguments]
        success = all(item.get("status") in ("complete", "dry_run") for item in outputs)
        result = outputs[0] if len(outputs) == 1 else {
            "status": "dry_run" if parsed.dry_run else "complete", "scopes": outputs,
            "shared_frozen_extraction": selected.input_source == "frozen",
        }
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    finally:
        # Keep caches through every requested fold/candidate/scope. Closing a
        # prefetch context joins its worker before pipeline control returns here.
        failures = []
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
