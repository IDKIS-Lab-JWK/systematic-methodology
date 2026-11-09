"""Command-line interface for the isolated 3DINO workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from . import (
        DEFAULT_CACHE_NAMESPACE,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        UPSTREAM_REVISION,
    )
except ImportError:  # pragma: no cover - direct script execution
    from __init__ import (
        DEFAULT_CACHE_NAMESPACE,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        UPSTREAM_REVISION,
    )


METHOD_LABELS = {
    "linear_probe": "publication-backed frozen-encoder linear probing",
    "full_finetune": "project-adapted end-to-end fine-tuning",
    "lora": "related-study attention LoRA with a five-class MLP head",
}


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def canonical_stage(value: str) -> str:
    """Accept the former search spelling without exposing it as canonical."""

    return "grid" if value == "search" else value


def canonical_parameter_source(value: str) -> str:
    """Accept the former search source while storing the canonical grid name."""

    return "grid" if value == "search" else value


def methodology_root() -> Path:
    """Resolve repository defaults without importing the execution stack."""

    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    return methodology_root() / "Dataset"


def default_results_root() -> Path:
    return methodology_root() / "Results" / "3DINO"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # parser/help and dry-run remain usable without PyTorch
        return "cpu"


def build_parser(method: str) -> argparse.ArgumentParser:
    """Build a parser without importing orchestration/training dependencies."""

    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown 3DINO method: {method}")
    root = methodology_root()
    results_root = default_results_root()
    parser = argparse.ArgumentParser(
        description=(
            f"3DINO {METHOD_LABELS[method]}: "
            + ("fixed-profile five-fold " if method == "lora" else "development-only grid, five-fold ")
            + "CV, full-development refit, and one fixed held-out evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=canonical_stage,
        choices=("cv", "test", "all") if method == "lora" else ("grid", "cv", "test", "all"),
        default="all",
        help=("Development CV/refit, held-out test, or both." if method == "lora" else
              "Workflow stage. The legacy value 'search' is accepted as an alias for 'grid'."),
    )
    parser.add_argument("--percentage", type=int, choices=(20, 40, 60, 80, 100), default=40)
    parser.add_argument(
        "--parameter_source",
        type=canonical_parameter_source,
        choices=("default", "manual") if method == "lora" else ("default", "publication", "grid", "manual"),
        default="default",
        help=("Use the study-based default profile or an explicit manual JSON profile." if method == "lora" else
            "For linear probing, 'publication' selects the 52-head paper grid. "
            "'grid' selects the project-defined grid, while 'default' and "
            "'manual' select one fixed profile. The legacy value 'search' is "
            "accepted as an alias for 'grid'. 3DINO did not publish a "
            "classification full-fine-tuning profile."
        ),
    )
    parser.add_argument("--parameter_config", type=Path, default=None)
    parser.add_argument(
        "--grid_config",
        "--search_config",
        dest="grid_config",
        type=Path,
        default=None,
        help=(
            "Optional project-grid definition used only with "
            "--parameter_source grid. --search_config is a legacy alias."
        ),
    )
    parser.add_argument(
        "--max_grid_candidates",
        "--max_search_candidates",
        dest="max_grid_candidates",
        type=positive_int,
        default=None,
        help=(
            "Optional positive development limit for a project grid. It is "
            "not permitted for the complete publication grid. "
            "--max_search_candidates is a legacy alias."
        ),
    )
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=results_root)
    parser.add_argument("--cache_root", type=Path, default=results_root / "Cache")
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--upstream_root",
        type=Path,
        default=root / "tmp" / "upstream_3dino",
        help=(
            "Local official-source checkout. If absent during real execution, "
            "the pinned AICONSlab/3DINO commit is cloned automatically."
        ),
    )
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument(
        "--pretrained_checkpoint",
        "--base_checkpoint",
        dest="pretrained_checkpoint",
        type=Path,
        default=None,
        help=(
            f"Optional local override for {MODEL_FILENAME}. When omitted, the "
            f"pinned gated {MODEL_REPO_ID} file is downloaded through the "
            "normal Hugging Face credential store and reused from its cache."
        ),
    )
    parser.add_argument(
        "--pretrained_checkpoint_sha256",
        default=MODEL_SHA256,
        help="Required SHA-256 for the official pinned checkpoint.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help=(
            "Disable Git/network acquisition. Existing source plus either a "
            "local checkpoint override or the pinned Hugging Face cache entry "
            "must already be present."
        ),
    )
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=None,
        help=(
            "Append-only token-sanitized source/checkpoint acquisition audit. "
            "Defaults to <results_root>/Setup/asset_provisioning.jsonl."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Execution device. When omitted, real execution detects CUDA lazily; "
            "dry-run planning uses CPU without probing PyTorch."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=nonnegative_int,
        default=0,
        help=(
            "CPU worker processes for training data preparation (default: 0). "
            "Values above zero use persistent training workers; validation and "
            "test loading remain fixed at zero workers. This is an operational "
            "throughput setting, not scientific experiment identity."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--model_source",
        choices=("refit_full_development", "cv_ensemble", "best_cv_fold", "provided_checkpoint"),
        default="refit_full_development",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained classifier checkpoint, used only with provided_checkpoint.",
    )
    parser.add_argument("--force_test", action="store_true")
    if method == "full_finetune":
        parser.add_argument(
            "--allow_experimental_full_finetune",
            action="store_true",
            help=(
                "Explicitly acknowledge that this is a project-adapted, "
                "non-publication workflow and that upstream licence/compute "
                "requirements have been reviewed."
            ),
        )
    return parser


def main(method: str, argv: list[str] | None = None) -> int:
    parser = build_parser(method)
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    if method == "lora" and (
        args.parameter_source not in {"default", "manual"}
        or args.stage == "grid" or args.grid_config is not None
        or args.max_grid_candidates is not None
    ):
        parser.error("LoRA supports fixed default/manual profiles and cv/test/all stages; no study grid is defined.")
    if args.stage == "grid" and args.parameter_source == "default":
        args.parameter_source = "grid"
    if args.parameter_source == "manual" and args.parameter_config is None:
        parser.error("--parameter_source manual requires --parameter_config.")
    if args.stage == "grid" and args.parameter_source == "manual":
        parser.error(
            "--stage grid requires --parameter_source grid or publication; "
            "manual is a fixed-parameter CV/test source."
        )
    if method == "full_finetune" and args.parameter_source == "publication":
        parser.error(
            "3DINO has no publication classification full-fine-tuning profile; "
            "use --parameter_source default, grid, or manual."
        )
    if args.parameter_source == "publication":
        if args.grid_config is not None:
            parser.error(
                "--parameter_source publication does not accept --grid_config; "
                "the audited publication grid is fixed."
            )
        if args.max_grid_candidates is not None:
            parser.error(
                "--parameter_source publication does not accept "
                "--max_grid_candidates; all publication candidates are required."
            )
    if args.device is None:
        args.device = "cpu" if args.dry_run else _default_device()
    try:
        # Keep parser construction and ``--help`` independent of the heavy
        # training stack.  The orchestrator is needed only after arguments have
        # been accepted and an execution/dry-run plan was actually requested.
        if __package__:
            from .orchestrator import run_pipeline
        else:  # pragma: no cover - direct invocation from this folder.
            from orchestrator import run_pipeline

        result = run_pipeline(args, method)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


__all__ = ["METHOD_LABELS", "build_parser", "main"]

