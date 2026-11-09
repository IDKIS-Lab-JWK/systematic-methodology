"""Shared command-line interface for Pillar-0 trained adaptations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from . import DEFAULT_CACHE_NAMESPACE
    from .model import MODEL_REPO_ID, MODEL_REVISION
    from .orchestrator import default_dataset_root, default_results_root, run_pipeline
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE
    from model import MODEL_REPO_ID, MODEL_REVISION
    from orchestrator import default_dataset_root, default_results_root, run_pipeline


METHOD_LABELS = {
    "linear_probe": "Frozen encoder + linear probe",
    "full_finetune": "End-to-end supervised fine-tuning",
}


def build_parser(method: str) -> argparse.ArgumentParser:
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")
    results_root = default_results_root()
    parser = argparse.ArgumentParser(
        description=f"Pillar-0 {METHOD_LABELS[method]}: efficient search, five-fold CV, refit, and held-out test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("grid", "search", "cv", "test", "all"),
        default="all",
        help="'grid' is the Merlin-compatible alias for the curated search stage.",
    )
    parser.add_argument("--percentage", type=int, choices=(20, 40, 60, 80, 100), default=40)
    parser.add_argument(
        "--parameter_source",
        choices=("default", "publication", "grid", "search", "manual"),
        default="default",
    )
    parser.add_argument("--parameter_config", type=Path, default=None)
    parser.add_argument(
        "--search_config", "--grid_config", dest="search_config", type=Path, default=None
    )
    parser.add_argument(
        "--max_search_candidates",
        "--max_grid_candidates",
        dest="max_search_candidates",
        type=int,
        default=None,
    )
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=results_root)
    parser.add_argument("--cache_root", type=Path, default=results_root / "Cache")
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument("--feature_cache_root", type=Path, default=None)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--model_source",
        choices=("refit_full_development", "cv_ensemble", "best_cv_fold", "provided_checkpoint"),
        default="refit_full_development",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--force_test", action="store_true")
    return parser


def main(method: str, argv: list[str] | None = None) -> int:
    parser = build_parser(method)
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    if args.stage == "grid":
        args.stage = "search"
    if args.parameter_source == "grid":
        args.parameter_source = "search"
    if args.stage == "search":
        args.parameter_source = "search"
    try:
        result = run_pipeline(args, method)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0

