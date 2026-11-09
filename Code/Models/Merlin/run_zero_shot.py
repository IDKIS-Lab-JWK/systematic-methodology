#!/usr/bin/env python
"""Run frozen Merlin zero-shot classification on the fixed held-out cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator import default_dataset_root, default_results_root
from zero_shot import run_zero_shot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Merlin image-text zero-shot evaluation. No training, grid, "
            "cross-validation, refit, or prompt selection is performed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["both", "five_class_adapted", "author_binary"],
        default="both",
        help=(
            "five_class_adapted gives one mutually exclusive benchmark label; "
            "author_binary independently compares finding-present and "
            "finding-absent prompts and reports one-vs-rest proxy metrics; "
            "both writes both result sets from one pass."
        ),
    )
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=default_results_root())
    parser.add_argument(
        "--prompt_config",
        type=Path,
        default=Path(__file__).with_name("zero_shot_prompts.json"),
        help=(
            "Fixed, versioned prompt profile. Both its file SHA-256 and its "
            "canonical JSON-content SHA-256 are recorded in results."
        ),
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=None,
        help=(
            "Base directory for the shared deterministic Merlin cache. "
            "Defaults to <results_root>/Cache."
        ),
    )
    parser.add_argument(
        "--cache_namespace",
        default="merlin_ct_ras_1p5x1p5x3_224x224x160_v1",
        help="Use a new name only if deterministic preprocessing changes.",
    )
    parser.add_argument(
        "--cache_temp_dir",
        type=Path,
        default=None,
        help="Optional temporary directory for persistent-cache writes.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA FP16 autocast for image encoding. Text encoding and all "
            "similarity calculations remain FP32."
        ),
    )
    parser.add_argument(
        "--skip_checkpoint_hash",
        action="store_true",
        help="Skip SHA-256 of the downloaded Merlin checkpoint to save disk I/O.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate manifests, NIfTI paths, prompt schema, cache paths, and "
            "outputs without loading/downloading Merlin or writing results."
        ),
    )
    parser.add_argument(
        "--validate_setup",
        action="store_true",
        help=(
            "Load Merlin and one held-out CT, validate 512-D image/text scoring, "
            "and exit without writing metrics."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace result artifacts for the same mode and prompt-set ID.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not getattr(args, 'dry_run', False):
        raise RuntimeError('Legacy zero-shot testing requires persistent test caches and is disabled in this distribution. The raw inference wrapper supports the published supervised classifiers.')
    result = run_zero_shot(args)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
