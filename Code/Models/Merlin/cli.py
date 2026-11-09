"""Shared command-line interface for Merlin variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orchestrator import (
    default_dataset_root,
    default_results_root,
    run_pipeline,
)

PARAMETER_ARGUMENTS = {
    "adapter_lr",
    "batch_size",
    "conv_lora_alpha",
    "conv_lora_rank",
    "dropout_prob",
    "early_stopping_patience",
    "embedding_dim",
    "encoder_lr",
    "encoder_weight_decay",
    "epochs",
    "grad_accum_steps",
    "gradient_checkpointing",
    "head_lr",
    "head_weight_decay",
    "image_layer_decay",
    "max_grad_norm",
    "min_delta",
    "min_lr_factor",
    "exclude_norm_bias_from_weight_decay",
    "scheduler_step_unit",
    "scheduler_decay_epochs",
    "train_image_projection",
    "warmup_epochs",
}

VARIANT_LABELS = {
    "linear_probe": "Linear Probe",
    "full_finetune": "Full Fine-Tune",
    "conv_lora": "ConvLoRA",
    "conv_lora_llrd": "ConvLoRA + LLRD",
}


def _explicit_parameter_names(
    parser: argparse.ArgumentParser, arguments: list[str]
) -> list[str]:
    """Return parameter destinations explicitly supplied on the command line."""
    option_to_destination = {
        option: action.dest
        for action in parser._actions
        if action.dest in PARAMETER_ARGUMENTS
        for option in action.option_strings
    }
    supplied = {
        option_to_destination[token.split("=", 1)[0]]
        for token in arguments
        if token.split("=", 1)[0] in option_to_destination
    }
    return sorted(supplied)


def build_parser(variant: str) -> argparse.ArgumentParser:
    try:
        label = VARIANT_LABELS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown Merlin variant: {variant}") from exc
    publication_inspired_llrd = variant == "conv_lora_llrd"
    linear_probe = variant == "linear_probe"
    parser = argparse.ArgumentParser(
        description=f"Merlin {label}: grid, five-fold CV, and held-out test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["grid", "cv", "test", "all"], default="all")
    parser.add_argument("--percentage", type=int, default=40)
    parser.add_argument(
        "--grid",
        default="true",
        help="true: select parameters from a grid; false: use provided CLI parameters.",
    )
    parser.add_argument(
        "--parameter_source",
        choices=[
            "auto",
            "grid",
            "default",
            "publication",
            "publication_derived",
            "manual",
        ],
        default="auto",
        help=(
            "Parameter-selection source. auto preserves the legacy --grid behavior; "
            "publication loads the variant's audited profile (including any "
            "explicitly documented task adaptations); publication_derived loads "
            "that profile and applies explicit parameter overrides."
        ),
    )
    parser.add_argument(
        "--publication_override_reason",
        default="user_requested_override",
        help=(
            "Reason recorded for explicit overrides applied with "
            "--parameter_source publication_derived."
        ),
    )
    parser.add_argument(
        "--percentage_specific_grid",
        default="false",
        help="true: use/run this percentage's grid; false: prefer the shared 40%% grid.",
    )
    parser.add_argument("--default_grid_percentage", type=int, default=40)
    parser.add_argument("--grid_config", type=Path, default=None)
    parser.add_argument("--max_grid_candidates", type=int, default=None)
    parser.add_argument("--force_grid", action="store_true")
    parser.add_argument("--force_cv", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--model_source",
        choices=[
            "refit_full_development",
            "cv_ensemble",
            "best_cv_fold",
            "provided_checkpoint",
        ],
        default="refit_full_development",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=default_results_root())
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=None,
        help=(
            "Base directory for the shared persistent preprocessing cache. "
            "Defaults to <results_root>/Cache."
        ),
    )
    parser.add_argument(
        "--cache_namespace",
        default="merlin_ct_ras_1p5x1p5x3_224x224x160_v1",
        help=(
            "Versioned shared-cache namespace. Use a new value whenever "
            "deterministic image preprocessing changes."
        ),
    )
    parser.add_argument(
        "--cache_temp_dir",
        type=Path,
        default=None,
        help=(
            "Temporary directory used while writing shared cache tensors. "
            "Defaults beside the shared cache instead of Windows %%TEMP%%."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--feature_extraction_batch_size",
        type=int,
        default=1,
        help=(
            "CT batch size used only while materializing frozen Merlin features "
            "for linear probing. Probe-head batch_size remains independent."
        ),
    )

    # Existing variant defaults are intentionally unchanged. The independent
    # LLRD variant defaults to the publication-inspired image-only recipe.
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.0 if publication_inspired_llrd or linear_probe else 0.1,
    )
    parser.add_argument(
        "--encoder_weight_decay",
        type=float,
        default=0.01 if publication_inspired_llrd else 1e-4,
    )
    parser.add_argument(
        "--head_weight_decay",
        type=float,
        default=0.01 if publication_inspired_llrd else 1e-4,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0 if publication_inspired_llrd or linear_probe else 2,
    )
    parser.add_argument(
        "--min_lr_factor",
        type=float,
        default=(
            0.0
            if publication_inspired_llrd
            else 1.0 if linear_probe else 0.1
        ),
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0 if publication_inspired_llrd or linear_probe else 1.0,
    )
    parser.add_argument(
        "--batch_size", type=int, default=2 if publication_inspired_llrd else 1
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=(8 if publication_inspired_llrd else 1 if linear_probe else 4),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=(
            50
            if variant == "full_finetune"
            else 100 if linear_probe
            else 300 if publication_inspired_llrd else 100
        ),
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=(
            8
            if variant == "full_finetune"
            else 10 if linear_probe
            else 5 if publication_inspired_llrd else 10
        ),
    )
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=2048)
    parser.add_argument(
        "--gradient_checkpointing", default="false" if linear_probe else "true"
    )
    parser.add_argument(
        "--scheduler_decay_epochs",
        type=int,
        default=100 if linear_probe else None,
        help=(
            "Cosine decay horizon for linear probing. This is independent of "
            "the number of epochs actually trained."
        ),
    )

    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--conv_lora_rank", type=int, default=2)
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0)
    parser.add_argument(
        "--train_image_projection",
        default="false" if publication_inspired_llrd else "true",
    )
    if publication_inspired_llrd:
        parser.add_argument("--image_layer_decay", type=float, default=0.7)
        parser.add_argument(
            "--exclude_norm_bias_from_weight_decay", default="true"
        )
        parser.add_argument(
            "--scheduler_step_unit",
            choices=["optimizer_update"],
            default="optimizer_update",
        )
    return parser


def main(variant: str) -> None:
    parser = build_parser(variant)
    arguments = sys.argv[1:]
    args = parser.parse_args(arguments)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    args.explicit_parameter_names = _explicit_parameter_names(parser, arguments)
    result = run_pipeline(args, variant)
    print(json.dumps(result, indent=2, default=str), flush=True)

