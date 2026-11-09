"""Grid, five-fold CV, model-selection, and held-out-test orchestration."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import statistics
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import (
    classes_from_manifest,
    cross_validation_split,
    development_mapping,
    heldout_mapping,
    load_manifest,
    rows_from_mapping,
    validate_no_overlap,
)
from metrics import (
    compute_metrics,
    fold_selection_score,
    format_predictions,
    grid_selection_score,
)
VARIANT_FOLDERS = {
    "linear_probe": "LinearProbe",
    "full_finetune": "FullFineTune",
    "conv_lora": "ConvLoRA",
    "conv_lora_llrd": "ConvLoRA_LLRD",
}
CV_PARAMETERS_FILENAME = "cv_parameters.json"
CV_REGISTRY_FILENAME = "cv_runs.json"
CACHE_LAYOUT_VERSION = 1
PUBLICATION_PROFILE_FILENAMES = {
    "linear_probe": "publication_linear_probe.json",
    "full_finetune": "publication_full_finetune.json",
    "conv_lora": "publication_conv_lora.json",
    "conv_lora_llrd": "publication_conv_lora_llrd.json",
}
GRID_CONFIG_FILENAMES = {
    "linear_probe": "grid_linear_probe.json",
    "full_finetune": "grid_full_finetune.json",
    "conv_lora": "grid_conv_lora.json",
    "conv_lora_llrd": "grid_conv_lora_llrd.json",
}
PROJECT_DEFAULTS_REFERENCE = {
    "source": "original_project_defaults",
    "description": (
        "Original Merlin classification defaults restored on 2026-07-28."
    ),
}
LLRD_DEFAULTS_REFERENCE = {
    "source": "publication_inspired_image_only_llrd_defaults",
    "description": (
        "Practical supervised image-only ConvLoRA defaults with the paper/code "
        "LLRD, optimizer, and regularization settings. This is not the paper's "
        "joint image-text objective."
    ),
}
LINEAR_PROBE_DEFAULTS_REFERENCE = {
    "source": "legacy_merlin_frozen_classifier_project_defaults",
    "description": (
        "Project-grounded frozen-encoder defaults from the legacy Merlin "
        "classifier, adapted here to a strict single-layer linear probe and "
        "the leakage-safe systematic CV/test protocol."
    ),
    "legacy_code": "merlin/merlin_frozen_ct_classifier.py",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    return repository_root() / "Dataset"


def default_results_root() -> Path:
    return repository_root() / "Results" / "Merlin"


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected true or false, received: {value}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _train_run(*args, **kwargs):
    from training import train_run

    return train_run(*args, **kwargs)


def _predict_checkpoint(*args, **kwargs):
    from training import predict_checkpoint

    return predict_checkpoint(*args, **kwargs)


def candidate_id(parameters: dict) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def grid_candidates(grid_path: Path, maximum: int | None = None) -> list[dict]:
    definition = read_json(grid_path)
    keys = list(definition)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(definition[key] for key in keys))
    ]
    if maximum is not None:
        combinations = combinations[:maximum]
    return combinations


def _variant_root(results_root: Path, variant: str) -> Path:
    return results_root / VARIANT_FOLDERS[variant]


def _shared_cache_paths(args) -> tuple[Path, Path]:
    results_root = Path(args.results_root).resolve()
    cache_base = (
        Path(args.cache_root).resolve()
        if getattr(args, "cache_root", None)
        else results_root / "Cache"
    )
    namespace = str(
        getattr(
            args,
            "cache_namespace",
            "merlin_ct_ras_1p5x1p5x3_224x224x160_v1",
        )
    ).strip()
    if (
        not namespace
        or namespace in {".", ".."}
        or Path(namespace).name != namespace
    ):
        raise ValueError(
            "--cache_namespace must be one non-empty directory name."
        )
    shared_root = cache_base / "Shared"
    cache_dir = shared_root / namespace
    temp_dir = (
        Path(args.cache_temp_dir).resolve()
        if getattr(args, "cache_temp_dir", None)
        else shared_root / "_tmp"
    )
    return cache_dir, temp_dir


def _configure_shared_cache(args, create: bool) -> tuple[Path, Path]:
    cache_dir, temp_dir = _shared_cache_paths(args)
    args.shared_cache_dir = cache_dir
    args.shared_cache_temp_dir = temp_dir
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for variable in ("TMP", "TEMP", "TMPDIR"):
            os.environ[variable] = str(temp_dir)
        os.environ["MERLIN_CACHE_TEMP_DIR"] = str(temp_dir)
        manifest_path = cache_dir / "cache_manifest.json"
        expected = {
            "cache_layout_version": CACHE_LAYOUT_VERSION,
            "namespace": cache_dir.name,
            "scope": (
                "Shared by FullFineTune and ConvLoRA across grid, CV, "
                "refit, and held-out test stages."
            ),
            "cache_key": "Merlin CTPersistentDataset hash of source image path",
            "deterministic_preprocessing": {
                "orientation": "RAS",
                "spacing": [1.5, 1.5, 3.0],
                "intensity_input": [-1000, 1000],
                "intensity_output": [0.0, 1.0],
                "spatial_size": [224, 224, 160],
                "crop": "center",
            },
        }
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if existing != expected:
                raise RuntimeError(
                    "Shared-cache manifest differs from the requested "
                    f"configuration: {manifest_path}. Use a new "
                    "--cache_namespace."
                )
        else:
            save_json(manifest_path, expected)
    return cache_dir, temp_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime(args) -> dict:
    cache_dir, cache_temp_dir = _shared_cache_paths(args)
    return {
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "amp": bool(args.amp),
        "resume": bool(args.resume),
        "feature_extraction_batch_size": int(
            getattr(args, "feature_extraction_batch_size", 1)
        ),
        "shared_cache_dir": str(cache_dir),
        "cache_temp_dir": str(cache_temp_dir),
    }


def _provided_parameters(args, variant: str) -> dict:
    if variant == "linear_probe":
        return {
            "head_lr": args.head_lr,
            "head_weight_decay": args.head_weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "min_lr_factor": args.min_lr_factor,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "scheduler_decay_epochs": args.scheduler_decay_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "min_delta": args.min_delta,
            "embedding_dim": args.embedding_dim,
        }

    parameters = {
        "head_lr": args.head_lr,
        "dropout_prob": args.dropout_prob,
        "encoder_weight_decay": args.encoder_weight_decay,
        "head_weight_decay": args.head_weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_factor": args.min_lr_factor,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "embedding_dim": args.embedding_dim,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if variant == "full_finetune":
        parameters["encoder_lr"] = args.encoder_lr
    elif variant in {"conv_lora", "conv_lora_llrd"}:
        parameters.update(
            {
                "adapter_lr": args.adapter_lr,
                "conv_lora_rank": args.conv_lora_rank,
                "conv_lora_alpha": args.conv_lora_alpha,
                "train_image_projection": args.train_image_projection,
            }
        )
        if variant == "conv_lora_llrd":
            parameters.update(
                {
                    "image_layer_decay": args.image_layer_decay,
                    "exclude_norm_bias_from_weight_decay": (
                        args.exclude_norm_bias_from_weight_decay
                    ),
                    "scheduler_step_unit": args.scheduler_step_unit,
                }
            )
    else:
        raise ValueError(f"Unknown Merlin variant: {variant}")
    return parameters


def _publication_profile_path(variant: str) -> Path:
    return Path(__file__).resolve().parent / PUBLICATION_PROFILE_FILENAMES[variant]


def _publication_parameters(args, variant: str) -> tuple[dict, dict]:
    """Load the complete audited publication-backed/adapted variant profile."""
    profile_path = _publication_profile_path(variant)
    profile = read_json(profile_path)
    if profile.get("variant") != variant:
        raise ValueError(
            f"Publication profile variant mismatch in {profile_path}: "
            f"{profile.get('variant')!r}"
        )

    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Publication profile has no parameter dictionary: {profile_path}")

    expected = set(_provided_parameters(args, variant))
    supplied = set(parameters)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            f"Publication profile keys do not match {variant}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    provenance = {
        "parameter_source": "publication_parameters",
        "parameter_origin": profile.get(
            "parameter_origin", "publication_backed"
        ),
        "profile_name": profile.get("profile_name"),
        "profile_path": str(profile_path),
        "profile_schema_version": profile.get("schema_version"),
        "publication": profile.get("publication"),
        "sources": profile.get("sources"),
        "evidence": profile.get("parameter_evidence"),
        "known_methodological_differences": profile.get(
            "known_methodological_differences", []
        ),
        "source_percentage": None,
        "parameters": dict(parameters),
    }
    if profile.get("publication_fidelity") is not None:
        provenance["publication_fidelity"] = profile[
            "publication_fidelity"
        ]
    return dict(parameters), provenance


def _publication_derived_parameters(
    args, variant: str, explicit_names: list[str]
) -> tuple[dict, dict]:
    """Apply explicit CLI overrides to an immutable publication profile."""
    publication_parameters, publication_provenance = _publication_parameters(
        args, variant
    )
    cli_parameters = _provided_parameters(args, variant)
    unsupported = sorted(set(explicit_names) - set(publication_parameters))
    if unsupported:
        raise ValueError(
            f"Publication-derived overrides are not valid for {variant}: "
            f"{', '.join(unsupported)}"
        )

    effective_parameters = dict(publication_parameters)
    changed_parameters = {}
    explicitly_unchanged = []
    reason = str(
        getattr(
            args,
            "publication_override_reason",
            "user_requested_override",
        )
    ).strip()
    if not reason:
        raise ValueError("--publication_override_reason cannot be empty.")
    for name in sorted(explicit_names):
        publication_value = publication_parameters[name]
        effective_value = cli_parameters[name]
        effective_parameters[name] = effective_value
        if effective_value == publication_value:
            explicitly_unchanged.append(name)
            continue
        changed_parameters[name] = {
            "publication_value": publication_value,
            "effective_value": effective_value,
            "display": f"{publication_value} -> {effective_value}",
            "reason": reason,
        }
    if not changed_parameters:
        raise ValueError(
            "--parameter_source publication_derived requires at least one "
            "explicit parameter whose value differs from the publication "
            "profile. Use --parameter_source publication for an exact run."
        )

    publication_effective_batch = int(
        publication_parameters["batch_size"]
    ) * int(publication_parameters["grad_accum_steps"])
    derived_effective_batch = int(effective_parameters["batch_size"]) * int(
        effective_parameters["grad_accum_steps"]
    )
    comparison = {
        "highlight": "DEVIATION FROM PUBLICATION",
        "status": "publication_derived",
        "publication_profile_unchanged": True,
        "changed_parameter_count": len(changed_parameters),
        "changed_parameters": changed_parameters,
        "explicit_but_unchanged_parameters": explicitly_unchanged,
        "publication_effective_batch_size": publication_effective_batch,
        "derived_effective_batch_size": derived_effective_batch,
        "effective_batch_size_changed": (
            publication_effective_batch != derived_effective_batch
        ),
    }
    return effective_parameters, {
        **publication_provenance,
        "parameter_source": "publication_derived_parameters",
        "parameter_origin": "publication_derived",
        "publication_fidelity": "modified",
        "base_publication_parameters": publication_parameters,
        "publication_parameter_overrides": comparison,
        "explicit_parameter_names": sorted(explicit_names),
        "parameters": effective_parameters,
    }


def _requested_parameter_source(args, explicit_names: list[str]) -> str:
    """Resolve the new source selector while preserving legacy --grid behavior."""
    requested = str(getattr(args, "parameter_source", "auto"))
    if requested == "auto":
        if args.grid:
            return "grid"
        return "manual" if explicit_names else "default"

    if requested == "manual" and not explicit_names:
        raise ValueError(
            "--parameter_source manual requires at least one explicit parameter option."
        )
    if requested == "publication_derived" and not explicit_names:
        raise ValueError(
            "--parameter_source publication_derived requires at least one "
            "explicit parameter option."
        )
    if requested in {"grid", "default", "publication"} and explicit_names:
        raise ValueError(
            f"--parameter_source {requested} cannot be combined with explicit "
            f"parameter options: {', '.join(explicit_names)}"
        )
    return requested


def _parameter_origin(parameter_provenance: dict) -> str:
    origin = parameter_provenance.get("parameter_origin")
    if origin:
        return str(origin)
    source = parameter_provenance.get("parameter_source")
    if source in {"percentage_specific_grid", "shared_grid"}:
        return "grid_search"
    if source == "default_parameters":
        return "default"
    return "provided_manually"


def _cv_number(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _cv_parameter_documents(cv_base: Path) -> list[tuple[Path, dict]]:
    documents = []
    for cv_dir in sorted(
        (path for path in cv_base.glob("CV_*") if path.is_dir()),
        key=lambda path: _cv_number(path.name),
    ):
        parameter_path = cv_dir / CV_PARAMETERS_FILENAME
        if parameter_path.is_file():
            documents.append((cv_dir, read_json(parameter_path)))
    return documents


def _write_cv_registry(cv_base: Path, variant: str) -> dict:
    runs = []
    for cv_dir, document in _cv_parameter_documents(cv_base):
        runs.append(
            {
                "cv_id": document["cv_id"],
                "parameter_fingerprint": document["parameter_fingerprint"],
                "training_implementation": document.get("training_implementation"),
                "parameter_origin": document["parameter_origin"],
                "parameter_origins_seen": document.get(
                    "parameter_origins_seen", [document["parameter_origin"]]
                ),
                "publication_parameter_overrides": document.get(
                    "latest_publication_parameter_overrides",
                    document.get("publication_parameter_overrides"),
                ),
                "parameters": document["parameters"],
                "percentage_runs": document.get("percentage_runs", {}),
                "parameters_file": str(cv_dir / CV_PARAMETERS_FILENAME),
            }
        )
    registry = {
        "schema_version": 1,
        "variant": variant,
        "description": (
            "Each CV_x identifies exact hyperparameters and the training implementation. "
            "Both must match before resuming or reusing a CV_x."
        ),
        "updated_at": _utc_now(),
        "cv_count": len(runs),
        "cv_runs": runs,
    }
    save_json(cv_base / CV_REGISTRY_FILENAME, registry)
    return registry


def _record_parameter_observation(
    cv_dir: Path, parameter_provenance: dict, runtime: dict
) -> dict:
    parameter_path = cv_dir / CV_PARAMETERS_FILENAME
    document = read_json(parameter_path)
    origin = _parameter_origin(parameter_provenance)
    origins = list(document.get("parameter_origins_seen", []))
    if origin not in origins:
        origins.append(origin)
    history = list(document.get("parameter_provenance_history", []))
    encoded = json.dumps(parameter_provenance, sort_keys=True)
    if all(json.dumps(item, sort_keys=True) != encoded for item in history):
        history.append(parameter_provenance)
    document["parameter_origins_seen"] = origins
    document["parameter_provenance_history"] = history
    document["latest_parameter_provenance"] = parameter_provenance
    document["latest_publication_parameter_overrides"] = (
        parameter_provenance.get("publication_parameter_overrides")
    )
    runtime_history = list(document.get("runtime_history", []))
    runtime_encoded = json.dumps(runtime, sort_keys=True)
    if all(
        json.dumps(item, sort_keys=True) != runtime_encoded
        for item in runtime_history
    ):
        runtime_history.append(runtime)
    document["runtime_history"] = runtime_history
    document["last_matched_at"] = _utc_now()
    save_json(parameter_path, document)
    return document


def resolve_cv_run(
    args,
    variant: str,
    parameters: dict,
    parameter_provenance: dict,
    create: bool,
    record_observation: bool = True,
) -> dict | None:
    """Match exact parameters to an existing CV_x or create the next CV_x."""
    results_root = Path(args.results_root).resolve()
    cv_base = _variant_root(results_root, variant) / "CV"
    if not cv_base.exists() and not create:
        return None
    cv_base.mkdir(parents=True, exist_ok=True)
    fingerprint = candidate_id(parameters)
    implementation = (
        "conv_lora_checkpoint_input_gradients_v1"
        if variant in {"conv_lora", "conv_lora_llrd"} else None
    )
    for cv_dir, document in _cv_parameter_documents(cv_base):
        if (document.get("parameters") == parameters
                and document.get("training_implementation") == implementation):
            if record_observation:
                document = _record_parameter_observation(
                    cv_dir, parameter_provenance, _runtime(args)
                )
                _write_cv_registry(cv_base, variant)
            return {
                "cv_id": document["cv_id"],
                "cv_dir": cv_dir,
                "document": document,
                "created": False,
            }
    if not create:
        return None

    next_number = max(
        [_cv_number(path.name) for path in cv_base.glob("CV_*") if path.is_dir()]
        or [0]
    ) + 1
    cv_id = f"CV_{next_number}"
    cv_dir = cv_base / cv_id
    cv_dir.mkdir(parents=True, exist_ok=False)
    origin = _parameter_origin(parameter_provenance)
    document = {
        "schema_version": 1,
        "cv_id": cv_id,
        "variant": variant,
        "parameter_fingerprint": fingerprint,
        "parameter_origin": origin,
        "parameter_origins_seen": [origin],
        "parameter_provenance": parameter_provenance,
        "parameter_provenance_history": [parameter_provenance],
        "publication_parameter_overrides": parameter_provenance.get(
            "publication_parameter_overrides"
        ),
        "latest_parameter_provenance": parameter_provenance,
        "latest_publication_parameter_overrides": parameter_provenance.get(
            "publication_parameter_overrides"
        ),
        "parameters": parameters,
        "runtime_at_creation": _runtime(args),
        "runtime_history": [_runtime(args)],
        "percentage_runs": {},
        "created_at": _utc_now(),
        "last_matched_at": _utc_now(),
    }
    if implementation is not None:
        document["training_implementation"] = implementation
    save_json(cv_dir / CV_PARAMETERS_FILENAME, document)
    _write_cv_registry(cv_base, variant)
    return {
        "cv_id": cv_id,
        "cv_dir": cv_dir,
        "document": document,
        "created": True,
    }


def _update_cv_percentage(
    cv_context: dict,
    variant: str,
    percentage: int,
    status: str,
    completed_folds: list[int],
    **extra,
) -> None:
    cv_dir = Path(cv_context["cv_dir"])
    parameter_path = cv_dir / CV_PARAMETERS_FILENAME
    document = read_json(parameter_path)
    key = f"{percentage}_percent"
    value = {
        "status": status,
        "completed_folds": sorted(set(completed_folds)),
        "result_dir": str(cv_dir / key),
        "updated_at": _utc_now(),
    }
    value.update(extra)
    document.setdefault("percentage_runs", {})[key] = value
    save_json(parameter_path, document)
    _write_cv_registry(cv_dir.parent, variant)


def _fold_rows(dataset_root: Path, manifest: dict, train_mapping, val_mapping):
    classes = classes_from_manifest(manifest)
    train_rows = rows_from_mapping(dataset_root, train_mapping, classes)
    validation_rows = rows_from_mapping(dataset_root, val_mapping, classes)
    validate_no_overlap(train_rows, validation_rows)
    return classes, train_rows, validation_rows


def _write_incremental_grid_summary(
    results_root: Path,
    grid_root: Path,
    variant: str,
    percentage: int,
    candidate_summaries: list[dict],
    planned_candidate_count: int,
) -> dict:
    """Persist the current ranking after every completed grid candidate."""
    ranked = sorted(
        candidate_summaries,
        key=lambda value: value["aggregate"]["grid_selection_score"],
        reverse=True,
    )
    for rank, summary in enumerate(ranked, 1):
        summary["rank"] = rank
    best = ranked[0]
    grid_complete = len(ranked) == planned_candidate_count
    best_document = {
        "cv_protocol": "single_loop_5_fold",
        "selection_formula": (
            "0.7*(mean_f1-std_f1)+"
            "0.3*(mean_balanced_accuracy-std_balanced_accuracy)"
        ),
        "variant": variant,
        "percentage": percentage,
        "candidate_count": len(ranked),
        "planned_candidate_count": planned_candidate_count,
        "grid_complete": grid_complete,
        "updated_after_each_completed_candidate": True,
        "best_candidate": best,
        "cv_note": (
            "Each candidate is evaluated on five train/validation pairs. "
            "The candidate with the best aggregate score is selected."
        ),
    }
    save_json(
        grid_root / "grid_summary.json",
        {
            "completed_candidate_count": len(ranked),
            "planned_candidate_count": planned_candidate_count,
            "grid_complete": grid_complete,
            "candidates": ranked,
        },
    )
    grid_csv_rows = [
        {
            "rank": summary["rank"],
            "candidate_id": summary["candidate_id"],
            **summary["aggregate"],
            "parameters": json.dumps(summary["parameters"], sort_keys=True),
        }
        for summary in ranked
    ]
    save_csv(
        grid_root / "grid_summary_by_selection_score.csv",
        grid_csv_rows,
        [
            "rank",
            "candidate_id",
            "grid_selection_score",
            "mean_f1",
            "std_f1",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "parameters",
        ],
    )
    save_json(grid_root / "best_hyperparameters.json", best_document)
    save_json(
        _variant_root(results_root, variant)
        / "Grid"
        / "best_hyperparameters.json",
        best_document,
    )
    return best_document


def run_grid(args, variant: str, percentage: int) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    manifest = load_manifest(dataset_root, percentage)
    grid_path = (
        Path(args.grid_config).resolve()
        if args.grid_config
        else Path(__file__).resolve().parent / GRID_CONFIG_FILENAMES[variant]
    )
    candidates = grid_candidates(grid_path, args.max_grid_candidates)
    expected_keys = set(_provided_parameters(args, variant))
    for index, candidate in enumerate(candidates, 1):
        candidate_keys = set(candidate)
        if candidate_keys != expected_keys:
            raise ValueError(
                f"Grid candidate {index} keys do not match {variant}; "
                f"missing={sorted(expected_keys - candidate_keys)}, "
                f"unexpected={sorted(candidate_keys - expected_keys)}"
            )
    grid_root = _variant_root(results_root, variant) / "Grid" / f"{percentage}_percent"
    grid_root.mkdir(parents=True, exist_ok=True)
    save_json(
        grid_root / "grid_index.json",
        {
            "cv_protocol": "single_loop_5_fold",
            "variant": variant,
            "percentage": percentage,
            "grid_config": str(grid_path),
            "candidate_count": len(candidates),
            "candidates": [
                {"candidate_id": candidate_id(value), "parameters": value}
                for value in candidates
            ],
        },
    )
    if args.dry_run:
        cache_dir, cache_temp_dir = _shared_cache_paths(args)
        return {
            "dry_run": True,
            "percentage": percentage,
            "candidate_count": len(candidates),
            "planned_cv_training_runs": len(candidates) * 5,
            "shared_cache_dir": str(cache_dir),
            "cache_temp_dir": str(cache_temp_dir),
        }

    candidate_summaries = []
    for candidate_index, parameters in enumerate(candidates, 1):
        identifier = candidate_id(parameters)
        candidate_root = grid_root / f"candidate_{identifier}"
        summary_path = candidate_root / "candidate_summary.json"
        existing_candidate = (
            read_json(summary_path) if summary_path.is_file() else None
        )
        if (
            existing_candidate
            and existing_candidate.get("cv_protocol") == "single_loop_5_fold"
            and not args.force_grid
        ):
            candidate_summaries.append(existing_candidate)
            _write_incremental_grid_summary(
                results_root,
                grid_root,
                variant,
                percentage,
                candidate_summaries,
                len(candidates),
            )
            continue

        fold_results = []
        for validation_fold in range(1, 6):
            train_mapping, validation_mapping = cross_validation_split(
                manifest, validation_fold
            )
            classes, train_rows, validation_rows = _fold_rows(
                dataset_root, manifest, train_mapping, validation_mapping
            )
            output_dir = candidate_root / f"fold_{validation_fold:02d}"
            metrics_path = output_dir / "metrics_summary.json"
            summary = read_json(metrics_path) if metrics_path.is_file() else None
            if (
                args.force_grid
                or summary is None
                or summary.get("parameters") != parameters
            ):
                summary = _train_run(
                    variant,
                    train_rows,
                    validation_rows,
                    classes,
                    parameters,
                    _runtime(args),
                    output_dir,
                    _shared_cache_paths(args)[0],
                )
            metrics = summary["validation_metrics"]
            fold_results.append(
                {
                    "fold": validation_fold,
                    "checkpoint": str(output_dir / "best_model.pt"),
                    "best_epoch": summary["best_epoch"],
                    "f1_macro": metrics["f1_macro"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "average_precision_macro": metrics[
                        "average_precision_macro"
                    ],
                    "selection_score": metrics["selection_score"],
                    "output_dir": str(output_dir),
                }
            )

        f1_values = np.asarray([row["f1_macro"] for row in fold_results])
        ba_values = np.asarray(
            [row["balanced_accuracy"] for row in fold_results]
        )
        aggregate = {
            "mean_f1": float(f1_values.mean()),
            "std_f1": float(f1_values.std(ddof=0)),
            "mean_balanced_accuracy": float(ba_values.mean()),
            "std_balanced_accuracy": float(ba_values.std(ddof=0)),
        }
        aggregate["grid_selection_score"] = grid_selection_score(
            aggregate["mean_f1"],
            aggregate["std_f1"],
            aggregate["mean_balanced_accuracy"],
            aggregate["std_balanced_accuracy"],
        )
        candidate_summary = {
            "cv_protocol": "single_loop_5_fold",
            "candidate_index": candidate_index,
            "candidate_id": identifier,
            "variant": variant,
            "percentage": percentage,
            "parameters": parameters,
            "aggregate": aggregate,
            "folds": fold_results,
        }
        save_json(summary_path, candidate_summary)
        candidate_summaries.append(candidate_summary)
        best_document = _write_incremental_grid_summary(
            results_root,
            grid_root,
            variant,
            percentage,
            candidate_summaries,
            len(candidates),
        )

    return best_document


def resolve_parameters(args, variant: str) -> tuple[dict, dict]:
    explicit_names = list(getattr(args, "explicit_parameter_names", []))
    supported_names = set(_provided_parameters(args, variant))
    unsupported_names = sorted(set(explicit_names) - supported_names)
    if unsupported_names:
        raise ValueError(
            f"Parameter options are not valid for {variant}: "
            f"{', '.join(unsupported_names)}"
        )
    requested_source = _requested_parameter_source(args, explicit_names)

    if requested_source == "publication":
        return _publication_parameters(args, variant)
    if requested_source == "publication_derived":
        return _publication_derived_parameters(
            args, variant, explicit_names
        )

    if requested_source in {"default", "manual"}:
        parameters = _provided_parameters(args, variant)
        origin = "provided_manually" if requested_source == "manual" else "default"
        return parameters, {
            "parameter_source": (
                "provided_parameters"
                if requested_source == "manual"
                else "default_parameters"
            ),
            "parameter_origin": origin,
            "explicit_parameter_names": explicit_names,
            "defaults_reference": (
                LINEAR_PROBE_DEFAULTS_REFERENCE
                if variant == "linear_probe"
                else (
                    LLRD_DEFAULTS_REFERENCE
                    if variant == "conv_lora_llrd"
                    else PROJECT_DEFAULTS_REFERENCE
                )
            ),
            "source_percentage": None,
            "parameters": parameters,
        }

    results_root = Path(args.results_root).resolve()
    grid_root = _variant_root(results_root, variant) / "Grid"
    if args.percentage_specific_grid:
        source_percentage = int(args.percentage)
        best_path = (
            grid_root / f"{source_percentage}_percent" / "best_hyperparameters.json"
        )
        existing = read_json(best_path) if best_path.is_file() else None
        document = (
            run_grid(args, variant, source_percentage)
            if (
                args.force_grid
                or existing is None
                or existing.get("cv_protocol") != "single_loop_5_fold"
                or existing.get("grid_complete") is False
            )
            else existing
        )
        source = "percentage_specific_grid"
    else:
        source_percentage = int(args.default_grid_percentage)
        preferred = (
            grid_root / f"{source_percentage}_percent" / "best_hyperparameters.json"
        )
        preferred_document = read_json(preferred) if preferred.is_file() else None
        if (
            preferred_document
            and preferred_document.get("cv_protocol") == "single_loop_5_fold"
            and preferred_document.get("grid_complete") is not False
            and not args.force_grid
        ):
            document = preferred_document
        else:
            available = [
                path
                for path in sorted(
                    grid_root.glob("*_percent/best_hyperparameters.json")
                )
                if (
                    read_json(path).get("cv_protocol")
                    == "single_loop_5_fold"
                    and read_json(path).get("grid_complete") is not False
                )
            ]
            if available and not args.force_grid:
                document = read_json(available[0])
                source_percentage = int(document["percentage"])
            else:
                original_percentage = args.percentage
                args.percentage = source_percentage
                document = run_grid(args, variant, source_percentage)
                args.percentage = original_percentage
        source = "shared_grid"

    if document.get("dry_run"):
        return {}, {"parameter_source": "dry_run", **document}
    parameters = document["best_candidate"]["parameters"]
    return parameters, {
        "parameter_source": source,
        "parameter_origin": "grid_search",
        "source_percentage": source_percentage,
        "selection_formula": document["selection_formula"],
        "grid_selection": document["best_candidate"]["aggregate"],
        "candidate_id": document["best_candidate"]["candidate_id"],
        "parameters": parameters,
    }


def _materialize_checkpoint(source: Path, destination: Path) -> str:
    """Place a checkpoint in CV, preferring a zero-copy same-volume hard link."""
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == source.stat().st_size
    ):
        return "existing"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
        storage = "hard_link"
    except OSError:
        shutil.copy2(source, temporary)
        storage = "copied"
    os.replace(temporary, destination)
    return storage


def _publish_grid_fold(
    source_dir: Path,
    cv_fold_dir: Path,
    candidate_id_value: str,
) -> Path:
    cv_fold_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "metrics_summary.json",
        "validation_patient_predictions.json",
        "validation_probabilities.csv",
        "training_history.csv",
        "training_state.json",
    ):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, cv_fold_dir / filename)
    checkpoint = cv_fold_dir / "best_model.pt"
    checkpoint_storage = _materialize_checkpoint(
        source_dir / "best_model.pt", checkpoint
    )
    save_json(
        cv_fold_dir / "selected_run.json",
        {
            "source": "selected_grid_candidate",
            "candidate_id": candidate_id_value,
            "source_output_dir": str(source_dir),
            "source_checkpoint": str(source_dir / "best_model.pt"),
            "checkpoint": str(checkpoint),
            "checkpoint_storage": checkpoint_storage,
        },
    )
    return checkpoint


def _import_legacy_cv_percentage(
    results_root: Path,
    variant: str,
    percentage: int,
    parameters: dict,
    cv_context: dict,
) -> bool:
    """Import a completed pre-CV_x percentage folder without retraining."""
    legacy_root = (
        _variant_root(results_root, variant)
        / "CV"
        / f"{percentage}_percent"
    )
    target_root = Path(cv_context["cv_dir"]) / f"{percentage}_percent"
    legacy_summary = legacy_root / "best_fold_model.json"
    if target_root.exists() or not legacy_summary.is_file():
        return False

    document = read_json(legacy_summary)
    if document.get("parameters") != parameters:
        return False
    fold_documents = {
        int(item["fold"]): item for item in document.get("folds", [])
    }
    if set(fold_documents) != set(range(1, 6)):
        return False

    artifact_names = (
        "metrics_summary.json",
        "validation_patient_predictions.json",
        "validation_probabilities.csv",
        "training_history.csv",
        "training_state.json",
    )
    for fold in range(1, 6):
        source_fold = legacy_root / f"fold_{fold:02d}"
        target_fold = target_root / f"fold_{fold:02d}"
        metrics_path = source_fold / "metrics_summary.json"
        checkpoint_source = Path(fold_documents[fold]["checkpoint"])
        if not metrics_path.is_file() or not checkpoint_source.is_file():
            return False
        target_fold.mkdir(parents=True, exist_ok=True)
        for filename in artifact_names:
            source = source_fold / filename
            if source.is_file():
                shutil.copy2(source, target_fold / filename)
        checkpoint = target_fold / "best_model.pt"
        checkpoint_storage = _materialize_checkpoint(
            checkpoint_source, checkpoint
        )
        save_json(
            target_fold / "selected_run.json",
            {
                "source": "legacy_cv_import",
                "legacy_output_dir": str(source_fold),
                "source_checkpoint": str(checkpoint_source),
                "checkpoint": str(checkpoint),
                "checkpoint_storage": checkpoint_storage,
            },
        )
    return True


def run_cross_validation(
    args,
    variant: str,
    parameters: dict,
    parameter_provenance: dict,
    cv_context: dict,
) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    percentage = int(args.percentage)
    manifest = load_manifest(dataset_root, percentage)
    cv_root = Path(cv_context["cv_dir"]) / f"{percentage}_percent"
    _import_legacy_cv_percentage(
        results_root,
        variant,
        percentage,
        parameters,
        cv_context,
    )
    completed_folds = [
        fold
        for fold in range(1, 6)
        if (cv_root / f"fold_{fold:02d}" / "metrics_summary.json").is_file()
        and (cv_root / f"fold_{fold:02d}" / "best_model.pt").is_file()
    ]
    existing_document_path = cv_root / "best_fold_model.json"
    if (
        not args.force_cv
        and existing_document_path.is_file()
        and len(completed_folds) == 5
    ):
        existing_document = read_json(existing_document_path)
        if existing_document.get("parameters") == parameters:
            # Preserve how the completed models were originally trained while
            # making the provenance of this reuse request visible. This is
            # especially important when publication-derived parameters match
            # an earlier manual/default run exactly.
            existing_document["latest_parameter_origin"] = _parameter_origin(
                parameter_provenance
            )
            existing_document["latest_parameter_provenance"] = (
                parameter_provenance
            )
            existing_document["latest_publication_parameter_overrides"] = (
                parameter_provenance.get("publication_parameter_overrides")
            )
            existing_document["last_reused_at"] = _utc_now()
            save_json(existing_document_path, existing_document)
            _update_cv_percentage(
                cv_context,
                variant,
                percentage,
                "completed",
                completed_folds,
                summary_file=str(existing_document_path),
            )
            return existing_document
    _update_cv_percentage(
        cv_context,
        variant,
        percentage,
        "running",
        completed_folds,
    )
    candidate_value = parameter_provenance.get("candidate_id")
    grid_source_percentage = parameter_provenance.get("source_percentage")
    selected_grid_root = None
    if (
        candidate_value
        and grid_source_percentage == int(args.percentage)
        and not args.force_cv
    ):
        selected_grid_root = (
            _variant_root(results_root, variant)
            / "Grid"
            / f"{args.percentage}_percent"
            / f"candidate_{candidate_value}"
        )
    fold_results = []
    for validation_fold in range(1, 6):
        train_mapping, validation_mapping = cross_validation_split(
            manifest, validation_fold
        )
        classes, train_rows, validation_rows = _fold_rows(
            dataset_root, manifest, train_mapping, validation_mapping
        )
        cv_fold_dir = cv_root / f"fold_{validation_fold:02d}"
        grid_fold_dir = (
            selected_grid_root / f"fold_{validation_fold:02d}"
            if selected_grid_root
            else None
        )
        grid_metrics = grid_fold_dir / "metrics_summary.json" if grid_fold_dir else None
        if grid_metrics and grid_metrics.is_file():
            summary = read_json(grid_metrics)
            if summary.get("parameters") != parameters:
                raise RuntimeError(
                    f"Selected grid fold parameters differ in {grid_metrics}"
                )
            output_dir = grid_fold_dir
            result_origin = "selected_grid_run"
            checkpoint = _publish_grid_fold(
                output_dir, cv_fold_dir, candidate_value
            )
        else:
            output_dir = cv_fold_dir
            metrics_path = output_dir / "metrics_summary.json"
            summary = read_json(metrics_path) if metrics_path.is_file() else None
            if (
                args.force_cv
                or summary is None
                or summary.get("parameters") != parameters
                or not (output_dir / "best_model.pt").is_file()
            ):
                summary = _train_run(
                    variant,
                    train_rows,
                    validation_rows,
                    classes,
                    parameters,
                    _runtime(args),
                    output_dir,
                    _shared_cache_paths(args)[0],
                    parameter_provenance=parameter_provenance,
                )
            selected_run_path = output_dir / "selected_run.json"
            selected_run = (
                read_json(selected_run_path)
                if selected_run_path.is_file()
                else {}
            )
            result_origin = selected_run.get("source", "cv_run")
            checkpoint = output_dir / "best_model.pt"
        metrics = summary["validation_metrics"]
        fold_results.append(
            {
                "fold": validation_fold,
                "checkpoint": str(checkpoint),
                "output_dir": str(output_dir),
                "published_cv_dir": str(cv_fold_dir),
                "result_origin": result_origin,
                "best_epoch": summary["best_epoch"],
                "parameter_origin": _parameter_origin(
                    parameter_provenance
                ),
                "publication_override_count": (
                    parameter_provenance.get(
                        "publication_parameter_overrides", {}
                    ).get("changed_parameter_count", 0)
                ),
                "parameters": parameters,
                "grid_candidate_id": candidate_value,
                "f1_macro": metrics["f1_macro"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "average_precision_macro": metrics["average_precision_macro"],
                "selection_score": fold_selection_score(metrics),
            }
        )
        completed_folds.append(validation_fold)
        _update_cv_percentage(
            cv_context,
            variant,
            percentage,
            "running",
            completed_folds,
        )

    best_fold = max(fold_results, key=lambda value: value["selection_score"])
    f1_values = [row["f1_macro"] for row in fold_results]
    ba_values = [row["balanced_accuracy"] for row in fold_results]
    document = {
        "cv_id": cv_context["cv_id"],
        "variant": variant,
        "percentage": percentage,
        "fold_selection_formula": "0.7*f1_macro+0.3*balanced_accuracy",
        "parameter_origin": _parameter_origin(parameter_provenance),
        "parameter_provenance": parameter_provenance,
        "publication_parameter_overrides": parameter_provenance.get(
            "publication_parameter_overrides"
        ),
        "parameters": parameters,
        "folds": fold_results,
        "cv_summary": {
            "mean_f1": statistics.mean(f1_values),
            "std_f1": statistics.pstdev(f1_values),
            "mean_balanced_accuracy": statistics.mean(ba_values),
            "std_balanced_accuracy": statistics.pstdev(ba_values),
        },
        "best_fold": best_fold,
    }
    save_json(cv_root / "best_fold_model.json", document)
    save_csv(
        cv_root / "cv_fold_summary.csv",
        fold_results,
        [
            "fold",
            "selection_score",
            "f1_macro",
            "balanced_accuracy",
            "average_precision_macro",
            "best_epoch",
            "parameter_origin",
            "publication_override_count",
            "grid_candidate_id",
            "result_origin",
            "checkpoint",
            "output_dir",
            "published_cv_dir",
            "parameters",
        ],
    )
    _update_cv_percentage(
        cv_context,
        variant,
        percentage,
        "completed",
        completed_folds,
        summary_file=str(cv_root / "best_fold_model.json"),
        best_fold=int(best_fold["fold"]),
        best_checkpoint=best_fold["checkpoint"],
    )
    return document


def _test_rows(dataset_root: Path, manifest: dict):
    classes = classes_from_manifest(manifest)
    return classes, rows_from_mapping(
        dataset_root, heldout_mapping(manifest), classes
    )


def _write_test_outputs(
    test_root: Path,
    variant: str,
    percentage: int,
    metrics: dict,
    predictions: list[dict],
    frame,
    selection: dict,
) -> dict:
    test_root.mkdir(parents=True, exist_ok=True)
    save_json(test_root / "test_patient_predictions.json", predictions)
    frame.to_csv(test_root / "test_probabilities.csv", index=False)
    save_json(test_root / "model_selection.json", selection)
    summary = {
        "model_name": f"merlin_{variant}",
        "model_family": "Merlin",
        "variant": variant,
        "percentage": percentage,
        "cv_id": selection.get("cv_id"),
        "parameter_origin": selection.get("parameter_origin"),
        "publication_parameter_overrides": selection.get(
            "publication_parameter_overrides"
        ),
        "model_selection": selection,
        "test_metrics": metrics,
    }
    save_json(test_root / "metrics_summary.json", summary)
    return summary


def _validate_test_checkpoint_identity(
    checkpoint_state: dict,
    variant: str,
    classes: list[str],
    parameters: dict,
) -> None:
    """Reject checkpoints that would be mislabeled by the current test run."""
    if checkpoint_state.get("variant") != variant:
        raise ValueError(
            "Checkpoint variant "
            f"{checkpoint_state.get('variant')!r} does not match {variant!r}."
        )
    if checkpoint_state.get("classes") != classes:
        raise ValueError(
            "Checkpoint class order does not exactly match the current "
            "dataset manifest."
        )
    if checkpoint_state.get("parameters") != parameters:
        raise ValueError(
            "Checkpoint parameters do not exactly match the parameters "
            "selected for this test run."
        )


def run_heldout_test(
    args,
    variant: str,
    parameters: dict,
    parameter_provenance: dict,
    cv_document: dict | None,
    cv_context: dict,
) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    percentage = int(args.percentage)
    manifest = load_manifest(dataset_root, percentage)
    classes, test_rows = _test_rows(dataset_root, manifest)
    source = args.model_source
    test_root = (
        _variant_root(results_root, variant)
        / "Test"
        / cv_context["cv_id"]
        / source
        / f"{percentage}_percent"
    )
    cv_root = Path(cv_context["cv_dir"]) / f"{percentage}_percent"
    if cv_document is None and (cv_root / "best_fold_model.json").is_file():
        cv_document = read_json(cv_root / "best_fold_model.json")

    checkpoints: list[Path]
    if source == "provided_checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for provided_checkpoint")
        checkpoints = [Path(args.checkpoint).resolve()]
    elif source == "best_cv_fold":
        if cv_document is None:
            raise FileNotFoundError("Run CV before selecting best_cv_fold.")
        checkpoints = [Path(cv_document["best_fold"]["checkpoint"])]
    elif source == "cv_ensemble":
        if cv_document is None:
            raise FileNotFoundError("Run CV before selecting cv_ensemble.")
        checkpoints = [Path(row["checkpoint"]) for row in cv_document["folds"]]
    elif source == "refit_full_development":
        if cv_document is None:
            raise FileNotFoundError("Run CV before refitting the full development pool.")
        best_epochs = [
            int(row["best_epoch"])
            for row in cv_document["folds"]
            if row["best_epoch"] is not None
        ]
        refit_epochs = max(1, round(statistics.median(best_epochs)))
        development_rows = rows_from_mapping(
            dataset_root, development_mapping(manifest), classes
        )
        validate_no_overlap(development_rows, test_rows)
        refit_root = cv_root / "refit_full_development"
        refit_metrics = refit_root / "metrics_summary.json"
        refit_summary = read_json(refit_metrics) if refit_metrics.is_file() else None
        if (
            args.force_cv
            or refit_summary is None
            or refit_summary.get("parameters") != parameters
            or int(refit_summary.get("best_epoch") or 0) != refit_epochs
        ):
            _train_run(
                variant,
                development_rows,
                None,
                classes,
                parameters,
                _runtime(args),
                refit_root,
                _shared_cache_paths(args)[0],
                parameter_provenance=parameter_provenance,
                fixed_epochs=refit_epochs,
            )
        checkpoints = [refit_root / "best_model.pt"]
    else:
        raise ValueError(f"Unknown model source: {source}")

    probability_sets = []
    labels_reference = None
    checkpoint_metadata = []
    for index, checkpoint in enumerate(checkpoints, 1):
        metrics, _, _, labels, probabilities, checkpoint_state = _predict_checkpoint(
            checkpoint,
            test_rows,
            _shared_cache_paths(args)[0],
            int(parameters["batch_size"]),
            int(args.num_workers),
        )
        _validate_test_checkpoint_identity(
            checkpoint_state, variant, classes, parameters
        )
        if labels_reference is None:
            labels_reference = labels
        elif not np.array_equal(labels_reference, labels):
            raise RuntimeError("Ensemble checkpoint patient order differs.")
        probability_sets.append(probabilities)
        checkpoint_metadata.append(
            {
                "path": str(checkpoint),
                "variant": checkpoint_state["variant"],
                "epoch": checkpoint_state["epoch"],
                "best_epoch": checkpoint_state.get("best_epoch"),
            }
        )

    probabilities = np.mean(np.stack(probability_sets), axis=0)
    predictions_array = probabilities.argmax(axis=1)
    test_metrics = compute_metrics(
        labels_reference, predictions_array, probabilities, classes
    )
    test_metrics["selection_score"] = fold_selection_score(test_metrics)
    predictions, frame = format_predictions(
        [row["patient_id"] for row in test_rows],
        [row["label_str"] for row in test_rows],
        probabilities,
        classes,
    )
    selection = {
        "cv_id": cv_context["cv_id"],
        "cv_parameters_file": str(
            Path(cv_context["cv_dir"]) / CV_PARAMETERS_FILENAME
        ),
        "test_results_directory": str(test_root),
        "model_source": source,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoint_metadata,
        "parameter_origin": _parameter_origin(parameter_provenance),
        "parameter_provenance": parameter_provenance,
        "publication_parameter_overrides": parameter_provenance.get(
            "publication_parameter_overrides"
        ),
        "parameters": parameters,
        "shared_cache_dir": str(_shared_cache_paths(args)[0]),
        "heldout_test_used_during_selection": False,
    }
    return _write_test_outputs(
        test_root,
        variant,
        percentage,
        test_metrics,
        predictions,
        frame,
        selection,
    )


def run_pipeline(args, variant: str) -> dict:
    if variant not in VARIANT_FOLDERS:
        raise ValueError(f"Unknown Merlin variant: {variant}")
    args.grid = parse_bool(args.grid)
    args.percentage_specific_grid = parse_bool(args.percentage_specific_grid)
    args.train_image_projection = parse_bool(args.train_image_projection)
    args.gradient_checkpointing = parse_bool(args.gradient_checkpointing)
    if hasattr(args, "exclude_norm_bias_from_weight_decay"):
        args.exclude_norm_bias_from_weight_decay = parse_bool(
            args.exclude_norm_bias_from_weight_decay
        )
    if int(args.percentage) not in {20, 40, 60, 80, 100}:
        raise ValueError("--percentage must be one of 20, 40, 60, 80, 100")
    if int(getattr(args, "feature_extraction_batch_size", 1)) <= 0:
        raise ValueError("--feature_extraction_batch_size must be positive.")
    cache_dir, cache_temp_dir = _configure_shared_cache(
        args, create=not args.dry_run
    )

    if args.stage == "grid":
        if getattr(args, "parameter_source", "auto") not in {"auto", "grid"}:
            raise ValueError(
                "--stage grid requires --parameter_source auto or grid."
            )
        percentage = (
            int(args.percentage)
            if args.percentage_specific_grid
            else int(args.default_grid_percentage)
        )
        return run_grid(args, variant, percentage)

    parameters, provenance = resolve_parameters(args, variant)
    if args.dry_run:
        matching_cv = resolve_cv_run(
            args,
            variant,
            parameters,
            provenance,
            create=False,
            record_observation=False,
        )
        return {
            "dry_run": True,
            "stage": args.stage,
            "variant": variant,
            "percentage": int(args.percentage),
            "parameter_provenance": provenance,
            "publication_parameter_overrides": provenance.get(
                "publication_parameter_overrides"
            ),
            "matching_cv_id": (
                matching_cv["cv_id"] if matching_cv else None
            ),
            "shared_cache_dir": str(cache_dir),
            "cache_temp_dir": str(cache_temp_dir),
            "planned_test_output": (
                str(
                    _variant_root(
                        Path(args.results_root).resolve(), variant
                    )
                    / "Test"
                    / matching_cv["cv_id"]
                    / args.model_source
                    / f"{int(args.percentage)}_percent"
                )
                if matching_cv and args.stage in {"test", "all"}
                else None
            ),
        }

    cv_context = resolve_cv_run(
        args,
        variant,
        parameters,
        provenance,
        create=args.stage in {"cv", "all"},
    )
    if cv_context is None:
        raise FileNotFoundError(
            "No CV_x has parameters matching this test request. "
            "Run --stage cv (or --stage all) with the same parameters first."
        )

    cv_document = None
    if args.stage in {"cv", "all"}:
        cv_document = run_cross_validation(
            args, variant, parameters, provenance, cv_context
        )
    if args.stage in {"test", "all"}:
        return run_heldout_test(
            args,
            variant,
            parameters,
            provenance,
            cv_document,
            cv_context,
        )
    return cv_document
