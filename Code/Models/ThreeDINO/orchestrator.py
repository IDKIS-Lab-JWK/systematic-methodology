"""Leakage-safe search, five-fold CV, refit, and test orchestration for 3DINO."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .recipe import recipe_parameters
except ImportError:  # direct scripts
    from recipe import recipe_parameters

try:
    from . import (
        CLASS_NAMES,
        DEFAULT_CACHE_NAMESPACE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from .cache import load_cache_manifest, validate_cache
    from .data import (
        cross_validation_split,
        development_mapping,
        heldout_mapping,
        load_manifest,
        rows_from_mapping,
        validate_no_overlap,
    )
    from .metrics import compute_metrics, fold_selection_score, format_predictions
    from .model import resolve_checkpoint, validate_upstream_checkout
    from .provisioning import provision_assets
    from .search import aggregate_fold_records, candidate_id, load_search_definition, rank_records, search_candidates
    from .training import (
        evaluate_checkpoint,
        predict_checkpoint,
        refit_model,
        train_full_finetune,
        train_lora,
        train_linear_probe,
        train_linear_probe_grid,
    )
except ImportError:  # pragma: no cover - direct scripts
    from __init__ import (
        CLASS_NAMES,
        DEFAULT_CACHE_NAMESPACE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from cache import load_cache_manifest, validate_cache
    from data import cross_validation_split, development_mapping, heldout_mapping, load_manifest, rows_from_mapping, validate_no_overlap
    from metrics import compute_metrics, fold_selection_score, format_predictions
    from model import resolve_checkpoint, validate_upstream_checkout
    from provisioning import provision_assets
    from search import aggregate_fold_records, candidate_id, load_search_definition, rank_records, search_candidates
    from training import evaluate_checkpoint, predict_checkpoint, refit_model, train_full_finetune, train_lora, train_linear_probe, train_linear_probe_grid


METHOD_FOLDERS = {"linear_probe": "LinearProbe", "full_finetune": "FullFineTune", "lora": "LoRA"}
PROFILE_FILES = {
    "lora": {"default": "default_lora.json"},
    "linear_probe": {
        "default": "default_linear_probe.json",
        "publication": "publication_linear_probe.json",
        "grid": "search_linear_probe.json",
        "search": "search_linear_probe.json",
    },
    "full_finetune": {
        "default": "default_full_finetune.json",
        "grid": "search_full_finetune.json",
        "search": "search_full_finetune.json",
    },
}


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    return methodology_root() / "Dataset"


def default_results_root() -> Path:
    return methodology_root() / "Results" / "3DINO"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def save_json(path: str | Path, value: Mapping) -> None:
    """Atomically write one standards-compliant JSON document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    os.replace(temporary, target)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible provenance without depending on key order or paths."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_rows_fingerprint(rows: Sequence[Mapping]) -> str:
    """Mirror training.py's ordered, source-aware development-row identity."""

    identity = [
        {
            "position": index,
            "patient_id": str(row.get("patient_id", "")),
            "label": int(row["label"]),
            "label_str": str(row.get("label_str", "")),
            "source_fingerprint": row.get("source_fingerprint"),
            "source_size_bytes": row.get("source_size_bytes"),
            "source_mtime_ns": row.get("source_mtime_ns"),
            "source": str(
                row.get("source_path", row.get("cached_path", row.get("image", "")))
                if isinstance(
                    row.get(
                        "source_path", row.get("cached_path", row.get("image", ""))
                    ),
                    (str, Path),
                )
                else "in_memory"
            ),
        }
        for index, row in enumerate(rows)
    ]
    return canonical_sha256(identity)


def _runtime_plan_fingerprint(
    parameters: Mapping, runtime_plan: Mapping
) -> str:
    return canonical_sha256(
        {
            "parameters": dict(parameters),
            "runtime_plan": dict(runtime_plan),
        }
    )


def method_root(results_root: str | Path, method: str) -> Path:
    try:
        folder = METHOD_FOLDERS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown 3DINO method: {method}") from exc
    return Path(results_root) / folder


def _profile_path(method: str, source: str) -> Path:
    source = "grid" if source == "search" else source
    if method == "full_finetune" and source == "publication":
        raise ValueError(
            "3DINO published frozen linear-probe classification experiments, not "
            "a classification full-fine-tuning recipe. Use default, grid, or a "
            "reviewed manual full-fine-tuning profile."
        )
    try:
        filename = PROFILE_FILES[method][source]
    except KeyError as exc:
        raise ValueError(f"No {source!r} profile exists for 3DINO {method}") from exc
    return Path(__file__).resolve().parent / filename


def load_parameter_profile(
    method: str, source: str, config_path: str | Path | None = None
) -> tuple[dict, dict]:
    """Load one resolved profile; publication probing is intentionally a grid."""

    if source not in {"default", "manual"}:
        if source == "publication" and method == "linear_probe":
            raise ValueError(
                "The publication linear-probe profile is a 52-head search space, "
                "not one resolved parameter set. Run the search stage first."
            )
        raise ValueError(f"Cannot load a single parameter profile for source={source}")
    if source == "manual":
        if config_path is None:
            raise ValueError("--parameter_source manual requires --parameter_config.")
        path = Path(config_path)
    else:
        path = _profile_path(method, source)
    document = read_json(path)
    if document.get("method") != method:
        raise ValueError(f"Parameter profile method mismatch in {path}")
    parameters = document.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"No non-empty parameters object in {path}")
    provenance = {
        "parameter_source": source,
        "profile_path": str(path.resolve()),
        "profile_sha256": sha256_file(path),
        "profile_type": document.get("profile_type", source),
        "source": document.get("source"),
        "authorization": document.get("authorization"),
        "task_adaptations": document.get("task_adaptations", {}),
    }
    return dict(parameters), provenance


def _identity_fingerprint_fields(model_identity: Mapping) -> dict:
    """Exclude machine-local paths while retaining the exact code/weight identity."""

    return {
        "model_repo_id": model_identity.get("model_repo_id", MODEL_REPO_ID),
        "model_revision": model_identity.get("model_revision", MODEL_REVISION),
        "upstream_repository": model_identity.get("upstream_repository", UPSTREAM_REPOSITORY),
        "upstream_revision": model_identity.get("upstream_revision", UPSTREAM_REVISION),
        "pretrained_checkpoint_sha256": model_identity.get("pretrained_checkpoint_sha256"),
    }


def parameter_fingerprint(method: str, parameters: Mapping, model_identity: Mapping) -> str:
    payload = json.dumps(
        {
            "method": method,
            "parameters": recipe_parameters(method, parameters),
            "model_identity": _identity_fingerprint_fields(model_identity),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def register_cv_run(
    results_root: str | Path,
    method: str,
    parameters: Mapping,
    provenance: Mapping,
    model_identity: Mapping,
) -> tuple[str, Path]:
    """Map the exact parameters and pretrained identity to one stable ``CV_x``."""

    cv_root = method_root(results_root, method) / "CV"
    registry_path = cv_root / "cv_runs.json"
    registry = (
        read_json(registry_path)
        if registry_path.is_file()
        else {"schema_version": 1, "model_family": "3DINO", "method": method, "runs": []}
    )
    if registry.get("method") != method or not isinstance(registry.get("runs"), list):
        raise ValueError(f"Invalid 3DINO CV registry: {registry_path}")
    fingerprint = parameter_fingerprint(method, parameters, model_identity)
    existing = next(
        (item for item in registry["runs"] if item.get("parameter_fingerprint") == fingerprint),
        None,
    )
    if existing is None:
        used: list[int] = []
        for item in registry["runs"]:
            value = str(item.get("cv_id", ""))
            if value.startswith("CV_") and value[3:].isdigit():
                used.append(int(value[3:]))
        cv_id = f"CV_{max(used, default=0) + 1}"
        registry["runs"].append(
            {
                "cv_id": cv_id,
                "parameter_fingerprint": fingerprint,
                "parameters": recipe_parameters(method, parameters),
                "model_identity": dict(model_identity),
                "created_at_utc": utc_now(),
            }
        )
        save_json(registry_path, registry)
    else:
        cv_id = str(existing["cv_id"])

    cv_dir = cv_root / cv_id
    cv_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = cv_dir / "cv_parameters.json"
    document = (
        read_json(parameter_path)
        if parameter_path.is_file()
        else {
            "schema_version": 1,
            "model_family": "3DINO",
            "cv_id": cv_id,
            "method": method,
            "parameter_fingerprint": fingerprint,
            "parameters": recipe_parameters(method, parameters),
            "model_identity": dict(model_identity),
            "provenance_history": [],
            "percentages": {},
        }
    )
    history = document.setdefault("provenance_history", [])
    event_payload = dict(provenance)
    previous_payload = None
    if history and isinstance(history[-1], Mapping):
        previous_payload = dict(history[-1])
        previous_payload.pop("recorded_at_utc", None)
    if previous_payload != event_payload:
        history.append({"recorded_at_utc": utc_now(), **event_payload})
        save_json(parameter_path, document)
    elif not parameter_path.is_file():
        save_json(parameter_path, document)
    return cv_id, cv_dir


def _runtime_model_identity(args: Any, *, require_files: bool) -> dict:
    upstream_root = Path(args.upstream_root).resolve(strict=False)
    checkpoint_value = getattr(args, "pretrained_checkpoint", None)
    checkpoint = (
        Path(checkpoint_value).resolve(strict=False)
        if checkpoint_value is not None
        else None
    )
    repo_id = str(getattr(args, "model_repo_id", MODEL_REPO_ID))
    model_revision = str(getattr(args, "model_revision", MODEL_REVISION))
    expected = getattr(args, "pretrained_checkpoint_sha256", MODEL_SHA256)
    actual: str | None = None
    upstream_validation: dict | None = None
    provisioning: dict | None = None
    if require_files:
        provision_log_value = getattr(args, "provision_log", None)
        assets = provision_assets(
            upstream_root=upstream_root,
            upstream_revision=str(args.upstream_revision),
            checkpoint_path=checkpoint,
            repo_id=repo_id,
            model_revision=model_revision,
            expected_sha256=expected,
            expected_size_bytes=MODEL_SIZE_BYTES,
            local_files_only=bool(getattr(args, "local_files_only", False)),
            log_path=(
                Path(provision_log_value)
                if provision_log_value is not None
                else Path(args.results_root) / "Setup" / "asset_provisioning.jsonl"
            ),
        )
        upstream_root = assets.upstream_root.resolve(strict=False)
        checkpoint = assets.checkpoint_path.resolve(strict=False)
        args.upstream_root = upstream_root
        args.pretrained_checkpoint = checkpoint
        # The provisioner already performed the structural check plus Git
        # origin/HEAD/clean-tree attestation. Preserve that stronger report;
        # the model loader repeats its own structural/import checks later.
        upstream_validation = dict(assets.upstream)
        checkpoint_info = resolve_checkpoint(checkpoint, expected_sha256=expected)
        actual = checkpoint_info.sha256
        provisioning = assets.to_dict()
        args._asset_provisioning = provisioning
    checkpoint_reference = (
        str(checkpoint)
        if checkpoint is not None
        else f"hf://{repo_id}@{model_revision}/{getattr(args, 'model_filename', '3dino_vit_weights.pth')}"
    )
    return {
        "model_repo_id": repo_id,
        "model_revision": model_revision,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_root": str(upstream_root),
        "upstream_revision": str(args.upstream_revision),
        "pretrained_checkpoint": checkpoint_reference,
        "pretrained_checkpoint_sha256": actual or expected,
        "downloaded_automatically": bool(
            provisioning and provisioning.get("downloads_performed")
        ),
        "git_operations_performed": bool(
            provisioning and provisioning.get("git_operations_performed")
        ),
        "upstream_modified": (
            upstream_validation.get("tracked_modifications_present")
            if isinstance(upstream_validation, Mapping)
            else None
        ),
        "upstream_modification_status": (
            upstream_validation.get("modification_status")
            if isinstance(upstream_validation, Mapping)
            else "planned_not_inspected"
        ),
        "upstream_validation": upstream_validation,
        "asset_provisioning": provisioning,
    }


def _runtime_execution_context(
    args: Any,
    manifest: Mapping,
    cache_manifest: Mapping,
    model_identity: Mapping,
) -> dict:
    """Bind stochastic, split, cache, and pretrained identities to every run.

    Paths are deliberately excluded from the fingerprinted context so an
    unchanged experiment can be relocated. File content, patient/cache
    membership, and the effective stochastic seed remain binding. DataLoader
    worker counts are operational provenance and deliberately excluded.
    """

    split_path = (
        Path(args.dataset_root) / "Splits" / f"{int(args.percentage)}_percent.json"
    )
    entries = cache_manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Validated 3DINO cache manifest has no entries list.")
    artifact_integrity = cache_manifest.get("artifact_integrity")
    cache_artifact_seal = (
        artifact_integrity.get("seal_sha256")
        if isinstance(artifact_integrity, Mapping)
        else None
    )
    if (
        not isinstance(cache_artifact_seal, str)
        or len(cache_artifact_seal) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in cache_artifact_seal)
    ):
        raise ValueError(
            "Validated 3DINO cache manifest has no valid artifact-integrity seal SHA-256."
        )
    cache_membership = sorted(
        (
            str(entry.get("patient_id")),
            int(entry.get("label")),
            str(entry.get("cache_key")),
        )
        for entry in entries
    )
    return {
        "schema_version": 2,
        "percentage": int(args.percentage),
        "seed": int(args.seed),
        "split_manifest_sha256": sha256_file(split_path),
        "split_manifest_semantic_sha256": canonical_sha256(dict(manifest)),
        "cache_plan_fingerprint": str(cache_manifest.get("plan_fingerprint")),
        "cache_artifact_integrity_sha256": cache_artifact_seal.lower(),
        "cache_split_manifest_sha256": str(
            cache_manifest.get("split_manifest_sha256")
        ),
        "cache_membership_sha256": canonical_sha256(cache_membership),
        "cache_source_count": len(entries),
        "model_identity": _identity_fingerprint_fields(model_identity),
    }


def _apply_execution_context(
    parameters: Mapping,
    args: Any,
    execution_context: Mapping,
) -> dict:
    """Resolve CLI stochastic settings and reject stale search selections."""

    result = dict(parameters)
    expected = dict(execution_context)
    recorded = result.get("_execution_context")
    if recorded is not None and recorded != expected:
        raise RuntimeError(
            "Selected parameters were produced with a different split, cache "
            "plan, seed, execution-context schema, or pretrained checkpoint. "
            "Rerun the development-only search for the current execution context."
        )
    result["seed"] = int(args.seed)
    result["_execution_context"] = expected
    return result


def _authorization_status(args: Any, authorization: Mapping | None) -> dict:
    authorization = authorization if isinstance(authorization, Mapping) else {}
    status = {
        "cli_acknowledged": bool(
            getattr(args, "allow_experimental_full_finetune", False)
        ),
        "profile_license_review": authorization.get("license_approval_claimed")
        is True,
        "profile_memory_preflight": authorization.get("memory_preflight_passed")
        is True,
    }
    status["authorized"] = all(status.values())
    return status


def _common_training_kwargs(args: Any) -> dict:
    import torch

    train_workers = int(args.num_workers)
    if train_workers < 0:
        raise ValueError("3DINO training num_workers must be non-negative.")
    return {
        "device": torch.device(args.device),
        "upstream_root": Path(args.upstream_root),
        "upstream_revision": str(args.upstream_revision),
        "checkpoint_path": Path(args.pretrained_checkpoint),
        "expected_checkpoint_sha256": getattr(args, "pretrained_checkpoint_sha256", None),
        "cache_dir": Path(args.cache_dir),
        "classes": list(CLASS_NAMES),
        "seed": int(args.seed),
        "num_workers": train_workers,
        "resume": bool(args.resume),
        "asset_provisioning": getattr(args, "_asset_provisioning", None),
    }


def _operational_loader_provenance(args: Any) -> dict:
    """Describe loader throughput settings without making them experiment identity."""

    train_workers = int(args.num_workers)
    if train_workers < 0:
        raise ValueError("3DINO training num_workers must be non-negative.")
    return {
        "schema_version": 1,
        "semantic_identity": False,
        "train_num_workers": train_workers,
        "persistent_train_workers": train_workers > 0,
        "validation_num_workers": 0,
        "test_num_workers": 0,
    }


def _split_rows(args: Any, mapping: Mapping) -> list[dict]:
    return rows_from_mapping(Path(args.dataset_root), mapping, require_files=True)


def _authorise_parameters(
    method: str,
    parameters: Mapping,
    args: Any,
    authorization: Mapping | None = None,
) -> dict:
    result = dict(parameters)
    if method == "full_finetune":
        if not bool(getattr(args, "allow_experimental_full_finetune", False)):
            raise PermissionError(
                "3DINO full fine-tuning is project-adapted and disabled by default. "
                "After reviewing upstream licence terms and memory requirements, "
                "pass --allow_experimental_full_finetune explicitly."
            )
        authorization = authorization if isinstance(authorization, Mapping) else {}
        missing_reviews = [
            key
            for key in ("license_approval_claimed", "memory_preflight_passed")
            if authorization.get(key) is not True
        ]
        if missing_reviews:
            raise PermissionError(
                "The full-fine-tuning profile is still marked unauthorized. "
                "Use a reviewed manual profile whose top-level authorization "
                "sets license_approval_claimed=true and "
                "memory_preflight_passed=true; missing approvals: "
                f"{missing_reviews}. The CLI switch alone is not permission."
            )
        # Use the canonical gate understood by training.py and retain the
        # compatibility alias accepted by model.py.
        result["allow_experimental_full_finetune"] = True
        result["allow_full_finetune"] = True
    return result


def _is_completed(
    output_dir: Path,
    parameters: Mapping | None = None,
    model_identity: Mapping | None = None,
) -> bool:
    marker = output_dir / "training_state.json"
    if not marker.is_file():
        return False
    state = read_json(marker)
    if not bool(state.get("completed")):
        return False
    required = (output_dir / "best_model.pt", output_dir / "metrics_summary.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Completed training state has missing artifacts in {output_dir}: {missing}"
        )
    if parameters is not None:
        expected_context = parameters.get("_execution_context")
        recorded_parameters = state.get("parameters")
        recorded_context = (
            recorded_parameters.get("_execution_context")
            if isinstance(recorded_parameters, Mapping)
            else None
        )
        if expected_context is not None and recorded_context != expected_context:
            raise RuntimeError(
                f"Completed training state has incompatible execution provenance: {marker}"
            )
        if not isinstance(recorded_parameters, Mapping) or dict(
            recorded_parameters
        ) != dict(parameters):
            raise RuntimeError(
                f"Completed training state has different effective parameters: {marker}"
            )
        if state.get("seed") is not None and int(state["seed"]) != int(
            parameters.get("seed", state["seed"])
        ):
            raise RuntimeError(f"Completed training state has a different seed: {marker}")
        expected_parameters_fingerprint = canonical_sha256(dict(parameters))
        recorded_parameters_fingerprint = state.get("parameters_fingerprint")
        if (
            recorded_parameters_fingerprint is not None
            and recorded_parameters_fingerprint != expected_parameters_fingerprint
        ):
            raise RuntimeError(
                f"Completed training state has a different parameter fingerprint: {marker}"
            )
    if model_identity is not None:
        source = state.get("source_provenance")
        expected_sha = model_identity.get("pretrained_checkpoint_sha256")
        if not isinstance(source, Mapping) or source.get("checkpoint_sha256") != expected_sha:
            raise RuntimeError(
                f"Completed training state has a different pretrained checkpoint: {marker}"
            )
        if source.get("upstream_revision") != model_identity.get("upstream_revision"):
            raise RuntimeError(
                f"Completed training state has a different upstream revision: {marker}"
            )
    binding = state.get("best_model_binding")
    legacy_sha = state.get("best_model_sha256")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"Completed training state has an invalid classifier binding: {marker}")
        recorded_sha = binding.get("checkpoint_sha256")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
            raise RuntimeError(f"Completed training state has no bound best-model SHA: {marker}")
        actual_sha = sha256_file(output_dir / "best_model.pt")
        if actual_sha != recorded_sha:
            raise RuntimeError(
                f"Completed training best_model.pt changed after CV sealing: {output_dir}"
            )
        if legacy_sha is not None and legacy_sha != recorded_sha:
            raise RuntimeError(f"Completed training state has conflicting best-model SHAs: {marker}")
        recorded_parameters = state.get("parameters")
        if not isinstance(recorded_parameters, Mapping):
            raise RuntimeError(f"Completed training state has no bound parameters: {marker}")
        if binding.get("parameters_fingerprint") != canonical_sha256(
            dict(recorded_parameters)
        ):
            raise RuntimeError(
                f"Completed training classifier parameter binding changed: {marker}"
            )
        runtime_plan = state.get("runtime_plan")
        if not isinstance(runtime_plan, Mapping):
            raise RuntimeError(f"Completed training state has no bound runtime plan: {marker}")
        expected_runtime_fingerprint = _runtime_plan_fingerprint(
            recorded_parameters, runtime_plan
        )
        if binding.get("runtime_plan_fingerprint") != expected_runtime_fingerprint:
            raise RuntimeError(
                f"Completed training classifier runtime-plan binding changed: {marker}"
            )
        if state.get("runtime_plan_fingerprint") != expected_runtime_fingerprint:
            raise RuntimeError(
                f"Completed training state runtime-plan fingerprint changed: {marker}"
            )
        if binding.get("training_rows_fingerprint") != runtime_plan.get(
            "training_rows_fingerprint"
        ):
            raise RuntimeError(
                f"Completed training classifier row binding changed: {marker}"
            )
    elif legacy_sha is not None:
        if sha256_file(output_dir / "best_model.pt") != legacy_sha:
            raise RuntimeError(
                f"Completed training best_model.pt changed after CV sealing: {output_dir}"
            )
    return True


def _normalise_grid_candidate_results(result: Mapping, candidates: Sequence[Mapping]) -> list[dict]:
    """Accept the documented list or mapping form of grid head summaries."""

    raw = result.get("candidates")
    values: list[dict] = []
    if isinstance(raw, Mapping):
        for identity, payload in raw.items():
            item = dict(payload) if isinstance(payload, Mapping) else {"metrics": payload}
            item.setdefault("candidate_id", str(identity))
            values.append(item)
    elif isinstance(raw, list):
        values = [dict(value) for value in raw]
    else:
        raise RuntimeError("Linear-probe grid did not return per-candidate results.")
    expected = {str(value["candidate_id"]) for value in candidates}
    observed = {str(value.get("candidate_id", "")) for value in values}
    if observed != expected:
        raise RuntimeError(
            "Linear-probe grid candidate mismatch: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    return values


def _metric_record(identity: str, fold: int, parameters: Mapping, payload: Mapping) -> dict:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"Candidate {identity} fold {fold} returned no metrics mapping")
    for key in ("f1_macro", "balanced_accuracy", "per_class"):
        if key not in metrics:
            raise RuntimeError(f"Candidate {identity} fold {fold} metrics lack {key}")
    return {
        "candidate_id": identity,
        "fold": int(fold),
        "nominal_lr": parameters.get("nominal_lr"),
        **dict(metrics),
    }


def _rank_fold_records(fold_records: Sequence[Mapping], parameters_by_id: Mapping[str, Mapping]) -> list[dict]:
    ranking = aggregate_fold_records(fold_records)
    for record in ranking:
        identity = str(record["candidate_id"])
        record["parameters"] = dict(parameters_by_id[identity])
        nominal = parameters_by_id[identity].get("nominal_lr")
        if nominal is not None:
            record["nominal_lr"] = float(nominal)
    # aggregate_fold_records ranks before parameters are attached; re-apply the
    # fully specified deterministic tie-break order through the public helper.
    return rank_records(ranking)


_GRID_FOLD_CORE_ARTIFACTS = (
    "best_model.pt",
    "training_state.json",
    "metrics_summary.json",
    "validation_patient_predictions.json",
    "validation_probabilities.csv",
)


def _canonical_grid_source(source: str) -> str:
    return "grid" if str(source) == "search" else str(source)


def _grid_config_arg(args: Any) -> Path | None:
    value = getattr(args, "grid_config", None)
    if value is None:
        value = getattr(args, "search_config", None)
    return Path(value) if value is not None else None


def _max_grid_candidates_arg(args: Any) -> int | None:
    value = getattr(args, "max_grid_candidates", None)
    if value is None:
        value = getattr(args, "max_search_candidates", None)
    return int(value) if value is not None else None


def _grid_folder_name(source: str) -> str:
    source = _canonical_grid_source(source)
    if source == "publication":
        return "Publication_Grid"
    if source == "grid":
        return "Grid"
    raise ValueError(f"No grid folder exists for parameter source {source!r}.")


def _grid_root(args: Any, method: str, source: str) -> Path:
    return (
        method_root(args.results_root, method)
        / _grid_folder_name(source)
        / f"{int(args.percentage)}_percent"
    )


def _search_path(args: Any, method: str, source: str) -> Path:
    source = _canonical_grid_source(source)
    if source == "publication":
        return _profile_path(method, "publication")
    if source != "grid":
        raise ValueError(f"Grid configuration requested for source={source!r}.")
    configured = _grid_config_arg(args)
    if configured is not None:
        return configured
    return _profile_path(method, "grid")


def _grid_provenance_fields(source: str, definition: Mapping) -> dict:
    source = _canonical_grid_source(source)
    if source == "publication":
        return {
            "parameter_source": "publication",
            "search_type": "publication_grid",
            "profile_type": "publication_adapted",
            "publication_search_space_preserved": True,
            "publication_fixed_recipe_preserved": True,
            "publication_exact": False,
        }
    return {
        "parameter_source": "grid",
        "search_type": "grid",
        "profile_type": definition.get("profile_type", "project_defined_grid"),
    }


def _materialize_file(source: Path, destination: Path) -> dict:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise RuntimeError(f"Required grid artifact is missing: {source}")
    source_sha = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_sha:
            raise RuntimeError(
                "Refusing to overwrite a different existing grid/CV artifact: "
                f"{destination}"
            )
        action = "existing"
    else:
        try:
            os.link(source, destination)
            action = "hard_link"
        except OSError:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".materializing.tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                shutil.copy2(source, temporary_path)
                if sha256_file(temporary_path) != source_sha:
                    raise RuntimeError(
                        f"Temporary materialized artifact hash mismatch: {destination}"
                    )
                # A concurrent/resumed publisher may have completed while the
                # temporary copy was being written. Never replace different data.
                if destination.exists():
                    if not destination.is_file() or sha256_file(destination) != source_sha:
                        raise RuntimeError(
                            "Refusing to overwrite a different existing grid/CV artifact: "
                            f"{destination}"
                        )
                    action = "existing"
                else:
                    os.replace(temporary_path, destination)
                    temporary_path = None
                    action = "copied"
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    destination_sha = sha256_file(destination)
    if destination_sha != source_sha:
        raise RuntimeError(f"Materialized artifact hash mismatch: {destination}")
    return {
        "source_sha256": source_sha,
        "destination_sha256": destination_sha,
        "size_bytes": int(destination.stat().st_size),
        "materialization": action,
    }


def _validate_grid_candidate_fold(
    directory: Path,
    *,
    expected_candidate_id: str,
    expected_parameters: Mapping,
) -> tuple[dict, dict]:
    directory = Path(directory)
    missing = [
        name for name in _GRID_FOLD_CORE_ARTIFACTS
        if not (directory / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Grid candidate fold is incomplete in {directory}: {missing}"
        )
    state = read_json(directory / "training_state.json")
    if state.get("completed") is not True or state.get("method") != "linear_probe":
        raise RuntimeError(f"Grid candidate training state is not complete: {directory}")
    if str(state.get("candidate_id")) != str(expected_candidate_id):
        raise RuntimeError(f"Grid candidate identity mismatch in {directory}")
    recorded_parameters = state.get("parameters")
    if not isinstance(recorded_parameters, Mapping) or dict(recorded_parameters) != dict(
        expected_parameters
    ):
        raise RuntimeError(f"Grid candidate parameters mismatch in {directory}")
    expected_fingerprint = canonical_sha256(dict(expected_parameters))
    if state.get("parameters_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Grid candidate parameter fingerprint mismatch in {directory}")
    state_runtime_plan = state.get("runtime_plan")
    if not isinstance(state_runtime_plan, Mapping):
        raise RuntimeError(f"Grid candidate runtime plan is missing in {directory}")
    runtime_fingerprint = _runtime_plan_fingerprint(
        expected_parameters, state_runtime_plan
    )
    if state.get("runtime_plan_fingerprint") != runtime_fingerprint:
        raise RuntimeError(f"Grid candidate runtime-plan fingerprint mismatch in {directory}")
    if (
        state_runtime_plan.get("head_initialization_schema")
        != "threedino-candidate-sha256-seed-v1"
    ):
        raise RuntimeError(
            f"Grid candidate uses an incompatible head initialization schema: {directory}"
        )
    if list(state.get("classes", [])) != list(CLASS_NAMES):
        raise RuntimeError(f"Grid candidate class order mismatch in {directory}")
    metrics_document = read_json(directory / "metrics_summary.json")
    metrics = metrics_document.get("validation_metrics", metrics_document)
    if not isinstance(metrics, Mapping) or any(
        key not in metrics for key in ("f1_macro", "balanced_accuracy", "per_class")
    ):
        raise RuntimeError(f"Grid candidate validation metrics are invalid in {directory}")
    if state.get("validation_metrics") != dict(metrics):
        raise RuntimeError(
            f"Grid candidate state/metrics disagreement in {directory}"
        )

    import torch

    checkpoint = torch.load(
        directory / "best_model.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Grid candidate checkpoint is invalid: {directory}")
    if checkpoint.get("method") != "linear_probe":
        raise RuntimeError(f"Grid candidate checkpoint method mismatch in {directory}")
    if str(checkpoint.get("candidate_id")) != str(expected_candidate_id):
        raise RuntimeError(f"Grid candidate checkpoint identity mismatch in {directory}")
    checkpoint_parameters = checkpoint.get("parameters")
    if not isinstance(checkpoint_parameters, Mapping) or dict(
        checkpoint_parameters
    ) != dict(expected_parameters):
        raise RuntimeError(f"Grid candidate checkpoint parameters mismatch in {directory}")
    if checkpoint.get("parameters_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Grid candidate checkpoint parameter fingerprint mismatch in {directory}"
        )
    checkpoint_runtime_plan = checkpoint.get("runtime_plan")
    if (
        not isinstance(checkpoint_runtime_plan, Mapping)
        or dict(checkpoint_runtime_plan) != dict(state_runtime_plan)
        or checkpoint.get("runtime_plan_fingerprint") != runtime_fingerprint
    ):
        raise RuntimeError(
            f"Grid candidate checkpoint runtime plan mismatch in {directory}"
        )
    if list(checkpoint.get("classes", [])) != list(CLASS_NAMES):
        raise RuntimeError(
            f"Grid candidate checkpoint class order mismatch in {directory}"
        )
    if not isinstance(checkpoint.get("head_state_dict"), Mapping):
        raise RuntimeError(f"Grid candidate checkpoint has no head state in {directory}")
    if checkpoint.get("validation_metrics") != dict(metrics):
        raise RuntimeError(
            f"Grid candidate checkpoint/metrics disagreement in {directory}"
        )
    expected_model_identity = (
        expected_parameters.get("_execution_context", {}).get("model_identity")
        if isinstance(expected_parameters.get("_execution_context"), Mapping)
        else None
    )
    if isinstance(expected_model_identity, Mapping):
        for document, label in ((state, "state"), (checkpoint, "checkpoint")):
            source = document.get("source_provenance")
            if (
                not isinstance(source, Mapping)
                or source.get("upstream_revision")
                != expected_model_identity.get("upstream_revision")
                or source.get("checkpoint_sha256")
                != expected_model_identity.get("pretrained_checkpoint_sha256")
            ):
                raise RuntimeError(
                    f"Grid candidate {label} model identity mismatch in {directory}"
                )
    return state, dict(checkpoint)


def _publish_grid_candidate_fold(
    source_dir: Path,
    destination_dir: Path,
    *,
    expected_candidate_id: str,
    expected_parameters: Mapping,
    shared_training_dir: Path,
) -> dict:
    _validate_grid_candidate_fold(
        source_dir,
        expected_candidate_id=expected_candidate_id,
        expected_parameters=expected_parameters,
    )
    destination_dir = Path(destination_dir)
    artifacts = {
        name: _materialize_file(Path(source_dir) / name, destination_dir / name)
        for name in _GRID_FOLD_CORE_ARTIFACTS
    }
    parameters_document = {
        "schema_version": 1,
        "model_family": "3DINO",
        "candidate_id": str(expected_candidate_id),
        "parameters": dict(expected_parameters),
    }
    reference_document = {
        "schema_version": 1,
        "model_family": "3DINO",
        "candidate_id": str(expected_candidate_id),
        "shared_training_directory": str(Path(shared_training_dir).resolve()),
        "shared_encoder_pass_per_candidate_bank": True,
    }
    for name, document in (
        ("parameters.json", parameters_document),
        ("shared_training_reference.json", reference_document),
    ):
        path = destination_dir / name
        existed = path.is_file()
        if existed and read_json(path) != document:
            raise RuntimeError(f"Existing candidate metadata differs: {path}")
        if not existed:
            save_json(path, document)
        digest = sha256_file(path)
        artifacts[name] = {
            "source_sha256": digest,
            "destination_sha256": digest,
            "size_bytes": int(path.stat().st_size),
            "materialization": "existing" if existed else "generated",
        }
    return {
        "candidate_id": str(expected_candidate_id),
        "fold_directory": str(destination_dir.resolve()),
        "shared_training_directory": str(Path(shared_training_dir).resolve()),
        "artifacts": artifacts,
    }


def _run_linear_probe_search(
    args: Any,
    manifest: Mapping,
    definition: Mapping,
    candidates: list[dict],
    grid_root: Path,
    *,
    source: str,
) -> tuple[dict, list[dict]]:
    if definition.get("rounds"):
        raise ValueError(
            "3DINO linear-probe grids use one complete five-fold candidate loop; "
            "round-based truncation is not supported."
        )
    parameters_by_id = {str(value["candidate_id"]): dict(value) for value in candidates}
    folds = [1, 2, 3, 4, 5]
    budget = int(
        definition.get("fixed_parameters", {}).get("max_iterations", 12500)
    )
    fold_records: list[dict] = []
    published_folds: dict[str, list[dict]] = {
        identity: [] for identity in parameters_by_id
    }
    fixed_parameters = dict(definition.get("fixed_parameters", {}))
    for fold in folds:
        training, validation = cross_validation_split(manifest, fold)
        train_rows = _split_rows(args, training)
        validation_rows = _split_rows(args, validation)
        validate_no_overlap(train_rows, validation_rows)
        shared_output = grid_root / "shared_training" / f"fold_{fold:02d}"
        if bool(getattr(args, "resume", False)) and _is_completed(
            shared_output, fixed_parameters
        ):
            shared_state = read_json(shared_output / "training_state.json")
            best_records = shared_state.get("candidate_best_records")
            if not isinstance(best_records, Mapping):
                raise RuntimeError(
                    f"Completed shared grid fold has no candidate records: {shared_output}"
                )
            recorded_candidates = {
                str(identity): dict(record.get("candidate", {}))
                for identity, record in best_records.items()
                if isinstance(record, Mapping)
            }
            if recorded_candidates != parameters_by_id:
                raise RuntimeError(
                    "Completed shared grid fold belongs to a different candidate set: "
                    f"{shared_output}"
                )
            result = {"candidates": dict(best_records)}
        else:
            result = train_linear_probe_grid(
                train_rows,
                validation_rows,
                fixed_parameters,
                list(parameters_by_id.values()),
                shared_output,
                budget=budget,
                **_common_training_kwargs(args),
            )
        for payload in _normalise_grid_candidate_results(
            result, list(parameters_by_id.values())
        ):
            identity = str(payload["candidate_id"])
            if bool(getattr(args, "resume", False)) and _is_completed(
                shared_output, fixed_parameters
            ):
                candidate_metrics = read_json(
                    shared_output
                    / "candidates"
                    / identity
                    / "metrics_summary.json"
                )
                if payload.get("metrics") != candidate_metrics:
                    raise RuntimeError(
                        "Completed shared grid state disagrees with the sealed "
                        f"candidate metrics for {identity}: {shared_output}"
                    )
            fold_record = _metric_record(
                identity, fold, parameters_by_id[identity], payload
            )
            fold_records.append(fold_record)
            published = _publish_grid_candidate_fold(
                shared_output / "candidates" / identity,
                grid_root / f"candidate_{identity}" / f"fold_{fold:02d}",
                expected_candidate_id=identity,
                expected_parameters=parameters_by_id[identity],
                shared_training_dir=shared_output,
            )
            published["metrics"] = {
                key: value
                for key, value in fold_record.items()
                if key not in {"candidate_id", "fold", "nominal_lr"}
            }
            published["fold"] = fold
            published_folds[identity].append(published)

    ranking = _rank_fold_records(fold_records, parameters_by_id)
    ranking_by_id = {str(record["candidate_id"]): record for record in ranking}
    for identity, parameters in parameters_by_id.items():
        aggregate = dict(ranking_by_id[identity])
        aggregate["grid_selection_score"] = aggregate["selection_score"]
        save_json(
            grid_root / f"candidate_{identity}" / "candidate_summary.json",
            {
                "schema_version": 1,
                "model_family": "3DINO",
                "method": "linear_probe",
                "parameter_source": _canonical_grid_source(source),
                "search_type": _grid_provenance_fields(source, definition)[
                    "search_type"
                ],
                "candidate_id": identity,
                "parameters": parameters,
                "cv_protocol": "single_loop_5_fold",
                "folds": published_folds[identity],
                "aggregate": aggregate,
                "heldout_test_used": False,
            },
        )
    return dict(ranking[0]["parameters"]), [
        {
            "folds": folds,
            "budget": budget,
            "keep": 1,
            "cv_protocol": "single_loop_5_fold",
            "shared_frozen_encoder_passes": True,
            "ranking": ranking,
        }
    ]


def _run_full_finetune_search(
    args: Any, manifest: Mapping, definition: Mapping, candidates: list[dict], grid_root: Path
) -> tuple[dict, list[dict]]:
    candidates = [
        _authorise_parameters(
            "full_finetune", value, args, definition.get("authorization")
        )
        for value in candidates
    ]
    parameters_by_id = {str(value.get("candidate_id") or candidate_id(value)): dict(value) for value in candidates}
    for identity, value in parameters_by_id.items():
        value["candidate_id"] = identity
    active = dict(parameters_by_id)
    rounds = definition.get("rounds") or [
        {
            "folds": [1, 2, 3, 4, 5],
            "budget": int(definition.get("fixed_parameters", {}).get("max_iterations", 0)),
            "keep": 1,
        }
    ]
    round_documents: list[dict] = []
    for round_number, round_definition in enumerate(rounds, start=1):
        folds = [int(value) for value in round_definition.get("folds", range(1, 6))]
        budget = int(round_definition.get("budget", 0)) or None
        fold_records: list[dict] = []
        for identity, parameters in active.items():
            for fold in folds:
                training, validation = cross_validation_split(manifest, fold)
                train_rows = _split_rows(args, training)
                validation_rows = _split_rows(args, validation)
                validate_no_overlap(train_rows, validation_rows)
                output = grid_root / f"round_{round_number:02d}" / identity / f"fold_{fold:02d}"
                result = train_full_finetune(
                    train_rows,
                    validation_rows,
                    parameters,
                    output,
                    budget=budget,
                    **_common_training_kwargs(args),
                )
                fold_records.append(_metric_record(identity, fold, parameters, result))
        ranking = _rank_fold_records(fold_records, parameters_by_id)
        keep = min(int(round_definition.get("keep", 1)), len(ranking))
        active = {str(item["candidate_id"]): parameters_by_id[str(item["candidate_id"])] for item in ranking[:keep]}
        document = {
            "round": round_number,
            "budget": budget,
            "folds": folds,
            "keep": keep,
            "ranking": ranking,
        }
        round_documents.append(document)
        save_json(grid_root / f"round_{round_number:02d}_summary.json", document)
    return dict(round_documents[-1]["ranking"][0]["parameters"]), round_documents


def _write_grid_ranking_csv(path: Path, ranking: Sequence[Mapping]) -> None:
    rows: list[dict] = []
    for record in ranking:
        parameters = record.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        rows.append(
            {
                "rank": int(record["rank"]),
                "candidate_id": str(record["candidate_id"]),
                "grid_selection_score": float(record["selection_score"]),
                "mean_f1_macro": float(record["mean_f1_macro"]),
                "std_f1_macro": float(record["std_f1_macro"]),
                "mean_balanced_accuracy": float(
                    record["mean_balanced_accuracy"]
                ),
                "std_balanced_accuracy": float(record["std_balanced_accuracy"]),
                "minority_perforation_recall": float(
                    record.get("minority_perforation_recall", -1.0)
                ),
                "nominal_lr": parameters.get("nominal_lr"),
                "effective_lr": parameters.get("lr"),
                "n_last_blocks": parameters.get("n_last_blocks"),
                "use_mean_patch_token": parameters.get(
                    "use_mean_patch_token"
                ),
            }
        )
    if not rows:
        raise RuntimeError("Cannot write an empty grid ranking.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grid_sealed_artifact_paths(grid_root: Path) -> list[Path]:
    paths = [
        grid_root / "grid_context.json",
        grid_root / "grid_index.json",
        grid_root / "grid_summary.json",
        grid_root / "grid_summary_by_selection_score.csv",
        grid_root / "best_hyperparameters.json",
        grid_root / "selected_parameters.json",
    ]
    index = read_json(grid_root / "grid_index.json")
    candidates = index.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Grid index has no candidates: {grid_root}")
    if int(index.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError(f"Grid index candidate count mismatch: {grid_root}")
    if index.get("method") == "linear_probe":
        observed: set[str] = set()
        for record in candidates:
            if not isinstance(record, Mapping):
                raise RuntimeError(f"Grid index candidate record is invalid: {grid_root}")
            identity = str(record.get("candidate_id", ""))
            expected_directory = f"candidate_{identity}"
            if (
                not identity
                or identity in observed
                or record.get("directory") != expected_directory
            ):
                raise RuntimeError(f"Grid index candidate identity is invalid: {grid_root}")
            observed.add(identity)
            candidate_dir = grid_root / expected_directory
            paths.append(candidate_dir / "candidate_summary.json")
            for fold in range(1, 6):
                fold_dir = candidate_dir / f"fold_{fold:02d}"
                for name in (
                    *_GRID_FOLD_CORE_ARTIFACTS,
                    "parameters.json",
                    "shared_training_reference.json",
                ):
                    paths.append(fold_dir / name)
        observed_directories = {
            path.name
            for path in grid_root.glob("candidate_*")
            if path.is_dir()
        }
        expected_directories = {f"candidate_{identity}" for identity in observed}
        if observed_directories != expected_directories:
            raise RuntimeError(
                "Grid required artifacts are missing because the candidate "
                "directory inventory differs from grid_index.json: "
                f"{grid_root}"
            )
        shared_training = grid_root / "shared_training"
        if not shared_training.is_dir():
            raise RuntimeError(f"Grid has no shared-training directory: {grid_root}")
        paths.extend(
            path
            for path in sorted(shared_training.rglob("*"))
            if path.is_file() and not path.name.endswith(".tmp")
        )
        allowed_root_entries = {
            "shared_training",
            *(f"candidate_{identity}" for identity in observed),
            "grid_context.json",
            "grid_index.json",
            "grid_summary.json",
            "grid_summary_by_selection_score.csv",
            "best_hyperparameters.json",
            "selected_parameters.json",
            "grid_completion.json",
        }
        unexpected_root_entries = {
            path.name for path in grid_root.iterdir()
        } - allowed_root_entries
        if unexpected_root_entries:
            raise RuntimeError(
                "Grid root contains undeclared artifacts: "
                f"{sorted(unexpected_root_entries)}"
            )
    else:
        # Experimental full fine-tuning can use real successive-halving rounds;
        # bind every produced round artifact instead of inventing candidate views.
        paths.extend(sorted(grid_root.glob("round_*_summary.json")))
        paths.extend(
            path
            for round_dir in sorted(grid_root.glob("round_*"))
            if round_dir.is_dir()
            for path in sorted(round_dir.rglob("*"))
            if path.is_file() and not path.name.endswith(".tmp")
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Grid cannot be sealed because required artifacts are missing: "
            f"{missing[:10]}"
        )
    return paths


def _write_grid_completion(
    grid_root: Path,
    *,
    method: str,
    percentage: int,
    parameter_source: str,
    search_type: str,
    selected_candidate_id: str,
    candidate_count: int,
    context_fingerprint: str,
) -> dict:
    artifact_hashes = {
        path.relative_to(grid_root).as_posix(): sha256_file(path)
        for path in _grid_sealed_artifact_paths(grid_root)
    }
    completion = {
        "schema_version": 1,
        "model_family": "3DINO",
        "method": method,
        "percentage": int(percentage),
        "parameter_source": parameter_source,
        "search_type": search_type,
        "selected_candidate_id": str(selected_candidate_id),
        "candidate_count": int(candidate_count),
        "context_fingerprint": str(context_fingerprint),
        "artifacts": artifact_hashes,
        "complete": True,
        "completed_at_utc": utc_now(),
    }
    completion["seal_fingerprint"] = canonical_sha256(
        {key: value for key, value in completion.items() if key != "completed_at_utc"}
    )
    save_json(grid_root / "grid_completion.json", completion)
    return completion


def _validate_grid_completion(
    grid_root: Path,
    *,
    expected_source: str | None = None,
    expected_candidate_id: str | None = None,
    expected_context_fingerprint: str | None = None,
) -> dict:
    marker_path = Path(grid_root) / "grid_completion.json"
    if not marker_path.is_file():
        raise RuntimeError(
            "Selected grid output is partial because grid_completion.json is "
            f"missing: {grid_root}. Rerun the grid stage."
        )
    completion = read_json(marker_path)
    if completion.get("complete") is not True:
        raise RuntimeError(f"Grid completion marker is not complete: {marker_path}")
    if expected_source is not None and _canonical_grid_source(
        str(completion.get("parameter_source"))
    ) != _canonical_grid_source(expected_source):
        raise RuntimeError(f"Grid completion parameter source mismatch: {marker_path}")
    if expected_candidate_id is not None and str(
        completion.get("selected_candidate_id")
    ) != str(expected_candidate_id):
        raise RuntimeError(f"Grid completion selected candidate mismatch: {marker_path}")
    if (
        expected_context_fingerprint is not None
        and completion.get("context_fingerprint") != expected_context_fingerprint
    ):
        raise RuntimeError(f"Grid completion execution context mismatch: {marker_path}")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError(f"Grid completion has no artifact hash map: {marker_path}")
    expected_paths = _grid_sealed_artifact_paths(Path(grid_root))
    expected_relatives = {
        path.relative_to(grid_root).as_posix() for path in expected_paths
    }
    if set(map(str, artifacts)) != expected_relatives:
        raise RuntimeError(
            f"Grid completion artifact inventory mismatch: {marker_path}"
        )
    for relative, expected_sha in artifacts.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Grid completion has an unsafe artifact path: {relative}")
        path = Path(grid_root) / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Completed grid artifact changed or is missing: {path}")
    expected_seal = canonical_sha256(
        {
            key: value
            for key, value in completion.items()
            if key not in {"completed_at_utc", "seal_fingerprint"}
        }
    )
    if completion.get("seal_fingerprint") != expected_seal:
        raise RuntimeError(f"Grid completion seal changed: {marker_path}")
    return completion


def run_search(
    args: Any,
    method: str,
    manifest: Mapping,
    *,
    source: str,
    execution_context: Mapping | None = None,
) -> tuple[dict, dict]:
    source = _canonical_grid_source(source)
    if source not in {"publication", "grid"}:
        raise ValueError("3DINO grid execution requires publication or grid source.")
    configured_grid = _grid_config_arg(args)
    maximum_candidates = _max_grid_candidates_arg(args)
    if source == "publication":
        if method != "linear_probe":
            raise ValueError(
                "3DINO publication parameters define a linear-probe grid only."
            )
        if configured_grid is not None:
            raise ValueError(
                "The publication grid does not accept a custom grid config; "
                "use --parameter_source grid for a project grid."
            )
        if maximum_candidates is not None:
            raise ValueError(
                "The complete publication grid requires all 52 candidates; "
                "candidate truncation is not permitted."
            )
    path = _search_path(args, method, source)
    definition = load_search_definition(path)
    if execution_context is not None:
        definition = dict(definition)
        definition["fixed_parameters"] = _apply_execution_context(
            definition.get("fixed_parameters", {}), args, execution_context
        )
        # load_search_definition resolves candidates before runtime provenance is
        # known. Rebuild them so IDs and resume fingerprints include it.
        definition.pop("resolved_candidates", None)
    candidates = search_candidates(
        definition, maximum_candidates if source == "grid" else None
    )
    if not candidates:
        raise ValueError("Grid candidate limit produced an empty grid.")
    if source == "publication" and len(candidates) != 52:
        raise RuntimeError(
            "Pinned publication profile did not resolve to the required 52 heads."
        )
    grid_root = _grid_root(args, method, source)
    grid_fields = _grid_provenance_fields(source, definition)
    context_document = {
        "schema_version": 2,
        "model_family": "3DINO",
        "method": method,
        "percentage": int(args.percentage),
        **grid_fields,
        "grid_config_sha256": sha256_file(path),
        "candidate_count": len(candidates),
        "candidate_ids": [str(value["candidate_id"]) for value in candidates],
        "candidate_set_fingerprint": canonical_sha256(candidates),
        "head_initialization_schema": (
            "threedino-candidate-sha256-seed-v1"
            if method == "linear_probe"
            else None
        ),
        "execution_context": dict(execution_context or {}),
    }
    context_document["context_fingerprint"] = canonical_sha256(context_document)
    context_path = grid_root / "grid_context.json"
    grid_had_artifacts = grid_root.is_dir() and any(grid_root.iterdir())
    if context_path.is_file():
        existing_context = read_json(context_path)
        if existing_context.get("context_fingerprint") != context_document[
            "context_fingerprint"
        ]:
            raise RuntimeError(
                "Existing 3DINO grid artifacts belong to a different split, "
                "cache, seed, execution-context schema, checkpoint, or search "
                "profile/candidate set. Use a separate results root or archive "
                "that grid directory."
            )
    elif grid_had_artifacts:
        raise RuntimeError(
            "Partial grid artifacts have no compatible grid_context.json and "
            f"cannot be adopted safely: {grid_root}. Archive/remove that grid "
            "and start the current workflow from a clean grid directory."
        )
    else:
        save_json(context_path, context_document)
    completion_path = grid_root / "grid_completion.json"
    if completion_path.is_file():
        if not bool(getattr(args, "resume", False)):
            raise RuntimeError(
                f"A completed grid already exists at {grid_root}. Enable resume "
                "to reuse it, or archive/remove that grid before starting over."
            )
        return _selected_parameters_from_search(
            args, method, source, execution_context
        )
    if grid_had_artifacts and not bool(getattr(args, "resume", False)):
        raise RuntimeError(
            f"Partial grid artifacts already exist at {grid_root}. Enable resume "
            "to continue them, or archive/remove the grid before starting over."
        )
    if method == "linear_probe":
        selected, phases = _run_linear_probe_search(
            args,
            manifest,
            definition,
            candidates,
            grid_root,
            source=source,
        )
    else:
        selected, phases = _run_full_finetune_search(
            args, manifest, definition, candidates, grid_root
        )
    ranking = list(phases[-1]["ranking"])
    selected_identity = str(ranking[0]["candidate_id"])
    execution_identity_fingerprint = canonical_sha256(
        dict(execution_context or {})
    )
    candidate_index = []
    for value in candidates:
        identity = str(value["candidate_id"])
        record = {
            "candidate_id": identity,
            "parameters": dict(value),
        }
        if method == "linear_probe":
            record["directory"] = f"candidate_{identity}"
        else:
            record["output_pattern"] = f"round_*/{identity}/fold_*"
        candidate_index.append(record)
    cv_protocol = (
        "single_loop_5_fold"
        if method == "linear_probe"
        else "configured_successive_halving"
    )
    grid_index = {
        "schema_version": 1,
        "model_family": "3DINO",
        "method": method,
        "percentage": int(args.percentage),
        **grid_fields,
        "cv_protocol": cv_protocol,
        "candidate_count": len(candidates),
        "shared_frozen_encoder_passes": method == "linear_probe",
        "candidates": candidate_index,
        "heldout_test_used": False,
    }
    save_json(grid_root / "grid_index.json", grid_index)
    summary = {
        "schema_version": 1,
        "model_family": "3DINO",
        "method": method,
        "strategy": definition.get("strategy", "publication_multi_head"),
        **grid_fields,
        "grid_config": str(path.resolve()),
        "grid_config_sha256": sha256_file(path),
        "execution_context": dict(execution_context or {}),
        "execution_context_fingerprint": execution_identity_fingerprint,
        "grid_context_fingerprint": context_document["context_fingerprint"],
        "heldout_test_used": False,
        "authorization": definition.get("authorization"),
        "task_adaptations": definition.get("task_adaptations", {}),
        "phases": phases,
        "cv_protocol": cv_protocol,
        "candidate_count": len(candidates),
        "ranking": ranking,
        "selected_candidate_id": selected_identity,
        "selected_parameters": selected,
        "completed_at_utc": utc_now(),
    }
    summary_path = grid_root / "grid_summary.json"
    save_json(summary_path, summary)
    _write_grid_ranking_csv(
        grid_root / "grid_summary_by_selection_score.csv", ranking
    )
    save_json(
        grid_root / "best_hyperparameters.json",
        {
            "schema_version": 1,
            "model_family": "3DINO",
            "method": method,
            **grid_fields,
            "selection_rule": (
                "0.7*(mean_f1_macro-std_f1_macro)+"
                "0.3*(mean_balanced_accuracy-std_balanced_accuracy)"
            ),
            "best_candidate": {
                "candidate_id": selected_identity,
                "parameters": selected,
                "aggregate": ranking[0],
            },
            "heldout_test_used": False,
        },
    )
    selection_path = grid_root / "selected_parameters.json"
    save_json(
        selection_path,
        {
            "schema_version": 1,
            "model_family": "3DINO",
            "method": method,
            **grid_fields,
            "parameters": selected,
            "candidate_id": selected_identity,
            "grid_config_sha256": sha256_file(path),
            "execution_context": dict(execution_context or {}),
            "execution_context_fingerprint": execution_identity_fingerprint,
            "grid_context_fingerprint": context_document["context_fingerprint"],
            "authorization": definition.get("authorization"),
            "task_adaptations": definition.get("task_adaptations", {}),
            "grid_summary": str(summary_path.resolve()),
            "heldout_test_used": False,
        },
    )
    completion = _write_grid_completion(
        grid_root,
        method=method,
        percentage=int(args.percentage),
        parameter_source=source,
        search_type=str(grid_fields["search_type"]),
        selected_candidate_id=selected_identity,
        candidate_count=len(candidates),
        context_fingerprint=context_document["context_fingerprint"],
    )
    provenance = {
        **grid_fields,
        "profile_path": str(path.resolve()),
        "profile_sha256": sha256_file(path),
        "candidate_id": selected_identity,
        "selection_file": str(selection_path.resolve()),
        "selection_file_sha256": sha256_file(selection_path),
        "grid_root": str(grid_root.resolve()),
        "grid_completion": str((grid_root / "grid_completion.json").resolve()),
        "grid_completion_seal_fingerprint": completion["seal_fingerprint"],
        "execution_context": dict(execution_context or {}),
        "execution_context_fingerprint": execution_identity_fingerprint,
        "grid_context_fingerprint": context_document["context_fingerprint"],
        "authorization": definition.get("authorization"),
        "task_adaptations": definition.get("task_adaptations", {}),
    }
    return selected, provenance


def _selected_parameters_from_search(
    args: Any,
    method: str,
    source: str,
    execution_context: Mapping | None = None,
) -> tuple[dict, dict]:
    source = _canonical_grid_source(source)
    grid_root = _grid_root(args, method, source)
    path = grid_root / "selected_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Grid selection not found: {path}. Run --stage grid first."
        )
    document = read_json(path)
    recorded_source = _canonical_grid_source(str(document.get("parameter_source")))
    if recorded_source != source:
        raise RuntimeError(
            f"Grid selection source mismatch in {path}: requested {source}, found {recorded_source}."
        )
    expected_config = _search_path(args, method, source)
    expected_sha = sha256_file(expected_config)
    recorded_sha = document.get(
        "grid_config_sha256", document.get("search_config_sha256")
    )
    if recorded_sha != expected_sha:
        raise RuntimeError(
            "Grid selection was produced by a different configuration; rerun the grid stage."
        )
    if execution_context is not None and document.get("execution_context") != dict(
        execution_context
    ):
        raise RuntimeError(
            "Search selection was produced with a different split, cache plan, "
            "seed, execution-context schema, or pretrained checkpoint; rerun grid."
        )
    completion = _validate_grid_completion(
        grid_root,
        expected_source=source,
        expected_candidate_id=str(document["candidate_id"]),
        expected_context_fingerprint=document.get(
            "grid_context_fingerprint",
            document.get("execution_context_fingerprint"),
        ),
    )
    grid_fields = {
        key: document[key]
        for key in (
            "parameter_source",
            "search_type",
            "profile_type",
            "publication_search_space_preserved",
            "publication_fixed_recipe_preserved",
            "publication_exact",
        )
        if key in document
    }
    return dict(document["parameters"]), {
        **grid_fields,
        "parameter_source": source,
        "search_type": document.get(
            "search_type", "publication_grid" if source == "publication" else "grid"
        ),
        "profile_type": document.get("profile_type", source),
        "selection_file": str(path.resolve()),
        "selection_file_sha256": sha256_file(path),
        "candidate_id": document["candidate_id"],
        "grid_root": str(grid_root.resolve()),
        "grid_completion": str((grid_root / "grid_completion.json").resolve()),
        "grid_completion_seal_fingerprint": completion["seal_fingerprint"],
        "profile_path": str(expected_config.resolve()),
        "profile_sha256": expected_sha,
        "execution_context": document.get("execution_context"),
        "execution_context_fingerprint": document.get(
            "execution_context_fingerprint"
        ),
        "grid_context_fingerprint": document.get(
            "grid_context_fingerprint",
            document.get("execution_context_fingerprint"),
        ),
        "authorization": document.get("authorization"),
        "task_adaptations": document.get("task_adaptations", {}),
    }


def _train_single(
    method: str,
    train_rows: list[dict],
    validation_rows: list[dict],
    parameters: Mapping,
    output_dir: Path,
    args: Any,
    *,
    budget: int | None = None,
) -> dict:
    function = {"linear_probe": train_linear_probe, "full_finetune": train_full_finetune, "lora": train_lora}[method]
    return function(
        train_rows,
        validation_rows,
        dict(parameters),
        output_dir,
        budget=budget,
        **_common_training_kwargs(args),
    )


def _publish_selected_grid_fold(
    source_dir: Path,
    destination_dir: Path,
    *,
    expected_candidate_id: str,
    expected_parameters: Mapping,
    model_identity: Mapping,
    result_origin: str,
) -> dict:
    """Reuse one verified grid head as a CV fold without retraining it."""

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    parameters_path = source_dir / "parameters.json"
    reference_path = source_dir / "shared_training_reference.json"
    if not parameters_path.is_file() or not reference_path.is_file():
        raise RuntimeError(
            f"Selected grid candidate metadata is incomplete: {source_dir}"
        )
    parameters_document = read_json(parameters_path)
    if (
        str(parameters_document.get("candidate_id"))
        != str(expected_candidate_id)
        or dict(parameters_document.get("parameters", {}))
        != dict(expected_parameters)
    ):
        raise RuntimeError(
            f"Selected grid candidate identity/parameters mismatch: {source_dir}"
        )
    reference = read_json(reference_path)
    if (
        str(reference.get("candidate_id")) != str(expected_candidate_id)
        or reference.get("shared_encoder_pass_per_candidate_bank") is not True
    ):
        raise RuntimeError(
            f"Selected grid candidate has an invalid shared-training reference: {source_dir}"
        )
    _validate_grid_candidate_fold(
        source_dir,
        expected_candidate_id=expected_candidate_id,
        expected_parameters=expected_parameters,
    )
    if not _is_completed(source_dir, expected_parameters, model_identity):
        raise RuntimeError(f"Selected grid candidate fold is incomplete: {source_dir}")
    _validate_internal_classifier_checkpoint(
        source_dir / "best_model.pt",
        method="linear_probe",
        expected_parameters=expected_parameters,
        model_identity=model_identity,
    )

    artifact_names = (
        *_GRID_FOLD_CORE_ARTIFACTS,
        "parameters.json",
        "shared_training_reference.json",
    )
    if destination_dir.is_dir():
        unexpected = sorted(
            path.name
            for path in destination_dir.iterdir()
            if path.name not in set(artifact_names) | {"grid_reuse.json"}
        )
        if unexpected:
            raise RuntimeError(
                "Selected-grid CV destination contains unrelated artifacts and "
                f"cannot be repaired safely: {destination_dir}: {unexpected}"
            )
    artifacts = {
        name: _materialize_file(source_dir / name, destination_dir / name)
        for name in artifact_names
    }
    metadata = {
        "schema_version": 1,
        "model_family": "3DINO",
        "method": "linear_probe",
        "result_origin": str(result_origin),
        "source_directory": str(source_dir.resolve()),
        "destination_directory": str(destination_dir.resolve()),
        "candidate_id": str(expected_candidate_id),
        "parameters_fingerprint": canonical_sha256(dict(expected_parameters)),
        "artifacts": artifacts,
        "reused_without_training": True,
    }
    save_json(destination_dir / "grid_reuse.json", metadata)
    return metadata


def _validate_or_refresh_grid_reuse(
    marker_path: Path,
    *,
    expected_candidate_id: str,
    expected_result_origin: str | None = None,
    expected_source_grid_root: Path | None = None,
    refresh_after_cv_binding: bool = False,
) -> dict:
    """Verify a reuse marker and optionally refresh post-seal destination hashes."""

    marker_path = Path(marker_path)
    document = read_json(marker_path)
    if (
        document.get("reused_without_training") is not True
        or str(document.get("candidate_id")) != str(expected_candidate_id)
    ):
        raise RuntimeError(f"Invalid selected-grid reuse marker: {marker_path}")
    source_dir = Path(str(document.get("source_directory", ""))).resolve()
    destination_dir = Path(str(document.get("destination_directory", ""))).resolve()
    if destination_dir != marker_path.parent.resolve():
        raise RuntimeError(f"Grid-reuse destination mismatch: {marker_path}")
    result_origin = str(document.get("result_origin", ""))
    if expected_result_origin is not None and result_origin != str(
        expected_result_origin
    ):
        raise RuntimeError(f"Grid-reuse result origin mismatch: {marker_path}")
    source_grid_root = source_dir.parent.parent.resolve()
    if (
        source_dir.name != destination_dir.name
        or source_dir.parent.name != f"candidate_{expected_candidate_id}"
    ):
        raise RuntimeError(f"Grid-reuse source fold identity mismatch: {marker_path}")
    if (
        expected_source_grid_root is not None
        and source_grid_root != Path(expected_source_grid_root).resolve()
    ):
        raise RuntimeError(f"Grid-reuse source grid root mismatch: {marker_path}")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"Grid-reuse marker has no artifact map: {marker_path}")
    refreshed: dict[str, dict] = {}
    for name in (
        *_GRID_FOLD_CORE_ARTIFACTS,
        "parameters.json",
        "shared_training_reference.json",
    ):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Grid-reuse marker does not bind {name}: {marker_path}")
        source = source_dir / name
        destination = destination_dir / name
        if (
            not source.is_file()
            or sha256_file(source) != record.get("source_sha256")
            or not destination.is_file()
        ):
            raise RuntimeError(f"Grid-reuse artifact changed or is missing: {name}")
        destination_sha = sha256_file(destination)
        recorded_destination_sha = record.get("destination_sha256")
        if destination_sha != recorded_destination_sha:
            if not refresh_after_cv_binding or name != "training_state.json":
                raise RuntimeError(
                    f"Grid-reuse destination artifact changed: {destination}"
                )
            state = read_json(destination)
            binding = state.get("best_model_binding")
            if (
                not isinstance(binding, Mapping)
                or binding.get("checkpoint_sha256")
                != sha256_file(destination_dir / "best_model.pt")
                or state.get("best_model_sha256")
                != binding.get("checkpoint_sha256")
            ):
                raise RuntimeError(
                    f"Grid-reuse training-state change is not a valid CV binding: {destination}"
                )
        refreshed[name] = {
            **dict(record),
            "destination_sha256": destination_sha,
            "size_bytes": int(destination.stat().st_size),
        }
        if destination_sha != record.get("source_sha256"):
            refreshed[name]["post_materialization_update"] = "cv_seal_binding"
    if refresh_after_cv_binding:
        document["artifacts"] = refreshed
        document["cv_sealed"] = True
        save_json(marker_path, document)
    return document


def _best_budget(state: Mapping, parameters: Mapping) -> int:
    for key in ("best_optimizer_step", "best_iteration", "best_step", "best_epoch"):
        value = state.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    selected_id = state.get("selected_candidate_id")
    candidate_records = state.get("candidate_best_records", {})
    if selected_id is not None and isinstance(candidate_records, Mapping):
        selected = candidate_records.get(str(selected_id))
        if isinstance(selected, Mapping):
            for key in ("optimizer_step", "best_optimizer_step", "epoch"):
                value = selected.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
    for key in ("max_iterations", "epochs"):
        value = parameters.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    raise RuntimeError("Training state contains no positive refit budget.")


def _fold_summary(percentage_dir: Path, parameters: Mapping) -> list[dict]:
    rows: list[dict] = []
    for fold in range(1, 6):
        directory = percentage_dir / f"fold_{fold:02d}"
        state = read_json(directory / "training_state.json")
        metric_document = read_json(directory / "metrics_summary.json")
        metrics = metric_document.get("validation_metrics", metric_document)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Invalid metrics summary in {directory}")
        rows.append(
            {
                "fold": fold,
                "best_epoch": state.get("best_epoch"),
                "best_optimizer_step": state.get("best_optimizer_step", state.get("best_iteration", state.get("best_step"))),
                "refit_budget": _best_budget(state, parameters),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "average_precision_macro": metrics.get("average_precision_macro"),
                "selection_score": fold_selection_score(metrics),
                "checkpoint": str((directory / "best_model.pt").resolve()),
            }
        )
    return rows


def run_cv(
    args: Any,
    method: str,
    manifest: Mapping,
    parameters: Mapping,
    provenance: Mapping,
    model_identity: Mapping,
) -> dict:
    search_type = str(provenance.get("search_type", ""))
    selected_grid_reuse = method == "linear_probe" and search_type in {
        "publication_grid",
        "grid",
    }
    selected_candidate_id = (
        str(provenance.get("candidate_id")) if selected_grid_reuse else None
    )
    selected_grid_root: Path | None = None
    result_origin: str | None = None
    if selected_grid_reuse:
        if not selected_candidate_id or selected_candidate_id == "None":
            raise RuntimeError("Selected grid provenance has no candidate identity.")
        source = _canonical_grid_source(
            str(provenance.get("parameter_source", ""))
        )
        expected_grid_root = _grid_root(args, method, source).resolve()
        recorded_grid_root = Path(
            provenance.get("grid_root", expected_grid_root)
        ).resolve()
        if recorded_grid_root != expected_grid_root:
            raise RuntimeError(
                "Selected grid provenance points outside the requested results tree."
            )
        selected_grid_root = expected_grid_root
        _validate_grid_completion(
            selected_grid_root,
            expected_source=source,
            expected_candidate_id=selected_candidate_id,
            expected_context_fingerprint=provenance.get(
                "grid_context_fingerprint"
            ),
        )
        result_origin = (
            "selected_publication_grid_run"
            if source == "publication"
            else "selected_grid_run"
        )
    # Validate selected-grid provenance before allocating a CV registry entry.
    cv_id, cv_dir = register_cv_run(
        args.results_root, method, parameters, provenance, model_identity
    )
    percentage_dir = cv_dir / f"{args.percentage}_percent"
    completion_path = percentage_dir / _CV_COMPLETION_FILENAME
    if (
        not args.resume
        and not completion_path.is_file()
        and percentage_dir.is_dir()
        and any(percentage_dir.iterdir())
    ):
        raise RuntimeError(
            f"Partial CV artifacts already exist at {percentage_dir}. Enable "
            "resume to continue them, or archive/remove that percentage result "
            "before starting over."
        )
    if completion_path.is_file():
        if not args.resume:
            raise RuntimeError(
                f"A completed CV result already exists at {percentage_dir}. "
                "Enable resume to reuse it or archive/remove it before starting over."
            )
        completed = _validate_cv_completion_seal(
            percentage_dir,
            cv_id=cv_id,
            method=method,
            percentage=int(args.percentage),
            parameters=parameters,
            model_identity=model_identity,
        )
        completed_reuse_records: list[dict] = []
        completed_origin = "dedicated_cv_training"
        completed_candidate: str | None = None
        if completed.get("selected_grid_folds_reused_without_training") is True:
            for fold in range(1, 6):
                marker = read_json(
                    percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json"
                )
                completed_reuse_records.append(marker)
            completed_origin = str(completed_reuse_records[0]["result_origin"])
            completed_candidate = str(completed_reuse_records[0]["candidate_id"])
        return {
            "cv_id": cv_id,
            "cv_directory": str(cv_dir),
            "percentage_directory": str(percentage_dir),
            "best_fold": int(completed["best_fold"]),
            "refit_budget": int(completed["refit_budget"]),
            "cv_completion": str(completion_path),
            "cv_completion_seal_fingerprint": completed["seal_fingerprint"],
            "result_origin": completed_origin,
            "selected_grid_candidate_id": completed_candidate,
            "selected_grid_folds_reused_without_training": bool(
                completed_reuse_records
            ),
            "grid_reuse_records": completed_reuse_records,
            "resumed_completed_result_without_training": True,
        }
    fold_training_rows_fingerprints: dict[int, str] = {}
    grid_reuse_records: list[dict] = []
    for fold in range(1, 6):
        training, validation = cross_validation_split(manifest, fold)
        train_rows = _split_rows(args, training)
        validation_rows = _split_rows(args, validation)
        validate_no_overlap(train_rows, validation_rows)
        fold_training_rows_fingerprints[fold] = _training_rows_fingerprint(
            train_rows
        )
        output = percentage_dir / f"fold_{fold:02d}"
        if args.resume:
            if selected_grid_reuse:
                marker = output / "grid_reuse.json"
                if marker.is_file():
                    if not _is_completed(output, parameters, model_identity):
                        raise RuntimeError(
                            f"Grid-reuse marker exists for an incomplete CV fold: {marker}"
                        )
                    record = _validate_or_refresh_grid_reuse(
                        marker,
                        expected_candidate_id=selected_candidate_id,
                        expected_result_origin=result_origin,
                        expected_source_grid_root=selected_grid_root,
                        refresh_after_cv_binding=True,
                    )
                    grid_reuse_records.append(record)
                    continue
                # A crash may leave any prefix of the verified artifact set
                # before the atomic marker write. Do not call _is_completed:
                # idempotent, hash-checked materialization repairs that prefix.
            elif _is_completed(output, parameters, model_identity):
                continue
        if selected_grid_reuse:
            assert selected_grid_root is not None
            source_fold = (
                selected_grid_root
                / f"candidate_{selected_candidate_id}"
                / f"fold_{fold:02d}"
            )
            source_state = read_json(source_fold / "training_state.json")
            runtime_plan = source_state.get("runtime_plan")
            if (
                not isinstance(runtime_plan, Mapping)
                or runtime_plan.get("training_rows_fingerprint")
                != fold_training_rows_fingerprints[fold]
            ):
                raise RuntimeError(
                    "Selected grid fold was trained on different development rows: "
                    f"{source_fold}"
                )
            grid_reuse_records.append(
                _publish_selected_grid_fold(
                    source_fold,
                    output,
                    expected_candidate_id=selected_candidate_id,
                    expected_parameters=parameters,
                    model_identity=model_identity,
                    result_origin=str(result_origin),
                )
            )
        else:
            _train_single(
                method, train_rows, validation_rows, parameters, output, args
            )

    summary_rows = _fold_summary(percentage_dir, parameters)
    percentage_dir.mkdir(parents=True, exist_ok=True)
    with (percentage_dir / "cv_fold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    best = max(summary_rows, key=lambda value: (float(value["selection_score"]), -int(value["fold"])))
    save_json(
        percentage_dir / "best_fold_model.json",
        {
            "schema_version": 1,
            "model_family": "3DINO",
            "cv_id": cv_id,
            "method": method,
            "percentage": args.percentage,
            "best_fold": best["fold"],
            "selection_rule": "0.7*f1_macro+0.3*balanced_accuracy",
            "checkpoint": best["checkpoint"],
            "parameters": dict(parameters),
            "parameter_provenance": dict(provenance),
            "result_origin": result_origin or "dedicated_cv_training",
            "selected_grid_candidate_id": selected_candidate_id,
            "selected_grid_folds_reused_without_training": bool(
                selected_grid_reuse
            ),
            "heldout_test_used_during_selection": False,
        },
    )
    refit_budget = max(1, int(round(statistics.median(int(row["refit_budget"]) for row in summary_rows))))
    refit_dir = percentage_dir / "refit_full_development"
    refit_effective_parameters = dict(parameters)
    refit_effective_parameters["evaluation_interval_iterations"] = refit_budget
    refit_effective_parameters["early_stopping_patience"] = 0
    development_rows = _split_rows(args, development_mapping(manifest))
    refit_training_rows_fingerprint = _training_rows_fingerprint(
        development_rows
    )
    if not (
        args.resume
        and _is_completed(
            refit_dir, refit_effective_parameters, model_identity
        )
    ):
        refit_model(
            development_rows,
            dict(parameters),
            refit_dir,
            method=method,
            refit_budget=refit_budget,
            **_common_training_kwargs(args),
        )
    parameter_doc = read_json(cv_dir / "cv_parameters.json")
    parameter_doc.setdefault("percentages", {})[str(args.percentage)] = {
        "completed": True,
        "best_fold": best["fold"],
        "refit_budget": refit_budget,
        "execution_context": parameters.get("_execution_context"),
        "cv_fold_summary_sha256": sha256_file(
            percentage_dir / "cv_fold_summary.csv"
        ),
        "best_fold_model_sha256": sha256_file(
            percentage_dir / "best_fold_model.json"
        ),
        "completed_at_utc": utc_now(),
    }
    save_json(cv_dir / "cv_parameters.json", parameter_doc)
    completion = _write_cv_completion_seal(
        percentage_dir,
        cv_id=cv_id,
        method=method,
        percentage=int(args.percentage),
        parameters=parameters,
        model_identity=model_identity,
        best_fold=int(best["fold"]),
        refit_budget=refit_budget,
        fold_training_rows_fingerprints=fold_training_rows_fingerprints,
        refit_training_rows_fingerprint=refit_training_rows_fingerprint,
        selected_grid_reuse=selected_grid_reuse,
        selected_grid_candidate_id=selected_candidate_id,
        selected_grid_result_origin=result_origin,
        selected_grid_root=selected_grid_root,
    )
    if selected_grid_reuse:
        grid_reuse_records = [
            _validate_or_refresh_grid_reuse(
                percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json",
                expected_candidate_id=str(selected_candidate_id),
                expected_result_origin=result_origin,
                expected_source_grid_root=selected_grid_root,
            )
            for fold in range(1, 6)
        ]
    return {
        "cv_id": cv_id,
        "cv_directory": str(cv_dir),
        "percentage_directory": str(percentage_dir),
        "best_fold": best["fold"],
        "refit_budget": refit_budget,
        "cv_completion": str(completion_path),
        "cv_completion_seal_fingerprint": completion["seal_fingerprint"],
        "result_origin": result_origin or "dedicated_cv_training",
        "selected_grid_candidate_id": selected_candidate_id,
        "selected_grid_folds_reused_without_training": bool(selected_grid_reuse),
        "grid_reuse_records": grid_reuse_records,
    }


def _resolve_cv(
    args: Any, method: str, parameters: Mapping, model_identity: Mapping
) -> tuple[str, Path]:
    registry_path = method_root(args.results_root, method) / "CV" / "cv_runs.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"CV registry not found: {registry_path}")
    fingerprint = parameter_fingerprint(method, parameters, model_identity)
    record = next(
        (
            item
            for item in read_json(registry_path).get("runs", [])
            if item.get("parameter_fingerprint") == fingerprint
        ),
        None,
    )
    if record is None:
        raise RuntimeError("No CV_x matches the requested parameters and pretrained checkpoint identity.")
    cv_id = str(record["cv_id"])
    return cv_id, method_root(args.results_root, method) / "CV" / cv_id


def _prediction_kwargs(args: Any) -> dict:
    common = _common_training_kwargs(args)
    common.pop("resume", None)
    common.pop("seed", None)
    common.pop("asset_provisioning", None)
    # Validation inside training and standalone prediction/test use fixed
    # single-process loading. The CLI worker count is a training-throughput
    # setting only and must not alter evaluation behavior or identity.
    common["num_workers"] = 0
    return common


_TEST_ARTIFACT_NAMES = (
    "metrics_summary.json",
    "test_patient_predictions.json",
    "test_probabilities.csv",
    "model_selection.json",
)


def _validate_test_completion(
    output_dir: Path,
    expected_fingerprint: str,
) -> dict:
    marker_path = output_dir / "test_completion.json"
    if not marker_path.is_file():
        raise RuntimeError(
            f"Held-out directory is partial because its final completion marker is missing: {output_dir}"
        )
    marker = read_json(marker_path)
    if marker.get("complete") is not True:
        raise RuntimeError(f"Held-out completion marker is not complete: {marker_path}")
    if marker.get("test_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Existing held-out result has different parameters, data/cache "
            "provenance, model source, or classifier checkpoint hashes."
        )
    recorded = marker.get("artifacts")
    if not isinstance(recorded, Mapping):
        raise RuntimeError(f"Held-out completion marker has no artifact hash map: {marker_path}")
    for name in _TEST_ARTIFACT_NAMES:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"Completed held-out result is missing {path}")
        if recorded.get(name) != sha256_file(path):
            raise RuntimeError(f"Completed held-out artifact changed after sealing: {path}")
    return marker


def _validate_internal_classifier_checkpoint(
    checkpoint: Path,
    *,
    method: str,
    expected_parameters: Mapping,
    model_identity: Mapping,
    expected_runtime_budget: int | None = None,
    expected_training_rows_fingerprint: str | None = None,
    expected_runtime_plan_fingerprint: str | None = None,
) -> dict:
    """Bind an internally selected classifier to its registered CV contract."""

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Classifier checkpoint root is not a mapping: {checkpoint}")
    if payload.get("method") != method:
        raise RuntimeError(
            f"Classifier checkpoint method mismatch in {checkpoint}: "
            f"expected {method}, found {payload.get('method')}"
        )
    if list(payload.get("classes", [])) != list(CLASS_NAMES):
        raise RuntimeError(f"Classifier checkpoint class order mismatch: {checkpoint}")
    recorded_parameters = payload.get("parameters", payload.get("base_parameters"))
    if not isinstance(recorded_parameters, Mapping) or dict(
        recorded_parameters
    ) != dict(expected_parameters):
        raise RuntimeError(
            "Internal classifier checkpoint parameters do not match the "
            f"registered CV/refit parameters: {checkpoint}"
        )
    parameters_fingerprint = canonical_sha256(dict(recorded_parameters))
    if payload.get("parameters_fingerprint") != parameters_fingerprint:
        raise RuntimeError(
            f"Classifier checkpoint parameter fingerprint mismatch: {checkpoint}"
        )
    source = payload.get("source_provenance")
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Classifier checkpoint lacks source provenance: {checkpoint}")
    if source.get("upstream_revision") != model_identity.get("upstream_revision"):
        raise RuntimeError(f"Classifier checkpoint upstream revision mismatch: {checkpoint}")
    if source.get("checkpoint_sha256") != model_identity.get(
        "pretrained_checkpoint_sha256"
    ):
        raise RuntimeError(f"Classifier checkpoint base-weight SHA mismatch: {checkpoint}")
    runtime_plan = payload.get("runtime_plan")
    if not isinstance(runtime_plan, Mapping):
        raise RuntimeError(f"Classifier checkpoint lacks a runtime plan: {checkpoint}")
    runtime_plan_fingerprint = _runtime_plan_fingerprint(
        recorded_parameters, runtime_plan
    )
    if payload.get("runtime_plan_fingerprint") != runtime_plan_fingerprint:
        raise RuntimeError(
            f"Classifier checkpoint runtime-plan fingerprint mismatch: {checkpoint}"
        )
    training_rows_fingerprint = runtime_plan.get("training_rows_fingerprint")
    if not isinstance(training_rows_fingerprint, str) or len(
        training_rows_fingerprint
    ) != 64:
        raise RuntimeError(
            f"Classifier checkpoint lacks a valid training-row fingerprint: {checkpoint}"
        )
    if (
        expected_training_rows_fingerprint is not None
        and training_rows_fingerprint != expected_training_rows_fingerprint
    ):
        raise RuntimeError(
            f"Classifier checkpoint training-row fingerprint mismatch: {checkpoint}"
        )
    if (
        expected_runtime_plan_fingerprint is not None
        and runtime_plan_fingerprint != expected_runtime_plan_fingerprint
    ):
        raise RuntimeError(
            f"Classifier checkpoint runtime-plan identity differs from the CV seal: {checkpoint}"
        )
    if expected_runtime_budget is not None:
        if int(runtime_plan.get("max_iterations", -1)) != int(
            expected_runtime_budget
        ):
            raise RuntimeError(
                f"Refit checkpoint runtime budget mismatch: {checkpoint}"
            )
    return {
        "parameters_fingerprint": parameters_fingerprint,
        "runtime_plan_fingerprint": runtime_plan_fingerprint,
        "training_rows_fingerprint": training_rows_fingerprint,
        "effective_checkpoint_parameters": dict(recorded_parameters),
        "runtime_plan": dict(runtime_plan),
        "method": payload.get("method"),
        "classes": list(payload.get("classes", [])),
        "binding_verified": True,
    }


_CV_COMPLETION_FILENAME = "cv_completion.json"


def _classifier_binding_from_output(
    percentage_dir: Path,
    output_dir: Path,
    *,
    method: str,
    classifier_identity: Mapping,
    expected_parameters: Mapping,
    expected_training_rows_fingerprint: str,
    model_identity: Mapping,
    expected_runtime_budget: int | None = None,
) -> dict:
    """Validate one completed training output and bind its best checkpoint."""

    if not _is_completed(output_dir, expected_parameters, model_identity):
        raise RuntimeError(f"Cannot seal incomplete classifier output: {output_dir}")
    checkpoint = output_dir / "best_model.pt"
    state_path = output_dir / "training_state.json"
    state = read_json(state_path)
    if state.get("method") != method:
        raise RuntimeError(f"Training-state method mismatch while sealing: {state_path}")
    if list(state.get("classes", [])) != list(CLASS_NAMES):
        raise RuntimeError(f"Training-state class order mismatch while sealing: {state_path}")
    if dict(state.get("parameters", {})) != dict(expected_parameters):
        raise RuntimeError(f"Training-state parameters changed before sealing: {state_path}")
    runtime_plan = state.get("runtime_plan")
    if not isinstance(runtime_plan, Mapping):
        raise RuntimeError(f"Training state lacks a runtime plan: {state_path}")
    runtime_plan_fingerprint = _runtime_plan_fingerprint(
        expected_parameters, runtime_plan
    )
    if state.get("runtime_plan_fingerprint") != runtime_plan_fingerprint:
        raise RuntimeError(f"Training-state runtime-plan fingerprint mismatch: {state_path}")
    if runtime_plan.get("training_rows_fingerprint") != expected_training_rows_fingerprint:
        raise RuntimeError(f"Training-state row fingerprint mismatch: {state_path}")

    checkpoint_metadata = _validate_internal_classifier_checkpoint(
        checkpoint,
        method=method,
        expected_parameters=expected_parameters,
        model_identity=model_identity,
        expected_runtime_budget=expected_runtime_budget,
        expected_training_rows_fingerprint=expected_training_rows_fingerprint,
        expected_runtime_plan_fingerprint=runtime_plan_fingerprint,
    )
    if checkpoint_metadata["runtime_plan"] != dict(runtime_plan):
        raise RuntimeError(
            f"Checkpoint and training state have different runtime plans: {checkpoint}"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    identity = dict(classifier_identity)
    binding = {
        "schema_version": 1,
        "classifier_identity": identity,
        "method": method,
        "fold": identity.get("fold"),
        "role": identity.get("role"),
        "relative_path": checkpoint.relative_to(percentage_dir).as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "effective_checkpoint_parameters": dict(expected_parameters),
        "parameters_fingerprint": checkpoint_metadata["parameters_fingerprint"],
        "training_rows_fingerprint": expected_training_rows_fingerprint,
        "runtime_plan": dict(runtime_plan),
        "runtime_plan_fingerprint": runtime_plan_fingerprint,
        "pretrained_identity": _identity_fingerprint_fields(model_identity),
        "classes": list(CLASS_NAMES),
    }
    state["best_model_sha256"] = checkpoint_sha256
    state["best_model_binding"] = dict(binding)
    save_json(state_path, state)
    binding["training_state_sha256"] = sha256_file(state_path)
    return binding


def _write_cv_completion_seal(
    percentage_dir: Path,
    *,
    cv_id: str,
    method: str,
    percentage: int,
    parameters: Mapping,
    model_identity: Mapping,
    best_fold: int,
    refit_budget: int,
    fold_training_rows_fingerprints: Mapping[int, str],
    refit_training_rows_fingerprint: str,
    selected_grid_reuse: bool = False,
    selected_grid_candidate_id: str | None = None,
    selected_grid_result_origin: str | None = None,
    selected_grid_root: Path | None = None,
) -> dict:
    """Write the final CV seal after every constituent artifact is complete."""

    marker_paths = [
        percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json"
        for fold in range(1, 6)
    ]
    present_markers = [path for path in marker_paths if path.is_file()]
    if selected_grid_reuse:
        if (
            len(present_markers) != 5
            or not selected_grid_candidate_id
            or not selected_grid_result_origin
            or selected_grid_root is None
        ):
            raise RuntimeError(
                "Selected-grid CV sealing requires five reuse markers plus a "
                "candidate, result origin, and source grid root."
            )
    elif present_markers:
        raise RuntimeError(
            "Dedicated CV output contains unclaimed selected-grid reuse markers: "
            f"{[str(path) for path in present_markers]}"
        )

    fold_models: list[dict] = []
    for fold in range(1, 6):
        expected_rows_fingerprint = fold_training_rows_fingerprints.get(fold)
        if not expected_rows_fingerprint:
            raise RuntimeError(f"Missing fold-{fold} training-row fingerprint.")
        fold_models.append(
            _classifier_binding_from_output(
                percentage_dir,
                percentage_dir / f"fold_{fold:02d}",
                method=method,
                classifier_identity={"role": "cv_fold", "fold": fold},
                expected_parameters=parameters,
                expected_training_rows_fingerprint=expected_rows_fingerprint,
                model_identity=model_identity,
            )
        )

    refit_effective_parameters = dict(parameters)
    refit_effective_parameters["evaluation_interval_iterations"] = int(
        refit_budget
    )
    refit_effective_parameters["early_stopping_patience"] = 0
    refit_model_binding = _classifier_binding_from_output(
        percentage_dir,
        percentage_dir / "refit_full_development",
        method=method,
        classifier_identity={"role": "refit_full_development", "fold": None},
        expected_parameters=refit_effective_parameters,
        expected_training_rows_fingerprint=refit_training_rows_fingerprint,
        model_identity=model_identity,
        expected_runtime_budget=int(refit_budget),
    )
    selected = next(
        record for record in fold_models if int(record["fold"]) == int(best_fold)
    )
    best_fold_path = percentage_dir / "best_fold_model.json"
    best_fold_document = read_json(best_fold_path)
    selected_checkpoint = percentage_dir / str(selected["relative_path"])
    if (
        best_fold_document.get("cv_id") != cv_id
        or best_fold_document.get("method") != method
        or int(best_fold_document.get("best_fold", 0)) != int(best_fold)
        or Path(str(best_fold_document.get("checkpoint", ""))).resolve()
        != selected_checkpoint.resolve()
    ):
        raise RuntimeError(
            f"Best-fold selection is not bound to fold {best_fold}: {best_fold_path}"
        )
    artifacts = {
        "cv_fold_summary.csv": sha256_file(
            percentage_dir / "cv_fold_summary.csv"
        ),
        "best_fold_model.json": sha256_file(
            best_fold_path
        ),
    }
    grid_reuse_markers: dict[str, str] | None = None
    if selected_grid_reuse:
        for fold in range(1, 6):
            marker_path = percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json"
            _validate_or_refresh_grid_reuse(
                marker_path,
                expected_candidate_id=selected_grid_candidate_id,
                expected_result_origin=selected_grid_result_origin,
                expected_source_grid_root=selected_grid_root,
                refresh_after_cv_binding=True,
            )
        grid_reuse_markers = {
            f"fold_{fold:02d}/grid_reuse.json": sha256_file(
                percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json"
            )
            for fold in range(1, 6)
        }
    seal = {
        "schema_version": 1,
        "model_family": "3DINO",
        "cv_id": cv_id,
        "method": method,
        "percentage": int(percentage),
        "parameter_fingerprint": parameter_fingerprint(
            method, parameters, model_identity
        ),
        "effective_parameters": dict(parameters),
        "execution_context": parameters.get("_execution_context"),
        "pretrained_identity": _identity_fingerprint_fields(model_identity),
        "fold_models": fold_models,
        "refit_model": refit_model_binding,
        "best_fold": int(best_fold),
        "best_fold_binding": {
            "fold": int(best_fold),
            "relative_path": selected["relative_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "training_rows_fingerprint": selected[
                "training_rows_fingerprint"
            ],
            "runtime_plan_fingerprint": selected[
                "runtime_plan_fingerprint"
            ],
        },
        "refit_budget": int(refit_budget),
        "artifacts": artifacts,
        "complete": True,
    }
    if grid_reuse_markers is not None:
        seal["grid_reuse_markers"] = grid_reuse_markers
        seal["selected_grid_folds_reused_without_training"] = True
        seal["selected_grid_candidate_id"] = str(selected_grid_candidate_id)
        seal["grid_reuse_result_origin"] = str(selected_grid_result_origin)
        seal["grid_reuse_source_root"] = str(Path(selected_grid_root).resolve())
    completion_path = percentage_dir / _CV_COMPLETION_FILENAME
    sealed_at = utc_now()
    if completion_path.is_file():
        existing = read_json(completion_path)
        existing_identity = dict(existing)
        existing_identity.pop("seal_fingerprint", None)
        existing_sealed_at = existing_identity.pop("sealed_at_utc", None)
        if existing_identity == seal and isinstance(existing_sealed_at, str):
            sealed_at = existing_sealed_at
    seal["sealed_at_utc"] = sealed_at
    seal["seal_fingerprint"] = canonical_sha256(seal)
    save_json(completion_path, seal)
    return seal


def _validate_cv_completion_seal(
    percentage_dir: Path,
    *,
    cv_id: str,
    method: str,
    percentage: int,
    parameters: Mapping,
    model_identity: Mapping,
) -> dict:
    """Load and structurally validate the final CV completion seal."""

    path = percentage_dir / _CV_COMPLETION_FILENAME
    if not path.is_file():
        raise RuntimeError(
            f"Internal classifier evaluation requires the final CV completion seal: {path}"
        )
    seal = read_json(path)
    recorded_fingerprint = seal.get("seal_fingerprint")
    unsigned = dict(seal)
    unsigned.pop("seal_fingerprint", None)
    if recorded_fingerprint != canonical_sha256(unsigned):
        raise RuntimeError(f"CV completion seal fingerprint mismatch: {path}")
    expected_top_level = {
        "complete": True,
        "model_family": "3DINO",
        "cv_id": cv_id,
        "method": method,
        "percentage": int(percentage),
        "parameter_fingerprint": parameter_fingerprint(
            method, parameters, model_identity
        ),
        "effective_parameters": dict(parameters),
        "execution_context": parameters.get("_execution_context"),
        "pretrained_identity": _identity_fingerprint_fields(model_identity),
    }
    for key, expected in expected_top_level.items():
        if seal.get(key) != expected:
            raise RuntimeError(f"CV completion seal has incompatible {key}: {path}")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"CV completion seal has no artifact hash map: {path}")
    for name in ("cv_fold_summary.csv", "best_fold_model.json"):
        artifact = percentage_dir / name
        if not artifact.is_file() or artifacts.get(name) != sha256_file(artifact):
            raise RuntimeError(f"CV completion artifact changed after sealing: {artifact}")

    fold_models = seal.get("fold_models")
    if not isinstance(fold_models, list) or len(fold_models) != 5:
        raise RuntimeError(f"CV completion seal must bind five fold models: {path}")
    observed_folds: set[int] = set()
    for record in fold_models:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Invalid fold classifier record in {path}")
        fold = int(record.get("fold", 0))
        observed_folds.add(fold)
        _validate_cv_classifier_record(
            percentage_dir,
            record,
            method=method,
            expected_identity={"role": "cv_fold", "fold": fold},
            expected_parameters=parameters,
            model_identity=model_identity,
            verify_checkpoint=True,
        )
    if observed_folds != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"CV completion seal has invalid fold identities: {path}")
    grid_reuse_markers = seal.get("grid_reuse_markers")
    grid_reuse_claim = seal.get("selected_grid_folds_reused_without_training")
    on_disk_grid_reuse_markers = [
        percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json"
        for fold in range(1, 6)
        if (percentage_dir / f"fold_{fold:02d}" / "grid_reuse.json").is_file()
    ]
    if grid_reuse_claim is True:
        if not isinstance(grid_reuse_markers, Mapping) or set(grid_reuse_markers) != {
            f"fold_{fold:02d}/grid_reuse.json" for fold in range(1, 6)
        }:
            raise RuntimeError(f"CV completion seal has invalid grid-reuse markers: {path}")
        expected_candidate = str(seal.get("selected_grid_candidate_id", ""))
        expected_result_origin = str(seal.get("grid_reuse_result_origin", ""))
        expected_source_root_raw = seal.get("grid_reuse_source_root")
        if (
            not expected_candidate
            or expected_result_origin not in {
                "selected_publication_grid_run",
                "selected_grid_run",
            }
            or not isinstance(expected_source_root_raw, str)
            or not expected_source_root_raw
        ):
            raise RuntimeError(
                f"CV completion seal has incomplete grid-reuse provenance: {path}"
            )
        expected_source_root = Path(expected_source_root_raw).resolve()
        for fold in range(1, 6):
            relative = f"fold_{fold:02d}/grid_reuse.json"
            marker = percentage_dir / relative
            if not marker.is_file() or sha256_file(marker) != grid_reuse_markers[relative]:
                raise RuntimeError(f"CV grid-reuse marker changed after sealing: {marker}")
            document = read_json(marker)
            candidate = str(document.get("candidate_id", ""))
            if candidate != expected_candidate:
                raise RuntimeError(f"CV grid-reuse candidate mismatch: {marker}")
            _validate_or_refresh_grid_reuse(
                marker,
                expected_candidate_id=candidate,
                expected_result_origin=expected_result_origin,
                expected_source_grid_root=expected_source_root,
            )
    elif grid_reuse_markers is not None or on_disk_grid_reuse_markers:
        raise RuntimeError(
            f"CV completion has unclaimed grid-reuse markers: {path}"
        )
    refit_budget = int(seal.get("refit_budget", 0))
    if refit_budget <= 0:
        raise RuntimeError(f"CV completion seal has no positive refit budget: {path}")
    refit_parameters = dict(parameters)
    refit_parameters["evaluation_interval_iterations"] = refit_budget
    refit_parameters["early_stopping_patience"] = 0
    refit_record = seal.get("refit_model")
    if not isinstance(refit_record, Mapping):
        raise RuntimeError(f"CV completion seal has no refit classifier: {path}")
    _validate_cv_classifier_record(
        percentage_dir,
        refit_record,
        method=method,
        expected_identity={"role": "refit_full_development", "fold": None},
        expected_parameters=refit_parameters,
        model_identity=model_identity,
        expected_runtime_budget=refit_budget,
        verify_checkpoint=True,
    )
    best_fold = int(seal.get("best_fold", 0))
    if best_fold not in observed_folds:
        raise RuntimeError(f"CV completion seal has an invalid best fold: {path}")
    selected = next(
        record for record in fold_models if int(record["fold"]) == best_fold
    )
    expected_best_binding = {
        "fold": best_fold,
        "relative_path": selected["relative_path"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "training_rows_fingerprint": selected["training_rows_fingerprint"],
        "runtime_plan_fingerprint": selected["runtime_plan_fingerprint"],
    }
    if seal.get("best_fold_binding") != expected_best_binding:
        raise RuntimeError(f"Best-fold binding differs from its fold record: {path}")
    best_fold_document = read_json(percentage_dir / "best_fold_model.json")
    selected_checkpoint = percentage_dir / str(selected["relative_path"])
    if (
        best_fold_document.get("cv_id") != cv_id
        or best_fold_document.get("method") != method
        or int(best_fold_document.get("best_fold", 0)) != best_fold
        or Path(str(best_fold_document.get("checkpoint", ""))).resolve()
        != selected_checkpoint.resolve()
    ):
        raise RuntimeError(
            "Best-fold selection document differs from the final CV completion seal."
        )
    return seal


def _validate_cv_classifier_record(
    percentage_dir: Path,
    record: Mapping,
    *,
    method: str,
    expected_identity: Mapping,
    expected_parameters: Mapping,
    model_identity: Mapping,
    expected_runtime_budget: int | None = None,
    verify_checkpoint: bool = True,
) -> dict:
    """Validate one classifier record, optionally including the checkpoint bytes."""

    identity = dict(expected_identity)
    if record.get("classifier_identity") != identity:
        raise RuntimeError("Classifier identity differs from the CV completion seal.")
    if record.get("method") != method:
        raise RuntimeError("Classifier method differs from the CV completion seal.")
    if record.get("role") != identity.get("role") or record.get("fold") != identity.get(
        "fold"
    ):
        raise RuntimeError("Classifier fold/refit identity is inconsistent.")
    expected_relative = (
        f"fold_{int(identity['fold']):02d}/best_model.pt"
        if identity.get("role") == "cv_fold"
        else "refit_full_development/best_model.pt"
    )
    if record.get("relative_path") != expected_relative:
        raise RuntimeError("Classifier path differs from its fold/refit identity.")
    if record.get("effective_checkpoint_parameters") != dict(expected_parameters):
        raise RuntimeError("Classifier effective parameters differ from the CV seal.")
    parameters_fingerprint = canonical_sha256(dict(expected_parameters))
    if record.get("parameters_fingerprint") != parameters_fingerprint:
        raise RuntimeError("Classifier parameter fingerprint differs from the CV seal.")
    runtime_plan = record.get("runtime_plan")
    if not isinstance(runtime_plan, Mapping):
        raise RuntimeError("Classifier CV seal record has no runtime plan.")
    runtime_plan_fingerprint = _runtime_plan_fingerprint(
        expected_parameters, runtime_plan
    )
    if record.get("runtime_plan_fingerprint") != runtime_plan_fingerprint:
        raise RuntimeError("Classifier runtime-plan fingerprint differs from the CV seal.")
    training_rows_fingerprint = runtime_plan.get("training_rows_fingerprint")
    if record.get("training_rows_fingerprint") != training_rows_fingerprint:
        raise RuntimeError("Classifier training-row fingerprint differs from the CV seal.")
    if record.get("pretrained_identity") != _identity_fingerprint_fields(model_identity):
        raise RuntimeError("Classifier pretrained identity differs from the CV seal.")
    state_path = percentage_dir / expected_relative.rsplit("/", 1)[0] / "training_state.json"
    if not state_path.is_file() or record.get("training_state_sha256") != sha256_file(
        state_path
    ):
        raise RuntimeError(f"Classifier training state changed after CV sealing: {state_path}")
    state = read_json(state_path)
    state_binding = state.get("best_model_binding")
    if not isinstance(state_binding, Mapping):
        raise RuntimeError(f"Classifier training state lacks its CV seal binding: {state_path}")
    for key in (
        "classifier_identity",
        "method",
        "fold",
        "role",
        "relative_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "effective_checkpoint_parameters",
        "parameters_fingerprint",
        "training_rows_fingerprint",
        "runtime_plan",
        "runtime_plan_fingerprint",
        "pretrained_identity",
        "classes",
    ):
        if state_binding.get(key) != record.get(key):
            raise RuntimeError(
                f"Classifier training-state binding differs from the CV seal for {key}: {state_path}"
            )
    if not verify_checkpoint:
        return dict(record)

    checkpoint = percentage_dir / expected_relative
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Sealed classifier checkpoint not found: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != record.get("checkpoint_sha256"):
        raise RuntimeError(f"Classifier checkpoint changed after CV sealing: {checkpoint}")
    if int(record.get("checkpoint_size_bytes", -1)) != int(checkpoint.stat().st_size):
        raise RuntimeError(f"Classifier checkpoint size differs from the CV seal: {checkpoint}")
    if not _is_completed(checkpoint.parent, expected_parameters, model_identity):
        raise RuntimeError(f"Sealed classifier output is incomplete: {checkpoint.parent}")
    checkpoint_metadata = _validate_internal_classifier_checkpoint(
        checkpoint,
        method=method,
        expected_parameters=expected_parameters,
        model_identity=model_identity,
        expected_runtime_budget=expected_runtime_budget,
        expected_training_rows_fingerprint=str(training_rows_fingerprint),
        expected_runtime_plan_fingerprint=runtime_plan_fingerprint,
    )
    if checkpoint_metadata["runtime_plan"] != dict(runtime_plan):
        raise RuntimeError("Classifier runtime plan differs from the sealed plan.")
    result = dict(record)
    result.update(checkpoint_metadata)
    result["path"] = str(checkpoint.resolve())
    result["sha256"] = checkpoint_sha256
    result["size_bytes"] = int(checkpoint.stat().st_size)
    result["cv_seal_binding_verified"] = True
    return result


def run_test(
    args: Any,
    method: str,
    manifest: Mapping,
    parameters: Mapping,
    provenance: Mapping,
    model_identity: Mapping,
) -> dict:
    import torch

    cv_id, cv_dir = _resolve_cv(args, method, parameters, model_identity)
    percentage_dir = cv_dir / f"{args.percentage}_percent"
    output_dir = method_root(args.results_root, method) / "Test" / cv_id / args.model_source / f"{args.percentage}_percent"
    best_fold_document: dict | None = None
    if args.model_source == "refit_full_development":
        checkpoints = [percentage_dir / "refit_full_development" / "best_model.pt"]
    elif args.model_source == "best_cv_fold":
        best_fold_document = read_json(percentage_dir / "best_fold_model.json")
        checkpoints = [Path(best_fold_document["checkpoint"])]
    elif args.model_source == "cv_ensemble":
        checkpoints = [percentage_dir / f"fold_{fold:02d}" / "best_model.pt" for fold in range(1, 6)]
    elif args.model_source == "provided_checkpoint":
        if args.checkpoint is None:
            raise ValueError("provided_checkpoint requires --checkpoint.")
        checkpoints = [Path(args.checkpoint)]
    else:  # defensive: argparse already constrains this value
        raise ValueError(f"Unknown model source: {args.model_source}")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint}")
    checkpoint_records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in checkpoints
    ]
    checkpoint_training_parameters: dict | None = None
    cv_completion: dict | None = None
    cv_completion_path: Path | None = None
    if args.model_source == "provided_checkpoint":
        for record in checkpoint_records:
            record["binding_verified"] = False
            record["binding_status"] = (
                "external_checkpoint_parameters_and_selection_history_unverified"
            )
    else:
        cv_completion_path = percentage_dir / _CV_COMPLETION_FILENAME
        cv_completion = _validate_cv_completion_seal(
            percentage_dir,
            cv_id=cv_id,
            method=method,
            percentage=int(args.percentage),
            parameters=parameters,
            model_identity=model_identity,
        )
        expected_checkpoint_parameters = dict(parameters)
        expected_runtime_budget: int | None = None
        sealed_records: list[Mapping]
        expected_identities: list[dict]
        if args.model_source == "refit_full_development":
            expected_runtime_budget = int(cv_completion["refit_budget"])
            expected_checkpoint_parameters["evaluation_interval_iterations"] = (
                expected_runtime_budget
            )
            expected_checkpoint_parameters["early_stopping_patience"] = 0
            sealed_records = [cv_completion["refit_model"]]
            expected_identities = [
                {"role": "refit_full_development", "fold": None}
            ]
        elif args.model_source == "best_cv_fold":
            if best_fold_document is None:
                raise RuntimeError("Best-fold selection document was not loaded.")
            selected_fold = int(best_fold_document.get("best_fold", 0))
            if selected_fold != int(cv_completion["best_fold"]):
                raise RuntimeError(
                    "Best-fold selection is not bound to the fold in the CV completion seal."
                )
            sealed_records = [
                next(
                    record
                    for record in cv_completion["fold_models"]
                    if int(record["fold"]) == selected_fold
                )
            ]
            expected_identities = [
                {"role": "cv_fold", "fold": selected_fold}
            ]
        else:
            sealed_records = sorted(
                cv_completion["fold_models"], key=lambda record: int(record["fold"])
            )
            expected_identities = [
                {"role": "cv_fold", "fold": fold} for fold in range(1, 6)
            ]
            sealed_hashes = [str(record["checkpoint_sha256"]) for record in sealed_records]
            if len(set(sealed_hashes)) != len(sealed_hashes):
                raise RuntimeError(
                    "CV ensemble is invalid because multiple folds have duplicate classifier SHA-256 values."
                )
        checkpoint_training_parameters = dict(expected_checkpoint_parameters)
        verified_records: list[dict] = []
        for checkpoint, sealed_record, expected_identity in zip(
            checkpoints, sealed_records, expected_identities
        ):
            verified = _validate_cv_classifier_record(
                percentage_dir,
                sealed_record,
                method=method,
                expected_identity=expected_identity,
                expected_parameters=expected_checkpoint_parameters,
                model_identity=model_identity,
                expected_runtime_budget=expected_runtime_budget,
                verify_checkpoint=True,
            )
            if Path(checkpoint).resolve() != Path(verified["path"]).resolve():
                raise RuntimeError(
                    "Selected classifier path differs from its fold/refit CV seal binding."
                )
            verified["binding_status"] = (
                "verified_against_final_cv_completion_seal"
            )
            verified_records.append(verified)
        checkpoint_records = verified_records
        if args.model_source == "cv_ensemble":
            current_hashes = [str(record["sha256"]) for record in checkpoint_records]
            if len(set(current_hashes)) != len(current_hashes):
                raise RuntimeError(
                    "CV ensemble is invalid because multiple folds resolve to duplicate classifier SHA-256 values."
                )
    test_identity = {
        "schema_version": 1,
        "cv_id": cv_id,
        "method": method,
        "percentage": int(args.percentage),
        "model_source": str(args.model_source),
        "parameter_fingerprint": parameter_fingerprint(
            method, parameters, model_identity
        ),
        "execution_context": parameters.get("_execution_context"),
        "pretrained_identity": _identity_fingerprint_fields(model_identity),
        "classifier_checkpoint_sha256": [
            value["sha256"] for value in checkpoint_records
        ],
        "cv_completion_seal_fingerprint": (
            cv_completion.get("seal_fingerprint")
            if isinstance(cv_completion, Mapping)
            else None
        ),
    }
    test_fingerprint = canonical_sha256(test_identity)
    marker_path = output_dir / "test_completion.json"
    existing_files = output_dir.is_dir() and any(output_dir.iterdir())
    if existing_files and not args.force_test:
        marker = _validate_test_completion(output_dir, test_fingerprint)
        if args.resume:
            return {
                "cv_id": cv_id,
                "output_directory": str(output_dir),
                "metrics": read_json(output_dir / "metrics_summary.json"),
                "completion": marker,
                "reused_completed_result": True,
            }
        raise FileExistsError(
            f"Held-out result already exists and is sealed: {output_dir}. "
            "Use --resume to reuse it or --force_test to evaluate again."
        )

    heldout_rows = _split_rows(args, heldout_mapping(manifest))
    development_rows = _split_rows(args, development_mapping(manifest))
    validate_no_overlap(development_rows, heldout_rows)

    if len(checkpoints) == 1:
        metrics = evaluate_checkpoint(
            heldout_rows,
            checkpoints[0],
            output_dir,
            method=method,
            **_prediction_kwargs(args),
        )
    else:
        probabilities: list[np.ndarray] = []
        labels = None
        for checkpoint in checkpoints:
            logits, current_labels = predict_checkpoint(
                heldout_rows,
                checkpoint,
                method=method,
                **_prediction_kwargs(args),
            )
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            labels = current_labels if labels is None else labels
            if not torch.equal(labels, current_labels):
                raise RuntimeError("Ensemble checkpoints produced inconsistent patient ordering.")
        averaged = np.mean(probabilities, axis=0)
        truth = labels.detach().cpu().numpy()
        metrics = compute_metrics(truth, averaged.argmax(axis=1), averaged, list(CLASS_NAMES))
        predictions, frame = format_predictions(
            [row["patient_id"] for row in heldout_rows],
            [row["label_str"] for row in heldout_rows],
            averaged,
            list(CLASS_NAMES),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "metrics_summary.json", metrics)
        save_json(output_dir / "test_patient_predictions.json", predictions)
        frame.to_csv(output_dir / "test_probabilities.csv", index=False)

    cache_manifest_path = Path(args.cache_dir) / "cache_manifest.json"
    save_json(
        output_dir / "model_selection.json",
        {
            "schema_version": 1,
            "model_family": "3DINO",
            "cv_id": cv_id,
            "method": method,
            "model_source": args.model_source,
            "parameters": (
                None
                if args.model_source == "provided_checkpoint"
                else dict(parameters)
            ),
            "cv_selection_parameters": dict(parameters),
            "checkpoint_training_parameters": checkpoint_training_parameters,
            "parameters_role": (
                "requested_cv_context_not_attributed_to_external_checkpoint"
                if args.model_source == "provided_checkpoint"
                else "cv_registered_training_parameters"
            ),
            "checkpoint_parameter_binding": (
                "external_checkpoint_not_asserted_to_match_requested_cv_parameters"
                if args.model_source == "provided_checkpoint"
                else "internal_classifier_bound_by_final_cv_completion_seal"
            ),
            "selection_provenance_status": (
                "unknown_external_checkpoint_history"
                if args.model_source == "provided_checkpoint"
                else "development_only_selection_recorded_by_3dino_workflow"
            ),
            "parameter_provenance": dict(provenance),
            "model_identity": dict(model_identity),
            "checkpoints": [str(path.resolve()) for path in checkpoints],
            "classifier_checkpoints": checkpoint_records,
            "cv_completion_seal": (
                str(cv_completion_path.resolve())
                if cv_completion_path is not None
                else None
            ),
            "cv_completion_seal_sha256": (
                sha256_file(cv_completion_path)
                if cv_completion_path is not None
                else None
            ),
            "cv_completion_seal_fingerprint": (
                cv_completion.get("seal_fingerprint")
                if isinstance(cv_completion, Mapping)
                else None
            ),
            "test_identity": test_identity,
            "test_fingerprint": test_fingerprint,
            "test_results_directory": str(output_dir.resolve()),
            "cache_manifest": str(cache_manifest_path.resolve()),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
            "heldout_test_used_during_selection": (
                None if args.model_source == "provided_checkpoint" else False
            ),
            "evaluated_at_utc": utc_now(),
        },
    )
    missing_artifacts = [
        name for name in _TEST_ARTIFACT_NAMES if not (output_dir / name).is_file()
    ]
    if missing_artifacts:
        raise RuntimeError(
            "Held-out evaluation did not produce its complete artifact contract: "
            f"{missing_artifacts}"
        )
    completion = {
        "schema_version": 1,
        "model_family": "3DINO",
        "cv_id": cv_id,
        "method": method,
        "percentage": int(args.percentage),
        "model_source": args.model_source,
        "test_fingerprint": test_fingerprint,
        "artifacts": {
            name: sha256_file(output_dir / name) for name in _TEST_ARTIFACT_NAMES
        },
        "complete": True,
        "completed_at_utc": utc_now(),
    }
    save_json(marker_path, completion)
    return {
        "cv_id": cv_id,
        "output_directory": str(output_dir),
        "metrics": metrics,
        "completion": completion,
    }


def _dry_run_profile(args: Any, method: str) -> tuple[dict, dict]:
    source = _canonical_grid_source(args.parameter_source)
    if source in {"grid", "publication"}:
        configured_grid = _grid_config_arg(args)
        maximum_candidates = _max_grid_candidates_arg(args)
        if source == "publication":
            if method != "linear_probe":
                raise ValueError(
                    "3DINO publication parameters define a linear-probe grid only."
                )
            if configured_grid is not None or maximum_candidates is not None:
                raise ValueError(
                    "The complete publication grid does not accept a custom grid "
                    "configuration or candidate truncation."
                )
        path = _search_path(args, method, source)
        definition = load_search_definition(path)
        candidates = search_candidates(
            definition, maximum_candidates if source == "grid" else None
        )
        grid_fields = _grid_provenance_fields(source, definition)
        return {
            "candidate_count": len(candidates),
            "strategy": definition.get("strategy", "publication_multi_head"),
        }, {
            **grid_fields,
            "profile_path": str(path.resolve()),
            "profile_sha256": sha256_file(path),
            "authorization": definition.get("authorization"),
            "task_adaptations": definition.get("task_adaptations", {}),
        }
    return load_parameter_profile(method, source, args.parameter_config)


def run_pipeline(args: Any, method: str) -> dict:
    """Run the requested stages while keeping held-out patients selection-blind."""

    if method not in METHOD_FOLDERS:
        raise ValueError(f"Unknown 3DINO method: {method}")
    if method == "lora" and (
        args.parameter_source not in {"default", "manual"}
        or args.stage in {"grid", "search"}
        or _grid_config_arg(args) is not None or _max_grid_candidates_arg(args) is not None
    ):
        raise ValueError("LoRA supports fixed default/manual profiles and cv/test/all stages only.")
    args.stage = "grid" if str(args.stage) == "search" else str(args.stage)
    args.parameter_source = _canonical_grid_source(args.parameter_source)
    if args.stage == "grid" and args.parameter_source == "default":
        args.parameter_source = "grid"
    if args.stage == "grid" and args.parameter_source == "manual":
        raise ValueError(
            "The grid stage requires parameter_source=grid or publication; "
            "manual supplies one fixed CV/test configuration."
        )
    args.dataset_root = Path(args.dataset_root).resolve()
    args.results_root = Path(args.results_root).resolve()
    args.cache_dir = Path(args.cache_root).resolve() / "Shared" / str(
        getattr(args, "cache_namespace", DEFAULT_CACHE_NAMESPACE)
    )
    manifest = load_manifest(args.dataset_root, args.percentage)
    preview_parameters, preview_provenance = _dry_run_profile(args, method)
    if method == "lora":
        try:
            from .lora import validate_parameters
        except ImportError:
            from lora import validate_parameters
        validate_parameters(preview_parameters)
    operational_loader = _operational_loader_provenance(args)
    if "candidate_count" in preview_parameters:
        preview_parameters = {
            **preview_parameters,
            "effective_seed": int(args.seed),
        }
    else:
        preview_parameters = dict(preview_parameters)
        preview_parameters["seed"] = int(args.seed)
    authorization_status = (
        _authorization_status(args, preview_provenance.get("authorization"))
        if method == "full_finetune"
        else {
            "applicable": False,
            "authorized": False,
        }
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "model_family": "3DINO",
            "method": method,
            "stage": args.stage,
            "percentage": args.percentage,
            "classes": list(CLASS_NAMES),
            "parameters_or_search": preview_parameters,
            "parameter_provenance": preview_provenance,
            "operational_loader": operational_loader,
            "model_identity": _runtime_model_identity(args, require_files=False),
            "cache_directory": str(args.cache_dir),
            "cache_manifest_exists": (args.cache_dir / "cache_manifest.json").is_file(),
            "full_finetune_authorization": authorization_status,
            "full_finetune_authorized": bool(
                authorization_status.get("authorized")
            ),
            "heldout_test_planned_only": True,
            "writes_performed": False,
        }

    if method == "full_finetune":
        # Reject an unauthorized profile before hashing a large checkpoint or
        # exhaustively validating cache artifacts. The same gate is repeated on
        # the resolved parameters below.
        _authorise_parameters(
            method, {}, args, preview_provenance.get("authorization")
        )
    cache_manifest = load_cache_manifest(args.cache_dir)
    validate_cache(cache_manifest, verify_artifacts=True)
    model_identity = _runtime_model_identity(args, require_files=True)
    execution_context = _runtime_execution_context(
        args, manifest, cache_manifest, model_identity
    )

    source = _canonical_grid_source(args.parameter_source)
    if source in {"grid", "publication"}:
        if args.stage in {"grid", "all"}:
            parameters, provenance = run_search(
                args,
                method,
                manifest,
                source=source,
                execution_context=execution_context,
            )
        else:
            parameters, provenance = _selected_parameters_from_search(
                args, method, source, execution_context
            )
    else:
        parameters, provenance = load_parameter_profile(method, source, args.parameter_config)
    parameters = _apply_execution_context(parameters, args, execution_context)
    provenance = {
        **dict(provenance),
        "execution_context": execution_context,
        "execution_context_fingerprint": canonical_sha256(execution_context),
        "runtime_overrides": {
            "seed": int(args.seed),
        },
        "operational_loader": operational_loader,
    }
    parameters = _authorise_parameters(
        method, parameters, args, provenance.get("authorization")
    )

    result: dict[str, Any] = {
        "model_family": "3DINO",
        "method": method,
        "percentage": args.percentage,
        "parameters": parameters,
        "parameter_provenance": provenance,
        "operational_loader": operational_loader,
        "model_identity": model_identity,
    }
    if args.stage == "grid":
        return result
    if args.stage in {"cv", "all"}:
        result["cv"] = run_cv(args, method, manifest, parameters, provenance, model_identity)
    if args.stage in {"test", "all"}:
        result["test"] = run_test(args, method, manifest, parameters, provenance, model_identity)
    return result


__all__ = [
    "METHOD_FOLDERS",
    "PROFILE_FILES",
    "default_dataset_root",
    "default_results_root",
    "load_parameter_profile",
    "methodology_root",
    "parameter_fingerprint",
    "register_cv_run",
    "run_cv",
    "run_pipeline",
    "run_search",
    "run_test",
]
