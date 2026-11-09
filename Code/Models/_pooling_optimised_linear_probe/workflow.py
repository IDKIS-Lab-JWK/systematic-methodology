"""COLIPRI pooling-optimised grid, CV, refit, and held-out orchestration.

The COLIPRI backend owns preprocessing, feature caching, pooling-head training,
and artifact loading. This isolated module preserves the shared experiment
semantics while assigning learned-pooling outputs their own method namespace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _linear_probe.common import (
    CLASS_NAMES,
    ContractError,
    atomic_write_csv,
    atomic_write_json,
    candidate_id,
    close_logger,
    completion_seal,
    cross_validation_split,
    development_mapping,
    ensure_same_or_write,
    fingerprint,
    heldout_mapping,
    load_manifest,
    materialize_file,
    read_json,
    rows_fingerprint,
    rows_from_mapping,
    setup_logger,
    sha256_file,
    tree_completion_seal,
    utc_now,
    validate_completion,
    validate_tree_completion,
)
from _linear_probe.metrics import (
    compute_metrics,
    fold_selection_score,
    prediction_documents,
)


CANONICAL_METHOD_LABEL = "PoolingOptimisedLinearProbe"
LEGACY_METHOD_LABEL = "LinearProbe"
MIGRATION_MANIFEST = "METHOD_IDENTITY_MIGRATION.json"


def _method_root(args: Any) -> Path:
    return Path(args.results_root) / CANONICAL_METHOD_LABEL


def _migration_document(args: Any) -> dict:
    path = _method_root(args) / MIGRATION_MANIFEST
    if not path.is_file():
        raise ContractError(
            "Legacy pooling-optimised resume requires its migration manifest: "
            f"{path}"
        )
    document = read_json(path)
    required = {
        "schema_version": 1,
        "migration_type": "metadata_only_result_namespace",
        "model": "COLIPRI",
        "legacy_method_label": LEGACY_METHOD_LABEL,
        "canonical_method_id": "pooling_optimised_linear_probe",
        "canonical_method_label": CANONICAL_METHOD_LABEL,
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ContractError(f"Incompatible method-identity migration manifest: {path}")
    legacy_implementation = document.get("legacy_probe_implementation_identity")
    if not isinstance(legacy_implementation, dict):
        raise ContractError(
            f"Migration manifest has no exact legacy implementation identity: {path}"
        )
    return document


def _identity_without_probe_implementation(identity: Mapping[str, Any]) -> dict:
    comparable = dict(identity)
    comparable.pop("probe_implementation_identity", None)
    return comparable


def _legacy_alias(path: Path | str) -> str:
    text = str(Path(path).resolve())
    return text.replace(CANONICAL_METHOD_LABEL, LEGACY_METHOD_LABEL, 1)


def _resolve_migrated_path(path: Path | str) -> Path:
    requested = Path(path).resolve()
    if requested.exists():
        return requested
    text = str(requested)
    if LEGACY_METHOD_LABEL in text:
        canonical = Path(
            text.replace(LEGACY_METHOD_LABEL, CANONICAL_METHOD_LABEL, 1)
        ).resolve()
        if canonical.exists():
            return canonical
    return requested


def _effective_grid_context(
    args: Any,
    root: Path,
    current: Mapping[str, Any],
) -> dict:
    """Adopt an exact legacy context only for the documented namespace migration."""

    context_path = root / "grid_context.json"
    if not context_path.is_file():
        return dict(current)
    stored = read_json(context_path)
    if stored == dict(current):
        return dict(current)
    if not args.resume:
        return dict(current)
    migration = _migration_document(args)
    if (
        stored.get("method") != LEGACY_METHOD_LABEL
        or current.get("method") != CANONICAL_METHOD_LABEL
    ):
        raise ContractError(
            f"Existing grid context is not covered by the declared migration: {root}"
        )
    allowed_differences = {"method", "profile", "profile_sha256", "model_identity"}
    for key in set(stored) | set(current):
        if key not in allowed_differences and stored.get(key) != current.get(key):
            raise ContractError(
                "Legacy grid resume rejected because a scientific/runtime identity "
                f"field changed ({key}): {root}"
            )
    stored_identity = stored.get("model_identity")
    current_identity = current.get("model_identity")
    if not isinstance(stored_identity, dict) or not isinstance(current_identity, dict):
        raise ContractError(f"Grid context has no valid model identity: {root}")
    if (
        stored_identity.get("probe_implementation_identity")
        != migration["legacy_probe_implementation_identity"]
    ):
        raise ContractError(
            f"Legacy grid implementation identity is not declared by the migration: {root}"
        )
    if _identity_without_probe_implementation(
        stored_identity
    ) != _identity_without_probe_implementation(current_identity):
        raise ContractError(
            f"Legacy grid feature/cache/runtime identity changed: {root}"
        )
    if stored.get("profile_sha256") != migration.get("legacy_profile_sha256"):
        raise ContractError(
            f"Legacy grid parameter profile is not declared by the migration: {root}"
        )
    return stored


def _grid_folder(parameter_source: str) -> str:
    if parameter_source == "publication":
        return "Publication_Grid"
    if parameter_source == "grid":
        return "Grid"
    raise ContractError(f"{parameter_source!r} is not a grid parameter source")


def _profile_for_source(args: Any, backend: Any) -> tuple[dict, Path]:
    source = args.parameter_source
    if source == "publication":
        path = Path(backend.publication_profile_path)
    elif source == "default":
        path = Path(args.parameter_config or backend.default_profile_path)
    elif source == "manual":
        if args.parameter_config is None:
            raise ContractError("manual parameter source requires --parameter_config")
        path = Path(args.parameter_config)
    elif source == "grid":
        path = Path(args.grid_config or backend.search_profile_path)
    else:
        raise ContractError(f"Unsupported parameter source: {source}")
    return read_json(path), path.resolve()


def _candidate_documents(args: Any, backend: Any) -> tuple[list[dict], dict, Path]:
    profile, path = _profile_for_source(args, backend)
    candidates = [dict(item) for item in backend.build_candidates(profile)]
    if not candidates:
        raise ContractError("The grid contains no candidates")
    seen: set[str] = set()
    for parameters in candidates:
        identifier = candidate_id(parameters)
        if identifier in seen:
            raise ContractError(f"Duplicate canonical grid candidate {identifier}")
        seen.add(identifier)
        parameters["candidate_id"] = identifier
    if args.parameter_source == "publication":
        declared = profile.get("search_space", {}).get("candidate_count")
        if declared is not None and int(declared) != len(candidates):
            raise ContractError(
                f"Publication profile declares {declared} candidates but builds {len(candidates)}"
            )
        if args.max_grid_candidates is not None:
            raise ContractError("The publication-backed grid cannot be truncated")
    elif args.max_grid_candidates is not None:
        candidates = candidates[: int(args.max_grid_candidates)]
    return candidates, profile, path


def _grid_root(args: Any, backend: Any) -> Path:
    return (
        Path(args.results_root)
        / CANONICAL_METHOD_LABEL
        / _grid_folder(args.parameter_source)
        / f"{args.percentage}_percent"
    )


def _grid_has_durable_training_artifacts(root: Path) -> bool:
    """Return whether a partial grid contains anything beyond its preflight scaffold."""

    if not root.exists():
        return False
    allowed_root_files = {"grid_context.json", "grid_index.json"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "Logs":
            continue
        if len(relative.parts) == 1 and relative.name in allowed_root_files:
            continue
        return True
    return False


def _labelled_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return fingerprint(
        [(str(row["patient_id"]), int(row["label"])) for row in rows]
    )


def _fold_partition_identities(args: Any, manifest: Mapping[str, Any]) -> list[dict]:
    partitions = []
    for fold_number in range(1, 6):
        training_mapping, validation_mapping = cross_validation_split(
            manifest, fold_number
        )
        training_rows = rows_from_mapping(
            args.dataset_root, training_mapping, require_files=False
        )
        validation_rows = rows_from_mapping(
            args.dataset_root, validation_mapping, require_files=False
        )
        partitions.append(
            {
                "fold": fold_number,
                "training_rows_sha256": _labelled_rows_sha256(training_rows),
                "validation_rows_sha256": _labelled_rows_sha256(validation_rows),
            }
        )
    return partitions


def _load_fixed_parameters(args: Any, backend: Any) -> tuple[dict, dict]:
    profile, path = _profile_for_source(args, backend)
    parameters = profile.get("parameters") or profile.get("fixed_parameters")
    if not isinstance(parameters, dict):
        raise ContractError(f"Fixed profile has no parameter object: {path}")
    validated = backend.build_candidates({"parameters": dict(parameters)})
    if len(validated) != 1:
        raise ContractError(f"Fixed profile did not resolve to exactly one candidate: {path}")
    return dict(validated[0]), {
        "parameter_source": args.parameter_source,
        "profile": str(path),
        "profile_sha256": sha256_file(path),
    }


def _grid_context(
    args: Any,
    backend: Any,
    manifest: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    profile_path: Path,
) -> dict:
    ids = [str(item["candidate_id"]) for item in candidates]
    return {
        "schema_version": 1,
        "model": backend.name,
        "method": CANONICAL_METHOD_LABEL,
        "parameter_source": args.parameter_source,
        "search_type": "publication_grid" if args.parameter_source == "publication" else "grid",
        "profile_type": "publication_adapted" if args.parameter_source == "publication" else "project_grid",
        "publication_search_space_preserved": args.parameter_source == "publication",
        "publication_fixed_recipe_preserved": args.parameter_source == "publication",
        "publication_exact": False,
        "percentage": int(args.percentage),
        "seed": int(args.seed),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "candidate_ids": ids,
        "candidate_count": len(ids),
        "candidate_set_sha256": fingerprint(ids),
        "model_identity": dict(model_identity),
        "fold_partitions": _fold_partition_identities(args, manifest),
    }


def _partition_for_fold(context: Mapping[str, Any], fold_number: int) -> dict:
    partitions = context.get("fold_partitions")
    if not isinstance(partitions, list) or len(partitions) != 5:
        raise ContractError("Experiment context has no exact five-fold partition identity")
    matches = [
        item
        for item in partitions
        if isinstance(item, dict) and item.get("fold") == int(fold_number)
    ]
    if len(matches) != 1:
        raise ContractError(f"Experiment context has no unique fold {fold_number}")
    return matches[0]


def _validate_grid_fold_identity(
    fold_root: Path,
    *,
    grid_context: Mapping[str, Any],
    parameters: Mapping[str, Any],
    candidate: str,
    fold_number: int,
) -> dict:
    seal = validate_completion(fold_root, "completion.json")
    state = read_json(fold_root / "training_state.json")
    state_context = state.get("context")
    if not isinstance(state_context, dict):
        raise ContractError(f"Grid fold has malformed training context: {fold_root}")
    if any(state_context.get(key) != value for key, value in grid_context.items()):
        raise ContractError(f"Grid fold belongs to a different root context: {fold_root}")
    if (
        state_context.get("fold") != int(fold_number)
        or state_context.get("candidate_id") != candidate
    ):
        raise ContractError(f"Grid fold number/candidate identity changed: {fold_root}")
    partition = _partition_for_fold(grid_context, fold_number)
    if (
        state.get("training_rows_sha256") != partition["training_rows_sha256"]
        or state.get("validation_rows_sha256")
        != partition["validation_rows_sha256"]
    ):
        raise ContractError(f"Grid fold patient partition changed: {fold_root}")
    if state.get("feature_identity") != grid_context.get("model_identity"):
        raise ContractError(f"Grid fold feature/model identity changed: {fold_root}")
    if read_json(fold_root / "parameters.json") != dict(parameters):
        raise ContractError(f"Grid fold parameter document changed: {fold_root}")
    metrics = read_json(fold_root / "metrics_summary.json")
    metric_status = metrics.get("candidate_status")
    if metric_status is not None and metric_status != state.get("status"):
        raise ContractError(f"Grid fold failure/completion status changed: {fold_root}")
    return seal


def _load_completed_grid(
    grid_root: Path, *, expected_context: Mapping[str, Any] | None = None
) -> dict:
    seal = validate_completion(grid_root, "grid_completion.json")
    context = read_json(grid_root / "grid_context.json")
    if expected_context is not None and context != dict(expected_context):
        raise ContractError(
            f"Completed grid identity does not match the current split/model/profile: {grid_root}"
        )
    if seal.get("context", {}).get("grid_context_sha256") != fingerprint(context):
        raise ContractError(f"Grid completion seal has an incompatible context: {grid_root}")
    candidate_ids = context.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) != int(context.get("candidate_count", -1))
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(
            not isinstance(identifier, str)
            or len(identifier) != 12
            or any(character not in "0123456789abcdef" for character in identifier)
            for identifier in candidate_ids
        )
    ):
        raise ContractError(f"Completed grid has an unsafe candidate set: {grid_root}")
    index = read_json(grid_root / "grid_index.json")
    indexed_ids = [item.get("candidate_id") for item in index.get("candidates", [])]
    if indexed_ids != candidate_ids or int(index.get("candidate_count", -1)) != len(
        candidate_ids
    ):
        raise ContractError(f"Grid index differs from its sealed context: {grid_root}")
    summary = read_json(grid_root / "grid_summary.json")
    ranking = summary.get("ranking")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(candidate_ids)
        or not all(isinstance(item, dict) for item in ranking)
        or {item.get("candidate_id") for item in ranking} != set(candidate_ids)
    ):
        raise ContractError(f"Grid summary has an incomplete candidate set: {grid_root}")
    ranked = {item["candidate_id"]: item for item in ranking}
    for identifier in candidate_ids:
        candidate_root = grid_root / f"candidate_{identifier}"
        candidate_summary = read_json(candidate_root / "candidate_summary.json")
        expected_candidate_summary = {
            key: value for key, value in ranked[identifier].items() if key != "rank"
        }
        if candidate_summary != expected_candidate_summary:
            raise ContractError(
                f"Candidate summary differs from the sealed grid ranking: {candidate_root}"
            )
        actual_fold_metrics = []
        for fold_number in range(1, 6):
            fold_root = candidate_root / f"fold_{fold_number:02d}"
            _validate_grid_fold_identity(
                fold_root,
                grid_context=context,
                parameters=candidate_summary["parameters"],
                candidate=identifier,
                fold_number=fold_number,
            )
            actual_fold_metrics.append(read_json(fold_root / "metrics_summary.json"))
        if candidate_summary.get("fold_metrics") != actual_fold_metrics:
            raise ContractError(
                f"Candidate ranking metrics differ from its sealed fold metrics: {candidate_root}"
            )
    selection = read_json(grid_root / "selected_parameters.json")
    if selection.get("candidate_id") is None or not isinstance(selection.get("parameters"), dict):
        raise ContractError("Completed grid selection is malformed")
    if selection.get("grid_context_sha256") != fingerprint(context):
        raise ContractError(f"Grid selection is not bound to its grid context: {grid_root}")
    if selection["candidate_id"] not in context.get("candidate_ids", []):
        raise ContractError(f"Selected candidate is absent from the sealed grid: {grid_root}")
    if ranked[selection["candidate_id"]].get("status", "complete") != "complete":
        raise ContractError(f"Grid selected a failed candidate: {grid_root}")
    return selection


def run_grid(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    candidates, profile, profile_path = _candidate_documents(args, backend)
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, development_rows)
    model_identity = backend.feature_identity(args)
    root = _grid_root(args, backend)
    context = _grid_context(
        args, backend, manifest, model_identity, candidates, profile_path
    )
    context = _effective_grid_context(args, root, context)
    if (root / "grid_completion.json").is_file():
        if not args.resume:
            raise ContractError(f"Completed grid exists and resume is disabled: {root}")
        return _load_completed_grid(root, expected_context=context)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ContractError(f"Partial grid exists and resume is disabled: {root}")
    context_path = root / "grid_context.json"
    if (
        args.resume
        and root.exists()
        and any(root.iterdir())
        and not context_path.is_file()
    ):
        raise ContractError(
            f"Partial grid artifacts exist without their root context: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    refreshed_pretraining_scaffold = False
    if context_path.is_file() and read_json(context_path) != context:
        if _grid_has_durable_training_artifacts(root):
            raise ContractError(
                f"Partial grid context changed after training artifacts were created: {root}"
            )
        atomic_write_json(context_path, context)
        refreshed_pretraining_scaffold = True
    else:
        ensure_same_or_write(context_path, context)
    logger = setup_logger(root / "Logs", f"{backend.name}.publication_grid")
    try:
        index = {
            "schema_version": 1,
            "candidate_count": len(candidates),
            "candidates": [
                {"candidate_id": item["candidate_id"], "parameters": dict(item)}
                for item in candidates
            ],
        }
        if refreshed_pretraining_scaffold:
            atomic_write_json(root / "grid_index.json", index)
            logger.info(
                "Refreshed a pre-training-only grid scaffold after implementation/profile identity changed"
            )
        else:
            ensure_same_or_write(root / "grid_index.json", index)
        logger.info("Starting %s-candidate grid at %s%%", len(candidates), args.percentage)

        for fold_number in range(1, 6):
            train_mapping, validation_mapping = cross_validation_split(manifest, fold_number)
            train_rows = rows_from_mapping(args.dataset_root, train_mapping, require_files=True)
            validation_rows = rows_from_mapping(
                args.dataset_root, validation_mapping, require_files=True
            )
            pending: list[dict] = []
            for parameters in candidates:
                candidate_root = root / f"candidate_{parameters['candidate_id']}"
                fold_root = candidate_root / f"fold_{fold_number:02d}"
                completion_path = fold_root / "completion.json"
                if completion_path.is_file() and args.resume:
                    validate_completion(fold_root, "completion.json")
                else:
                    if completion_path.is_file() and not args.resume:
                        raise ContractError(f"Grid fold exists and resume is disabled: {fold_root}")
                    pending.append(parameters)
            if pending:
                logger.info("Fold %s: fitting %s pending candidates", fold_number, len(pending))
                backend.fit_grid_fold(
                    candidates,
                    train_rows,
                    validation_rows,
                    root,
                    fold_number,
                    args,
                    {**context, "pending_candidate_ids": [item["candidate_id"] for item in pending]},
                    logger,
                )
            for parameters in candidates:
                _validate_grid_fold_identity(
                    root
                    / f"candidate_{parameters['candidate_id']}"
                    / f"fold_{fold_number:02d}",
                    grid_context=context,
                    parameters=parameters,
                    candidate=str(parameters["candidate_id"]),
                    fold_number=fold_number,
                )

        summaries: list[dict] = []
        for parameters in candidates:
            identifier = parameters["candidate_id"]
            fold_metrics = [
                read_json(
                    root
                    / f"candidate_{identifier}"
                    / f"fold_{fold:02d}"
                    / "metrics_summary.json"
                )
                for fold in range(1, 6)
            ]
            summary = {
                "candidate_id": identifier,
                "parameters": dict(parameters),
                "fold_metrics": fold_metrics,
                **backend.selection_summary(parameters, fold_metrics),
            }
            summary.setdefault("status", "complete")
            summary.setdefault("eligible_for_selection", True)
            summary.setdefault("fold_failures", [])
            atomic_write_json(root / f"candidate_{identifier}" / "candidate_summary.json", summary)
            summaries.append(summary)
        ranking = sorted(summaries, key=backend.ranking_key)
        eligible = [
            item
            for item in ranking
            if item.get("status") == "complete"
            and item.get("eligible_for_selection") is True
        ]
        if not eligible:
            raise ContractError("Every grid candidate failed; no parameter set is selectable")
        for rank, item in enumerate(eligible, start=1):
            item["rank"] = rank
        for item in ranking:
            if item not in eligible:
                item["rank"] = None
        best = eligible[0]
        grid_summary = {
            "schema_version": 1,
            "status": "complete",
            "candidate_count": len(ranking),
            "successful_candidate_count": len(eligible),
            "failed_candidate_count": len(ranking) - len(eligible),
            "selection_metric": backend.selection_metric,
            "ranking": ranking,
        }
        atomic_write_json(root / "grid_summary.json", grid_summary)
        csv_rows = [backend.summary_csv_row(item) for item in ranking]
        atomic_write_csv(
            root / "grid_summary_by_selection_score.csv",
            csv_rows,
            list(csv_rows[0]),
        )
        selected = {
            "schema_version": 1,
            "selected_at": utc_now(),
            "candidate_id": best["candidate_id"],
            "parameters": best["parameters"],
            "rank": 1,
            "selection_metric": backend.selection_metric,
            "selection_value": best["selection_value"],
            "parameter_source": args.parameter_source,
            "search_type": context["search_type"],
            "publication_backed_grid": args.parameter_source == "publication",
            "grid_context_sha256": fingerprint(context),
        }
        atomic_write_json(root / "best_hyperparameters.json", selected)
        atomic_write_json(root / "selected_parameters.json", selected)
        seal_paths = [
            root / "grid_context.json",
            root / "grid_index.json",
            root / "grid_summary.json",
            root / "grid_summary_by_selection_score.csv",
            root / "best_hyperparameters.json",
            root / "selected_parameters.json",
        ]
        atomic_write_json(
            root / "grid_completion.json",
            completion_seal(
                seal_paths,
                context={
                    "model": backend.name,
                    "percentage": args.percentage,
                    "candidate_id": best["candidate_id"],
                    "candidate_count": len(candidates),
                    "grid_context_sha256": fingerprint(context),
                },
            ),
        )
        logger.info("Selected publication candidate %s", best["candidate_id"])
        return selected
    finally:
        close_logger(logger)


def _selected_parameters(
    args: Any,
    backend: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
    model_identity: Mapping[str, Any] | None = None,
) -> tuple[dict, dict]:
    if args.parameter_source in ("publication", "grid"):
        root = _grid_root(args, backend)
        expected_context = None
        if manifest is not None and model_identity is not None:
            candidates, _, profile_path = _candidate_documents(args, backend)
            expected_context = _grid_context(
                args,
                backend,
                manifest,
                model_identity,
                candidates,
                profile_path,
            )
            expected_context = _effective_grid_context(
                args, root, expected_context
            )
        selection = _load_completed_grid(root, expected_context=expected_context)
        grid_context = read_json(root / "grid_context.json")
        provenance = {
            "parameter_source": args.parameter_source,
            "candidate_id": selection["candidate_id"],
            "profile": grid_context["profile"],
            "profile_sha256": grid_context["profile_sha256"],
            "candidate_set_sha256": grid_context["candidate_set_sha256"],
            "selection_file": str((root / "selected_parameters.json").resolve()),
            "selection_file_sha256": sha256_file(root / "selected_parameters.json"),
            "grid_root": str(root.resolve()),
        }
        if grid_context.get("method") == LEGACY_METHOD_LABEL:
            provenance.update(
                {
                    "selection_file": _legacy_alias(
                        root / "selected_parameters.json"
                    ),
                    "grid_root": _legacy_alias(root),
                    "_resolved_grid_root": str(root.resolve()),
                    "_model_identity_override": dict(grid_context["model_identity"]),
                    "_method_label_override": LEGACY_METHOD_LABEL,
                }
            )
        return dict(selection["parameters"]), provenance
    return _load_fixed_parameters(args, backend)


def _cv_provenance_identity(provenance: Mapping[str, Any]) -> dict:
    """Return the stable, percentage-independent parameter-source identity."""
    identity = {"parameter_source": provenance.get("parameter_source")}
    if provenance.get("profile_sha256") is not None:
        identity["profile_sha256"] = provenance["profile_sha256"]
    if provenance.get("candidate_id") is not None:
        identity["candidate_id"] = provenance["candidate_id"]
    if not isinstance(identity["parameter_source"], str):
        raise ContractError("Parameter provenance has no valid source")
    return identity


def _register_cv(
    results_root: Path,
    backend: Any,
    parameters: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    seed: int,
    method_label: str = CANONICAL_METHOD_LABEL,
    create: bool = True,
) -> tuple[str, Path]:
    cv_base = results_root / CANONICAL_METHOD_LABEL / "CV"
    registry_path = cv_base / "cv_runs.json"
    provenance_identity = _cv_provenance_identity(provenance)
    if registry_path.exists():
        registry = read_json(registry_path)
    else:
        if not create:
            raise ContractError(
                f"No CV registry exists for the requested test configuration: {registry_path}"
            )
        registry = {"schema_version": 1, "model": backend.name, "runs": []}
    if (
        registry.get("schema_version") != 1
        or registry.get("model") != backend.name
        or not isinstance(registry.get("runs"), list)
    ):
        raise ContractError(f"CV registry has an incompatible schema/model: {registry_path}")
    observed_ids: set[str] = set()
    observed_identities: set[str] = set()
    for run in registry["runs"]:
        if not isinstance(run, dict):
            raise ContractError(f"CV registry contains a malformed run: {registry_path}")
        cv_id = str(run.get("cv_id", ""))
        if not cv_id.startswith("CV_") or not cv_id[3:].isdigit():
            raise ContractError(f"CV registry contains an unsafe CV id: {cv_id!r}")
        run_identity = str(run.get("identity_sha256", ""))
        if len(run_identity) != 64:
            raise ContractError(f"CV registry contains an invalid identity: {cv_id}")
        if not isinstance(run.get("parameter_provenance_identity"), dict):
            raise ContractError(
                f"CV registry contains malformed parameter provenance: {cv_id}"
            )
        if cv_id in observed_ids or run_identity in observed_identities:
            raise ContractError(f"CV registry contains duplicate runs: {registry_path}")
        observed_ids.add(cv_id)
        observed_identities.add(run_identity)
    identity = {
        "method": method_label,
        "parameters": dict(parameters),
        "model_identity": dict(model_identity),
        "seed": int(seed),
        "parameter_provenance_identity": provenance_identity,
    }
    identity_sha = fingerprint(identity)

    def ensure_parameters_document(cv_id: str) -> Path:
        cv_root = cv_base / cv_id
        path = cv_root / "cv_parameters.json"
        expected_core = {
            "schema_version": 1,
            "cv_id": cv_id,
            "identity_sha256": identity_sha,
            "parameters": dict(parameters),
            "model_identity": dict(model_identity),
            "seed": int(seed),
            "parameter_provenance_identity": provenance_identity,
        }
        if path.is_file():
            observed = read_json(path)
            for key, value in expected_core.items():
                if observed.get(key) != value:
                    raise ContractError(f"CV parameter identity changed: {path}")
        else:
            if not create:
                raise ContractError(
                    f"CV parameter identity document is missing: {path}"
                )
            cv_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, expected_core)
        return cv_root

    for run in registry["runs"]:
        if run["identity_sha256"] == identity_sha:
            if (
                run.get("parameters") != dict(parameters)
                or run.get("model_identity") != dict(model_identity)
                or run.get("seed") != int(seed)
                or run.get("parameter_provenance_identity") != provenance_identity
            ):
                raise ContractError(
                    f"CV registry identity payload changed: {registry_path}"
                )
            cv_id = run["cv_id"]
            return cv_id, ensure_parameters_document(cv_id)
    if not create:
        raise ContractError(
            "No completed CV identity matches the requested parameters and feature cache"
        )
    cv_base.mkdir(parents=True, exist_ok=True)
    numbers = [int(item["cv_id"].split("_")[1]) for item in registry["runs"]]
    cv_id = f"CV_{max(numbers, default=0) + 1}"
    registry["runs"].append(
        {
            "cv_id": cv_id,
            "identity_sha256": identity_sha,
            "parameters": dict(parameters),
            "model_identity": dict(model_identity),
            "seed": int(seed),
            "parameter_provenance_identity": provenance_identity,
            "created_at": utc_now(),
        }
    )
    atomic_write_json(registry_path, registry)
    cv_root = ensure_parameters_document(cv_id)
    return cv_id, cv_root


def _reuse_grid_fold(
    source: Path,
    destination: Path,
    backend: Any,
    candidate: str,
    grid_root: Path,
    method_label: str,
) -> None:
    validate_completion(source, "completion.json")
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        backend.artifact_filename,
        "parameters.json",
        "metrics_summary.json",
        "validation_predictions.json",
        "validation_probabilities.csv",
        "training_state.json",
        "completion.json",
    ]
    materialized = {}
    for filename in files:
        path = source / filename
        if path.is_file():
            materialized[filename] = materialize_file(path, destination / filename)
    atomic_write_json(
        destination / "grid_reuse.json",
        {
            "schema_version": 1,
            "candidate_id": candidate,
            "source_grid_root": _legacy_alias(grid_root)
            if method_label == LEGACY_METHOD_LABEL
            else str(grid_root.resolve()),
            "source_fold": _legacy_alias(source)
            if method_label == LEGACY_METHOD_LABEL
            else str(source.resolve()),
            "destination_fold": _legacy_alias(destination)
            if method_label == LEGACY_METHOD_LABEL
            else str(destination.resolve()),
            "reused_without_training": True,
            "artifacts": materialized,
        },
    )


def _cv_completion_context(
    *,
    args: Any,
    backend: Any,
    manifest: Mapping[str, Any],
    cv_id: str,
    parameters: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    development_rows: Sequence[Mapping[str, Any]],
    selected_candidate: str | None,
    cv_parameters_path: Path,
    parameter_provenance: Mapping[str, Any],
    method_label: str = CANONICAL_METHOD_LABEL,
) -> dict:
    return {
        "model": backend.name,
        "method": method_label,
        "cv_id": cv_id,
        "percentage": int(args.percentage),
        "seed": int(args.seed),
        "split_manifest_sha256": manifest["_sha256"],
        "development_rows_sha256": rows_fingerprint(development_rows),
        "development_row_count": len(development_rows),
        "development_training_rows_sha256": _labelled_rows_sha256(
            development_rows
        ),
        "fold_partitions": _fold_partition_identities(args, manifest),
        "parameters": dict(parameters),
        "parameters_sha256": fingerprint(parameters),
        "model_identity": dict(model_identity),
        "model_identity_sha256": fingerprint(model_identity),
        "selected_candidate_id": selected_candidate,
        "parameter_provenance": dict(parameter_provenance),
        "cv_parameters_sha256": sha256_file(cv_parameters_path),
    }


def _validate_cv_fold_identity(
    fold_root: Path,
    *,
    expected_context: Mapping[str, Any],
    fold_number: int,
) -> None:
    candidate = expected_context.get("selected_candidate_id")
    reuse_path = fold_root / "grid_reuse.json"
    if candidate is not None:
        if not reuse_path.is_file():
            raise ContractError(f"Grid-backed CV fold lacks its reuse record: {fold_root}")
        reuse = read_json(reuse_path)
        provenance = expected_context.get("parameter_provenance", {})
        source_grid_root = Path(str(provenance.get("grid_root", ""))).resolve()
        expected_source = (
            source_grid_root
            / f"candidate_{candidate}"
            / f"fold_{fold_number:02d}"
        ).resolve()
        if (
            reuse.get("candidate_id") != candidate
            or reuse.get("reused_without_training") is not True
            or Path(str(reuse.get("source_grid_root", ""))).resolve()
            != source_grid_root
            or Path(str(reuse.get("source_fold", ""))).resolve() != expected_source
            or Path(str(reuse.get("destination_fold", ""))).resolve()
            != Path(
                _legacy_alias(fold_root)
                if expected_context.get("method") == LEGACY_METHOD_LABEL
                else str(fold_root.resolve())
            ).resolve()
        ):
            raise ContractError(f"Grid-backed CV reuse identity changed: {fold_root}")
        resolved_grid_root = _resolve_migrated_path(source_grid_root)
        grid_context = read_json(resolved_grid_root / "grid_context.json")
        _validate_grid_fold_identity(
            fold_root,
            grid_context=grid_context,
            parameters=expected_context["parameters"],
            candidate=str(candidate),
            fold_number=fold_number,
        )
        return
    if reuse_path.exists():
        raise ContractError(f"Fixed-profile CV fold has an unexpected grid reuse record: {fold_root}")
    validate_completion(fold_root, "completion.json")
    state = read_json(fold_root / "training_state.json")
    state_context = state.get("context")
    expected_state_context = {
        "cv_id": expected_context["cv_id"],
        "percentage": expected_context["percentage"],
        "fold": int(fold_number),
        "model_identity": expected_context["model_identity"],
    }
    if not isinstance(state_context, dict) or any(
        state_context.get(key) != value
        for key, value in expected_state_context.items()
    ):
        raise ContractError(f"CV fold belongs to a different run context: {fold_root}")
    partition = _partition_for_fold(expected_context, fold_number)
    if (
        state.get("training_rows_sha256") != partition["training_rows_sha256"]
        or state.get("validation_rows_sha256")
        != partition["validation_rows_sha256"]
    ):
        raise ContractError(f"CV fold patient partition changed: {fold_root}")
    if state.get("feature_identity") != expected_context["model_identity"]:
        raise ContractError(f"CV fold feature/model identity changed: {fold_root}")
    if read_json(fold_root / "parameters.json") != expected_context["parameters"]:
        raise ContractError(f"CV fold parameter document changed: {fold_root}")


def _validate_refit_identity(
    refit_root: Path, *, expected_context: Mapping[str, Any]
) -> None:
    validate_completion(refit_root, "completion.json")
    state = read_json(refit_root / "training_state.json")
    state_context = state.get("context")
    expected_state_context = {
        "cv_id": expected_context["cv_id"],
        "percentage": expected_context["percentage"],
        "model_identity": expected_context["model_identity"],
    }
    if not isinstance(state_context, dict) or any(
        state_context.get(key) != value
        for key, value in expected_state_context.items()
    ):
        raise ContractError(f"CV refit belongs to a different run context: {refit_root}")
    if (
        state.get("row_count") != expected_context["development_row_count"]
        or state.get("rows_sha256")
        != expected_context["development_training_rows_sha256"]
    ):
        raise ContractError(f"CV refit patient partition changed: {refit_root}")
    if state.get("feature_identity") != expected_context["model_identity"]:
        raise ContractError(f"CV refit feature/model identity changed: {refit_root}")
    if read_json(refit_root / "parameters.json") != expected_context["parameters"]:
        raise ContractError(f"CV refit parameter document changed: {refit_root}")


def _validate_cv_completion(
    cv_percentage_root: Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> dict:
    seal = validate_tree_completion(cv_percentage_root, "cv_completion.json")
    if expected_context is not None and seal.get("context") != dict(expected_context):
        raise ContractError(
            f"Completed CV identity does not match the current split/model/seed: "
            f"{cv_percentage_root}"
        )
    # The outer seal intentionally records the nested completion documents so
    # that the CV tree remains compact.  Validate every nested seal as well;
    # otherwise a changed metrics/predictions/training-state file could remain
    # undetected while its unchanged completion document still satisfies the
    # outer tree hash.
    for fold_number in range(1, 6):
        if expected_context is None:
            validate_completion(
                cv_percentage_root / f"fold_{fold_number:02d}", "completion.json"
            )
        else:
            _validate_cv_fold_identity(
                cv_percentage_root / f"fold_{fold_number:02d}",
                expected_context=expected_context,
                fold_number=fold_number,
            )
    if expected_context is None:
        validate_completion(
            cv_percentage_root / "refit_full_development", "completion.json"
        )
    else:
        _validate_refit_identity(
            cv_percentage_root / "refit_full_development",
            expected_context=expected_context,
        )
    return seal


def run_cv(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    all_development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, all_development_rows)
    live_model_identity = backend.feature_identity(args)
    parameters, provenance = _selected_parameters(
        args,
        backend,
        manifest=manifest,
        model_identity=live_model_identity,
    )
    resolved_grid_root = provenance.pop("_resolved_grid_root", None)
    model_identity = provenance.pop(
        "_model_identity_override", live_model_identity
    )
    method_label = provenance.pop(
        "_method_label_override", CANONICAL_METHOD_LABEL
    )
    cv_id, cv_root = _register_cv(
        Path(args.results_root),
        backend,
        parameters,
        model_identity,
        provenance,
        seed=args.seed,
        method_label=method_label,
    )
    percentage_root = cv_root / f"{args.percentage}_percent"
    completion_path = percentage_root / "cv_completion.json"
    selected_candidate = provenance.get("candidate_id")
    expected_completion_context = _cv_completion_context(
        args=args,
        backend=backend,
        manifest=manifest,
        cv_id=cv_id,
        parameters=parameters,
        model_identity=model_identity,
        development_rows=all_development_rows,
        selected_candidate=selected_candidate,
        cv_parameters_path=cv_root / "cv_parameters.json",
        parameter_provenance=provenance,
        method_label=method_label,
    )
    if completion_path.is_file():
        if not args.resume:
            raise ContractError(f"Completed CV exists and resume is disabled: {percentage_root}")
        _validate_cv_completion(
            percentage_root, expected_context=expected_completion_context
        )
        return {"cv_id": cv_id, "cv_root": str(cv_root), "status": "reused"}
    if percentage_root.exists() and any(percentage_root.iterdir()) and not args.resume:
        raise ContractError(f"Partial CV exists and resume is disabled: {percentage_root}")
    run_context_path = percentage_root / "cv_run_context.json"
    if (
        args.resume
        and percentage_root.exists()
        and any(percentage_root.iterdir())
        and not run_context_path.is_file()
    ):
        raise ContractError(
            f"Partial CV artifacts exist without their root context: {percentage_root}"
        )
    percentage_root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(
        run_context_path,
        {
            "schema_version": 1,
            "context": expected_completion_context,
        },
    )
    logger = setup_logger(percentage_root / "Logs", f"{backend.name}.cv")
    try:
        fold_rows: list[dict] = []
        artifacts: list[Path] = [run_context_path]
        source_grid_root = (
            Path(resolved_grid_root)
            if resolved_grid_root
            else Path(provenance["grid_root"])
            if provenance.get("grid_root")
            else None
        )
        for fold_number in range(1, 6):
            fold_root = percentage_root / f"fold_{fold_number:02d}"
            if (fold_root / "completion.json").is_file():
                # A resumed fixed-profile CV must be a genuine no-op for folds
                # that were already completed and sealed.  Publication-backed
                # folds likewise retain their original grid-reuse record.
                validate_completion(fold_root, "completion.json")
            else:
                if selected_candidate and source_grid_root is not None:
                    source = (
                        source_grid_root
                        / f"candidate_{selected_candidate}"
                        / f"fold_{fold_number:02d}"
                    )
                    _reuse_grid_fold(
                        source,
                        fold_root,
                        backend,
                        selected_candidate,
                        source_grid_root,
                        method_label,
                    )
                else:
                    train_mapping, validation_mapping = cross_validation_split(
                        manifest, fold_number
                    )
                    backend.fit_one(
                        parameters,
                        rows_from_mapping(
                            args.dataset_root, train_mapping, require_files=True
                        ),
                        rows_from_mapping(
                            args.dataset_root, validation_mapping, require_files=True
                        ),
                        fold_root,
                        args,
                        {
                            "cv_id": cv_id,
                            "percentage": args.percentage,
                            "fold": fold_number,
                            "model_identity": model_identity,
                            "method": method_label,
                        },
                        logger,
                    )
            _validate_cv_fold_identity(
                fold_root,
                expected_context=expected_completion_context,
                fold_number=fold_number,
            )
            metrics = read_json(fold_root / "metrics_summary.json")
            fold_rows.append(
                {
                    "fold": fold_number,
                    "selection_score": fold_selection_score(metrics),
                    "f1_macro": metrics["f1_macro"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "average_precision_macro": metrics.get("average_precision_macro"),
                }
            )
            artifacts.extend([fold_root / backend.artifact_filename, fold_root / "completion.json"])
            if (fold_root / "grid_reuse.json").is_file():
                artifacts.append(fold_root / "grid_reuse.json")
        atomic_write_csv(
            percentage_root / "cv_fold_summary.csv",
            fold_rows,
            list(fold_rows[0]),
        )
        best = sorted(fold_rows, key=lambda item: (-item["selection_score"], item["fold"]))[0]
        atomic_write_json(
            percentage_root / "best_fold_model.json",
            {
                "schema_version": 1,
                "fold": best["fold"],
                "selection_score": best["selection_score"],
                "artifact": str(
                    (percentage_root / f"fold_{best['fold']:02d}" / backend.artifact_filename).resolve()
                ),
            },
        )
        refit_root = percentage_root / "refit_full_development"
        if not (refit_root / "completion.json").is_file():
            backend.refit(
                parameters,
                all_development_rows,
                refit_root,
                args,
                {
                    "cv_id": cv_id,
                    "percentage": args.percentage,
                    "model_identity": model_identity,
                    "method": method_label,
                },
                logger,
            )
        _validate_refit_identity(
            refit_root, expected_context=expected_completion_context
        )
        artifacts.extend(
            [
                percentage_root / "cv_fold_summary.csv",
                percentage_root / "best_fold_model.json",
                refit_root / backend.artifact_filename,
                refit_root / "completion.json",
            ]
        )
        seal = tree_completion_seal(
            percentage_root,
            artifacts,
            context=expected_completion_context,
        )
        atomic_write_json(completion_path, seal)
        logger.info("Completed %s-fold CV and full-development refit", 5)
        return {"cv_id": cv_id, "cv_root": str(cv_root), "status": "complete"}
    finally:
        close_logger(logger)


def _locate_cv(
    args: Any, backend: Any, manifest: Mapping[str, Any]
) -> tuple[str, Path, dict, dict, str]:
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, development_rows)
    live_model_identity = backend.feature_identity(args)
    parameters, provenance = _selected_parameters(
        args,
        backend,
        manifest=manifest,
        model_identity=live_model_identity,
    )
    provenance.pop("_resolved_grid_root", None)
    model_identity = provenance.pop(
        "_model_identity_override", live_model_identity
    )
    method_label = provenance.pop(
        "_method_label_override", CANONICAL_METHOD_LABEL
    )
    cv_id, cv_root = _register_cv(
        Path(args.results_root),
        backend,
        parameters,
        model_identity,
        provenance,
        seed=args.seed,
        method_label=method_label,
        create=False,
    )
    percentage_root = cv_root / f"{args.percentage}_percent"
    expected_context = _cv_completion_context(
        args=args,
        backend=backend,
        manifest=manifest,
        cv_id=cv_id,
        parameters=parameters,
        model_identity=model_identity,
        development_rows=development_rows,
        selected_candidate=provenance.get("candidate_id"),
        cv_parameters_path=cv_root / "cv_parameters.json",
        parameter_provenance=provenance,
        method_label=method_label,
    )
    _validate_cv_completion(percentage_root, expected_context=expected_context)
    return cv_id, percentage_root, parameters, model_identity, method_label


def _model_source_artifacts(
    args: Any, backend: Any, cv_percentage_root: Path
) -> list[Path]:
    if args.model_source == "refit_full_development":
        artifacts = [
            cv_percentage_root / "refit_full_development" / backend.artifact_filename
        ]
    elif args.model_source == "best_cv_fold":
        best = read_json(cv_percentage_root / "best_fold_model.json")
        fold = best.get("fold")
        if not isinstance(fold, int) or fold not in range(1, 6):
            raise ContractError("best_fold_model.json has an invalid fold")
        artifacts = [
            cv_percentage_root
            / f"fold_{fold:02d}"
            / backend.artifact_filename
        ]
    elif args.model_source == "cv_ensemble":
        artifacts = [
            cv_percentage_root / f"fold_{fold:02d}" / backend.artifact_filename
            for fold in range(1, 6)
        ]
    else:
        raise ContractError(f"Unsupported model source: {args.model_source}")
    hashes = [sha256_file(path) for path in artifacts]
    if args.model_source == "cv_ensemble" and len(set(hashes)) != len(hashes):
        raise ContractError("CV ensemble contains duplicate classifier artifacts")
    return artifacts


def _test_selection_document(
    *,
    backend: Any,
    cv_id: str,
    args: Any,
    artifacts: Sequence[Path],
    parameters: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    heldout_rows: Sequence[Mapping[str, Any]],
    method_label: str = CANONICAL_METHOD_LABEL,
) -> dict:
    return {
        "schema_version": 1,
        "model": backend.name,
        "method": method_label,
        "cv_id": cv_id,
        "percentage": args.percentage,
        "model_source": args.model_source,
        "classifier_artifacts": [
            {
                "path": _legacy_alias(path)
                if method_label == LEGACY_METHOD_LABEL
                else str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
        "parameters": dict(parameters),
        "model_identity": dict(model_identity),
        "heldout_rows_sha256": rows_fingerprint(heldout_rows),
    }


def _validate_test_completion(
    test_root: Path,
    *,
    expected_selection: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict:
    seal = validate_completion(test_root, "test_completion.json")
    selection = read_json(test_root / "model_selection.json")
    if selection != dict(expected_selection):
        raise ContractError(f"Held-out result belongs to a different model source: {test_root}")
    expected_context = {
        "cv_id": expected_selection["cv_id"],
        "model_source": expected_selection["model_source"],
        "percentage": expected_selection["percentage"],
        "selection_sha256": fingerprint(selection),
    }
    if seal.get("context") != expected_context:
        raise ContractError(f"Held-out completion context changed: {test_root}")
    predictions_path = test_root / "test_patient_predictions.json"
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    expected_truth = [
        {
            "patient_id": str(row["patient_id"]),
            "true_index": int(row["label"]),
            "true_class": str(row["class_name"]),
        }
        for row in expected_rows
    ]
    if not isinstance(predictions, list) or [
        {
            "patient_id": item.get("patient_id"),
            "true_index": item.get("true_index"),
            "true_class": item.get("true_class"),
        }
        for item in predictions
        if isinstance(item, dict)
    ] != expected_truth or len(predictions) != len(expected_truth):
        raise ContractError(f"Held-out prediction patient/order identity changed: {test_root}")
    return seal


def run_test(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    cv_id, cv_percentage_root, parameters, model_identity, method_label = _locate_cv(
        args, backend, manifest
    )
    test_root = (
        Path(args.results_root)
        / CANONICAL_METHOD_LABEL
        / "Test"
        / cv_id
        / args.model_source
        / f"{args.percentage}_percent"
    )
    artifacts = _model_source_artifacts(args, backend, cv_percentage_root)
    expected_test_rows = rows_from_mapping(
        args.dataset_root, heldout_mapping(manifest), require_files=False
    )
    expected_selection = _test_selection_document(
        backend=backend,
        cv_id=cv_id,
        args=args,
        artifacts=artifacts,
        parameters=parameters,
        model_identity=model_identity,
        heldout_rows=expected_test_rows,
        method_label=method_label,
    )
    if (test_root / "test_completion.json").is_file() and not args.force_test:
        _validate_test_completion(
            test_root,
            expected_selection=expected_selection,
            expected_rows=expected_test_rows,
        )
        return {"cv_id": cv_id, "test_root": str(test_root), "status": "reused"}
    if test_root.exists() and any(test_root.iterdir()) and not args.force_test:
        raise ContractError(
            f"Partial held-out output exists; use --force_test to replace it: {test_root}"
        )
    # Held-out rows are deliberately materialized only after the exact CV
    # identity and its completion seal have been validated above.  Finish all
    # read-only preflight checks before creating the held-out output directory.
    test_rows = rows_from_mapping(
        args.dataset_root, heldout_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, test_rows)
    if rows_fingerprint(test_rows) != rows_fingerprint(expected_test_rows):
        raise ContractError("Held-out row identity changed between preflight and execution")
    test_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(test_root / "Logs", f"{backend.name}.test")
    try:
        logger.info(
            "Evaluating %s held-out rows from model source %s",
            len(test_rows),
            args.model_source,
        )
        probabilities = [
            backend.predict(
                path,
                test_rows,
                args,
                {"model_identity": model_identity, "method": method_label},
            )
            for path in artifacts
        ]
        combined = np.mean(np.stack(probabilities, axis=0), axis=0)
        combined = combined / combined.sum(axis=1, keepdims=True)
        truth = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
        predicted = combined.argmax(axis=1)
        metrics = compute_metrics(truth, predicted, combined, CLASS_NAMES)
        json_rows, csv_rows = prediction_documents(test_rows, combined, CLASS_NAMES)
        atomic_write_json(test_root / "metrics_summary.json", metrics)
        atomic_write_json(test_root / "test_patient_predictions.json", json_rows)
        atomic_write_csv(test_root / "test_probabilities.csv", csv_rows, list(csv_rows[0]))
        selection = expected_selection
        atomic_write_json(test_root / "model_selection.json", selection)
        seal_paths = [
            test_root / "metrics_summary.json",
            test_root / "test_patient_predictions.json",
            test_root / "test_probabilities.csv",
            test_root / "model_selection.json",
        ]
        atomic_write_json(
            test_root / "test_completion.json",
            completion_seal(
                seal_paths,
                context={
                    "cv_id": cv_id,
                    "model_source": args.model_source,
                    "percentage": args.percentage,
                    "selection_sha256": fingerprint(selection),
                },
            ),
        )
        logger.info("Held-out evaluation completed and sealed")
        return {"cv_id": cv_id, "test_root": str(test_root), "status": "complete"}
    finally:
        close_logger(logger)


def dry_run_plan(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    candidate_count = None
    profile = None
    if args.parameter_source in ("publication", "grid"):
        candidates, _, profile_path = _candidate_documents(args, backend)
        candidate_count = len(candidates)
        profile = str(profile_path)
    else:
        _, provenance = _load_fixed_parameters(args, backend)
        profile = provenance["profile"]
    base = Path(args.results_root) / CANONICAL_METHOD_LABEL
    return {
        "status": "dry_run",
        "side_effects": False,
        "model": backend.name,
        "method": CANONICAL_METHOD_LABEL,
        "stage": args.stage,
        "percentage": args.percentage,
        "parameter_source": args.parameter_source,
        "publication_backed_grid": args.parameter_source == "publication",
        "candidate_count": candidate_count,
        "profile": profile,
        "model_source": args.model_source,
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "planned_grid_root": str(_grid_root(args, backend))
        if args.parameter_source in ("publication", "grid")
        else None,
        "planned_cv_root": str(base / "CV" / "CV_<resolved-by-identity>"),
        "planned_test_root": str(
            base
            / "Test"
            / "CV_<resolved-by-identity>"
            / args.model_source
            / f"{args.percentage}_percent"
        ),
        "model_claim": backend.dry_run_model_claim(args),
    }


def run_pipeline(args: Any, backend: Any) -> dict:
    manifest = load_manifest(args.dataset_root, args.percentage)
    if args.dry_run:
        return dry_run_plan(args, backend, manifest)
    outputs: dict[str, Any] = {}
    if args.stage in ("grid", "all"):
        if args.parameter_source not in ("publication", "grid"):
            raise ContractError("grid stage requires publication or grid parameter source")
        outputs["grid"] = run_grid(args, backend, manifest)
    if args.stage in ("cv", "all"):
        outputs["cv"] = run_cv(args, backend, manifest)
    if args.stage in ("test", "all"):
        outputs["test"] = run_test(args, backend, manifest)
    return {
        "status": "complete",
        "model": backend.name,
        "method": CANONICAL_METHOD_LABEL,
        "stage": args.stage,
        "percentage": args.percentage,
        "outputs": outputs,
    }


__all__ = ["dry_run_plan", "run_cv", "run_grid", "run_pipeline", "run_test"]

