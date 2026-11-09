"""Publication/grid, five-fold CV, refit, and held-out adaptation workflow."""

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
from _linear_probe.metrics import compute_metrics, fold_selection_score, prediction_documents

from .profiles import load_profile
from .safe_io import same_or_write_csv, same_or_write_json


def _method_root(args: Any, backend: Any) -> Path:
    return Path(args.results_root) / backend.method_label


def _profile_path(args: Any, backend: Any) -> Path:
    if args.parameter_source == "publication":
        if args.parameter_config is not None or args.grid_config is not None:
            raise ContractError("publication uses the packaged immutable profile")
        return Path(backend.publication_profile_path)
    if args.parameter_source == "grid":
        return Path(args.grid_config or backend.search_profile_path)
    if args.parameter_source == "default":
        return Path(args.parameter_config or backend.default_profile_path)
    raise ContractError(f"Unsupported parameter source: {args.parameter_source!r}")


def _load_candidates(args: Any, backend: Any) -> tuple[list[dict], dict, Path]:
    path = _profile_path(args, backend).resolve()
    profile, _raw = load_profile(
        path,
        expected_model=backend.name,
        expected_method=backend.method_label,
        expected_source=args.parameter_source,
    )
    candidates = backend.build_candidates(profile)
    if args.parameter_source == "publication" and args.max_grid_candidates is not None:
        raise ContractError("The publication profile cannot be truncated")
    if args.parameter_source == "grid" and args.max_grid_candidates is not None:
        candidates = candidates[: int(args.max_grid_candidates)]
    if not candidates:
        raise ContractError("The requested profile resolves to no candidates")
    return candidates, profile, path


def _parameter_selection_mode(
    args: Any,
    profile: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Resolve whether this source is fixed or requires development-set search.

    ``grid`` is always an explicitly requested project search and ``default``
    is always fixed.  Publication profiles may opt into a genuine
    publication-specified search with ``parameter_selection=publication_grid``.
    New fixed publication profiles declare ``parameter_selection=fixed``.
    Missing metadata defaults to fixed selection; multiple publication candidates
    require an explicit declaration of publication-specified parameter selection.
    """

    source = str(args.parameter_source)
    if source == "grid":
        return "grid"
    if source == "default":
        mode = "fixed"
    elif source == "publication":
        declared = profile.get("parameter_selection")
        if declared is None:
            mode = "fixed"
        elif declared == "fixed":
            mode = "fixed"
        elif declared == "publication_grid":
            mode = "grid"
            if len(candidates) < 2:
                raise ContractError(
                    "An explicit publication_grid must contain at least two candidates"
                )
        else:
            raise ContractError(
                "Publication parameter_selection must be 'fixed' or "
                f"'publication_grid', got {declared!r}"
            )
    else:
        raise ContractError(f"Unsupported parameter source: {source!r}")

    if mode == "fixed" and len(candidates) != 1:
        raise ContractError(
            "Fixed parameter selection must resolve to exactly one candidate"
        )
    return mode


def _grid_folder(source: str) -> str:
    if source == "publication":
        return "Publication_Grid"
    if source == "grid":
        return "Grid"
    raise ContractError("Only publication and grid sources have a grid stage")


def _grid_root(args: Any, backend: Any) -> Path:
    return (
        _method_root(args, backend)
        / _grid_folder(args.parameter_source)
        / f"{args.percentage}_percent"
    )


def _fold_partitions(args: Any, manifest: Mapping[str, Any]) -> list[dict]:
    result = []
    for fold in range(1, 6):
        training, validation = cross_validation_split(manifest, fold)
        train_rows = rows_from_mapping(args.dataset_root, training, require_files=False)
        validation_rows = rows_from_mapping(
            args.dataset_root, validation, require_files=False
        )
        result.append(
            {
                "fold": fold,
                "training_rows_sha256": rows_fingerprint(train_rows),
                "validation_rows_sha256": rows_fingerprint(validation_rows),
            }
        )
    return result


def _grid_context(
    args: Any,
    backend: Any,
    manifest: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    profile_path: Path,
) -> dict:
    identifiers = [str(item["candidate_id"]) for item in candidates]
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    return {
        "schema_version": 1,
        "model": backend.name,
        "method": backend.method_label,
        "method_id": backend.method_id,
        "parameter_source": args.parameter_source,
        "publication_relation": "publication-backed and/or explicitly adapted"
        if args.parameter_source == "publication"
        else None,
        "percentage": int(args.percentage),
        "seed": int(args.seed),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "candidate_ids": identifiers,
        "candidate_count": len(identifiers),
        "candidate_set_sha256": fingerprint(identifiers),
        "model_identity": dict(model_identity),
        "development_input_identity": backend.input_rows_identity(development_rows),
        "fold_partitions": _fold_partitions(args, manifest),
    }


def _validate_fold(
    root: Path,
    *,
    parameters: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    training_rows_sha256: str,
    validation_rows_sha256: str,
) -> dict:
    seal = validate_completion(root, "completion.json")
    if read_json(root / "parameters.json") != dict(parameters):
        raise ContractError(f"Fold parameters changed: {root}")
    state = read_json(root / "training_state.json")
    if state.get("feature_identity") != dict(model_identity):
        raise ContractError(f"Fold pretrained/model identity changed: {root}")
    if state.get("training_rows_sha256") != training_rows_sha256:
        raise ContractError(f"Fold training patients changed: {root}")
    if state.get("validation_rows_sha256") != validation_rows_sha256:
        raise ContractError(f"Fold validation patients changed: {root}")
    return seal


def _grid_partition(context: Mapping[str, Any], fold: int) -> dict:
    matches = [
        item
        for item in context.get("fold_partitions", [])
        if isinstance(item, Mapping) and int(item.get("fold", -1)) == int(fold)
    ]
    if len(matches) != 1:
        raise ContractError(f"Grid context has no unique fold {fold}")
    return dict(matches[0])


def _validate_grid(
    root: Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> dict:
    validate_tree_completion(root, "grid_completion.json")
    context = read_json(root / "grid_context.json")
    if expected_context is not None and context != dict(expected_context):
        raise ContractError(f"Completed grid identity differs: {root}")
    index = read_json(root / "grid_index.json")
    candidates = index.get("candidates")
    if not isinstance(candidates, list) or [
        item.get("candidate_id") for item in candidates
    ] != context.get("candidate_ids"):
        raise ContractError(f"Grid candidate index differs from its context: {root}")
    for item in candidates:
        identifier = str(item["candidate_id"])
        parameters = dict(item["parameters"])
        for fold in range(1, 6):
            partition = _grid_partition(context, fold)
            _validate_fold(
                root / f"candidate_{identifier}" / f"fold_{fold:02d}",
                parameters=parameters,
                model_identity=context["model_identity"],
                training_rows_sha256=partition["training_rows_sha256"],
                validation_rows_sha256=partition["validation_rows_sha256"],
            )
    selection = read_json(root / "selected_parameters.json")
    if (
        selection.get("candidate_id") not in context.get("candidate_ids", [])
        or selection.get("grid_context_sha256") != fingerprint(context)
    ):
        raise ContractError(f"Grid selection is not bound to the grid context: {root}")
    return selection


def run_grid(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    if args.parameter_source not in ("publication", "grid"):
        raise ContractError("grid stage requires publication or grid parameter source")
    candidates, profile, profile_path = _load_candidates(args, backend)
    if _parameter_selection_mode(args, profile, candidates) != "grid":
        raise ContractError(
            "This profile has fixed parameters and bypasses grid selection; "
            "use stage cv/test/all, or choose parameter_source=grid"
        )
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, development_rows)
    model_identity = backend.feature_identity(args)
    root = _grid_root(args, backend)
    context = _grid_context(
        args, backend, manifest, model_identity, candidates, profile_path
    )
    if (root / "grid_completion.json").is_file():
        if not args.resume:
            raise ContractError(f"Completed grid exists and resume is disabled: {root}")
        return {"status": "reused", **_validate_grid(root, expected_context=context)}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ContractError(f"Partial grid exists and resume is disabled: {root}")
    if root.exists() and any(root.iterdir()):
        durable = [
            path
            for path in root.rglob("*")
            if path.is_file() and "Logs" not in path.relative_to(root).parts
        ]
        if durable and not (root / "grid_context.json").is_file():
            raise ContractError(f"Partial grid artifacts have no immutable context: {root}")
        candidate_artifacts = [
            path for path in durable if "candidate_" in path.relative_to(root).parts[0]
        ]
        if candidate_artifacts and not (root / "grid_index.json").is_file():
            raise ContractError(f"Partial grid candidates have no immutable index: {root}")
    root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(root / "grid_context.json", context)
    index = {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "candidates": [
            {"candidate_id": item["candidate_id"], "parameters": dict(item)}
            for item in candidates
        ],
    }
    ensure_same_or_write(root / "grid_index.json", index)
    logger = setup_logger(root / "Logs", f"{backend.name}.{backend.method_id}.grid")
    try:
        for fold in range(1, 6):
            training_mapping, validation_mapping = cross_validation_split(manifest, fold)
            training_rows = rows_from_mapping(
                args.dataset_root, training_mapping, require_files=True
            )
            validation_rows = rows_from_mapping(
                args.dataset_root, validation_mapping, require_files=True
            )
            partition = _grid_partition(context, fold)
            for parameters in candidates:
                identifier = str(parameters["candidate_id"])
                fold_root = root / f"candidate_{identifier}" / f"fold_{fold:02d}"
                if (fold_root / "completion.json").is_file():
                    if not args.resume:
                        raise ContractError(
                            f"Completed grid fold exists and resume is disabled: {fold_root}"
                        )
                else:
                    backend.fit_one(
                        parameters,
                        training_rows,
                        validation_rows,
                        fold_root,
                        args,
                        {
                            "kind": "grid_fold",
                            "grid_context_sha256": fingerprint(context),
                            "percentage": int(args.percentage),
                            "fold": fold,
                            "candidate_id": identifier,
                        },
                        logger,
                    )
                _validate_fold(
                    fold_root,
                    parameters=parameters,
                    model_identity=model_identity,
                    training_rows_sha256=partition["training_rows_sha256"],
                    validation_rows_sha256=partition["validation_rows_sha256"],
                )

        summaries = []
        sealed_paths: list[Path] = [root / "grid_context.json", root / "grid_index.json"]
        for parameters in candidates:
            identifier = str(parameters["candidate_id"])
            candidate_root = root / f"candidate_{identifier}"
            metrics = [
                read_json(candidate_root / f"fold_{fold:02d}" / "metrics_summary.json")
                for fold in range(1, 6)
            ]
            summary = {
                "candidate_id": identifier,
                "parameters": dict(parameters),
                "fold_metrics": metrics,
                **backend.selection_summary(parameters, metrics),
            }
            same_or_write_json(candidate_root / "candidate_summary.json", summary)
            summaries.append(summary)
            sealed_paths.append(candidate_root / "candidate_summary.json")
            sealed_paths.extend(
                candidate_root / f"fold_{fold:02d}" / "completion.json"
                for fold in range(1, 6)
            )
        ranking = sorted(summaries, key=backend.ranking_key)
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        best = ranking[0]
        grid_summary = {
            "schema_version": 1,
            "selection_rule": (
                "0.7*(mean macro-F1 - population SD) + 0.3*(mean balanced accuracy "
                "- population SD); tie-break mean Perforation recall then candidate ID"
            ),
            "ranking": ranking,
        }
        same_or_write_json(root / "grid_summary.json", grid_summary)
        same_or_write_csv(
            root / "grid_summary.csv",
            [backend.summary_csv_row(item) for item in ranking],
            list(backend.summary_csv_row(ranking[0])),
        )
        selection = {
            "schema_version": 1,
            "model": backend.name,
            "method": backend.method_label,
            "parameter_source": args.parameter_source,
            "candidate_id": best["candidate_id"],
            "parameters": best["parameters"],
            "grid_root": str(root.resolve()),
            "grid_context_sha256": fingerprint(context),
            "selection_rule": grid_summary["selection_rule"],
        }
        same_or_write_json(root / "selected_parameters.json", selection)
        sealed_paths.extend(
            [
                root / "grid_summary.json",
                root / "grid_summary.csv",
                root / "selected_parameters.json",
            ]
        )
        atomic_write_json(
            root / "grid_completion.json",
            tree_completion_seal(
                root,
                sealed_paths,
                context={"grid_context_sha256": fingerprint(context)},
            ),
        )
        logger.info("Completed and sealed %s candidates across five folds", len(candidates))
        return {"status": "complete", **selection}
    finally:
        close_logger(logger)


def _selected_parameters(
    args: Any,
    backend: Any,
    manifest: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> tuple[dict, dict]:
    candidates, profile, profile_path = _load_candidates(args, backend)
    if _parameter_selection_mode(args, profile, candidates) == "grid":
        context = _grid_context(
            args, backend, manifest, model_identity, candidates, profile_path
        )
        root = _grid_root(args, backend)
        selection = _validate_grid(root, expected_context=context)
        return dict(selection["parameters"]), {
            "parameter_source": args.parameter_source,
            "candidate_id": selection["candidate_id"],
            "grid_root": str(root.resolve()),
            "grid_completion_sha256": sha256_file(root / "grid_completion.json"),
            "profile": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
        }
    parameters = dict(candidates[0])
    return parameters, {
        "parameter_source": args.parameter_source,
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
    }


def _read_registry(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "runs": []}
    document = read_json(path)
    if document.get("schema_version") != 1 or not isinstance(document.get("runs"), list):
        raise ContractError(f"Malformed CV registry: {path}")
    return document


def _register_cv(
    root: Path,
    *,
    backend: Any,
    identity: Mapping[str, Any],
    create: bool,
) -> tuple[str, Path]:
    cv_base = root / "CV"
    registry_path = cv_base / "cv_registry.json"
    registry = _read_registry(registry_path)
    identity_sha = fingerprint(identity)
    matches = [item for item in registry["runs"] if item.get("identity_sha256") == identity_sha]
    if matches:
        if len(matches) != 1 or matches[0].get("identity") != dict(identity):
            raise ContractError("CV registry contains a fingerprint collision or duplicate")
        cv_id = str(matches[0]["cv_id"])
        cv_root = cv_base / cv_id
        document = {
            "schema_version": 1,
            "cv_id": cv_id,
            "model": backend.name,
            "method": backend.method_label,
            "identity": dict(identity),
        }
        if create:
            cv_root.mkdir(parents=True, exist_ok=True)
            ensure_same_or_write(cv_root / "cv_parameters.json", document)
        elif not (cv_root / "cv_parameters.json").is_file():
            raise ContractError(f"Registered CV has no immutable parameter document: {cv_root}")
        elif read_json(cv_root / "cv_parameters.json") != document:
            raise ContractError(f"Registered CV parameter identity changed: {cv_root}")
        return cv_id, cv_root
    if not create:
        raise ContractError("No CV run matches the requested model/parameters")
    used = []
    for item in registry["runs"]:
        value = str(item.get("cv_id", ""))
        if not value.startswith("CV_") or not value[3:].isdigit() or int(value[3:]) <= 0:
            raise ContractError(f"Malformed CV identifier in registry: {value!r}")
        used.append(int(value[3:]))
    cv_id = f"CV_{max(used, default=0) + 1}"
    cv_root = cv_base / cv_id
    cv_root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(
        cv_root / "cv_parameters.json",
        {
            "schema_version": 1,
            "cv_id": cv_id,
            "model": backend.name,
            "method": backend.method_label,
            "identity": dict(identity),
        },
    )
    registry["runs"].append(
        {
            "cv_id": cv_id,
            "identity_sha256": identity_sha,
            "identity": dict(identity),
            "registered_at": utc_now(),
        }
    )
    atomic_write_json(registry_path, registry)
    return cv_id, cv_root


def _cv_identity(
    backend: Any,
    args: Any,
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> dict:
    from .recipe import recipe_identity
    return recipe_identity({
        "schema_version": 1,
        "model": backend.name,
        "method": backend.method_label,
        "method_id": backend.method_id,
        "parameters": dict(parameters),
        "parameter_provenance_identity": _cv_provenance_identity(provenance),
        "model_identity": dict(model_identity),
        "seed": int(args.seed),
    })


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


def _cv_run_context(
    *,
    args: Any,
    backend: Any,
    manifest: Mapping[str, Any],
    cv_id: str,
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    development_rows: Sequence[Mapping[str, Any]],
) -> dict:
    return {
        "schema_version": 1,
        "model": backend.name,
        "method": backend.method_label,
        "method_id": backend.method_id,
        "cv_id": cv_id,
        "percentage": int(args.percentage),
        "seed": int(args.seed),
        "split_manifest_sha256": manifest["_sha256"],
        "parameters": dict(parameters),
        "parameter_provenance": dict(provenance),
        "parameter_provenance_identity": _cv_provenance_identity(provenance),
        "model_identity": dict(model_identity),
        "development_rows_sha256": rows_fingerprint(development_rows),
        "development_input_identity": backend.input_rows_identity(development_rows),
        "fold_partitions": _fold_partitions(args, manifest),
    }


def _refit_selection(fold_states: Sequence[Mapping[str, Any]], backend: Any) -> dict:
    selected_steps = [int(item["selected_optimizer_step"]) for item in fold_states]
    return {
        "schema_version": 1,
        "rule": "median selected optimizer step across the five CV folds",
        "fold_selected_optimizer_steps": selected_steps,
        "refit_optimizer_steps": backend.refit_budget(fold_states),
    }


def _cv_completion_context(context: Mapping[str, Any], refit: Mapping[str, Any]) -> dict:
    return {
        **dict(context),
        "refit_selection_sha256": fingerprint(refit),
        "refit_optimizer_steps": int(refit["refit_optimizer_steps"]),
    }


def _materialize_grid_fold(source: Path, destination: Path) -> None:
    seal = validate_completion(source, "completion.json")
    materialized = []
    for record in seal["artifacts"]:
        materialized.append(
            {
                "path": record["path"],
                **materialize_file(source / record["path"], destination / record["path"]),
            }
        )
    materialize_file(source / "completion.json", destination / "completion.json")
    atomic_write_json(
        destination / "grid_reuse.json",
        {
            "schema_version": 1,
            "source": str(source.resolve()),
            "source_completion_sha256": sha256_file(source / "completion.json"),
            "materialized": materialized,
        },
    )


def _fold_expected_rows(
    args: Any, manifest: Mapping[str, Any], fold: int
) -> tuple[list[dict], list[dict]]:
    training, validation = cross_validation_split(manifest, fold)
    return (
        rows_from_mapping(args.dataset_root, training, require_files=False),
        rows_from_mapping(args.dataset_root, validation, require_files=False),
    )


def _validate_cv(
    root: Path,
    *,
    expected_context: Mapping[str, Any],
    backend: Any,
    args: Any,
    manifest: Mapping[str, Any],
) -> dict:
    seal = validate_tree_completion(root, "cv_completion.json")
    if read_json(root / "cv_run_context.json") != dict(expected_context):
        raise ContractError(f"CV run context changed: {root}")
    parameters = expected_context["parameters"]
    model_identity = expected_context["model_identity"]
    fold_states = []
    for fold in range(1, 6):
        training, validation = _fold_expected_rows(args, manifest, fold)
        _validate_fold(
            root / f"fold_{fold:02d}",
            parameters=parameters,
            model_identity=model_identity,
            training_rows_sha256=rows_fingerprint(training),
            validation_rows_sha256=rows_fingerprint(validation),
        )
        fold_states.append(read_json(root / f"fold_{fold:02d}" / "training_state.json"))
    refit_selection = read_json(root / "refit_selection.json")
    expected_refit_selection = _refit_selection(fold_states, backend)
    if refit_selection != expected_refit_selection:
        raise ContractError(f"Full-development refit selection changed: {root}")
    completion_context = _cv_completion_context(expected_context, refit_selection)
    if seal.get("context") != completion_context:
        raise ContractError(f"CV completion context changed: {root}")
    validate_completion(root / "refit_full_development", "completion.json")
    if read_json(root / "refit_full_development" / "parameters.json") != dict(parameters):
        raise ContractError(f"Full-development refit parameters changed: {root}")
    refit_state = read_json(root / "refit_full_development" / "training_state.json")
    if (
        refit_state.get("feature_identity") != model_identity
        or refit_state.get("training_rows_sha256")
        != expected_context["development_rows_sha256"]
        or int(refit_state.get("selected_optimizer_step", -1))
        != int(refit_selection["refit_optimizer_steps"])
    ):
        raise ContractError(f"Full-development refit identity changed: {root}")
    return seal


def run_cv(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, development_rows)
    model_identity = backend.feature_identity(args)
    parameters, provenance = _selected_parameters(
        args, backend, manifest, model_identity
    )
    identity = _cv_identity(
        backend, args, parameters, provenance, model_identity
    )
    cv_id, cv_root = _register_cv(
        _method_root(args, backend), backend=backend, identity=identity, create=True
    )
    percentage_root = cv_root / f"{args.percentage}_percent"
    if percentage_root.exists() and any(percentage_root.iterdir()) and not args.resume:
        raise ContractError(f"Partial/completed CV exists and resume is disabled: {percentage_root}")
    context = _cv_run_context(
        args=args,
        backend=backend,
        manifest=manifest,
        cv_id=cv_id,
        parameters=parameters,
        provenance=provenance,
        model_identity=model_identity,
        development_rows=development_rows,
    )
    if percentage_root.exists() and any(percentage_root.iterdir()):
        durable = [
            path
            for path in percentage_root.rglob("*")
            if path.is_file() and "Logs" not in path.relative_to(percentage_root).parts
        ]
        if durable and not (percentage_root / "cv_run_context.json").is_file():
            raise ContractError(
                f"Nonempty partial CV has no immutable run context: {percentage_root}"
            )
    percentage_root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(percentage_root / "cv_run_context.json", context)
    if (percentage_root / "cv_completion.json").is_file():
        _validate_cv(
            percentage_root,
            expected_context=context,
            backend=backend,
            args=args,
            manifest=manifest,
        )
        return {"status": "reused", "cv_id": cv_id, "cv_root": str(cv_root)}
    logger = setup_logger(percentage_root / "Logs", f"{backend.name}.{backend.method_id}.cv")
    try:
        source_grid = Path(provenance["grid_root"]) if provenance.get("grid_root") else None
        selected = provenance.get("candidate_id")
        fold_rows = []
        fold_states = []
        sealed_paths: list[Path] = []
        for fold in range(1, 6):
            fold_root = percentage_root / f"fold_{fold:02d}"
            training_mapping, validation_mapping = cross_validation_split(manifest, fold)
            training_rows = rows_from_mapping(
                args.dataset_root, training_mapping, require_files=True
            )
            validation_rows = rows_from_mapping(
                args.dataset_root, validation_mapping, require_files=True
            )
            if not (fold_root / "completion.json").is_file():
                if source_grid is not None and selected:
                    _materialize_grid_fold(
                        source_grid / f"candidate_{selected}" / f"fold_{fold:02d}",
                        fold_root,
                    )
                else:
                    backend.fit_one(
                        parameters,
                        training_rows,
                        validation_rows,
                        fold_root,
                        args,
                        {
                            "kind": "cv_fold",
                            "cv_id": cv_id,
                            "percentage": int(args.percentage),
                            "fold": fold,
                        },
                        logger,
                    )
            _validate_fold(
                fold_root,
                parameters=parameters,
                model_identity=model_identity,
                training_rows_sha256=rows_fingerprint(training_rows),
                validation_rows_sha256=rows_fingerprint(validation_rows),
            )
            metrics = read_json(fold_root / "metrics_summary.json")
            state = read_json(fold_root / "training_state.json")
            fold_states.append(state)
            fold_rows.append(
                {
                    "fold": fold,
                    "selection_score": fold_selection_score(metrics),
                    "f1_macro": metrics["f1_macro"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "selected_optimizer_step": state["selected_optimizer_step"],
                }
            )
            sealed_paths.extend(
                [fold_root / "completion.json", fold_root / backend.artifact_filename]
            )
            if (fold_root / "grid_reuse.json").is_file():
                sealed_paths.append(fold_root / "grid_reuse.json")

        refit_selection = _refit_selection(fold_states, backend)
        refit_steps = int(refit_selection["refit_optimizer_steps"])
        same_or_write_json(percentage_root / "refit_selection.json", refit_selection)

        same_or_write_csv(
            percentage_root / "cv_fold_summary.csv", fold_rows, list(fold_rows[0])
        )
        best = sorted(
            fold_rows, key=lambda item: (-float(item["selection_score"]), int(item["fold"]))
        )[0]
        same_or_write_json(
            percentage_root / "best_fold_model.json",
            {
                "schema_version": 1,
                "fold": best["fold"],
                "selection_score": best["selection_score"],
                "artifact": str(
                    (
                        percentage_root
                        / f"fold_{int(best['fold']):02d}"
                        / backend.artifact_filename
                    ).resolve()
                ),
            },
        )
        refit_root = percentage_root / "refit_full_development"
        if not (refit_root / "completion.json").is_file():
            backend.refit(
                parameters,
                development_rows,
                refit_root,
                args,
                {
                    "kind": "full_development_refit",
                    "cv_id": cv_id,
                    "percentage": int(args.percentage),
                    "refit_optimizer_steps": refit_steps,
                    "cv_context_sha256": fingerprint(context),
                },
                logger,
            )
        validate_completion(refit_root, "completion.json")
        sealed_paths.extend(
            [
                percentage_root / "cv_run_context.json",
                percentage_root / "refit_selection.json",
                percentage_root / "cv_fold_summary.csv",
                percentage_root / "best_fold_model.json",
                refit_root / "completion.json",
                refit_root / backend.artifact_filename,
            ]
        )
        atomic_write_json(
            percentage_root / "cv_completion.json",
            tree_completion_seal(
                percentage_root,
                sealed_paths,
                context=_cv_completion_context(context, refit_selection),
            ),
        )
        _validate_cv(
            percentage_root,
            expected_context=context,
            backend=backend,
            args=args,
            manifest=manifest,
        )
        logger.info("Completed five folds and full-development refit at %s steps", refit_steps)
        return {"status": "complete", "cv_id": cv_id, "cv_root": str(cv_root)}
    finally:
        close_logger(logger)


def _locate_cv(
    args: Any, backend: Any, manifest: Mapping[str, Any]
) -> tuple[str, Path, dict, dict, dict]:
    development_rows = rows_from_mapping(
        args.dataset_root, development_mapping(manifest), require_files=True
    )
    backend.ensure_feature_rows(args, development_rows)
    model_identity = backend.feature_identity(args)
    parameters, provenance = _selected_parameters(
        args, backend, manifest, model_identity
    )
    identity = _cv_identity(backend, args, parameters, provenance, model_identity)
    cv_id, cv_root = _register_cv(
        _method_root(args, backend), backend=backend, identity=identity, create=False
    )
    percentage_root = cv_root / f"{args.percentage}_percent"
    context = _cv_run_context(
        args=args,
        backend=backend,
        manifest=manifest,
        cv_id=cv_id,
        parameters=parameters,
        provenance=provenance,
        model_identity=model_identity,
        development_rows=development_rows,
    )
    _validate_cv(
        percentage_root,
        expected_context=context,
        backend=backend,
        args=args,
        manifest=manifest,
    )
    return cv_id, percentage_root, parameters, model_identity, provenance


def _source_artifacts(args: Any, backend: Any, cv_root: Path) -> list[Path]:
    if args.model_source == "refit_full_development":
        paths = [cv_root / "refit_full_development" / backend.artifact_filename]
    elif args.model_source == "best_cv_fold":
        fold = int(read_json(cv_root / "best_fold_model.json")["fold"])
        if fold not in range(1, 6):
            raise ContractError("best_fold_model.json has an invalid fold")
        paths = [cv_root / f"fold_{fold:02d}" / backend.artifact_filename]
    elif args.model_source == "cv_ensemble":
        paths = [
            cv_root / f"fold_{fold:02d}" / backend.artifact_filename
            for fold in range(1, 6)
        ]
    else:
        raise ContractError(f"Unsupported model source: {args.model_source!r}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = [sha256_file(path) for path in paths]
    if len(paths) > 1 and len(hashes) != len(set(hashes)):
        raise ContractError("CV ensemble contains duplicate adaptation artifacts")
    return paths


def _validate_test(
    root: Path,
    *,
    selection: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict:
    seal = validate_completion(root, "test_completion.json")
    if read_json(root / "test_context.json") != dict(selection):
        raise ContractError(f"Held-out execution context changed: {root}")
    if read_json(root / "model_selection.json") != dict(selection):
        raise ContractError(f"Held-out model selection changed: {root}")
    predictions = json.loads((root / "test_patient_predictions.json").read_text(encoding="utf-8"))
    expected = [
        (str(row["patient_id"]), int(row["label"]), str(row["class_name"]))
        for row in expected_rows
    ]
    observed = [
        (str(item["patient_id"]), int(item["true_index"]), str(item["true_class"]))
        for item in predictions
    ]
    if observed != expected:
        raise ContractError(f"Held-out patient/order identity changed: {root}")
    expected_seal_context = {
        "cv_id": selection["cv_id"],
        "percentage": int(selection["percentage"]),
        "model_source": selection["model_source"],
        "selection_sha256": fingerprint(selection),
    }
    if seal.get("context") != expected_seal_context:
        raise ContractError(f"Held-out completion context changed: {root}")
    return seal


def run_test(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    cv_id, cv_percentage_root, parameters, model_identity, provenance = _locate_cv(
        args, backend, manifest
    )
    artifacts = _source_artifacts(args, backend, cv_percentage_root)
    expected_rows = rows_from_mapping(
        args.dataset_root, heldout_mapping(manifest), require_files=False
    )
    rows = rows_from_mapping(args.dataset_root, heldout_mapping(manifest), require_files=True)
    backend.ensure_feature_rows(args, rows)
    if rows_fingerprint(rows) != rows_fingerprint(expected_rows):
        raise ContractError("Held-out identity changed between preflight and execution")
    selection = {
        "schema_version": 1,
        "model": backend.name,
        "method": backend.method_label,
        "cv_id": cv_id,
        "percentage": int(args.percentage),
        "model_source": args.model_source,
        "artifacts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in artifacts
        ],
        "parameters": dict(parameters),
        "parameter_provenance": dict(provenance),
        "model_identity": dict(model_identity),
        "heldout_rows_sha256": rows_fingerprint(expected_rows),
        "heldout_input_identity": backend.input_rows_identity(rows),
    }
    root = (
        _method_root(args, backend)
        / "Test"
        / cv_id
        / args.model_source
        / f"{args.percentage}_percent"
    )
    if (root / "test_completion.json").is_file():
        if not args.resume:
            raise ContractError(f"Completed held-out output exists and resume is disabled: {root}")
        _validate_test(root, selection=selection, expected_rows=expected_rows)
        return {"status": "reused", "cv_id": cv_id, "test_root": str(root)}
    if root.exists():
        durable = [
            path
            for path in root.rglob("*")
            if path.is_file() and "Logs" not in path.relative_to(root).parts
        ]
        allowed_partial = {
            "test_context.json",
            "metrics_summary.json",
            "test_patient_predictions.json",
            "test_probabilities.csv",
            "model_selection.json",
        }
        unknown = [path for path in durable if path.relative_to(root).as_posix() not in allowed_partial]
        if durable and not args.resume:
            raise ContractError(f"Partial held-out output exists and resume is disabled: {root}")
        if unknown:
            raise ContractError(
                f"Unknown unsealed held-out artifacts exist; refusing overwrite: {unknown[0]}"
            )
    root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(root / "test_context.json", selection)
    logger = setup_logger(root / "Logs", f"{backend.name}.{backend.method_id}.test")
    try:
        predictions = [
            backend.predict(path, rows, args, {"model_identity": model_identity})
            for path in artifacts
        ]
        probabilities = np.mean(np.stack(predictions, axis=0), axis=0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        metrics = compute_metrics(
            truth, probabilities.argmax(axis=1), probabilities, CLASS_NAMES
        )
        json_rows, csv_rows = prediction_documents(rows, probabilities, CLASS_NAMES)
        same_or_write_json(root / "metrics_summary.json", metrics)
        same_or_write_json(root / "test_patient_predictions.json", json_rows)
        same_or_write_csv(root / "test_probabilities.csv", csv_rows, list(csv_rows[0]))
        same_or_write_json(root / "model_selection.json", selection)
        atomic_write_json(
            root / "test_completion.json",
            completion_seal(
                [
                    root / "test_context.json",
                    root / "metrics_summary.json",
                    root / "test_patient_predictions.json",
                    root / "test_probabilities.csv",
                    root / "model_selection.json",
                ],
                context={
                    "cv_id": cv_id,
                    "percentage": int(args.percentage),
                    "model_source": args.model_source,
                    "selection_sha256": fingerprint(selection),
                },
            ),
        )
        logger.info("Held-out evaluation completed and sealed")
        return {"status": "complete", "cv_id": cv_id, "test_root": str(root)}
    finally:
        close_logger(logger)


def dry_run_plan(args: Any, backend: Any, manifest: Mapping[str, Any]) -> dict:
    candidates, profile, path = _load_candidates(args, backend)
    selection_mode = _parameter_selection_mode(args, profile, candidates)
    if args.stage == "grid" and selection_mode != "grid":
        raise ContractError(
            "This profile has fixed parameters and bypasses grid selection; "
            "use stage cv/test/all, or choose parameter_source=grid"
        )
    base = _method_root(args, backend)
    return {
        "status": "dry_run",
        "side_effects": False,
        "model": backend.name,
        "method": backend.method_label,
        "method_id": backend.method_id,
        "stage": args.stage,
        "percentage": int(args.percentage),
        "parameter_source": args.parameter_source,
        "parameter_selection": selection_mode,
        "candidate_count": len(candidates),
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "profile": str(path),
        "profile_sha256": sha256_file(path),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "model_source": args.model_source,
        "planned_grid_root": str(_grid_root(args, backend))
        if selection_mode == "grid"
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
    manifest = load_manifest(args.dataset_root, int(args.percentage))
    if args.dry_run:
        return dry_run_plan(args, backend, manifest)
    candidates, profile, _path = _load_candidates(args, backend)
    selection_mode = _parameter_selection_mode(args, profile, candidates)
    outputs: dict[str, Any] = {}
    if args.stage == "grid" and selection_mode != "grid":
        raise ContractError(
            "This profile has fixed parameters and bypasses grid selection; "
            "use stage cv/test/all, or choose parameter_source=grid"
        )
    if args.stage == "grid" or (args.stage == "all" and selection_mode == "grid"):
        outputs["grid"] = run_grid(args, backend, manifest)
    if args.stage in ("cv", "all"):
        outputs["cv"] = run_cv(args, backend, manifest)
    if args.stage in ("test", "all"):
        outputs["test"] = run_test(args, backend, manifest)
    return {
        "status": "complete",
        "model": backend.name,
        "method": backend.method_label,
        "stage": args.stage,
        "percentage": int(args.percentage),
        "outputs": outputs,
    }


__all__ = ["dry_run_plan", "run_cv", "run_grid", "run_pipeline", "run_test"]
