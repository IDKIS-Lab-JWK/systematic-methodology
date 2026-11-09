"""Search, five-fold CV, refit, and held-out orchestration for Pillar-0."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:
    from . import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from .cache import load_cached_rows
    from .data import (
        cross_validation_split,
        development_mapping,
        heldout_mapping,
        load_manifest,
        rows_from_mapping,
        validate_no_overlap,
    )
    from .metrics import compute_metrics, fold_selection_score, format_predictions, grid_selection_score
    from .model import MODEL_REPO_ID, MODEL_REVISION
    from .search import candidate_id, load_search_definition, rank_records, search_candidates
    from .training import (
        FeatureStore,
        evaluate_checkpoint,
        predict_checkpoint,
        refit_model,
        save_json,
        train_full_finetune,
        train_linear_probe,
    )
except ImportError:  # pragma: no cover - direct scripts
    from __init__ import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from cache import load_cached_rows
    from data import cross_validation_split, development_mapping, heldout_mapping, load_manifest, rows_from_mapping, validate_no_overlap
    from metrics import compute_metrics, fold_selection_score, format_predictions, grid_selection_score
    from model import MODEL_REPO_ID, MODEL_REVISION
    from search import candidate_id, load_search_definition, rank_records, search_candidates
    from training import FeatureStore, evaluate_checkpoint, predict_checkpoint, refit_model, save_json, train_full_finetune, train_linear_probe


METHOD_FOLDERS = {"linear_probe": "LinearProbe", "full_finetune": "FullFineTune"}
PROFILE_FILES = {
    "linear_probe": {"default": "default_linear_probe.json", "publication": "publication_linear_probe.json", "search": "search_linear_probe.json"},
    "full_finetune": {"default": "default_full_finetune.json", "publication": "publication_full_finetune.json", "search": "search_full_finetune.json"},
}


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    return methodology_root() / "Dataset"


def default_results_root() -> Path:
    return methodology_root() / "Results" / "Pillar0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_root(results_root: Path, method: str) -> Path:
    return Path(results_root) / METHOD_FOLDERS[method]


def load_parameter_profile(method: str, source: str, config_path: Path | None = None) -> tuple[dict, dict]:
    if source not in {"default", "publication", "manual"}:
        raise ValueError(f"Cannot load a single parameter profile for source={source}")
    if source == "manual":
        if config_path is None:
            raise ValueError("--parameter_source manual requires --parameter_config.")
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent / PROFILE_FILES[method][source]
    document = read_json(path)
    if document.get("method") != method:
        raise ValueError(f"Parameter profile method mismatch in {path}")
    parameters = document.get("parameters", document if source == "manual" else None)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"No parameters object in {path}")
    provenance = {
        "parameter_source": source,
        "profile_path": str(path.resolve()),
        "profile_sha256": sha256_file(path),
        "profile_type": document.get("profile_type", source),
        "source": document.get("source"),
        "task_adaptations": document.get("task_adaptations", []),
    }
    return parameters, provenance


def parameter_fingerprint(method: str, parameters: dict, repo_id: str, revision: str) -> str:
    payload = json.dumps(
        {"method": method, "parameters": parameters, "model_repo_id": repo_id, "model_revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def register_cv_run(results_root: Path, method: str, parameters: dict, provenance: dict, repo_id: str, revision: str) -> tuple[str, Path]:
    cv_root = method_root(results_root, method) / "CV"
    registry_path = cv_root / "cv_runs.json"
    registry = read_json(registry_path) if registry_path.is_file() else {"schema_version": 1, "method": method, "runs": []}
    fingerprint = parameter_fingerprint(method, parameters, repo_id, revision)
    existing = next((item for item in registry["runs"] if item["parameter_fingerprint"] == fingerprint), None)
    if existing:
        cv_id = existing["cv_id"]
    else:
        used = [int(item["cv_id"].split("_", 1)[1]) for item in registry["runs"]]
        cv_id = f"CV_{max(used, default=0) + 1}"
        registry["runs"].append({"cv_id": cv_id, "parameter_fingerprint": fingerprint, "parameters": parameters, "model_repo_id": repo_id, "model_revision": revision, "created_at_utc": utc_now()})
        save_json(registry_path, registry)
    cv_dir = cv_root / cv_id
    cv_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = cv_dir / "cv_parameters.json"
    document = read_json(parameter_path) if parameter_path.is_file() else {"schema_version": 1, "cv_id": cv_id, "method": method, "parameter_fingerprint": fingerprint, "parameters": parameters, "model_repo_id": repo_id, "model_revision": revision, "provenance_history": [], "percentages": {}}
    event = {"recorded_at_utc": utc_now(), **provenance}
    if event not in document["provenance_history"]:
        document["provenance_history"].append(event)
    save_json(parameter_path, document)
    return cv_id, cv_dir


def _feature_store(args, raw_manifest: Path) -> FeatureStore:
    signature = f"{Path(args.cache_namespace).name}_{args.model_revision[:12]}_{sha256_file(raw_manifest)[:12]}"
    root = Path(args.feature_cache_root) if args.feature_cache_root else Path(args.results_root) / "Cache" / "Features" / signature
    return FeatureStore(root, raw_manifest, args.model_repo_id, args.model_revision)


def _split_cached_rows(args, mapping: dict) -> list[dict]:
    source = rows_from_mapping(Path(args.dataset_root), mapping)
    return load_cached_rows(Path(args.cache_dir), source, expected_namespace=args.cache_namespace)


def _train_method(method: str, train_rows: list[dict], val_rows: list[dict], parameters: dict, output_dir: Path, args, feature_store: FeatureStore | None, budget: int | None = None):
    common = {"device": torch.device(args.device), "repo_id": args.model_repo_id, "revision": args.model_revision, "seed": args.seed, "budget": budget}
    if method == "linear_probe":
        if feature_store is None:
            raise RuntimeError("Linear probe requires a feature store.")
        return train_linear_probe(train_rows, val_rows, parameters, output_dir, feature_store, **common)
    return train_full_finetune(train_rows, val_rows, parameters, output_dir, num_workers=args.num_workers, **common)


def run_search(args, method: str, manifest: dict) -> tuple[dict, dict]:
    path = Path(args.search_config) if args.search_config else Path(__file__).resolve().parent / PROFILE_FILES[method]["search"]
    definition = load_search_definition(path)
    candidates = search_candidates(definition, args.max_search_candidates)
    fixed = definition.get("fixed_parameters", {})
    candidates = [{**fixed, **candidate} for candidate in candidates]
    active = {candidate_id(item): item for item in candidates}
    grid_root = method_root(args.results_root, method) / "Grid" / f"{args.percentage}_percent"
    raw_manifest = Path(args.cache_dir) / "cache_manifest.json"
    feature_store = _feature_store(args, raw_manifest) if method == "linear_probe" else None
    all_rounds = []
    for round_number, round_definition in enumerate(definition["rounds"], start=1):
        records = []
        for identity, parameters in active.items():
            fold_metrics = []
            for fold in [int(value) for value in round_definition["folds"]]:
                training, validation = cross_validation_split(manifest, fold)
                train_rows, val_rows = _split_cached_rows(args, training), _split_cached_rows(args, validation)
                validate_no_overlap(train_rows, val_rows)
                output = grid_root / f"round_{round_number:02d}" / identity / f"fold_{fold:02d}"
                state_path = output / "training_state.json"
                if args.resume and state_path.is_file() and read_json(state_path).get("completed"):
                    result = {"metrics": read_json(output / "metrics_summary.json")}
                else:
                    result = _train_method(method, train_rows, val_rows, parameters, output, args, feature_store, int(round_definition["budget"]))
                fold_metrics.append(result["metrics"])
            f1_values = [float(item["f1_macro"]) for item in fold_metrics]
            ba_values = [float(item["balanced_accuracy"]) for item in fold_metrics]
            perforation = [float(item["per_class"]["Perforation"]["sensitivity"]) for item in fold_metrics]
            records.append({"candidate_id": identity, "parameters": parameters, "folds": list(round_definition["folds"]), "budget": int(round_definition["budget"]), "mean_f1_macro": float(np.mean(f1_values)), "std_f1_macro": float(np.std(f1_values)), "mean_balanced_accuracy": float(np.mean(ba_values)), "std_balanced_accuracy": float(np.std(ba_values)), "minority_perforation_recall": float(np.mean(perforation))})
        ranked = rank_records(records)
        keep = min(int(round_definition["keep"]), len(ranked))
        active = {item["candidate_id"]: item["parameters"] for item in ranked[:keep]}
        round_doc = {"round": round_number, "budget": int(round_definition["budget"]), "folds": list(round_definition["folds"]), "keep": keep, "ranking": ranked}
        all_rounds.append(round_doc)
        save_json(grid_root / f"round_{round_number:02d}_summary.json", round_doc)
    winner = all_rounds[-1]["ranking"][0]
    summary = {"schema_version": 1, "method": method, "strategy": definition["strategy"], "search_config": str(path.resolve()), "search_config_sha256": sha256_file(path), "heldout_test_used": False, "rounds": all_rounds, "selected_candidate_id": winner["candidate_id"], "selected_parameters": winner["parameters"], "completed_at_utc": utc_now()}
    save_json(grid_root / "grid_search_summary.json", summary)
    save_json(grid_root / "selected_parameters.json", {"parameters": winner["parameters"], "candidate_id": winner["candidate_id"], "source": "curated_successive_halving", "search_summary": str((grid_root / "grid_search_summary.json").resolve())})
    provenance = {"parameter_source": "search", "profile_path": str(path.resolve()), "profile_sha256": sha256_file(path), "candidate_id": winner["candidate_id"], "task_adaptations": definition.get("task_adaptations", [])}
    return winner["parameters"], provenance


def _fold_summary(cv_percentage_dir: Path) -> list[dict]:
    rows = []
    for fold in range(1, 6):
        directory = cv_percentage_dir / f"fold_{fold:02d}"
        state, metrics = read_json(directory / "training_state.json"), read_json(directory / "metrics_summary.json")
        rows.append({"fold": fold, "best_epoch": state.get("best_epoch"), "best_optimizer_step": state.get("best_optimizer_step"), "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "f1_macro": metrics["f1_macro"], "average_precision_macro": metrics["average_precision_macro"], "selection_score": fold_selection_score(metrics), "checkpoint": str((directory / "best_model.pt").resolve())})
    return rows


def run_cv(args, method: str, manifest: dict, parameters: dict, provenance: dict) -> dict:
    cv_id, cv_dir = register_cv_run(args.results_root, method, parameters, provenance, args.model_repo_id, args.model_revision)
    percentage_dir = cv_dir / f"{args.percentage}_percent"
    raw_manifest = Path(args.cache_dir) / "cache_manifest.json"
    feature_store = _feature_store(args, raw_manifest) if method == "linear_probe" else None
    for fold in range(1, 6):
        training, validation = cross_validation_split(manifest, fold)
        train_rows, val_rows = _split_cached_rows(args, training), _split_cached_rows(args, validation)
        validate_no_overlap(train_rows, val_rows)
        output = percentage_dir / f"fold_{fold:02d}"
        marker = output / "training_state.json"
        if args.resume and marker.is_file() and read_json(marker).get("completed"):
            continue
        _train_method(method, train_rows, val_rows, parameters, output, args, feature_store)
    summary_rows = _fold_summary(percentage_dir)
    percentage_dir.mkdir(parents=True, exist_ok=True)
    with (percentage_dir / "cv_fold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    best = max(summary_rows, key=lambda item: (item["selection_score"], -item["fold"]))
    save_json(percentage_dir / "best_fold_model.json", {"schema_version": 1, "cv_id": cv_id, "method": method, "percentage": args.percentage, "best_fold": best["fold"], "selection_rule": "0.7*f1_macro+0.3*balanced_accuracy", "checkpoint": best["checkpoint"], "parameters": parameters, "parameter_provenance": provenance, "heldout_test_used_during_selection": False})
    budget_key = "best_epoch" if method == "linear_probe" else "best_optimizer_step"
    budget = max(1, int(round(statistics.median(int(row[budget_key]) for row in summary_rows if row[budget_key] is not None))))
    refit_dir = percentage_dir / "refit_full_development"
    if not (args.resume and (refit_dir / "training_state.json").is_file() and read_json(refit_dir / "training_state.json").get("completed")):
        development = _split_cached_rows(args, development_mapping(manifest))
        refit_model(development, parameters, refit_dir, method=method, device=torch.device(args.device), refit_budget=budget, feature_store=feature_store, repo_id=args.model_repo_id, revision=args.model_revision, seed=args.seed, num_workers=args.num_workers)
    parameter_doc = read_json(cv_dir / "cv_parameters.json")
    parameter_doc["percentages"][str(args.percentage)] = {"completed": True, "best_fold": best["fold"], "refit_budget": budget, "completed_at_utc": utc_now()}
    save_json(cv_dir / "cv_parameters.json", parameter_doc)
    return {"cv_id": cv_id, "cv_directory": str(cv_dir), "percentage_directory": str(percentage_dir), "best_fold": best["fold"], "refit_budget": budget}


def _selected_parameters_from_search(args, method: str) -> tuple[dict, dict]:
    path = method_root(args.results_root, method) / "Grid" / f"{args.percentage}_percent" / "selected_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Search selection not found: {path}. Run --stage search first.")
    document = read_json(path)
    return document["parameters"], {"parameter_source": "search", "selection_file": str(path.resolve()), "selection_file_sha256": sha256_file(path), "candidate_id": document["candidate_id"]}


def _resolve_cv(args, method: str, parameters: dict) -> tuple[str, Path]:
    registry_path = method_root(args.results_root, method) / "CV" / "cv_runs.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"CV registry not found: {registry_path}")
    fingerprint = parameter_fingerprint(method, parameters, args.model_repo_id, args.model_revision)
    record = next((item for item in read_json(registry_path)["runs"] if item["parameter_fingerprint"] == fingerprint), None)
    if record is None:
        raise RuntimeError("No CV_x matches the requested parameters/model revision.")
    return record["cv_id"], method_root(args.results_root, method) / "CV" / record["cv_id"]


def run_test(args, method: str, manifest: dict, parameters: dict, provenance: dict) -> dict:
    cv_id, cv_dir = _resolve_cv(args, method, parameters)
    percentage_dir = cv_dir / f"{args.percentage}_percent"
    output_dir = method_root(args.results_root, method) / "Test" / cv_id / args.model_source / f"{args.percentage}_percent"
    if (output_dir / "metrics_summary.json").is_file() and not args.force_test:
        if args.resume:
            return {
                "cv_id": cv_id,
                "output_directory": str(output_dir),
                "metrics": read_json(output_dir / "metrics_summary.json"),
                "reused_completed_result": True,
            }
        raise FileExistsError(f"Held-out result already exists: {output_dir}. Use --force_test to replace it.")
    heldout = _split_cached_rows(args, heldout_mapping(manifest))
    development = _split_cached_rows(args, development_mapping(manifest))
    validate_no_overlap(development, heldout)
    raw_manifest = Path(args.cache_dir) / "cache_manifest.json"
    feature_store = _feature_store(args, raw_manifest) if method == "linear_probe" else None
    if args.model_source == "refit_full_development":
        checkpoints = [percentage_dir / "refit_full_development" / "best_model.pt"]
    elif args.model_source == "best_cv_fold":
        checkpoints = [Path(read_json(percentage_dir / "best_fold_model.json")["checkpoint"])]
    elif args.model_source == "cv_ensemble":
        checkpoints = [percentage_dir / f"fold_{fold:02d}" / "best_model.pt" for fold in range(1, 6)]
    elif args.model_source == "provided_checkpoint":
        if args.checkpoint is None:
            raise ValueError("provided_checkpoint requires --checkpoint.")
        checkpoints = [Path(args.checkpoint)]
    else:
        raise ValueError(f"Unknown model source: {args.model_source}")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if len(checkpoints) == 1:
        metrics = evaluate_checkpoint(heldout, checkpoints[0], output_dir, method=method, device=torch.device(args.device), feature_store=feature_store, repo_id=args.model_repo_id, revision=args.model_revision, prefix="test", num_workers=args.num_workers)
    else:
        probabilities, labels = [], None
        for checkpoint in checkpoints:
            logits, current_labels = predict_checkpoint(heldout, checkpoint, method=method, device=torch.device(args.device), feature_store=feature_store, repo_id=args.model_repo_id, revision=args.model_revision, num_workers=args.num_workers)
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            labels = current_labels if labels is None else labels
            if not torch.equal(labels, current_labels):
                raise RuntimeError("Ensemble checkpoints produced inconsistent patient ordering.")
        averaged = np.mean(probabilities, axis=0)
        truth = labels.numpy()
        metrics = compute_metrics(truth, averaged.argmax(axis=1), averaged, list(CLASS_NAMES))
        predictions, frame = format_predictions([row["patient_id"] for row in heldout], [row["label_str"] for row in heldout], averaged, list(CLASS_NAMES))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "metrics_summary.json", metrics)
        save_json(output_dir / "test_patient_predictions.json", predictions)
        frame.to_csv(output_dir / "test_probabilities.csv", index=False)
    selection = {"schema_version": 1, "cv_id": cv_id, "method": method, "model_source": args.model_source, "parameters": parameters, "parameter_provenance": provenance, "checkpoints": [str(path.resolve()) for path in checkpoints], "test_results_directory": str(output_dir.resolve()), "raw_cache_manifest": str(raw_manifest.resolve()), "raw_cache_manifest_sha256": sha256_file(raw_manifest), "heldout_test_used_during_selection": False, "evaluated_at_utc": utc_now()}
    save_json(output_dir / "model_selection.json", selection)
    return {"cv_id": cv_id, "output_directory": str(output_dir), "metrics": metrics}


def run_pipeline(args, method: str) -> dict:
    if method not in METHOD_FOLDERS:
        raise ValueError(f"Unknown Pillar-0 method: {method}")
    args.dataset_root, args.results_root = Path(args.dataset_root).resolve(), Path(args.results_root).resolve()
    args.cache_dir = (Path(args.cache_root).resolve() / "Shared" / args.cache_namespace)
    manifest = load_manifest(args.dataset_root, args.percentage)
    if args.dry_run:
        profile_parameters, provenance = ({}, {"parameter_source": "search"}) if args.parameter_source == "search" else load_parameter_profile(method, args.parameter_source, args.parameter_config)
        return {"status": "dry_run", "method": method, "stage": args.stage, "percentage": args.percentage, "classes": list(CLASS_NAMES), "parameters": profile_parameters, "parameter_provenance": provenance, "model_repo_id": args.model_repo_id, "model_revision": args.model_revision, "cache_directory": str(args.cache_dir), "cache_manifest_exists": (args.cache_dir / "cache_manifest.json").is_file(), "heldout_test_planned_only": True}
    if not (args.cache_dir / "cache_manifest.json").is_file():
        raise FileNotFoundError(f"Pillar-0 cache is not prepared: {args.cache_dir / 'cache_manifest.json'}")
    if args.parameter_source == "search":
        if args.stage in {"search", "all"}:
            parameters, provenance = run_search(args, method, manifest)
        else:
            parameters, provenance = _selected_parameters_from_search(args, method)
    else:
        parameters, provenance = load_parameter_profile(method, args.parameter_source, args.parameter_config)
    result = {"method": method, "percentage": args.percentage, "parameters": parameters, "parameter_provenance": provenance}
    if args.stage == "search":
        return result
    if args.stage in {"cv", "all"}:
        result["cv"] = run_cv(args, method, manifest, parameters, provenance)
    if args.stage in {"test", "all"}:
        result["test"] = run_test(args, method, manifest, parameters, provenance)
    return result
