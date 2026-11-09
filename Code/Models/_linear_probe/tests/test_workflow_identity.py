from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from _linear_probe.common import (  # noqa: E402
    CLASS_NAMES,
    ContractError,
    atomic_write_json,
    completion_seal,
    fingerprint,
    read_json,
    tree_completion_seal,
)
from _linear_probe.workflow import (  # noqa: E402
    _grid_has_durable_training_artifacts,
    _load_completed_grid,
    _register_cv,
    _validate_test_completion,
    _validate_cv_completion,
    run_cv,
    run_test,
)


def test_grid_pretraining_scaffold_is_refreshable_but_checkpoint_is_durable(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "grid_context.json", {"old": True})
    _write_json(tmp_path / "grid_index.json", {"candidates": []})
    (tmp_path / "Logs").mkdir()
    (tmp_path / "Logs" / "run.log").write_text("failed before checkpoint\n")
    (tmp_path / "shared_training" / "fold_01").mkdir(parents=True)

    assert _grid_has_durable_training_artifacts(tmp_path) is False

    (tmp_path / "shared_training" / "fold_01" / "last_checkpoint.pt").write_bytes(
        b"checkpoint"
    )
    assert _grid_has_durable_training_artifacts(tmp_path) is True


class _Backend:
    name = "SyntheticFM"
    artifact_filename = "model.bin"

    def __init__(self, profile: Path | None = None):
        self.default_profile_path = profile
        self.publication_profile_path = profile
        self.search_profile_path = profile
        self.validated_rows: list[list[dict]] = []

    def ensure_feature_rows(self, _args, rows):
        self.validated_rows.append(list(rows))

    def build_candidates(self, profile):
        return [dict(profile["parameters"])]

    def feature_identity(self, _args):
        return {"cache_manifest_sha256": "a" * 64}


class _InterruptingBackend(_Backend):
    def __init__(self, profile: Path):
        super().__init__(profile)
        self.fit_calls = 0

    def fit_one(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise RuntimeError("synthetic interruption")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_cv_registry_identity_binds_seed_and_lookup_does_not_create(tmp_path: Path) -> None:
    backend = _Backend()
    parameters = {"C": 1.0}
    model_identity = {"checkpoint_sha256": "b" * 64}
    provenance = {"parameter_source": "default"}

    first, _ = _register_cv(
        tmp_path,
        backend,
        parameters,
        model_identity,
        provenance,
        seed=42,
    )
    repeated, _ = _register_cv(
        tmp_path,
        backend,
        parameters,
        model_identity,
        provenance,
        seed=42,
    )
    changed_seed, _ = _register_cv(
        tmp_path,
        backend,
        parameters,
        model_identity,
        provenance,
        seed=43,
    )

    assert (first, repeated, changed_seed) == ("CV_1", "CV_1", "CV_2")
    with pytest.raises(ContractError, match="No completed CV identity"):
        _register_cv(
            tmp_path,
            backend,
            {"C": 2.0},
            model_identity,
            provenance,
            seed=42,
            create=False,
        )
    registry = json.loads(
        (tmp_path / "LinearProbe" / "CV" / "cv_runs.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["cv_id"] for item in registry["runs"]] == ["CV_1", "CV_2"]


def test_cv_registry_repairs_crash_only_in_create_mode_and_binds_provenance(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    parameters = {"C": 1.0}
    model_identity = {"checkpoint_sha256": "b" * 64}
    provenance = {"parameter_source": "publication", "profile_sha256": "c" * 64}
    _, root = _register_cv(
        tmp_path, backend, parameters, model_identity, provenance, seed=42
    )
    document = root / "cv_parameters.json"
    document.unlink()

    with pytest.raises(ContractError, match="identity document is missing"):
        _register_cv(
            tmp_path,
            backend,
            parameters,
            model_identity,
            provenance,
            seed=42,
            create=False,
        )
    assert not document.exists()

    _, repaired = _register_cv(
        tmp_path, backend, parameters, model_identity, provenance, seed=42
    )
    assert repaired == root
    assert json.loads(document.read_text(encoding="utf-8"))[
        "parameter_provenance_identity"
    ] == provenance

    different_source, _ = _register_cv(
        tmp_path,
        backend,
        parameters,
        model_identity,
        {"parameter_source": "manual"},
        seed=42,
    )
    assert different_source == "CV_2"


def test_cv_registry_keeps_one_id_across_percentage_specific_grid_provenance(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    common = {
        "parameter_source": "publication",
        "profile_sha256": "c" * 64,
        "candidate_id": "abc123def456",
    }
    first, _ = _register_cv(
        tmp_path,
        backend,
        {"C": 1.0},
        {"checkpoint_sha256": "b" * 64},
        {
            **common,
            "grid_root": "Publication_Grid/100_percent",
            "selection_file_sha256": "1" * 64,
        },
        seed=42,
    )
    second, _ = _register_cv(
        tmp_path,
        backend,
        {"C": 1.0},
        {"checkpoint_sha256": "b" * 64},
        {
            **common,
            "grid_root": "Publication_Grid/80_percent",
            "selection_file_sha256": "2" * 64,
        },
        seed=42,
    )

    assert first == second == "CV_1"


def test_completed_grid_must_match_exact_current_context(tmp_path: Path) -> None:
    identifier = "abc123def456"
    context = {
        "candidate_ids": [identifier],
        "candidate_count": 1,
        "split_manifest_sha256": "c" * 64,
        "model_identity": {"feature_cache_manifest_sha256": "d" * 64},
        "fold_partitions": [
            {
                "fold": fold,
                "training_rows_sha256": f"train-{fold}",
                "validation_rows_sha256": f"validation-{fold}",
            }
            for fold in range(1, 6)
        ],
    }
    parameters = {"C": 1.0, "candidate_id": identifier}
    fold_metrics = [{"fold": fold} for fold in range(1, 6)]
    ranked = {
        "candidate_id": identifier,
        "parameters": parameters,
        "fold_metrics": fold_metrics,
        "rank": 1,
    }
    selected = {
        "candidate_id": identifier,
        "parameters": parameters,
        "grid_context_sha256": fingerprint(context),
    }
    _write_json(tmp_path / "grid_context.json", context)
    _write_json(
        tmp_path / "grid_index.json",
        {
            "candidate_count": 1,
            "candidates": [{"candidate_id": identifier, "parameters": parameters}],
        },
    )
    _write_json(tmp_path / "grid_summary.json", {"ranking": [ranked]})
    _write_json(tmp_path / "selected_parameters.json", selected)
    candidate_root = tmp_path / f"candidate_{identifier}"
    _write_json(
        candidate_root / "candidate_summary.json",
        {key: value for key, value in ranked.items() if key != "rank"},
    )
    for fold in range(1, 6):
        fold_root = candidate_root / f"fold_{fold:02d}"
        artifact = fold_root / "model.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"fold:{fold}".encode("utf-8"))
        metrics_path = fold_root / "metrics_summary.json"
        _write_json(metrics_path, fold_metrics[fold - 1])
        parameters_path = fold_root / "parameters.json"
        _write_json(parameters_path, parameters)
        state_path = fold_root / "training_state.json"
        _write_json(
            state_path,
            {
                "context": {
                    **context,
                    "fold": fold,
                    "candidate_id": identifier,
                },
                "training_rows_sha256": f"train-{fold}",
                "validation_rows_sha256": f"validation-{fold}",
                "feature_identity": context["model_identity"],
            },
        )
        atomic_write_json(
            fold_root / "completion.json",
            completion_seal(
                [artifact, metrics_path, parameters_path, state_path],
                context={"fold": fold},
            ),
        )
    atomic_write_json(
        tmp_path / "grid_completion.json",
        completion_seal(
            [
                tmp_path / "grid_context.json",
                tmp_path / "grid_index.json",
                tmp_path / "grid_summary.json",
                tmp_path / "selected_parameters.json",
            ],
            context={"grid_context_sha256": fingerprint(context)},
        ),
    )

    assert _load_completed_grid(tmp_path, expected_context=context) == selected
    changed = {**context, "split_manifest_sha256": "e" * 64}
    with pytest.raises(ContractError, match="does not match"):
        _load_completed_grid(tmp_path, expected_context=changed)

    fold_02 = candidate_root / "fold_02"
    _write_json(
        fold_02 / "training_state.json",
        read_json(candidate_root / "fold_01" / "training_state.json"),
    )
    atomic_write_json(
        fold_02 / "completion.json",
        completion_seal(
            [
                fold_02 / "model.bin",
                fold_02 / "metrics_summary.json",
                fold_02 / "parameters.json",
                fold_02 / "training_state.json",
            ],
            context={"fold": 2},
        ),
    )
    with pytest.raises(ContractError, match="fold number/candidate"):
        _load_completed_grid(tmp_path, expected_context=context)


def test_completed_cv_must_match_exact_percentage_context(tmp_path: Path) -> None:
    parameters = {"C": 1.0}
    model_identity = {"feature_cache_manifest_sha256": "a" * 64}
    context = {
        "cv_id": "CV_1",
        "percentage": 100,
        "seed": 42,
        "split_manifest_sha256": "f" * 64,
        "parameters": parameters,
        "model_identity": model_identity,
        "selected_candidate_id": None,
        "parameter_provenance": {"parameter_source": "default"},
        "development_row_count": 25,
        "development_training_rows_sha256": "development-rows",
        "fold_partitions": [
            {
                "fold": fold,
                "training_rows_sha256": f"train-{fold}",
                "validation_rows_sha256": f"validation-{fold}",
            }
            for fold in range(1, 6)
        ],
    }
    nested_artifacts = []
    for fold in range(1, 6):
        nested = tmp_path / f"fold_{fold:02d}"
        artifact = nested / "model.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(f"classifier:fold-{fold}".encode("utf-8"))
        parameters_path = nested / "parameters.json"
        state_path = nested / "training_state.json"
        _write_json(parameters_path, parameters)
        _write_json(
            state_path,
            {
                "context": {
                    "cv_id": "CV_1",
                    "percentage": 100,
                    "fold": fold,
                    "model_identity": model_identity,
                },
                "training_rows_sha256": f"train-{fold}",
                "validation_rows_sha256": f"validation-{fold}",
                "feature_identity": model_identity,
            },
        )
        atomic_write_json(
            nested / "completion.json",
            completion_seal(
                [artifact, parameters_path, state_path], context={"fold": fold}
            ),
        )
        nested_artifacts.extend([artifact, nested / "completion.json"])
    refit = tmp_path / "refit_full_development"
    refit_artifact = refit / "model.bin"
    refit_artifact.parent.mkdir(parents=True)
    refit_artifact.write_bytes(b"classifier:refit")
    refit_parameters = refit / "parameters.json"
    refit_state = refit / "training_state.json"
    _write_json(refit_parameters, parameters)
    _write_json(
        refit_state,
        {
            "context": {
                "cv_id": "CV_1",
                "percentage": 100,
                "model_identity": model_identity,
            },
            "row_count": 25,
            "rows_sha256": "development-rows",
            "feature_identity": model_identity,
        },
    )
    atomic_write_json(
        refit / "completion.json",
        completion_seal(
            [refit_artifact, refit_parameters, refit_state],
            context={"refit": True},
        ),
    )
    nested_artifacts.extend([refit_artifact, refit / "completion.json"])
    atomic_write_json(
        tmp_path / "cv_completion.json",
        tree_completion_seal(tmp_path, nested_artifacts, context=context),
    )

    _validate_cv_completion(tmp_path, expected_context=context)
    with pytest.raises(ContractError, match="does not match"):
        _validate_cv_completion(
            tmp_path, expected_context={**context, "seed": 7}
        )

    # Simulate a self-consistent fold_01 tree transplanted into fold_02, then
    # reseal both levels.  Cryptographic integrity alone is insufficient; the
    # embedded fold/partition identity must reject it.
    fold_02 = tmp_path / "fold_02"
    transplanted_state = read_json(tmp_path / "fold_01" / "training_state.json")
    _write_json(fold_02 / "training_state.json", transplanted_state)
    atomic_write_json(
        fold_02 / "completion.json",
        completion_seal(
            [
                fold_02 / "model.bin",
                fold_02 / "parameters.json",
                fold_02 / "training_state.json",
            ],
            context={"fold": 2},
        ),
    )
    atomic_write_json(
        tmp_path / "cv_completion.json",
        tree_completion_seal(tmp_path, nested_artifacts, context=context),
    )
    with pytest.raises(ContractError, match="different run context"):
        _validate_cv_completion(tmp_path, expected_context=context)


def _synthetic_manifest() -> dict:
    folds = []
    for fold in range(1, 6):
        folds.append(
            {
                "fold": fold,
                "patients_by_class": {
                    name: [f"dev_{fold}_{class_index}"]
                    for class_index, name in enumerate(CLASS_NAMES)
                },
            }
        )
    return {
        "cross_validation": {"folds": folds},
        "heldout_test": {
            "patients_by_class": {
                name: [f"heldout_{class_index}"]
                for class_index, name in enumerate(CLASS_NAMES)
            }
        },
    }


def test_completed_test_reuse_rejects_transplanted_model_source(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "patient_id": "heldout_1",
            "label": 0,
            "class_name": CLASS_NAMES[0],
        }
    ]
    selection = {
        "schema_version": 1,
        "model": "SyntheticFM",
        "method": "LinearProbe",
        "cv_id": "CV_1",
        "percentage": 100,
        "model_source": "refit_full_development",
        "classifier_artifacts": [
            {"path": "C:/sealed/model.bin", "sha256": "a" * 64}
        ],
        "parameters": {"C": 1.0},
        "model_identity": {"cache": "b" * 64},
        "heldout_rows_sha256": "c" * 64,
    }
    predictions = [
        {
            "patient_id": "heldout_1",
            "true_index": 0,
            "true_class": CLASS_NAMES[0],
            "predicted_index": 0,
            "predicted_class": CLASS_NAMES[0],
            "probabilities": {name: 0.2 for name in CLASS_NAMES},
        }
    ]
    _write_json(tmp_path / "model_selection.json", selection)
    _write_json(tmp_path / "test_patient_predictions.json", predictions)
    _write_json(tmp_path / "metrics_summary.json", {"accuracy": 1.0})
    (tmp_path / "test_probabilities.csv").write_text("patient_id\nheldout_1\n", encoding="utf-8")
    artifacts = [
        tmp_path / "metrics_summary.json",
        tmp_path / "test_patient_predictions.json",
        tmp_path / "test_probabilities.csv",
        tmp_path / "model_selection.json",
    ]
    atomic_write_json(
        tmp_path / "test_completion.json",
        completion_seal(
            artifacts,
            context={
                "cv_id": "CV_1",
                "model_source": "refit_full_development",
                "percentage": 100,
                "selection_sha256": fingerprint(selection),
            },
        ),
    )

    _validate_test_completion(
        tmp_path, expected_selection=selection, expected_rows=rows
    )
    with pytest.raises(ContractError, match="different model source"):
        _validate_test_completion(
            tmp_path,
            expected_selection={**selection, "model_source": "cv_ensemble"},
            expected_rows=rows,
        )


def test_test_stage_does_not_touch_heldout_rows_or_create_ghost_cv(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "Dataset"
    split = dataset / "Splits" / "100_percent.json"
    _write_json(split, _synthetic_manifest())
    # Development files exist so the exact model/CV preflight can run.  Held-out
    # files deliberately do not exist: a missing CV registry must fail first.
    for fold in range(1, 6):
        for class_index, class_name in enumerate(CLASS_NAMES):
            image = dataset / "NIFTI" / class_name / f"dev_{fold}_{class_index}.nii.gz"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"synthetic")
    profile = tmp_path / "default.json"
    _write_json(profile, {"parameters": {"C": 1.0}})
    results = tmp_path / "Results"
    args = SimpleNamespace(
        parameter_source="default",
        parameter_config=profile,
        grid_config=None,
        max_grid_candidates=None,
        percentage=100,
        dataset_root=dataset,
        results_root=results,
        cache_root=tmp_path / "Cache",
        cache_namespace="synthetic",
        model_source="refit_full_development",
        seed=42,
        force_test=False,
    )
    backend = _Backend(profile)
    manifest = json.loads(split.read_text(encoding="utf-8"))
    manifest["_path"] = str(split.resolve())
    manifest["_sha256"] = "1" * 64

    with pytest.raises(ContractError, match="No CV registry"):
        run_test(args, backend, manifest)
    assert len(backend.validated_rows) == 1
    assert all(row["patient_id"].startswith("dev_") for row in backend.validated_rows[0])
    assert not results.exists()


def test_partial_cv_resume_rejects_changed_percentage_context(tmp_path: Path) -> None:
    dataset = tmp_path / "Dataset"
    split = dataset / "Splits" / "100_percent.json"
    _write_json(split, _synthetic_manifest())
    for fold in range(1, 6):
        for class_index, class_name in enumerate(CLASS_NAMES):
            image = dataset / "NIFTI" / class_name / f"dev_{fold}_{class_index}.nii.gz"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"synthetic")
    profile = tmp_path / "default.json"
    _write_json(profile, {"parameters": {"C": 1.0}})
    args = SimpleNamespace(
        parameter_source="default",
        parameter_config=profile,
        grid_config=None,
        max_grid_candidates=None,
        percentage=100,
        dataset_root=dataset,
        results_root=tmp_path / "Results",
        cache_root=tmp_path / "Cache",
        cache_namespace="synthetic",
        model_source="refit_full_development",
        seed=42,
        resume=True,
    )
    manifest = json.loads(split.read_text(encoding="utf-8"))
    manifest["_path"] = str(split.resolve())
    manifest["_sha256"] = "1" * 64
    backend = _InterruptingBackend(profile)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_cv(args, backend, manifest)
    assert backend.fit_calls == 1

    with pytest.raises(ContractError, match="identity document is incompatible"):
        run_cv(args, backend, {**manifest, "_sha256": "2" * 64})
    assert backend.fit_calls == 1
