from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from FlexiCT import CLASS_NAMES, FEATURE_DIM
from FlexiCT.backend import FlexiCTBackend
from _linear_probe.common import ContractError, candidate_id


class WarningPipeline(Pipeline):
    def fit(self, X, y=None, **params):
        warnings.warn("synthetic convergence boundary", ConvergenceWarning)
        return super().fit(X, y, **params)


class RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message, *args) -> None:
        self.warnings.append(message % args)

    def info(self, *_args) -> None:
        return None


def test_feature_identity_binds_local_probe_implementation(monkeypatch) -> None:
    backend = FlexiCTBackend()
    store = type("Store", (), {"identity": {"feature_contract_sha256": "a" * 64}})()
    monkeypatch.setattr(backend, "_feature_store", lambda _args: store)

    identity = backend.feature_identity(object())

    assert set(identity["probe_implementation_identity"]["files"]) == {
        "FlexiCT/backend.py",
        "_linear_probe/common.py",
        "_linear_probe/metrics.py",
        "_linear_probe/workflow.py",
    }
    assert identity["probe_runtime"]["python_version"]
    assert identity["probe_runtime"]["scipy_version"]


def test_probe_execution_rejects_stat_only_source_identity() -> None:
    backend = FlexiCTBackend()
    with pytest.raises(ContractError, match="SHA-256 source identity"):
        backend._feature_store(type("Args", (), {"source_fingerprint": "stat"})())


def native_parameters(C: float = 1.0) -> dict:
    return {
        "classifier": "multinomial_logistic_regression",
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 20000,
        "random_state": 0,
        "feature_standardization": "StandardScaler fit on training partition only",
        "encoder_frozen": True,
        "feature_dimension": FEATURE_DIM,
        "C": C,
    }


def test_explicit_candidate_id_must_be_canonical_fingerprint() -> None:
    parameters = native_parameters()
    with pytest.raises(ContractError, match="canonical 12-character"):
        FlexiCTBackend._validate_parameters(
            {**parameters, "candidate_id": "my_run"}
        )
    FlexiCTBackend._validate_parameters(
        {**parameters, "candidate_id": candidate_id(parameters)}
    )


def synthetic_matrix(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(len(CLASS_NAMES), dtype=np.int64), 3)
    features = np.zeros((len(labels), FEATURE_DIM), dtype=np.float32)
    for index, label in enumerate(labels):
        features[index, label] = 5.0 + index * 0.01
        features[index, 10] = offset + index
    return features, labels


def test_native_pipeline_has_exact_publication_recipe() -> None:
    pipeline = FlexiCTBackend._pipeline(native_parameters(C=0.1))
    assert list(pipeline.named_steps) == ["standard_scaler", "logistic_regression"]
    assert isinstance(pipeline.named_steps["standard_scaler"], StandardScaler)
    classifier = pipeline.named_steps["logistic_regression"]
    assert classifier.C == 0.1
    assert classifier.penalty == "l2"
    assert classifier.solver == "lbfgs"
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 20000
    assert classifier.random_state == 0
    assert classifier.multi_class == "multinomial"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("C", 0),
        ("C", float("nan")),
        ("solver", "saga"),
        ("class_weight", None),
        ("max_iter", 2000),
        ("encoder_frozen", False),
        ("feature_dimension", 1728),
    ],
)
def test_native_recipe_mutations_fail_closed(field: str, value) -> None:
    parameters = native_parameters()
    parameters[field] = value
    with pytest.raises(ContractError):
        FlexiCTBackend._pipeline(parameters)


def test_scaler_is_fit_only_on_training_partition() -> None:
    train_x, train_y = synthetic_matrix(offset=0.0)
    validation_x, _ = synthetic_matrix(offset=10_000.0)
    pipeline = FlexiCTBackend._pipeline(native_parameters())
    pipeline.fit(train_x, train_y)
    scaler = pipeline.named_steps["standard_scaler"]
    np.testing.assert_allclose(scaler.mean_, train_x.mean(axis=0), atol=1e-10)
    assert not np.isclose(scaler.mean_[10], np.vstack([train_x, validation_x]).mean(axis=0)[10])


def test_probability_contract_requires_all_five_classes_and_normalizes() -> None:
    features, labels = synthetic_matrix()
    complete = FlexiCTBackend._pipeline(native_parameters()).fit(features, labels)
    probabilities = FlexiCTBackend._aligned_probabilities(complete, features[:4])
    assert probabilities.shape == (4, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    incomplete = FlexiCTBackend._pipeline(native_parameters()).fit(
        features[labels != 4], labels[labels != 4]
    )
    with pytest.raises(RuntimeError, match="class order"):
        FlexiCTBackend._aligned_probabilities(incomplete, features[:4])


def test_training_partition_requires_all_canonical_classes() -> None:
    FlexiCTBackend._validate_training_labels(np.arange(5))
    with pytest.raises(ContractError, match="all canonical classes"):
        FlexiCTBackend._validate_training_labels(np.arange(4))


def test_arrays_reject_empty_and_out_of_range_partitions(monkeypatch) -> None:
    backend = FlexiCTBackend()
    with pytest.raises(ContractError, match="empty"):
        backend._arrays(object(), [])

    class Store:
        @staticmethod
        def load(_row):
            import torch

            return torch.zeros(FEATURE_DIM)

    backend._store = Store()
    with pytest.raises(ContractError, match="labels"):
        backend._arrays(object(), [{"label": 5, "patient_id": "x"}])


def test_fit_records_and_logs_convergence_warning(
    monkeypatch, tmp_path: Path
) -> None:
    train_x, train_y = synthetic_matrix(offset=0.0)
    validation_x, validation_y = synthetic_matrix(offset=1.0)
    train_rows = [
        {
            "partition": "train",
            "patient_id": f"tr-{index}",
            "label": int(label),
            "class_name": CLASS_NAMES[int(label)],
        }
        for index, label in enumerate(train_y)
    ]
    validation_rows = [
        {
            "partition": "validation",
            "patient_id": f"va-{index}",
            "label": int(label),
            "class_name": CLASS_NAMES[int(label)],
        }
        for index, label in enumerate(validation_y)
    ]
    backend = FlexiCTBackend()

    def arrays(_args, rows):
        return (train_x, train_y) if rows[0]["partition"] == "train" else (
            validation_x,
            validation_y,
        )

    def warning_pipeline(_parameters):
        return WarningPipeline(
            [
                ("standard_scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=20000,
                        random_state=0,
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

    monkeypatch.setattr(backend, "_arrays", arrays)
    monkeypatch.setattr(backend, "_pipeline", warning_pipeline)
    monkeypatch.setattr(
        backend,
        "feature_identity",
        lambda _args: {
            "feature_contract_sha256": "a" * 64,
            "feature_cache_manifest_sha256": "b" * 64,
            "feature_entries_sha256": "c" * 64,
        },
    )
    logger = RecordingLogger()
    output = tmp_path / "fit"
    backend.fit_one(
        native_parameters(),
        train_rows,
        validation_rows,
        output,
        object(),
        {"fold": 1},
        logger,
    )
    state = json.loads((output / "training_state.json").read_text(encoding="utf-8"))
    assert state["convergence_warnings"] == ["synthetic convergence boundary"]
    assert state["method_label"] == "LinearProbe"
    assert state["linear_classifier"] is True
    assert state["strict_raw_backbone_probe"] is False
    assert state["strict_linear"] is False
    assert any("synthetic convergence boundary" in item for item in logger.warnings)
    assert (output / "completion.json").is_file()
    restored = joblib.load(output / backend.artifact_filename)
    assert isinstance(restored, Pipeline)


def test_predict_rejects_non_pipeline_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "bad.joblib"
    joblib.dump({"not": "a pipeline"}, artifact)
    with pytest.raises(RuntimeError, match="Unexpected"):
        FlexiCTBackend().predict(artifact, [], object(), {})
