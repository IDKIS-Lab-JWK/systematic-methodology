from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from COLIPRI import CLASS_NAMES  # noqa: E402
from COLIPRI.backend import (  # noqa: E402
    COLIPRIBackend,
    DeterministicBatchStream,
    effective_number_weights,
)
from COLIPRI.cli import (  # noqa: E402
    _install_cpu_batch_stream_resume_compatibility,
    _install_failed_candidate_materialization_compatibility,
)
from _linear_probe.common import (  # noqa: E402
    candidate_id,
    read_json,
    sha256_file,
    validate_completion,
)


PACKAGE = Path(__file__).resolve().parents[1]


def _profile(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def test_publication_profile_builds_exact_ordered_twenty_candidate_grid() -> None:
    candidates = COLIPRIBackend().build_candidates(
        _profile("publication_pooling_optimised_linear_probe.json")
    )

    assert len(candidates) == 20
    assert [item["pooling"] for item in candidates[:4]] == ["average"] * 4
    assert [item["learning_rate"] for item in candidates[:4]] == [
        0.003,
        0.01,
        0.03,
        0.1,
    ]
    assert candidates[-1]["pooling"] == "multi_learned_attention"
    assert candidates[-1]["learning_rate"] == 0.1


def test_explicit_candidate_id_must_be_canonical_hex_fingerprint() -> None:
    parameters = COLIPRIBackend().build_candidates(
        _profile("default_pooling_optimised_linear_probe.json")
    )[0]
    with pytest.raises(ValueError, match="canonical 12-character"):
        COLIPRIBackend._validate_candidate_parameters(
            {**parameters, "candidate_id": "my_run"}
        )

    canonical = candidate_id(parameters)
    COLIPRIBackend._validate_candidate_parameters(
        {**parameters, "candidate_id": canonical}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["search_space"]["pooling"].reverse(), "pooling"),
        (
            lambda value: value["references"].update({"official_revision": "main"}),
            "revision",
        ),
        (
            lambda value: value["fixed_parameters"].update({"momentum": 0.9}),
            "recipe",
        ),
        (
            lambda value: value["search_space"].update({"candidate_count": 19}),
            "20 candidates",
        ),
    ],
)
def test_publication_profile_mutations_are_rejected(mutation, message: str) -> None:
    profile = copy.deepcopy(_profile("publication_pooling_optimised_linear_probe.json"))
    mutation(profile)

    with pytest.raises(ValueError, match=message):
        COLIPRIBackend().build_candidates(profile)


def test_fixed_parameter_contract_rejects_nonpublication_recipe_changes() -> None:
    profile = _profile("default_pooling_optimised_linear_probe.json")
    COLIPRIBackend().build_candidates(profile)
    profile["parameters"]["effective_batch_size"] = 8

    with pytest.raises(ValueError, match="effective_batch_size"):
        COLIPRIBackend().build_candidates(profile)


def test_effective_number_weights_are_training_fold_only_mean_one_weights() -> None:
    labels = [0] * 20 + [1] * 10 + [2] * 5 + [3] * 3 + [4]
    weights, counts = effective_number_weights(labels, 5, beta=0.999)

    assert counts.tolist() == [20, 10, 5, 3, 1]
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[4] > weights[3] > weights[2] > weights[1] > weights[0]


def test_weighted_microbatch_sum_matches_one_effective_batch_mean() -> None:
    logits = torch.randn(16, 5, generator=torch.Generator().manual_seed(3))
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 4, 4, 2, 1, 0])
    weights = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0])
    denominator = weights[targets].sum()
    accumulated = sum(
        torch.nn.functional.cross_entropy(
            logits[start : start + 2],
            targets[start : start + 2],
            weight=weights,
            reduction="sum",
        )
        / denominator
        for start in range(0, 16, 2)
    )
    full = torch.nn.functional.cross_entropy(logits, targets, weight=weights)

    assert accumulated.item() == pytest.approx(full.item(), rel=1e-6)


def test_batch_stream_resume_is_exact() -> None:
    stream = DeterministicBatchStream(length=7, seed=91)
    stream.next(11)
    state = stream.state_dict()
    expected = stream.next(31)

    resumed = DeterministicBatchStream(length=7, seed=999, state=state)

    assert resumed.next(31) == expected


def test_batch_stream_resume_compatibility_is_idempotent_and_exact() -> None:
    stream = DeterministicBatchStream(length=9, seed=117)
    stream.next(13)
    state = stream.state_dict()
    expected = stream.next(29)

    _install_cpu_batch_stream_resume_compatibility()
    installed = DeterministicBatchStream.load_state_dict
    _install_cpu_batch_stream_resume_compatibility()

    assert DeterministicBatchStream.load_state_dict is installed
    resumed = DeterministicBatchStream(length=9, seed=999)
    resumed.load_state_dict(state)
    assert resumed.next(29) == expected
    assert resumed.order.device.type == "cpu"


def test_failed_candidate_is_sealed_without_fabricated_probabilities(
    tmp_path: Path, monkeypatch
) -> None:
    _install_failed_candidate_materialization_compatibility()
    backend = COLIPRIBackend()
    monkeypatch.setattr(backend, "feature_identity", lambda _args: {"cache": "test"})
    parameters = backend.build_candidates(_profile("default_pooling_optimised_linear_probe.json"))[0]
    parameters["candidate_id"] = candidate_id(parameters)

    class ToyProbe(torch.nn.Module):
        strict_linear = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def architecture(self):
            return {"type": "toy"}

    rows = [
        {"patient_id": f"p{label}", "class_name": name, "label": label}
        for label, name in enumerate(CLASS_NAMES)
    ]
    failure = {
        "status": "failed_optimizer_divergence",
        "reason": "non_finite_loss",
    }
    output = tmp_path / "failed_fold"
    result = backend._complete_validation_run(
        model=ToyProbe(),
        parameters=parameters,
        probabilities=np.full((len(rows), len(CLASS_NAMES)), np.nan),
        train_rows=rows,
        validation_rows=rows,
        output_dir=output,
        args=SimpleNamespace(),
        context={"fold": 1, "candidate_id": parameters["candidate_id"]},
        training={"candidate_failure": failure},
        logger=SimpleNamespace(info=lambda *_args: None),
    )

    validate_completion(output, "completion.json")
    metrics = read_json(output / "metrics_summary.json")
    assert result["metrics"] == metrics
    assert metrics["candidate_status"] == "failed"
    assert metrics["average_precision_macro"] is None
    assert metrics["metrics_status"] == "not_computed_failed_optimizer_divergence"
    assert json.loads(
        (output / "validation_predictions.json").read_text(encoding="utf-8")
    ) == []
    assert (output / "validation_probabilities.csv").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    summary = backend.selection_summary(parameters, [metrics] * 5)
    assert summary["eligible_for_selection"] is False
    assert summary["selection_value"] is None


def test_ranking_prefers_lower_lr_before_pooling_order_after_metric_ties() -> None:
    common = {
        "selection_value": 0.7,
        "std_average_precision_macro": 0.02,
        "mean_perforation_recall": 0.8,
    }
    high_lr_average = {
        **common,
        "candidate_id": "a",
        "parameters": {"pooling": "average", "learning_rate": 0.1},
    }
    low_lr_max = {
        **common,
        "candidate_id": "b",
        "parameters": {"pooling": "max", "learning_rate": 0.003},
    }

    assert COLIPRIBackend.ranking_key(low_lr_max) < COLIPRIBackend.ranking_key(
        high_lr_average
    )


def test_probe_precision_excludes_fp16() -> None:
    assert COLIPRIBackend._resolve_training_precision("cpu", "auto") == ("fp32", None)
    with pytest.raises(ValueError, match="auto, bf16, or fp32"):
        COLIPRIBackend._resolve_training_precision("cpu", "fp16")


def test_probe_precision_rejects_an_unavailable_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA probe training"):
        COLIPRIBackend._resolve_training_precision("cuda", "auto")


def test_artifact_load_rejects_feature_identity_mismatch(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "COLIPRI",
            "method_label": "PoolingOptimisedLinearProbe",
            "class_names": list(CLASS_NAMES),
            "feature_identity": {"feature_cache_manifest_sha256": "old"},
        },
        artifact,
    )
    backend = COLIPRIBackend()
    monkeypatch.setattr(
        backend,
        "feature_identity",
        lambda args: {"feature_cache_manifest_sha256": "new"},
    )

    with pytest.raises(RuntimeError, match="feature identity mismatch"):
        backend._load_artifact_model(artifact, SimpleNamespace())


def test_artifact_load_rejects_recorded_failed_grid_trial(tmp_path: Path) -> None:
    artifact = tmp_path / "failed.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "COLIPRI",
            "method_label": "PoolingOptimisedLinearProbe",
            "class_names": list(CLASS_NAMES),
            "training": {"candidate_failure": {"reason": "non_finite_gradient"}},
        },
        artifact,
    )

    with pytest.raises(RuntimeError, match="recorded failed grid trial"):
        COLIPRIBackend()._load_artifact_model(artifact, SimpleNamespace())


def test_backend_identity_binds_probe_seed_precision_and_microbatch(monkeypatch) -> None:
    backend = COLIPRIBackend()
    backend._store = SimpleNamespace(identity={"feature_cache_manifest_sha256": "cache"})
    base = SimpleNamespace(
        device="cpu", training_precision="auto", seed=42, micro_batch_size=1
    )
    identity = backend.feature_identity(base)
    assert set(identity["probe_implementation_identity"]["files"]) == {
        "COLIPRI/backend.py",
        "COLIPRI/pooling.py",
        "_linear_probe/common.py",
        "_linear_probe/metrics.py",
        "_pooling_optimised_linear_probe/workflow.py",
    }

    assert identity["probe_runtime"]["seed"] == 42
    assert identity["probe_runtime"]["training_precision_requested"] == "auto"
    assert identity["probe_runtime"]["training_precision_resolved"] == "fp32"
    assert identity["probe_runtime"]["micro_batch_size"] == 1
    assert identity["probe_runtime"]["torch_version"] == torch.__version__
    assert identity["probe_runtime"]["python_version"]
    assert identity["probe_runtime"]["numpy_version"] == np.__version__
    assert identity["probe_runtime"]["scipy_version"]
    assert identity["probe_runtime"]["scikit_learn_version"]
    assert identity["probe_runtime"]["device_type"] == "cpu"
    assert identity["probe_runtime"]["cuda_runtime"] is None
    assert backend.feature_identity(
        SimpleNamespace(
            device="cpu", training_precision="fp32", seed=42, micro_batch_size=1
        )
    ) != identity
    assert backend.feature_identity(
        SimpleNamespace(
            device="cpu", training_precision="auto", seed=43, micro_batch_size=1
        )
    ) != identity
    assert backend.feature_identity(
        SimpleNamespace(
            device="cpu", training_precision="auto", seed=42, micro_batch_size=2
        )
    ) != identity


def test_backend_reuses_one_resolved_feature_store(monkeypatch) -> None:
    from COLIPRI import backend as backend_module

    calls = {"init": 0, "validate": 0}

    class FakeStore:
        identity = {"feature_cache_manifest_sha256": "cache"}

        def __init__(self, args):
            calls["init"] += 1

        def validate_rows(self, rows):
            calls["validate"] += 1

    monkeypatch.setattr(backend_module, "FeatureStore", FakeStore)
    backend = COLIPRIBackend()
    args = SimpleNamespace(
        device="cpu", training_precision="auto", seed=42, micro_batch_size=1
    )

    backend.ensure_feature_rows(args, [{"patient_id": "p1"}])
    backend.ensure_feature_rows(args, [{"patient_id": "p2"}])
    backend.feature_identity(args)

    assert calls == {"init": 1, "validate": 2}


def test_shared_fold_completion_is_immutable_when_republishing_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    backend = COLIPRIBackend()
    parameters = COLIPRIBackend().build_candidates(
        _profile("default_pooling_optimised_linear_probe.json")
    )[0]
    parameters["candidate_id"] = "abc123"
    checkpoint = tmp_path / "shared_training" / "fold_01" / "last_checkpoint.pt"

    def fake_train(candidates, train_rows, checkpoint_path, args, context, logger):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            torch.save({"complete": torch.tensor(1)}, checkpoint_path)
        return {"abc123": object()}, {
            "training_identity_sha256": "training-identity",
            "checkpoint": str(checkpoint_path),
        }

    monkeypatch.setattr(backend, "_train_candidates", fake_train)
    monkeypatch.setattr(
        backend,
        "_evaluate_models",
        lambda models, rows, args: {"abc123": np.ones((1, 5)) / 5.0},
    )
    monkeypatch.setattr(
        backend,
        "_complete_validation_run",
        lambda **kwargs: {"status": "complete"},
    )
    args = SimpleNamespace()
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    context = {"pending_candidate_ids": ["abc123"]}

    backend.fit_grid_fold(
        [parameters], [], [{}], tmp_path, 1, args, context, logger
    )
    completion = checkpoint.parent / "completion.json"
    first_hash = sha256_file(completion)
    first_bytes = completion.read_bytes()

    backend.fit_grid_fold(
        [parameters], [], [{}], tmp_path, 1, args, context, logger
    )

    assert sha256_file(completion) == first_hash
    assert completion.read_bytes() == first_bytes


def test_shared_training_isolates_nonfinite_publication_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    backend = COLIPRIBackend()
    base = backend.build_candidates(_profile("default_pooling_optimised_linear_probe.json"))[0]
    good = {**base, "max_steps": 2, "learning_rate": 0.003}
    bad = {**base, "max_steps": 2, "learning_rate": 0.1}
    good["candidate_id"] = candidate_id(good)
    bad["candidate_id"] = candidate_id(bad)

    class ToyProbe(torch.nn.Module):
        def __init__(self, fail: bool):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(5))
            self.fail = fail

        def forward_tokens(self, tokens):
            logits = self.logits.unsqueeze(0).expand(tokens.shape[0], -1)
            return logits * torch.tensor(float("nan")) if self.fail else logits

    monkeypatch.setattr(
        backend,
        "_build_probe",
        lambda parameters, **_kwargs: ToyProbe(
            float(parameters["learning_rate"]) == 0.1
        ),
    )
    monkeypatch.setattr(
        backend,
        "_load_batch",
        lambda _args, _rows, _indices: (
            torch.zeros(1, 864, 1, 1, 1),
            torch.zeros(1, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_feature_store",
        lambda _args: SimpleNamespace(identity={"cache": "synthetic"}),
    )
    args = SimpleNamespace(
        micro_batch_size=1,
        device="cpu",
        training_precision="auto",
        seed=42,
        resume=True,
        checkpoint_interval=1,
    )
    logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    rows = [{"patient_id": f"p{index}", "label": index % 5} for index in range(10)]

    models, training = backend._train_candidates(
        [good, bad], rows, tmp_path / "checkpoint.pt", args, {"fold": 1}, logger
    )
    assert set(models) == {good["candidate_id"], bad["candidate_id"]}
    assert training["optimizer_steps"] == 2
    assert training["successful_candidate_ids"] == [good["candidate_id"]]
    failure = training["candidate_failures"][bad["candidate_id"]]
    assert failure["status"] == "failed_optimizer_divergence"
    assert failure["reason"] == "non_finite_loss"
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["candidate_failures"] == training["candidate_failures"]


def test_failed_candidate_summary_is_ineligible_and_ranked_last() -> None:
    backend = COLIPRIBackend()
    parameters = backend.build_candidates(_profile("default_pooling_optimised_linear_probe.json"))[0]
    complete_metrics = {
        "average_precision_macro": 0.5,
        "f1_macro": 0.4,
        "balanced_accuracy": 0.45,
        "per_class": {name: {"recall": 0.5} for name in CLASS_NAMES},
        "candidate_status": "complete",
        "candidate_failure": None,
    }
    failed_metrics = {
        **complete_metrics,
        "candidate_status": "failed",
        "candidate_failure": {"reason": "non_finite_gradient"},
    }
    complete = {
        "candidate_id": "a" * 12,
        "parameters": parameters,
        **backend.selection_summary(parameters, [complete_metrics] * 5),
    }
    failed = {
        "candidate_id": "b" * 12,
        "parameters": parameters,
        **backend.selection_summary(
            parameters, [failed_metrics, *([complete_metrics] * 4)]
        ),
    }

    assert complete["eligible_for_selection"] is True
    assert failed["eligible_for_selection"] is False
    assert failed["selection_value"] is None
    assert backend.ranking_key(complete) < backend.ranking_key(failed)

