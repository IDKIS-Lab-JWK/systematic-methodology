from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from _linear_probe.common import ContractError, candidate_id
from _supervised_adaptation.profiles import build_candidates, strip_note_fields
from _supervised_adaptation.runtime import DeterministicBatchStream, torch_load
from _supervised_adaptation.safe_io import same_or_write_csv, same_or_write_json
from _supervised_adaptation.trainer import AdaptationBackend, input_rows_identity
from _supervised_adaptation.workflow import (
    _cv_provenance_identity,
    _parameter_selection_mode,
    _register_cv,
)


class Logger:
    def info(self, *_args, **_kwargs):
        return None


class TinyAdapter:
    name = "Tiny"
    method_label = "TinyPartialFineTune"
    method_id = "tiny_partial_finetune"

    def __init__(self, fail_on_training_call: int | None = None):
        self.publication_profile_path = Path(__file__)
        self.default_profile_path = Path(__file__)
        self.search_profile_path = Path(__file__)
        self.fail_on_training_call = fail_on_training_call
        self.training_calls = 0

    def implementation_paths(self):
        return [Path(__file__)]

    def planned_identity(self, _args):
        return {"synthetic": True}

    def preflight(self, _args, _rows):
        return None

    def resolve_identity(self, _args):
        return {"checkpoint": "synthetic-immutable-v1"}

    def representation_contract(self, _args):
        return {"input_dimension": 2}

    def prepare_rows(self, _args, rows):
        return [dict(item) for item in rows]

    def validate_parameters(self, parameters):
        assert parameters["optimizer"] == "sgd"

    def build_model(self, _args, _parameters, _device):
        return nn.Linear(2, 5)

    def validate_trainable_scope(self, model, _parameters):
        assert all(parameter.requires_grad for parameter in model.parameters())

    def optimizer_groups(self, model, parameters):
        return [
            {
                "group_name": "tiny",
                "params": list(model.parameters()),
                "lr": parameters["head_learning_rate"],
                "weight_decay": parameters["weight_decay"],
            }
        ]

    def set_training_mode(self, model, mode):
        model.train(mode)

    def move_inputs(self, inputs, device):
        return inputs.to(device)

    def load_batch(self, _args, rows, indices, *, training, generator):
        del generator
        if training:
            self.training_calls += 1
            if self.fail_on_training_call == self.training_calls:
                raise RuntimeError("synthetic interruption")
        inputs = torch.stack([rows[index]["features"] for index in indices])
        labels = torch.as_tensor([rows[index]["label"] for index in indices])
        return inputs, labels

    def architecture(self, _model, _parameters):
        return {"kind": "tiny-linear"}


def parameters() -> dict:
    value = {
        "optimizer": "sgd",
        "head_learning_rate": 0.05,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "nesterov": False,
        "scheduler": "cosine",
        "warmup_steps": 0,
        "minimum_lr_ratio": 0.0,
        "max_steps": 3,
        "effective_batch_size": 5,
        "eval_batch_size": 5,
        "validation_interval_steps": 1,
        "early_stopping_patience_steps": 0,
        "early_stopping_min_delta": 0.0,
        "class_weight_beta": 0.0,
        "gradient_clip_norm": 0.0,
    }
    return {**value, "candidate_id": candidate_id(value)}


def rows(prefix: str) -> list[dict]:
    return [
        {
            "patient_id": f"{prefix}_{index}",
            "class_name": ("Appendicitis", "Cholezystitis", "Normal", "Perforation", "Sigmoid")[index],
            "label": index,
            "features": torch.tensor([float(index), 1.0], dtype=torch.float32),
        }
        for index in range(5)
    ]


def args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        device="cpu",
        training_precision="fp32",
        micro_batch_size=5,
        checkpoint_interval=1,
        resume=True,
    )


def test_note_fields_are_inert_and_candidate_identity_is_executable_only():
    profile = {
        "# source (not executable)": "paper",
        "parameters": {
            "learning_rate": 0.1,
            "# explanation (not executable)": "ignored",
        },
    }
    cleaned = strip_note_fields(profile)
    assert "# source (not executable)" not in cleaned
    candidates = build_candidates(profile)
    assert candidates == [
        {"learning_rate": 0.1, "candidate_id": candidate_id({"learning_rate": 0.1})}
    ]


def test_parameter_selection_distinguishes_fixed_publication_and_manual_grid():
    one = [{"candidate_id": "fixed"}]
    many = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    assert _parameter_selection_mode(
        SimpleNamespace(parameter_source="publication"),
        {"parameter_selection": "fixed"},
        one,
    ) == "fixed"
    assert _parameter_selection_mode(
        SimpleNamespace(parameter_source="grid"), {}, many
    ) == "grid"
    assert _parameter_selection_mode(
        SimpleNamespace(parameter_source="publication"), {}, one
    ) == "fixed"
    with pytest.raises(ContractError, match="exactly one candidate"):
        _parameter_selection_mode(SimpleNamespace(parameter_source="publication"), {}, many)
    assert _parameter_selection_mode(
        SimpleNamespace(parameter_source="publication"),
        {"parameter_selection": "publication_grid"},
        many,
    ) == "grid"
    with pytest.raises(ContractError, match="at least two candidates"):
        _parameter_selection_mode(
            SimpleNamespace(parameter_source="publication"),
            {"parameter_selection": "publication_grid"},
            one,
        )
    with pytest.raises(ContractError, match="exactly one candidate"):
        _parameter_selection_mode(
            SimpleNamespace(parameter_source="publication"),
            {"parameter_selection": "fixed"},
            many,
        )


def test_batch_stream_resume_is_exact():
    stream = DeterministicBatchStream(7, 123)
    prefix = stream.next(9)
    state = stream.state_dict()
    expected = stream.next(25)
    resumed = DeterministicBatchStream(7, 999, state=state)
    assert resumed.next(25) == expected
    assert len(prefix) == 9


def test_interrupted_training_resumes_to_identical_artifact(tmp_path: Path):
    train_rows = rows("train")
    validation_rows = rows("validation")
    context = {"kind": "synthetic", "fold": 1, "percentage": 20}
    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        AdaptationBackend(TinyAdapter(fail_on_training_call=2)).fit_one(
            parameters(),
            train_rows,
            validation_rows,
            interrupted_root,
            args(),
            context,
            Logger(),
        )
    checkpoint = torch_load(interrupted_root / "last_checkpoint.pt")
    assert checkpoint["completed_optimizer_steps"] == 1

    AdaptationBackend(TinyAdapter()).fit_one(
        parameters(),
        train_rows,
        validation_rows,
        interrupted_root,
        args(),
        context,
        Logger(),
    )
    uninterrupted_root = tmp_path / "uninterrupted"
    AdaptationBackend(TinyAdapter()).fit_one(
        parameters(),
        train_rows,
        validation_rows,
        uninterrupted_root,
        args(),
        context,
        Logger(),
    )
    resumed = torch_load(interrupted_root / "model.pt")["trainable_state_dict"]
    uninterrupted = torch_load(uninterrupted_root / "model.pt")["trainable_state_dict"]
    resumed_training = __import__("json").loads((interrupted_root / "training_state.json").read_text())
    uninterrupted_training = __import__("json").loads((uninterrupted_root / "training_state.json").read_text())
    assert resumed_training["history"] == uninterrupted_training["history"]
    assert set(resumed) == set(uninterrupted)
    differences = {
        key: float((resumed[key] - uninterrupted[key]).abs().max())
        for key in resumed
    }
    assert all(torch.equal(resumed[key], uninterrupted[key]) for key in resumed), differences


def test_no_overwrite_writers_accept_identical_and_reject_different(tmp_path: Path):
    json_path = tmp_path / "result.json"
    same_or_write_json(json_path, {"b": [2, 3], "a": 1})
    same_or_write_json(json_path, {"a": 1, "b": [2, 3]})
    with pytest.raises(ContractError, match="refusing overwrite"):
        same_or_write_json(json_path, {"a": 2, "b": [2, 3]})

    csv_path = tmp_path / "result.csv"
    rows = [{"patient_id": "p1", "score": 0.25}]
    same_or_write_csv(csv_path, rows, ["patient_id", "score"])
    same_or_write_csv(csv_path, rows, ["patient_id", "score"])
    with pytest.raises(ContractError, match="refusing overwrite"):
        same_or_write_csv(
            csv_path,
            [{"patient_id": "p1", "score": 0.5}],
            ["patient_id", "score"],
        )


def test_cv_registry_repairs_only_its_missing_immutable_document(tmp_path: Path):
    backend = SimpleNamespace(name="Tiny", method_label="TinyPartialFineTune")
    identity = {"schema_version": 1, "parameters": {"learning_rate": 0.1}}
    cv_id, cv_root = _register_cv(
        tmp_path, backend=backend, identity=identity, create=True
    )
    assert cv_id == "CV_1"
    parameters_path = cv_root / "cv_parameters.json"
    parameters_path.unlink()

    resumed_id, resumed_root = _register_cv(
        tmp_path, backend=backend, identity=identity, create=True
    )
    assert (resumed_id, resumed_root) == (cv_id, cv_root)
    assert parameters_path.is_file()

    parameters_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="parameter identity changed"):
        _register_cv(tmp_path, backend=backend, identity=identity, create=False)


def test_cv_parameter_source_identity_ignores_percentage_specific_paths():
    first = {
        "parameter_source": "publication",
        "profile_sha256": "a" * 64,
        "candidate_id": "123456789abc",
        "grid_root": "Grid/20_percent",
        "grid_completion_sha256": "b" * 64,
    }
    second = {
        **first,
        "grid_root": "Grid/100_percent",
        "grid_completion_sha256": "c" * 64,
    }
    assert _cv_provenance_identity(first) == _cv_provenance_identity(second)


def test_input_file_identity_changes_with_concrete_source(tmp_path: Path):
    source = tmp_path / "patient.nii.gz"
    source.write_bytes(b"first")
    row = {
        "patient_id": "p1",
        "class_name": "Normal",
        "label": 2,
        "image_path": str(source),
    }
    first = input_rows_identity([row])
    assert input_rows_identity([row]) == first
    source.write_bytes(b"second-content")
    assert input_rows_identity([row]) != first
