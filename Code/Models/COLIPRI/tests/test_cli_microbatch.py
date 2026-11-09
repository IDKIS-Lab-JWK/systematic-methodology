from __future__ import annotations

import json
from pathlib import Path

import pytest

from COLIPRI import cli as probe_cli
from COLIPRI import train_primus_m_partial_finetune as partial_cli
from _supervised_adaptation import cli as shared_cli


PACKAGE = Path(__file__).resolve().parents[1]


def test_probe_default_reaches_pipeline_as_full_batch(monkeypatch):
    observed = []
    monkeypatch.setattr(
        probe_cli, "run_pipeline",
        lambda args, backend: observed.append(args.micro_batch_size) or {},
    )
    assert probe_cli.main(["--dry_run"]) == 0
    assert observed == [16]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_probe_accepts_full_batch_and_valid_accumulation_sizes(size):
    args = probe_cli.build_parser().parse_args(["--micro_batch_size", str(size)])
    assert args.micro_batch_size == size


@pytest.mark.parametrize("size, message", [
    (17, "equal to or lower than 16"),
    (32, "equal to or lower than 16"),
    (3, "divide the effective batch size (16) evenly"),
    (0, "must be positive"),
])
def test_probe_rejects_invalid_sizes_before_pipeline(size, message, monkeypatch, capsys):
    def unexpected_pipeline(*args):
        pytest.fail("Invalid microbatch reached the pipeline")

    monkeypatch.setattr(probe_cli, "run_pipeline", unexpected_pipeline)
    with pytest.raises(SystemExit) as exc:
        probe_cli.main(["--micro_batch_size", str(size), "--dry_run"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("scope", ["final_block", "final_stage"])
@pytest.mark.parametrize("source, expected", [("publication", 8), ("grid", 8), ("default", 4)])
def test_partial_default_follows_selected_profile(scope, source, expected, monkeypatch):
    observed = []
    monkeypatch.setattr(
        shared_cli, "run_pipeline",
        lambda args, backend: observed.append(args.micro_batch_size) or {},
    )
    assert partial_cli.main([
        "--adaptation_scope", scope, "--parameter_source", source, "--dry_run",
    ]) == 0
    assert observed == [expected]


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_partial_preserves_explicit_microbatch(size, monkeypatch):
    observed = []
    monkeypatch.setattr(
        shared_cli, "run_pipeline",
        lambda args, backend: observed.append(args.micro_batch_size) or {},
    )
    assert partial_cli.main(["--micro_batch_size", str(size), "--dry_run"]) == 0
    assert observed == [size]


@pytest.mark.parametrize("source, size, message", [
    ("publication", 16, "equal to or lower than 8"),
    ("publication", 32, "equal to or lower than 8"),
    ("default", 8, "equal to or lower than 4"),
    ("publication", 3, "divide the effective batch size (8) evenly"),
])
def test_partial_rejects_invalid_sizes_before_pipeline(source, size, message, monkeypatch, capsys):
    def unexpected_pipeline(*args):
        pytest.fail("Invalid microbatch reached the pipeline")

    monkeypatch.setattr(shared_cli, "run_pipeline", unexpected_pipeline)
    with pytest.raises(SystemExit) as exc:
        partial_cli.main([
            "--parameter_source", source, "--micro_batch_size", str(size), "--dry_run",
        ])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_partial_custom_profile_controls_default_and_limit(tmp_path, monkeypatch, capsys):
    profile = json.loads((PACKAGE / "default_primus_m_final_block_finetune.json").read_text())
    profile["parameters"]["effective_batch_size"] = 6
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(profile))
    observed = []
    monkeypatch.setattr(
        shared_cli, "run_pipeline",
        lambda args, backend: observed.append(args.micro_batch_size) or {},
    )
    argv = ["--parameter_source", "default", "--parameter_config", str(path), "--dry_run"]
    assert partial_cli.main(argv) == 0
    assert observed == [6]
    with pytest.raises(SystemExit):
        partial_cli.main([*argv, "--micro_batch_size", "8"])
    assert "equal to or lower than 6" in capsys.readouterr().err
    assert observed == [6]


def test_mixed_batch_grid_requires_explicit_common_microbatch(tmp_path, monkeypatch, capsys):
    profile = json.loads((PACKAGE / "default_primus_m_final_block_finetune.json").read_text())
    parameters = profile.pop("parameters")
    profile["parameter_source"] = "grid"
    profile["candidates"] = [
        {**parameters, "effective_batch_size": batch} for batch in (4, 8)
    ]
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(profile))
    observed = []
    monkeypatch.setattr(
        shared_cli, "run_pipeline",
        lambda args, backend: observed.append(args.micro_batch_size) or {},
    )
    argv = ["--parameter_source", "grid", "--grid_config", str(path), "--dry_run"]
    with pytest.raises(SystemExit):
        partial_cli.main(argv)
    assert "multiple effective batch sizes: [4, 8]" in capsys.readouterr().err
    assert observed == []
    assert partial_cli.main([*argv, "--micro_batch_size", "4"]) == 0
    assert observed == [4]
