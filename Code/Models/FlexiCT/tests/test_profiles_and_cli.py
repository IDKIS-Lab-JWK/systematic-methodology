from __future__ import annotations

import json
from pathlib import Path

import pytest

from FlexiCT import DEFAULT_CACHE_NAMESPACE, UPSTREAM_REVISION
from FlexiCT.backend import FlexiCTBackend
from FlexiCT.check_setup import main as check_setup_main
from FlexiCT.cli import build_parser, main as train_main
from FlexiCT.prepare_cache import main as cache_main
from _linear_probe.common import ContractError


PACKAGE = Path(__file__).resolve().parents[1]


def _json(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def test_publication_profile_is_complete_and_native() -> None:
    profile = _json("publication_linear_probe.json")
    backend = FlexiCTBackend()
    candidates = backend.build_candidates(profile)
    assert [item["C"] for item in candidates] == list(backend.publication_c_grid)
    assert len(candidates) == profile["search_space"]["candidate_count"] == 7
    fixed = profile["fixed_parameters"]
    assert fixed == {
        "classifier": "multinomial_logistic_regression",
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 20000,
        "random_state": 0,
        "feature_standardization": "StandardScaler fit on training partition only",
        "encoder_frozen": True,
        "feature_dimension": 1024,
    }
    assert profile["source"]["repository_revision"] == UPSTREAM_REVISION
    assert profile["search_space"]["ordered_grid_immutable_for_publication_claim"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: profile["search_space"].update(C=[1e-3] * 7),
        lambda profile: profile["fixed_parameters"].update(class_weight=None),
        lambda profile: profile["fixed_parameters"].pop("solver"),
        lambda profile: profile["source"].update(repository_revision="main"),
    ],
)
def test_publication_profile_tampering_fails_closed(mutation) -> None:
    profile = _json("publication_linear_probe.json")
    mutation(profile)
    with pytest.raises(ContractError):
        FlexiCTBackend().build_candidates(profile)


def test_default_and_project_grid_profiles_validate() -> None:
    backend = FlexiCTBackend()
    default = backend.build_candidates(_json("default_linear_probe.json"))
    project = backend.build_candidates(_json("search_linear_probe.json"))
    assert len(default) == 1 and default[0]["C"] == 1.0
    assert [item["C"] for item in project] == [0.01, 0.1, 1.0, 10.0]


def test_launcher_facing_defaults_are_scientific() -> None:
    args = build_parser().parse_args([])
    assert args.parameter_source == "publication"
    assert args.feature_precision == "auto"
    assert args.source_fingerprint == "sha256"
    assert args.cache_namespace == DEFAULT_CACHE_NAMESPACE
    assert args.upstream_revision == UPSTREAM_REVISION
    assert args.allow_trusted_legacy_checkpoint is False
    assert args.num_workers == 0


def test_train_dry_run_expands_seven_candidates_without_side_effects(
    split_dataset: Path, tmp_path: Path, capsys
) -> None:
    results = tmp_path / "never-created-results"
    code = train_main(
        [
            "--stage",
            "all",
            "--percentage",
            "100",
            "--parameter_source",
            "publication",
            "--dataset_root",
            str(split_dataset),
            "--results_root",
            str(results),
            "--cache_root",
            str(tmp_path / "cache"),
            "--dry_run",
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "dry_run"
    assert document["side_effects"] is False
    assert document["candidate_count"] == 7
    assert document["publication_backed_grid"] is True
    assert document["model_claim"]["representation"] == "vlm_projected_cls_mean_l2_1024"
    assert document["model_claim"]["source_fingerprint"] == "sha256"
    assert document["model_claim"]["text_tower_loaded"] is False
    assert not results.exists()


def test_cache_plan_is_dependency_light_and_uses_100_percent_inventory(
    split_dataset: Path, tmp_path: Path, capsys
) -> None:
    code = cache_main(
        [
            "--action",
            "plan",
            "--percentage",
            "100",
            "--dataset_root",
            str(split_dataset),
            "--cache_root",
            str(tmp_path / "cache"),
            "--dry_run",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["side_effects"] is False
    assert plan["inventory_percentage"] == 100
    assert plan["requested_percentage"] == 100
    assert plan["row_count"] == 30
    assert plan["contract"]["feature_shape"] == [1024]
    assert plan["contract"]["source_fingerprint"] == "sha256"
    assert plan["blockers"]
    assert not (tmp_path / "cache").exists()

    with pytest.raises(SystemExit):
        cache_main(["--percentage", "20", "--dry_run"])


def test_setup_dry_run_does_not_resolve_source_or_checkpoint(tmp_path: Path, capsys) -> None:
    code = check_setup_main(
        [
            "--upstream_root",
            str(tmp_path / "absent"),
            "--dry_run",
            "--load_model",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert result["side_effects"] is False
    assert result["load_model"] is True
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--parameter_source", "publication", "--max_grid_candidates", "1"],
        ["--parameter_source", "publication", "--grid_config", "grid.json"],
        ["--parameter_source", "manual"],
        ["--stage", "grid", "--parameter_source", "default"],
        ["--parameter_source", "grid", "--max_grid_candidates", "0"],
        ["--upstream_revision", "main"],
        ["--allow_trusted_legacy_checkpoint"],
        ["--num_workers", "1"],
        ["--source_fingerprint", "stat"],
    ],
)
def test_invalid_cli_contracts_are_rejected_before_execution(argv) -> None:
    with pytest.raises(SystemExit):
        train_main(argv)


def test_trusted_legacy_opt_in_requires_digest_in_all_asset_clis() -> None:
    with pytest.raises(SystemExit):
        cache_main(["--allow_trusted_legacy_checkpoint", "--dry_run"])
    with pytest.raises(SystemExit):
        check_setup_main(["--allow_trusted_legacy_checkpoint", "--dry_run"])
