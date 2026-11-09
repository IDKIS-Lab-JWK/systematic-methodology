from copy import deepcopy
from types import SimpleNamespace

import pytest

from _linear_probe.common import ContractError, atomic_write_json, fingerprint, sha256_file
from _supervised_adaptation.compatibility import (
    MANIFEST_NAME, migrated_model_identity, resolve_historical_artifact,
)
from _supervised_adaptation.migrate_fixed_publication import checked_write, contained, snapshot_tree
from _supervised_adaptation import workflow


def fixture(tmp_path):
    profile = tmp_path / "profile.json"
    atomic_write_json(profile, {"parameter_source": "publication"})
    old = {"method": "Example", "pretrained_model": "pinned", "implementation": {"sha256": "old"}}
    current = {**old, "implementation": {"sha256": "reviewed-current"}}
    args = SimpleNamespace(results_root=tmp_path, percentage=40, seed=42, parameter_source="publication")
    document = {"schema_version": 1, "entries": [{
        "percentage": 40, "seed": 42, "parameter_source": "publication",
        "legacy_model_identity": old, "accepted_implementation": current["implementation"],
        "profile": str(profile), "profile_sha256": sha256_file(profile),
    }], "path_mappings": []}
    atomic_write_json(tmp_path / "Example" / MANIFEST_NAME,
                      {**document, "manifest_sha256": fingerprint(document)})
    return old, current, args


def test_compatibility_is_exact_and_experiment_specific(tmp_path):
    old, current, args = fixture(tmp_path)
    assert migrated_model_identity(current, args) == old
    with pytest.raises(ContractError, match="checkpoint or representation"):
        migrated_model_identity({**current, "pretrained_model": "different"}, args)
    with pytest.raises(ContractError, match="Implementation differs"):
        migrated_model_identity({**current, "implementation": {"sha256": "different"}}, args)
    args.percentage = 20
    assert migrated_model_identity(current, args) == current
    args.percentage = 40
    atomic_write_json(tmp_path / "profile.json", {"changed": True})
    with pytest.raises(ContractError, match="profile changed"):
        migrated_model_identity(current, args)


def test_archive_json_independence_idempotent_rewrite_and_target_containment(tmp_path):
    source = tmp_path / "active"
    archive = tmp_path / "archive"
    atomic_write_json(source / "context.json", {"old": True})
    source.joinpath("model.pt").write_bytes(b"immutable checkpoint fixture")
    before = snapshot_tree(source, archive)
    checked_write(source / "context.json", {"fixed": True}, archive / "context.json")
    checked_write(source / "context.json", {"fixed": True}, archive / "context.json")
    assert sha256_file(archive / "context.json") == before["context.json"]
    assert sha256_file(source / "model.pt") == before["model.pt"]
    with pytest.raises(ContractError, match="changed after preparation"):
        checked_write(source / "context.json", {"unexpected": True}, archive / "context.json")
    with pytest.raises(ContractError, match="strictly inside"):
        contained(tmp_path, tmp_path)
    with pytest.raises(ContractError, match="strictly inside"):
        contained(tmp_path.parent, tmp_path)


@pytest.mark.parametrize("source,selection,expected", [
    ("publication", "fixed", ["cv", "test"]),
    ("default", "fixed", ["cv", "test"]),
    ("grid", "fixed", ["grid", "cv", "test"]),
    ("publication", "publication_grid", ["grid", "cv", "test"]),
])
def test_pipeline_runs_grid_only_for_selected_search(monkeypatch, source, selection, expected):
    args = SimpleNamespace(dataset_root="unused", percentage=40, dry_run=False,
                           parameter_source=source, stage="all")
    candidates = [{"candidate_id": "a"}]
    if selection == "publication_grid":
        candidates.append({"candidate_id": "b"})
    monkeypatch.setattr(workflow, "load_manifest", lambda *_: {})
    monkeypatch.setattr(workflow, "_load_candidates", lambda *_: (candidates, {"parameter_selection": selection}, None))
    calls = []
    for stage in ("grid", "cv", "test"):
        monkeypatch.setattr(workflow, "run_" + stage, lambda *_, s=stage: calls.append(s))
    workflow.run_pipeline(args, SimpleNamespace(name="test", method_label="test"))
    assert calls == expected
