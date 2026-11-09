from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from COLIPRI.backend import COLIPRIBackend
from COLIPRI.cache import _legacy_cache_contract
from _linear_probe.common import ContractError, fingerprint
from _pooling_optimised_linear_probe.workflow import (
    CANONICAL_METHOD_LABEL,
    LEGACY_METHOD_LABEL,
    _effective_grid_context,
)


LEGACY_IMPLEMENTATION = {
    "schema_version": 1,
    "files": {"legacy.py": "a" * 64},
}
LEGACY_PROFILE_SHA256 = "b" * 64
LEGACY_CACHE_IMPLEMENTATION = {
    "schema_version": 1,
    "files": {"__init__.py": "e" * 64},
}


def _write_migration(results_root: Path) -> None:
    path = results_root / CANONICAL_METHOD_LABEL
    path.mkdir(parents=True, exist_ok=True)
    (path / "METHOD_IDENTITY_MIGRATION.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "migration_type": "metadata_only_result_namespace",
                "model": "COLIPRI",
                "legacy_method_label": LEGACY_METHOD_LABEL,
                "canonical_method_id": "pooling_optimised_linear_probe",
                "canonical_method_label": CANONICAL_METHOD_LABEL,
                "legacy_profile_sha256": LEGACY_PROFILE_SHA256,
                "legacy_probe_implementation_identity": LEGACY_IMPLEMENTATION,
                "legacy_cache_implementation_identity": LEGACY_CACHE_IMPLEMENTATION,
            }
        ),
        encoding="utf-8",
    )


def _contexts() -> tuple[dict, dict]:
    shared = {
        "schema_version": 1,
        "model": "COLIPRI",
        "parameter_source": "publication",
        "percentage": 40,
        "seed": 42,
        "candidate_ids": ["0123456789ab"],
        "fold_partitions": [{"fold": 1, "training_rows_sha256": "x"}],
    }
    current = {
        **shared,
        "method": CANONICAL_METHOD_LABEL,
        "profile": "publication_pooling_optimised_linear_probe.json",
        "profile_sha256": "c" * 64,
        "model_identity": {
            "cache": "same",
            "runtime": "same",
            "probe_implementation_identity": {"files": {"current.py": "d" * 64}},
        },
    }
    legacy = {
        **shared,
        "method": LEGACY_METHOD_LABEL,
        "profile": "publication_linear_probe.json",
        "profile_sha256": LEGACY_PROFILE_SHA256,
        "model_identity": {
            "cache": "same",
            "runtime": "same",
            "probe_implementation_identity": LEGACY_IMPLEMENTATION,
        },
    }
    return current, legacy


def test_documented_legacy_context_is_adopted_without_rewriting(tmp_path: Path) -> None:
    results_root = tmp_path / "Results" / "COLIPRI"
    _write_migration(results_root)
    root = results_root / CANONICAL_METHOD_LABEL / "Publication_Grid" / "40_percent"
    root.mkdir(parents=True)
    current, legacy = _contexts()
    context_path = root / "grid_context.json"
    original = json.dumps(legacy, sort_keys=True)
    context_path.write_text(original, encoding="utf-8")

    effective = _effective_grid_context(
        SimpleNamespace(results_root=results_root, resume=True), root, current
    )

    assert effective == legacy
    assert context_path.read_text(encoding="utf-8") == original


def test_legacy_context_rejects_non_name_identity_changes(tmp_path: Path) -> None:
    results_root = tmp_path / "Results" / "COLIPRI"
    _write_migration(results_root)
    root = results_root / CANONICAL_METHOD_LABEL / "Publication_Grid" / "40_percent"
    root.mkdir(parents=True)
    current, legacy = _contexts()
    changed = copy.deepcopy(legacy)
    changed["seed"] = 7
    (root / "grid_context.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )

    with pytest.raises(ContractError, match="identity field changed"):
        _effective_grid_context(
            SimpleNamespace(results_root=results_root, resume=True), root, current
        )


def test_training_identity_preserves_legacy_checkpoint_method(monkeypatch) -> None:
    backend = COLIPRIBackend()
    monkeypatch.setattr(
        backend,
        "_probe_runtime",
        lambda _args: {"training_precision_resolved": "fp32"},
    )
    monkeypatch.setattr(
        backend,
        "_feature_store",
        lambda _args: SimpleNamespace(identity={"cache": "same"}),
    )
    args = SimpleNamespace(
        seed=42,
        training_precision="fp32",
        micro_batch_size=1,
    )

    identity = backend._training_identity(
        [{"candidate_id": "0123456789ab"}],
        [{"patient_id": "p1", "label": 0}],
        args,
        {"method": LEGACY_METHOD_LABEL, "fold": 2, "percentage": 40},
        "fp32",
    )

    assert identity["method"] == LEGACY_METHOD_LABEL
    assert identity["fold"] == 2
    assert identity["percentage"] == 40


def test_declared_legacy_cache_is_resolved_without_rewriting(tmp_path: Path) -> None:
    results_root = tmp_path / "Results" / "COLIPRI"
    cache_root = results_root / "Cache"
    _write_migration(results_root)
    current = {
        "schema_version": 4,
        "model": "COLIPRI",
        "dataset": "same",
        "implementation_identity": {"files": {"current.py": "f" * 64}},
    }
    legacy = {
        **current,
        "implementation_identity": LEGACY_CACHE_IMPLEMENTATION,
    }
    namespace = "test_cache"
    root = (
        cache_root
        / "Features"
        / f"{namespace}_{fingerprint(legacy, length=12)}"
    )
    root.mkdir(parents=True)
    (root / "feature_cache_manifest.json").write_text(
        json.dumps({"status": "complete", "contract": legacy}), encoding="utf-8"
    )
    completion = root / "feature_cache_completion.json"
    completion.write_text("{}", encoding="utf-8")
    original = (root / "feature_cache_manifest.json").read_text(encoding="utf-8")

    resolved = _legacy_cache_contract(
        SimpleNamespace(
            resume=True,
            results_root=results_root,
            cache_root=cache_root,
            cache_namespace=namespace,
        ),
        current,
    )

    assert resolved == (root, legacy)
    assert (root / "feature_cache_manifest.json").read_text(encoding="utf-8") == original


def test_legacy_cache_rejects_extraction_contract_changes(tmp_path: Path) -> None:
    results_root = tmp_path / "Results" / "COLIPRI"
    cache_root = results_root / "Cache"
    _write_migration(results_root)
    current = {
        "schema_version": 4,
        "model": "COLIPRI",
        "dataset": "current",
        "implementation_identity": {"files": {"current.py": "f" * 64}},
    }
    legacy = {
        **current,
        "dataset": "changed",
        "implementation_identity": LEGACY_CACHE_IMPLEMENTATION,
    }
    namespace = "test_cache"
    root = (
        cache_root
        / "Features"
        / f"{namespace}_{fingerprint(legacy, length=12)}"
    )
    root.mkdir(parents=True)
    (root / "feature_cache_manifest.json").write_text(
        json.dumps({"status": "complete", "contract": legacy}), encoding="utf-8"
    )
    (root / "feature_cache_completion.json").write_text("{}", encoding="utf-8")

    resolved = _legacy_cache_contract(
        SimpleNamespace(
            resume=True,
            results_root=results_root,
            cache_root=cache_root,
            cache_namespace=namespace,
        ),
        current,
    )

    assert resolved is None
