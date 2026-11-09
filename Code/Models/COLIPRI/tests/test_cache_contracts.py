from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from COLIPRI import MODEL_REVISION  # noqa: E402
from COLIPRI import cache as cache_module  # noqa: E402
from COLIPRI import cli as cli_module  # noqa: E402
from COLIPRI import prepare_cache as prepare_cache_module  # noqa: E402
from _linear_probe.common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    completion_seal,
    fingerprint,
    read_json,
    sha256_file,
    validate_completion,
)


def test_feature_store_identity_binds_sealed_manifest_and_entries(
    tmp_path: Path, monkeypatch
) -> None:
    shape = (2, 2, 2, 2)
    monkeypatch.setattr(cache_module, "FEATURE_SHAPE_CDHW", shape)
    row = {"patient_id": "p1", "image_path": str(tmp_path / "p1.nii.gz")}
    key = cache_module._entry_key(row)
    source_identity = {
        "path": str(Path(row["image_path"]).resolve()),
        "bytes": 7,
        "modified_ns": 11,
        "sha256": "a" * 64,
    }
    artifact = tmp_path / "artifacts" / f"{key}.pt"
    artifact.parent.mkdir()
    torch.save(
        {
            "patient_id": "p1",
            "contract_sha256": "contract",
            "source_identity": source_identity,
            "dense_features": torch.ones(shape),
        },
        artifact,
    )
    manifest = {
        "contract_sha256": "contract",
        "_sealed_manifest_sha256": "manifest-sha",
        "_entries_sha256": "entries-sha",
        "_completion_seal_sha256": "seal-sha",
        "_execution_ready_receipt": {"binding_sha256": "receipt-binding"},
        "checkpoint": {"sha256": "checkpoint-sha"},
        "source": {"revision": MODEL_REVISION},
        "contract": {
            "representation": "raw_unprojected_unpooled_primus_m_backbone",
            "feature_shape_cdhw": list(shape),
            "storage_dtype": "float32",
            "extraction_precision": "fp32",
            "extraction_runtime": {
                "schema_version": 1,
                "precision_resolved": "fp32",
                "torch_version": "test-torch",
            },
        },
        "entries": {
            key: {
                "artifact": f"artifacts/{key}.pt",
                "source_identity": source_identity,
            }
        },
    }
    monkeypatch.setattr(
        cache_module,
        "_resolve_existing_cache",
        lambda args, **kwargs: (tmp_path, manifest),
    )
    original_isfinite = torch.isfinite
    calls = {"count": 0}

    def counted_isfinite(value):
        calls["count"] += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    store = cache_module.FeatureStore(SimpleNamespace())

    assert store.identity["feature_cache_manifest_sha256"] == "manifest-sha"
    assert store.identity["feature_entries_sha256"] == "entries-sha"
    assert store.identity["execution_receipt_binding_sha256"] == "receipt-binding"
    assert store.identity["extraction_runtime"]["torch_version"] == "test-torch"
    assert "feature_cache" not in store.identity
    assert tuple(store.load(row).shape) == shape
    assert tuple(store.load(row).shape) == shape
    assert calls["count"] == 1


def test_structural_validation_avoids_hashing_and_exhaustive_validation_hashes_all(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "p1.nii.gz"
    source.write_bytes(b"source")
    artifact = tmp_path / "artifacts" / "p1.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"artifact")
    row = {"patient_id": "p1", "image_path": str(source)}
    key = cache_module._entry_key(row)
    stat = source.stat()
    source_identity = {
        "path": str(source.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "sha256": sha256_file(source),
    }
    manifest = {
        "contract_sha256": "contract",
        "entries": {
            key: {
                "artifact": "artifacts/p1.pt",
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
                "source_identity": source_identity,
            }
        },
    }
    monkeypatch.setattr(
        cache_module,
        "_resolve_existing_cache",
        lambda args, **kwargs: (tmp_path, manifest),
    )
    original_sha256 = cache_module.sha256_file
    calls: list[Path] = []

    def counted_sha256(path):
        calls.append(Path(path))
        return original_sha256(path)

    payload_calls = {"count": 0}

    def fake_validate_payload(*args, **kwargs):
        payload_calls["count"] += 1
        return {}

    monkeypatch.setattr(cache_module, "sha256_file", counted_sha256)
    monkeypatch.setattr(cache_module, "_validate_payload", fake_validate_payload)
    monkeypatch.setattr(
        cache_module,
        "all_rows",
        lambda dataset_root, require_files: ([row], {"_path": "x", "_sha256": "y"}),
    )
    receipt_calls = {"count": 0}

    def fake_write_receipt(root, manifest, *, validation_source):
        receipt_calls["count"] += 1
        assert validation_source == "exhaustive_sha256_validation"
        return {"binding_sha256": "receipt-binding"}

    monkeypatch.setattr(
        cache_module, "_write_execution_ready_receipt", fake_write_receipt
    )
    args = SimpleNamespace(source_fingerprint="sha256", dataset_root=tmp_path)

    fast = cache_module.validate_feature_cache(
        args, required_rows=[row], exhaustive=False
    )
    assert fast["status"] == "diagnostic_validation_only"
    assert fast["execution_ready"] is False
    assert calls == []
    assert payload_calls["count"] == 0

    full = cache_module.validate_feature_cache(args, exhaustive=True)
    assert full["status"] == "valid"
    assert full["execution_ready"] is True
    assert calls == [source, artifact]
    assert payload_calls["count"] == 1
    assert receipt_calls["count"] == 1


def _tiny_sealed_cache(tmp_path: Path) -> tuple[dict, dict, Path]:
    artifact = tmp_path / "artifacts" / "key.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"ABCD")
    entries = {
        "key": {
            "artifact": "artifacts/key.pt",
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "contract_sha256": "contract",
        "contract": {"source_fingerprint": "sha256"},
        "entries": entries,
    }
    manifest_path = tmp_path / "feature_cache_manifest.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        tmp_path / "feature_cache_completion.json",
        completion_seal(
            [manifest_path],
            context={
                "entries_sha256": fingerprint(entries),
                "contract_sha256": "contract",
            },
        ),
    )
    completion = validate_completion(tmp_path, "feature_cache_completion.json")
    return manifest, completion, artifact


def test_execution_receipt_rejects_missing_receipt(tmp_path: Path) -> None:
    manifest, completion, _ = _tiny_sealed_cache(tmp_path)

    with pytest.raises(ContractError, match="receipt is missing"):
        cache_module._validate_execution_ready_receipt(
            tmp_path, manifest, completion
        )


def test_execution_receipt_rejects_same_size_artifact_tamper(tmp_path: Path) -> None:
    manifest, completion, artifact = _tiny_sealed_cache(tmp_path)
    receipt = cache_module._write_execution_ready_receipt(
        tmp_path,
        manifest,
        validation_source="exhaustive_sha256_validation",
    )
    assert receipt["status"] == "execution_ready"
    cache_module._validate_execution_ready_receipt(tmp_path, manifest, completion)

    previous = artifact.stat()
    artifact.write_bytes(b"WXYZ")
    os.utime(
        artifact,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(ContractError, match="no longer matches"):
        cache_module._validate_execution_ready_receipt(
            tmp_path, manifest, completion
        )


def test_feature_store_requires_execution_ready_resolution(monkeypatch) -> None:
    observed = {"required": False}

    def fake_resolve(args, *, require_execution_ready=False):
        observed["required"] = require_execution_ready
        raise ContractError("receipt is missing")

    monkeypatch.setattr(cache_module, "_resolve_existing_cache", fake_resolve)

    with pytest.raises(ContractError, match="receipt is missing"):
        cache_module.FeatureStore(SimpleNamespace())
    assert observed["required"] is True


def test_prepare_source_sha_is_reused_but_exhaustive_mode_forces_rehash(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "p1.nii.gz"
    source.write_bytes(b"source")
    prior = cache_module._source_identity(source, "sha256")
    original_sha256 = cache_module.sha256_file
    calls = {"count": 0}

    def counted_sha256(path):
        calls["count"] += 1
        return original_sha256(path)

    monkeypatch.setattr(cache_module, "sha256_file", counted_sha256)

    reused = cache_module._source_identity(
        source, "sha256", prior=prior, force_hash=False
    )
    exhaustive = cache_module._source_identity(
        source, "sha256", prior=prior, force_hash=True
    )

    assert reused == prior
    assert exhaustive == prior
    assert calls["count"] == 1


def test_partial_artifact_must_match_recorded_size_and_sha(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "key.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"sealed partial artifact")
    row = {"patient_id": "p1"}
    prior = {
        "patient_id": "p1",
        "artifact": "artifacts/key.pt",
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }

    cache_module._validate_prior_artifact(artifact, row, prior)
    artifact.write_bytes(b"tampered partial artifact")

    with pytest.raises(ContractError, match="size changed|SHA-256 changed"):
        cache_module._validate_prior_artifact(artifact, row, prior)


def test_orphan_partial_artifact_is_not_reused(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "key.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"unrecorded orphan")
    row = {"patient_id": "p1"}

    assert cache_module._recorded_artifact_reusable(artifact, row, {}) is False


def test_cache_contract_binds_exact_hundred_percent_inventory(tmp_path: Path) -> None:
    rows = [
        {"patient_id": "p1", "image_path": str(tmp_path / "p1.nii.gz")},
        {"patient_id": "p2", "image_path": str(tmp_path / "p2.nii.gz")},
    ]
    manifest = {
        "_path": str(tmp_path / "Splits" / "100_percent.json"),
        "_sha256": "a" * 64,
    }
    args = SimpleNamespace(
        dataset_root=tmp_path,
        upstream_root=tmp_path,
        upstream_revision="pinned",
        local_files_only=True,
        provision_log=tmp_path / "provision.jsonl",
        checkpoint=None,
        device="cpu",
        feature_precision="fp32",
        source_fingerprint="sha256",
    )
    checkpoint = {"sha256": "b" * 64, "bytes": 123}
    runtime = {
        "schema_version": 1,
        "precision_resolved": "fp32",
        "torch_version": "2.8.0",
        "python_version": "3.11.9",
        "package_versions": {"numpy": "2.3.3", "torchio": "1.2.1"},
    }
    contract = cache_module._contract(
        args, checkpoint, "fp32", {}, runtime, rows, manifest
    )

    assert contract["schema_version"] == 4
    assert set(contract["implementation_identity"]["files"]) == {
        "__init__.py",
        "cache.py",
        "model.py",
    }
    assert contract["dataset_inventory"] == {
        "inventory_source_percentage": 100,
        "row_count": 2,
        "rows_sha256": fingerprint(
            [
                {
                    "entry_key": cache_module._entry_key(row),
                    "patient_id": row["patient_id"],
                    "source": str(Path(row["image_path"]).resolve()),
                }
                for row in rows
            ]
        ),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": "a" * 64,
    }
    changed_manifest = {**manifest, "_sha256": "c" * 64}
    changed = cache_module._contract(
        args, checkpoint, "fp32", {}, runtime, rows, changed_manifest
    )
    assert fingerprint(changed) != fingerprint(contract)

    changed_runtime = {
        **runtime,
        "package_versions": {**runtime["package_versions"], "torchio": "1.2.2"},
    }
    runtime_changed_contract = cache_module._contract(
        args, checkpoint, "fp32", {}, changed_runtime, rows, manifest
    )
    assert fingerprint(runtime_changed_contract) != fingerprint(contract)


def test_extraction_runtime_identity_binds_python_torch_device_and_packages(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_module.importlib.metadata,
        "version",
        lambda distribution: f"test-{distribution}-version",
    )

    runtime = cache_module.extraction_runtime_identity("cpu", "auto")

    assert runtime["schema_version"] == 1
    assert runtime["precision_requested"] == "auto"
    assert runtime["precision_resolved"] == "fp32"
    assert runtime["device_type"] == "cpu"
    assert runtime["device_resolved"] == "cpu"
    assert runtime["torch_version"] == str(torch.__version__)
    assert isinstance(runtime["python_build"], list)
    assert runtime["cuda_runtime"] is None
    assert runtime["cudnn_version"] is None
    assert set(runtime["package_versions"]) == set(
        cache_module.EXTRACTION_RUNTIME_DISTRIBUTIONS
    )


def test_cli_rejects_unsupported_probe_micro_batch_size_early() -> None:
    assert cli_module.build_parser().parse_args(
        ["--micro_batch_size", "16"]
    ).micro_batch_size == 16
    with pytest.raises(SystemExit):
        cli_module.build_parser().parse_args(["--micro_batch_size", "3"])
    with pytest.raises(SystemExit):
        cli_module.main(["--source_fingerprint", "stat"])
    with pytest.raises(SystemExit):
        prepare_cache_module.build_parser().parse_args(["--percentage", "20"])


def test_prepare_cache_validate_is_exhaustive_by_default(monkeypatch, capsys) -> None:
    calls: list[bool] = []

    def fake_validate(args, *, exhaustive):
        calls.append(exhaustive)
        return {"status": "valid", "execution_ready": exhaustive}

    monkeypatch.setattr(prepare_cache_module, "validate_feature_cache", fake_validate)

    assert prepare_cache_module.main(["--action", "validate", "--cache_target", "probe"]) == 0
    assert prepare_cache_module.main(
        ["--action", "validate", "--cache_target", "probe", "--skip_artifact_validation"]
    ) == 0
    capsys.readouterr()

    assert calls == [True, False]
    assert prepare_cache_module.build_parser().parse_args([]).source_fingerprint == "sha256"


def test_completed_prepare_reports_missing_execution_receipt_next_action(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "feature_cache_completion.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        dataset_root=tmp_path,
        upstream_root=tmp_path,
        upstream_revision="pinned",
        local_files_only=True,
        provision_log=tmp_path / "provision.jsonl",
        checkpoint=None,
        device="cpu",
        feature_precision="fp32",
        source_fingerprint="sha256",
    )
    monkeypatch.setattr(cache_module, "all_rows", lambda *_args, **_kwargs: ([{}], {}))
    monkeypatch.setattr(cache_module, "ensure_upstream", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cache_module, "resolve_checkpoint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        cache_module,
        "extraction_runtime_identity",
        lambda *_args, **_kwargs: {"precision_resolved": "fp32"},
    )
    monkeypatch.setattr(cache_module, "processor_config_identity", lambda *_args: {})
    monkeypatch.setattr(cache_module, "_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cache_module, "_cache_directory", lambda *_args: tmp_path)
    monkeypatch.setattr(cache_module, "validate_feature_cache", lambda *_args, **_kwargs: {})

    result = cache_module.prepare_feature_cache(args)

    assert result["status"] == "reused"
    assert result["execution_ready"] is False
    assert result["execution_receipt"] is None
    assert "--action validate" in result["next_action"]


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (prepare_cache_module, ["--num_workers", "1"]),
        (cli_module, ["--num_workers", "1", "--dry_run"]),
    ],
)
def test_streamed_cache_rejects_ignored_worker_setting(module, argv) -> None:
    with pytest.raises(SystemExit):
        module.main(argv)
