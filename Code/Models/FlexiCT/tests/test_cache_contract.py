from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from FlexiCT import FEATURE_DIM, UPSTREAM_REVISION
from FlexiCT import cache as cache_module
from FlexiCT.model import vision_encoder_contract
from FlexiCT.provisioning import resolve_checkpoint
from _linear_probe.common import (
    ContractError,
    atomic_write_json,
    fingerprint,
    read_json,
    sha256_file,
    tree_completion_seal,
)
from .conftest import cache_args


def build_complete_cache(args) -> tuple[Path, list[dict], dict]:
    rows, split_manifest = cache_module.all_rows(args.dataset_root, require_files=True)
    checkpoint = resolve_checkpoint(args.checkpoint)
    contract = cache_module._cache_contract(
        args, checkpoint, "fp32", rows, split_manifest
    )
    contract_sha = fingerprint(contract)
    root = cache_module._cache_directory(
        args, checkpoint, "fp32", rows, split_manifest
    )
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    entries = {}
    for row in rows:
        key = cache_module._entry_key(row)
        source_identity = cache_module._source_identity(
            Path(row["image_path"]), args.source_fingerprint
        )
        embedding = torch.zeros(FEATURE_DIM, dtype=torch.float32)
        embedding[int(row["label"])] = 1.0
        path = artifacts / f"{key}.pt"
        cache_module._atomic_torch_save(
            {
                "schema_version": 1,
                "patient_id": str(row["patient_id"]),
                "source_identity": source_identity,
                "contract_sha256": contract_sha,
                "preprocessing_metadata": {"synthetic": True},
                "extraction": {"precision": "fp32", "l2_norm": 1.0},
                "embedding": embedding,
            },
            path,
        )
        entries[key] = {
            "patient_id": str(row["patient_id"]),
            "artifact": f"artifacts/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_identity": source_identity,
            "extraction": {"precision": "fp32", "l2_norm": 1.0},
        }
    source = {
        "repository": "https://github.com/ricklisz/FlexiCT",
        "revision": UPSTREAM_REVISION,
        "path": str(args.upstream_root),
        "acquisition": "synthetic-test",
    }
    progress = {
        "schema_version": 2,
        "status": "partial",
        "contract": contract,
        "contract_sha256": contract_sha,
        "inventory": contract["dataset_inventory"],
        "entries": entries,
        "completed_rows": len(rows),
        "model_details": {
            "source": source,
            "checkpoint_load_mode": "torch_weights_only",
            **vision_encoder_contract(),
        },
    }
    atomic_write_json(root / "feature_cache_progress.json", progress)
    document = {
        **progress,
        "status": "complete",
        "row_count": len(rows),
        "split_manifest": split_manifest["_path"],
        "split_manifest_sha256": split_manifest["_sha256"],
        "checkpoint": checkpoint,
        "source": source,
        "torch_version": torch.__version__,
    }
    manifest_path = root / "feature_cache_manifest.json"
    atomic_write_json(manifest_path, document)
    seal_paths = [manifest_path]
    seal_paths.extend(root / entry["artifact"] for entry in entries.values())
    context = {
        "model": "FlexiCT",
        "contract_sha256": contract_sha,
        "row_count": len(rows),
        "entries_sha256": fingerprint(entries),
        "inventory_sha256": contract["dataset_inventory"]["rows_sha256"],
    }
    atomic_write_json(
        root / "feature_cache_completion.json",
        tree_completion_seal(root, seal_paths, context=context),
    )
    return root, rows, document


@pytest.fixture
def completed_cache(split_dataset_with_sources: Path, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    args = cache_args(split_dataset_with_sources, tmp_path, checkpoint)
    root, rows, manifest = build_complete_cache(args)
    return args, root, rows, manifest


def test_cache_contract_binds_model_preprocessing_inventory_and_policy(
    split_dataset_with_sources: Path, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"one")
    args = cache_args(split_dataset_with_sources, tmp_path, checkpoint_path)
    rows, manifest = cache_module.all_rows(args.dataset_root, require_files=True)
    checkpoint = resolve_checkpoint(checkpoint_path)
    base = cache_module._cache_contract(args, checkpoint, "fp32", rows, manifest)
    assert base["schema_version"] == 4
    assert set(base["implementation_identity"]["files"]) == {
        "__init__.py",
        "cache.py",
        "data.py",
        "model.py",
    }
    assert base["feature_shape"] == [1024]
    assert base["input_shape_dhw"] == [160, 160, 160]
    assert base["patch_size"] == 8
    assert base["expected_patch_tokens"] == 8000
    assert base["representation"] == "vlm_projected_cls_mean_l2_1024"
    assert base["dataset_inventory"]["row_count"] == 30
    assert base["checkpoint_deserialization_policy"] == "weights_only_only"
    assert base["extraction_runtime"]["torch_version"]

    changed_runtime = cache_module._cache_contract(
        args,
        checkpoint,
        "fp32",
        rows,
        manifest,
        {"torch_version": "different-runtime"},
    )
    assert fingerprint(changed_runtime) != fingerprint(base)

    legacy_args = copy.copy(args)
    legacy_args.allow_trusted_legacy_checkpoint = True
    legacy = cache_module._cache_contract(
        legacy_args, checkpoint, "fp32", rows, manifest
    )
    assert fingerprint(legacy) != fingerprint(base)

    stat_args = copy.copy(args)
    stat_args.source_fingerprint = "stat"
    stat_contract = cache_module._cache_contract(
        stat_args, checkpoint, "fp32", rows, manifest
    )
    assert fingerprint(stat_contract) != fingerprint(base)
    bf16 = cache_module._cache_contract(args, checkpoint, "bf16", rows, manifest)
    assert fingerprint(bf16) != fingerprint(base)

    changed_checkpoint = dict(checkpoint)
    changed_checkpoint["sha256"] = "f" * 64
    changed = cache_module._cache_contract(
        args, changed_checkpoint, "fp32", rows, manifest
    )
    assert fingerprint(changed) != fingerprint(base)


def test_entry_key_and_feature_cache_are_label_independent() -> None:
    row = {"patient_id": "p", "image_path": "C:/scan.nii.gz", "label": 0}
    relabelled = {**row, "label": 4, "class_name": "Sigmoid"}
    assert cache_module._entry_key(row) == cache_module._entry_key(relabelled)


def test_complete_cache_validates_and_identity_binds_manifest_and_entries(
    completed_cache,
) -> None:
    args, root, rows, manifest = completed_cache
    result = cache_module.validate_feature_cache(args)
    assert result["status"] == "valid"
    assert result["validated_rows"] == 30
    store = cache_module.FeatureStore(args, required_rows=rows[:2])
    identity = store.identity
    assert identity["feature_cache_manifest_sha256"] == sha256_file(
        root / "feature_cache_manifest.json"
    )
    assert identity["feature_entries_sha256"] == fingerprint(manifest["entries"])
    assert identity["feature_contract_sha256"] == manifest["contract_sha256"]
    feature = store.load(rows[0])
    assert feature.shape == (FEATURE_DIM,)
    assert feature.dtype == torch.float32


def test_normal_reuse_trusts_matching_source_stat_but_explicit_validate_rehashes(
    completed_cache, monkeypatch
) -> None:
    args, _root, rows, _manifest = completed_cache
    real_sha = cache_module.sha256_file
    source_hashes: list[Path] = []

    def recording_sha(path, *args_, **kwargs):
        candidate = Path(path)
        if candidate.name.endswith(".nii.gz"):
            source_hashes.append(candidate)
        return real_sha(path, *args_, **kwargs)

    monkeypatch.setattr(cache_module, "sha256_file", recording_sha)
    cache_module.validate_feature_cache(args, required_rows=rows[:3])
    assert source_hashes == []
    cache_module.validate_feature_cache(args)
    assert len(source_hashes) == len(rows)


def test_tampered_artifact_is_rejected_by_tree_seal(completed_cache) -> None:
    args, root, _rows, manifest = completed_cache
    first = next(iter(manifest["entries"].values()))
    path = root / first["artifact"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="Sealed artifact changed"):
        cache_module.validate_feature_cache(args)


def test_undeclared_artifact_is_rejected(completed_cache) -> None:
    args, root, _rows, _manifest = completed_cache
    torch.save({"embedding": torch.zeros(FEATURE_DIM)}, root / "artifacts" / "extra.pt")
    with pytest.raises(ContractError, match="undeclared"):
        cache_module.validate_feature_cache(args)


def test_changed_source_is_rejected_during_normal_stage_validation(completed_cache) -> None:
    args, _root, rows, _manifest = completed_cache
    source = Path(rows[0]["image_path"])
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ContractError, match="Source identity mismatch"):
        cache_module.validate_feature_cache(args, required_rows=[rows[0]])


def test_exact_inventory_mismatch_fails_even_with_a_new_valid_seal(completed_cache) -> None:
    args, root, _rows, _manifest = completed_cache
    manifest_path = root / "feature_cache_manifest.json"
    document = read_json(manifest_path)
    removed_key = next(iter(document["entries"]))
    removed = document["entries"].pop(removed_key)
    (root / removed["artifact"]).unlink()
    atomic_write_json(manifest_path, document)
    paths = [manifest_path]
    paths.extend(root / item["artifact"] for item in document["entries"].values())
    context = {
        "model": "FlexiCT",
        "contract_sha256": document["contract_sha256"],
        "row_count": len(document["entries"]),
        "entries_sha256": fingerprint(document["entries"]),
        "inventory_sha256": document["contract"]["dataset_inventory"]["rows_sha256"],
    }
    atomic_write_json(
        root / "feature_cache_completion.json",
        tree_completion_seal(root, paths, context=context),
    )
    with pytest.raises(ContractError, match="entry set"):
        cache_module.validate_feature_cache(args)


@pytest.mark.parametrize("relative", ["../escape.pt", "C:/absolute.pt", "artifacts/../../x.pt"])
def test_cache_artifact_paths_must_remain_inside_root(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ContractError, match="Unsafe"):
        cache_module._safe_artifact_path(tmp_path, relative)


def test_completed_cache_resume_never_touches_upstream_or_extractor(
    completed_cache, monkeypatch
) -> None:
    args, root, rows, _manifest = completed_cache
    monkeypatch.setattr(
        cache_module,
        "ensure_upstream",
        lambda *_args, **_kwargs: pytest.fail("completed cache must not provision source"),
    )
    monkeypatch.setattr(
        cache_module,
        "build_vision_only_encoder",
        lambda **_kwargs: pytest.fail("completed cache must not load a model"),
    )
    monkeypatch.setattr(
        cache_module,
        "preprocess_nifti",
        lambda *_args, **_kwargs: pytest.fail("completed cache must not read a volume"),
    )
    result = cache_module.prepare_feature_cache(args)
    assert result == {
        "status": "reused",
        "cache_directory": str(root),
        "row_count": len(rows),
    }


def test_partial_cache_with_all_artifacts_seals_without_model_or_volume_extraction(
    completed_cache, monkeypatch
) -> None:
    args, root, rows, _manifest = completed_cache
    (root / "feature_cache_completion.json").unlink()
    monkeypatch.setattr(
        cache_module,
        "ensure_upstream",
        lambda *_args, **_kwargs: {
            "repository": "synthetic",
            "revision": UPSTREAM_REVISION,
            "path": str(args.upstream_root),
            "acquisition": "existing",
        },
    )
    monkeypatch.setattr(
        cache_module,
        "build_vision_only_encoder",
        lambda **_kwargs: pytest.fail("valid partial artifacts must not load a model"),
    )
    monkeypatch.setattr(
        cache_module,
        "preprocess_nifti",
        lambda *_args, **_kwargs: pytest.fail("valid partial artifacts must not read scans"),
    )
    result = cache_module.prepare_feature_cache(args)
    assert result["status"] == "complete"
    assert result["row_count"] == len(rows)
    assert (root / "feature_cache_completion.json").is_file()
    assert cache_module.validate_feature_cache(args)["status"] == "valid"


def test_partial_cache_never_adopts_a_changed_finite_embedding(
    completed_cache, monkeypatch
) -> None:
    args, root, _rows, manifest = completed_cache
    (root / "feature_cache_completion.json").unlink()
    first = next(iter(manifest["entries"].values()))
    artifact = root / first["artifact"]
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["embedding"] = torch.roll(payload["embedding"], shifts=1)
    cache_module._atomic_torch_save(payload, artifact)
    monkeypatch.setattr(
        cache_module,
        "ensure_upstream",
        lambda *_args, **_kwargs: {
            "repository": "synthetic",
            "revision": UPSTREAM_REVISION,
            "path": str(args.upstream_root),
            "acquisition": "existing",
        },
    )
    monkeypatch.setattr(
        cache_module,
        "build_vision_only_encoder",
        lambda **_kwargs: pytest.fail("corrupt partial cache must fail before model load"),
    )
    with pytest.raises(ContractError, match="changed since its progress entry"):
        cache_module.prepare_feature_cache(args)


def test_partial_cache_regenerates_unrecorded_orphan_artifact(
    completed_cache, monkeypatch
) -> None:
    args, root, rows, _manifest = completed_cache
    (root / "feature_cache_completion.json").unlink()
    progress_path = root / "feature_cache_progress.json"
    progress = read_json(progress_path)
    orphan_row = rows[0]
    orphan_key = cache_module._entry_key(orphan_row)
    orphan_path = root / progress["entries"][orphan_key]["artifact"]
    progress["entries"].pop(orphan_key)
    atomic_write_json(progress_path, progress)

    malicious = torch.load(orphan_path, map_location="cpu", weights_only=True)
    malicious["embedding"] = torch.roll(malicious["embedding"], shifts=1)
    cache_module._atomic_torch_save(malicious, orphan_path)
    calls = {"model": 0, "preprocess": 0, "extract": 0}

    monkeypatch.setattr(
        cache_module,
        "ensure_upstream",
        lambda *_args, **_kwargs: {
            "repository": "synthetic",
            "revision": UPSTREAM_REVISION,
            "path": str(args.upstream_root),
            "acquisition": "existing",
        },
    )

    def fake_model(**_kwargs):
        calls["model"] += 1
        return object(), {"source": {"revision": UPSTREAM_REVISION}, **vision_encoder_contract()}

    def fake_preprocess(_path):
        calls["preprocess"] += 1
        return torch.zeros((1, 1, 1, 1)), {"synthetic": True}

    def fake_extract(_model, _volume, **_kwargs):
        calls["extract"] += 1
        embedding = torch.zeros(FEATURE_DIM, dtype=torch.float32)
        embedding[0] = 1.0
        return embedding, {"precision": "fp32", "l2_norm": 1.0}

    monkeypatch.setattr(cache_module, "build_vision_only_encoder", fake_model)
    monkeypatch.setattr(cache_module, "preprocess_nifti", fake_preprocess)
    monkeypatch.setattr(cache_module, "extract_embedding", fake_extract)

    result = cache_module.prepare_feature_cache(args)
    assert result["status"] == "complete"
    assert calls == {"model": 1, "preprocess": 1, "extract": 1}
    regenerated = torch.load(orphan_path, map_location="cpu", weights_only=True)
    assert regenerated["embedding"][0].item() == 1.0
    assert regenerated["embedding"][1:].count_nonzero().item() == 0


def test_payload_contract_rejects_wrong_shape_and_nonfinite(tmp_path: Path) -> None:
    row = {"patient_id": "p", "image_path": str(tmp_path / "p.nii.gz")}
    path = tmp_path / "feature.pt"
    valid_embedding = torch.zeros(FEATURE_DIM)
    valid_embedding[0] = 1.0
    base = {
        "patient_id": "p",
        "contract_sha256": "a" * 64,
        "embedding": valid_embedding,
    }
    cache_module._atomic_torch_save(base, path)
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
    assert cache_module._validate_payload(path, row, "a" * 64)["patient_id"] == "p"

    cache_module._atomic_torch_save({**base, "embedding": torch.zeros(1728)}, path)
    with pytest.raises(ContractError, match="dtype/shape"):
        cache_module._validate_payload(path, row, "a" * 64)
    invalid = torch.zeros(FEATURE_DIM)
    invalid[0] = torch.nan
    cache_module._atomic_torch_save({**base, "embedding": invalid}, path)
    with pytest.raises(ContractError, match="Non-finite"):
        cache_module._validate_payload(path, row, "a" * 64)
    cache_module._atomic_torch_save(
        {**base, "embedding": torch.full((FEATURE_DIM,), 0.5)}, path
    )
    with pytest.raises(ContractError, match="not L2-normalized"):
        cache_module._validate_payload(path, row, "a" * 64)
