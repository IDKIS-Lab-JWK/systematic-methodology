from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from COLIPRI import volume_cache as vc
from COLIPRI import prepare_cache as cli
from COLIPRI import train_primus_m_partial_finetune as ft_cli
from COLIPRI.cached_partial_finetune import CachedCOLIPRIPartialAdapter
from COLIPRI.partial_finetune import COLIPRIPartialAdapter
from _linear_probe.common import ContractError, sha256_file
from _supervised_adaptation import cli as shared_cli
from _supervised_adaptation.trainer import input_rows_identity


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    rows = []
    for index in range(2):
        source = tmp_path / f"patient{index}.nii.gz"
        source.write_bytes(f"source{index}".encode())
        rows.append({"patient_id": f"patient{index}", "image_path": str(source),
                     "label": index, "class_name": ["Appendicitis", "Cholezystitis"][index]})
    args = SimpleNamespace(dataset_root=tmp_path, results_root=tmp_path / "results",
        cache_root=tmp_path / "cache", upstream_root=tmp_path / "source", local_files_only=True,
        source_fingerprint="sha256", provision_log=None)
    monkeypatch.setattr(vc, "VOLUME_SHAPE", (1, 2, 2, 2))
    monkeypatch.setattr(vc, "all_rows", lambda *a, **kw: (rows, {"_path": "split", "_sha256": "sealed"}))
    monkeypatch.setattr(vc, "ensure_upstream", lambda *a, **kw: {})
    monkeypatch.setattr(vc, "processor_config_identity", lambda *a: {"sha256": "processor"})
    monkeypatch.setattr(vc, "_runtime", lambda: {"device": "cpu", "torch": "test"})
    monkeypatch.setattr(vc, "_processor", lambda args: object())
    calls = []
    expected = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2) / 8

    def preprocess(processor, path):
        calls.append(str(path))
        return expected.clone(), {"shape": [1, 1, 2, 2, 2]}

    monkeypatch.setattr(vc, "preprocess_nifti", preprocess)
    return args, rows, calls, expected


def test_prepare_is_lossless_and_complete_cache_reuses_without_preprocessing(tiny_cache):
    args, rows, calls, expected = tiny_cache
    result = vc.prepare_volume_cache(args)
    assert result["execution_ready"]
    store = vc.VolumeStore(args)
    assert torch.equal(store.load(rows[0]), expected[0])
    assert len(calls) == 2
    assert vc.prepare_volume_cache(args)["status"] == "reused"
    assert len(calls) == 2
    assert vc.validate_volume_cache(args)["execution_ready"]


def test_both_finetuning_scopes_load_same_cache_and_bind_input_files(tiny_cache, monkeypatch):
    args, rows, calls, expected = tiny_cache
    vc.prepare_volume_cache(args)
    monkeypatch.setattr(vc, "preprocess_nifti", lambda *a: pytest.fail("Training repeated preprocessing"))
    identities = []
    for scope in ("final_block", "final_stage"):
        adapter = CachedCOLIPRIPartialAdapter(scope)
        prepared = adapter.prepare_rows(args, rows)
        batch, labels = adapter.load_batch(args, prepared, [1, 0], training=True, generator=None)
        assert torch.equal(batch, torch.cat([expected, expected]))
        assert labels.tolist() == [1, 0]
        assert all(row["cache_sha256"] == sha256_file(row["cache_path"]) for row in prepared)
        identities.append(adapter._store(args).identity)
        assert input_rows_identity(prepared) != input_rows_identity(rows)
    assert identities[0] == identities[1]
    assert len(calls) == 2


def test_interrupted_prepare_reuses_verified_completed_patient(tiny_cache, monkeypatch):
    args, rows, calls, _expected = tiny_cache
    original = vc.preprocess_nifti

    def interrupt(processor, path):
        if str(path) == rows[1]["image_path"]:
            raise RuntimeError("interrupted")
        return original(processor, path)

    monkeypatch.setattr(vc, "preprocess_nifti", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        vc.prepare_volume_cache(args)
    with pytest.raises(ContractError, match="missing"):
        vc.VolumeStore(args)
    monkeypatch.setattr(vc, "preprocess_nifti", original)
    assert vc.prepare_volume_cache(args)["execution_ready"]
    assert calls.count(rows[0]["image_path"]) == 1
    assert calls.count(rows[1]["image_path"]) == 1


def test_missing_receipt_requires_exhaustive_validation(tiny_cache):
    args, _rows, _calls, _expected = tiny_cache
    vc.prepare_volume_cache(args)
    root = vc.VolumeStore(args).root
    (root / vc.RECEIPT).unlink()
    assert not vc.prepare_volume_cache(args)["execution_ready"]
    assert not vc.validate_volume_cache(args, exhaustive=False)["execution_ready"]
    with pytest.raises(ContractError, match="exhaustive validation"):
        vc.VolumeStore(args)
    vc.validate_volume_cache(args)
    vc.VolumeStore(args)


def test_diagnostic_stat_cache_cannot_train(tiny_cache):
    args, _rows, _calls, _expected = tiny_cache
    args.source_fingerprint = "stat"
    assert not vc.prepare_volume_cache(args)["execution_ready"]
    assert not vc.validate_volume_cache(args)["execution_ready"]
    with pytest.raises(ContractError, match="SHA-256"):
        vc.VolumeStore(args)


def test_exhaustive_validation_detects_source_changes_with_preserved_stats(tiny_cache):
    args, rows, _calls, _expected = tiny_cache
    vc.prepare_volume_cache(args)
    source = Path(rows[0]["image_path"])
    stat = source.stat()
    source.write_bytes(b"changed")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ContractError, match="Source changed"):
        vc.validate_volume_cache(args)


def test_artifact_corruption_and_manifest_tampering_are_rejected(tiny_cache):
    args, rows, _calls, _expected = tiny_cache
    vc.prepare_volume_cache(args)
    store = vc.VolumeStore(args)
    path = store.root / store.manifest["entries"][vc._entry_key(rows[0])]["artifact"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ContractError):
        vc.VolumeStore(args)
    with pytest.raises(ContractError, match="hash changed"):
        vc.validate_volume_cache(args)
    manifest_path = store.root / vc.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"].pop(vc._entry_key(rows[0]))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="seal"):
        vc.VolumeStore(args)


def test_changed_preprocessing_uses_a_distinct_cache(tiny_cache, monkeypatch):
    args, _rows, _calls, _expected = tiny_cache
    vc.prepare_volume_cache(args)
    previous = vc.VolumeStore(args).root
    monkeypatch.setattr(vc, "processor_config_identity", lambda *a: {"sha256": "different"})
    assert vc._context(args)[2] != previous
    with pytest.raises(ContractError, match="missing"):
        vc.VolumeStore(args)


@pytest.mark.parametrize("target, expected", [("probe", ["probe"]), ("finetune", ["finetune"]), ("both", ["finetune", "probe"])])
@pytest.mark.parametrize("action", ["prepare", "validate"])
def test_cache_target_dispatch(target, expected, action, monkeypatch, capsys):
    calls = []
    for name, label in [("prepare_feature_cache", "probe"), ("prepare_volume_cache", "finetune"),
                        ("validate_feature_cache", "probe"), ("validate_volume_cache", "finetune")]:
        monkeypatch.setattr(cli, name, lambda args, label=label, **kw: calls.append(label) or {"execution_ready": True})
    assert cli.main(["--action", action, "--cache_target", target]) == 0
    assert calls == expected
    assert json.loads(capsys.readouterr().out)["cache_target"] == target


def test_default_both_dry_run_plans_both_without_preparation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "volume_cache_plan", lambda args: {"estimated_raw_volume_gib": 1})
    monkeypatch.setattr(cli, "cache_plan", lambda args, **kw: {"estimated_raw_feature_gib": 2})
    monkeypatch.setattr(cli, "prepare_volume_cache", lambda args: pytest.fail("Dry run wrote cache"))
    assert cli.main(["--action", "prepare", "--dry_run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cache_target"] == "both"
    assert result["estimated_total_raw_gib"] == 3
    assert result["side_effects"] is False


@pytest.mark.parametrize("source, adapter_type", [("cache", CachedCOLIPRIPartialAdapter), ("nifti", COLIPRIPartialAdapter)])
def test_training_selects_cached_or_legacy_adapter(source, adapter_type, monkeypatch):
    observed = []
    monkeypatch.setattr(shared_cli, "run_pipeline", lambda args, backend: observed.append(type(backend.adapter)) or {})
    assert ft_cli.main(["--input_source", source, "--dry_run"]) == 0
    assert observed == [adapter_type]


def test_migrated_nifti_run_gets_actionable_input_mode_error(tiny_cache, monkeypatch):
    from COLIPRI import cached_partial_finetune as cached

    args, _rows, _calls, _expected = tiny_cache
    args.percentage, args.seed, args.parameter_source = 100, 42, "publication"
    monkeypatch.setattr(cached, "read_migration", lambda root: {"entries": [{
        "percentage": 100, "seed": 42, "parameter_source": "publication",
        "legacy_model_identity": {"pretrained_model": {}},
    }]})
    with pytest.raises(ContractError, match="InputSource nifti"):
        CachedCOLIPRIPartialAdapter()._store(args)


def test_released_processor_cache_round_trip_on_synthetic_ct(tmp_path, monkeypatch):
    """Exercise real CPU preprocessing and serialization, with no encoder or patient data."""
    import nibabel as nib
    import numpy as np

    root = Path(__file__).resolve().parents[4]
    upstream = root / "tmp" / "upstream_colipri"
    if not upstream.is_dir():
        pytest.skip("Pinned COLIPRI processor checkout is not available")
    source = tmp_path / "synthetic.nii.gz"
    array = np.linspace(-1200, 1200, 8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    nib.save(nib.Nifti1Image(array, np.diag([2., 2., 2., 1.])), source)
    rows = [{"patient_id": "synthetic", "class_name": "Normal", "label": 2, "image_path": str(source)}]
    monkeypatch.setattr(vc, "all_rows", lambda *a, **kw: (rows, {"_path": "synthetic", "_sha256": "synthetic"}))
    args = SimpleNamespace(dataset_root=tmp_path, results_root=tmp_path / "results",
        cache_root=tmp_path / "cache", upstream_root=upstream, local_files_only=True,
        source_fingerprint="sha256", provision_log=None)
    vc._context(args)
    expected, _metadata = vc.preprocess_nifti(vc._processor(args), source)
    vc.prepare_volume_cache(args)
    actual = vc.VolumeStore(args).load(rows[0])
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected[0])
