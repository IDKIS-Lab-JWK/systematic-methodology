import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from COLIPRI import frozen_cache as fc
from COLIPRI import train_primus_m_partial_finetune as cli
from COLIPRI.frozen_partial_finetune import FrozenCOLIPRIPartialAdapter
from _linear_probe.common import ContractError


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    args = SimpleNamespace(cache_root=tmp_path / "cache", results_root=tmp_path / "results",
                           percentage=100, seed=42, parameter_source="publication",
                           device="cpu", training_precision="fp32")
    rows = [{"patient_id": f"p{i}", "image_path": str(tmp_path / f"{i}.nii.gz"), "label": i} for i in range(3)]
    contract = {"boundaries": ["final_block", "final_stage"], "weights_sha256": "synthetic"}
    entries = {fc._entry_key(row): {"sha256": f"ct-{index}"} for index, row in enumerate(rows)}
    volumes = SimpleNamespace(manifest={"entries": entries}, load=lambda row: torch.ones(1, 2, 2, 2))
    calls = []
    def capture(encoder, images, scopes):
        calls.append(len(calls))
        return {scope: torch.full((1, 8, 6), float(index)) for index, scope in enumerate(scopes)}, (24, 24, 24)
    class Encoder(torch.nn.Module):
        def encode_image(self, *a, **kw):
            return torch.zeros(1, 768)
    monkeypatch.setattr(fc, "capture_prefixes", capture)
    monkeypatch.setattr(fc, "pooled_from_prefix", lambda *a: torch.zeros(1, 768))
    return args, rows, contract, volumes, calls, Encoder


def test_frozen_cache_reuses_both_scopes_and_removes_only_temporary_files(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    permanent = args.cache_root / "Volumes" / "keep.pt"
    permanent.parent.mkdir(parents=True)
    permanent.write_bytes(b"permanent")
    store = fc.FrozenStore(args, contract)
    identity = store.identity
    store.prepare(rows, volumes, factory)
    assert len(calls) == 3
    assert store.manifest["forward_verification"] == {"final_block": 0., "final_stage": 0.}
    store.prepare(rows, volumes, factory)
    assert len(calls) == 3
    assert torch.equal(store.load(rows[0], "final_stage"), torch.ones(8, 6))
    assert torch.equal(store.load(rows[0], "final_block"), torch.zeros(8, 6))
    store.finish(True)
    assert not store.root.exists()
    assert permanent.read_bytes() == b"permanent"
    regenerated = fc.FrozenStore(args, contract)
    assert regenerated.identity == identity
    regenerated.prepare(rows, volumes, factory)
    assert len(calls) == 6
    regenerated.finish(True)


def test_interrupted_cache_preparation_reuses_finished_patients(synthetic, monkeypatch):
    args, rows, contract, volumes, calls, factory = synthetic
    capture = fc.capture_prefixes
    def interrupted(*a):
        if len(calls) == 1:
            raise RuntimeError("interrupted")
        return capture(*a)
    monkeypatch.setattr(fc, "capture_prefixes", interrupted)
    store = fc.FrozenStore(args, contract)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.prepare(rows, volumes, factory)
    store.finish(False)
    assert store.root.is_dir()
    monkeypatch.setattr(fc, "capture_prefixes", capture)
    resumed = fc.FrozenStore(args, contract)
    resumed.prepare(rows, volumes, factory)
    assert len(calls) == 3
    resumed.finish(True)


def test_corrupt_frozen_payload_is_rejected_on_resume(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    store = fc.FrozenStore(args, contract)
    store.prepare(rows, volumes, factory)
    path = next((store.root / "artifacts").glob("*.pt"))
    store.finish(False)
    path.write_bytes(b"corrupt")
    resumed = fc.FrozenStore(args, contract)
    try:
        with pytest.raises(ContractError, match="hash changed"):
            resumed.prepare(rows, volumes, factory)
    finally:
        resumed.finish(False)


def test_failed_forward_verification_logs_error_and_writes_no_shards(synthetic, monkeypatch):
    args, rows, contract, volumes, calls, factory = synthetic
    monkeypatch.setattr(fc, "pooled_from_prefix", lambda *a: torch.full((1, 768), 0.125))
    store = fc.FrozenStore(args, contract)
    try:
        with pytest.raises(ContractError, match="maximum absolute error=0.125"):
            store.prepare(rows, volumes, factory)
        assert not store.manifest["entries"]
        assert not list(store.root.glob("artifacts/*.pt"))
    finally:
        store.finish(False)
    events = [json.loads(line) for line in store.log.read_text().splitlines()]
    diagnostic = next(event for event in events if event["event"] == "forward_verification_failed")
    assert diagnostic["maximum_absolute_error"] == 0.125
    assert diagnostic["rtol"] == 1e-5 and diagnostic["atol"] == 1e-6


def test_changed_source_or_contract_cannot_reuse_wrong_tensors(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    store = fc.FrozenStore(args, contract)
    store.prepare(rows, volumes, factory)
    volumes.manifest["entries"][fc._entry_key(rows[0])]["sha256"] = "changed"
    with pytest.raises(ContractError, match="CT input changed"):
        store.prepare(rows, volumes, factory)
    store.finish(False)
    changed = fc.FrozenStore(args, {**contract, "weights_sha256": "new"})
    assert changed.root != store.root and changed.identity != store.identity
    changed.finish(False)


def test_single_scope_identity_matches_combined_capture(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    combined = fc.FrozenStore(args, contract)
    single = fc.FrozenStore(args, {**contract, "boundaries": ["final_block"]})
    try:
        assert combined.identity_for("final_block") == single.identity_for("final_block")
        assert combined.identity_for("final_stage") != single.identity_for("final_block")
    finally:
        combined.finish(False)
        single.finish(False)


def test_value_digest_survives_serialization_and_detects_rounding(tmp_path):
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    first = fc.tensor_digest(tensor)
    path = tmp_path / "tokens.pt"
    torch.save(tensor, path)
    assert fc.tensor_digest(torch.load(path, weights_only=True)) == first
    tensor[0, 0] += 1e-6
    assert fc.tensor_digest(tensor) != first


def test_cleanup_refuses_unexpected_files(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    store = fc.FrozenStore(args, contract)
    store.prepare(rows, volumes, factory)
    extra = store.root / "user_notes.txt"
    extra.write_text("keep")
    with pytest.raises(ContractError, match="Unexpected"):
        store.finish(True)
    assert extra.is_file()
    assert list((store.root / "artifacts").glob("*.pt"))
    assert not store.lock.is_locked


def test_shared_cache_lease_waits_for_other_launch(synthetic):
    args, rows, contract, volumes, calls, factory = synthetic
    first = fc.FrozenStore(args, contract)
    first.prepare(rows, volumes, factory)
    ready = threading.Event()
    acquired = threading.Event()
    errors = []
    def second_launch():
        ready.set()
        try:
            second = fc.FrozenStore(args, contract)
            acquired.set()
            second.prepare(rows, volumes, factory)
            second.finish(True)
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=second_launch)
    worker.start()
    assert ready.wait(3)
    assert not acquired.wait(.1)
    first.finish(False)
    worker.join(5)
    assert not worker.is_alive() and acquired.is_set() and not errors
    assert len(calls) == 3


@pytest.mark.parametrize("fail", [False, True])
def test_cli_cleanup_happens_after_all_scopes_and_retains_on_failure(monkeypatch, fail):
    calls = []
    class Store:
        def finish(self, success):
            calls.append(("finish", success))
    def pipeline(args, backend):
        assert isinstance(backend.adapter, FrozenCOLIPRIPartialAdapter)
        assert args.prefetch is True
        assert args.frozen_scopes == ["final_stage", "final_block"]
        args.frozen_stores.setdefault("shared", Store())
        calls.append(args.adaptation_scope)
        if fail and args.adaptation_scope == "final_block":
            raise RuntimeError("training failed")
        return {"status": "complete"}
    monkeypatch.setattr(cli.shared_cli, "run_pipeline", pipeline)
    assert cli.main(["--adaptation_scope", "both"]) == int(fail)
    assert calls == ["final_stage", "final_block", ("finish", not fail)]


def test_default_frozen_dry_run_never_creates_a_cache(tmp_path, capsys):
    root = tmp_path / "not_created"
    assert cli.main(["--dry_run", "--cache_root", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_claim"]["input_source"] == "frozen"
    assert result["model_claim"]["prefetch_microbatches"] == 1
    assert not root.exists()


def test_prefetch_can_be_disabled_without_changing_input_mode(capsys):
    assert cli.main(["--dry_run", "--no-prefetch"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_claim"]["input_source"] == "frozen"
    assert result["model_claim"]["prefetch_microbatches"] == 0


def test_real_tiny_encoder_adapter_generation_shared_reuse_and_training(tmp_path, monkeypatch):
    """Exercise the actual cache + adapter + prefetch + trainer on synthetic CTs."""
    import copy
    from COLIPRI import frozen_partial_finetune as adapter_module
    from COLIPRI.cached_partial_finetune import CachedCOLIPRIPartialAdapter
    from COLIPRI.partial_finetune import COLIPRIPartialAdapter
    from COLIPRI.frozen_forward import FrozenPrimusMPartialClassifier
    from COLIPRI.prefetched_backends import PrefetchedAdaptationBackend
    from COLIPRI.tests.test_frozen_forward import tiny_encoder
    from _supervised_adaptation.tests.test_runtime_and_resume import Logger
    from _supervised_adaptation.profiles import build_candidates
    from _linear_probe.common import CLASS_NAMES, candidate_id, sha256_file

    torch.manual_seed(41)
    encoder = tiny_encoder()
    rows, entries = [], {}
    for index in range(5):
        path = tmp_path / f"ct{index}.pt"
        torch.save(torch.randn(1, 16, 16, 16), path)
        row = {"patient_id": str(index), "label": index, "class_name": CLASS_NAMES[index],
               "image_path": str(path), "cache_path": str(path), "cache_sha256": sha256_file(path)}
        rows.append(row)
        entries[fc._entry_key(row)] = {"sha256": row["cache_sha256"]}
    volumes = SimpleNamespace(manifest={"entries": entries}, identity={"synthetic_ct_cache": True},
                              load=lambda row: torch.load(row["cache_path"], weights_only=True),
                              prepare_rows=lambda rows: [dict(row) for row in rows])
    def preflight(self, args, rows):
        self._source_identity = {"synthetic_source": True}
        self._checkpoint_identity = {"path": "synthetic", "sha256": "synthetic"}
        self._processor_identity = {"synthetic_processor": True}
    monkeypatch.setattr(COLIPRIPartialAdapter, "preflight", preflight)
    monkeypatch.setattr(CachedCOLIPRIPartialAdapter, "_store", lambda self, args: volumes)
    monkeypatch.setattr(FrozenCOLIPRIPartialAdapter, "_encoder", lambda self, args: copy.deepcopy(encoder))
    monkeypatch.setattr(fc, "GRID_SHAPE", (2, 2, 2))
    monkeypatch.setattr(adapter_module, "FrozenPrimusMPartialClassifier",
                        lambda encoder, scope, dropout: FrozenPrimusMPartialClassifier(encoder, scope, dropout, (2, 2, 2)))
    run_args = SimpleNamespace(cache_root=tmp_path / "cache", results_root=tmp_path / "results",
                               percentage=100, seed=42, parameter_source="default", device="cpu",
                               training_precision="fp32", frozen_stores={}, frozen_scopes=["final_stage", "final_block"],
                               micro_batch_size=4, checkpoint_interval=1, prefetch=True, resume=True)
    try:
        for scope in run_args.frozen_scopes:
            adapter = FrozenCOLIPRIPartialAdapter(scope)
            backend = PrefetchedAdaptationBackend(adapter)
            backend.ensure_feature_rows(run_args, rows)
            assert len(run_args.frozen_stores) == 1
            profile = json.loads(adapter.default_profile_path.read_text())
            parameters = build_candidates(profile)[0]
            parameters.update(max_steps=1, warmup_steps=0, validation_interval_steps=1)
            parameters.pop("candidate_id", None)
            parameters["candidate_id"] = candidate_id(parameters)
            backend.fit_one(parameters, rows, rows, tmp_path / scope, run_args, {"fold": 1}, Logger())
            assert (tmp_path / scope / "completion.json").is_file()
            checkpoint = torch.load(tmp_path / scope / "last_checkpoint.pt", weights_only=True)
            assert checkpoint["identity"]["frozen_input_values"]["training"] == adapter.frozen_rows_identity(rows)
    finally:
        for store in run_args.frozen_stores.values():
            store.finish(True)
    assert all(Path(row["cache_path"]).is_file() for row in rows)
