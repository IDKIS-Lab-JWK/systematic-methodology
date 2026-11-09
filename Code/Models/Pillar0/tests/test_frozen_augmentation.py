import copy
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Pillar0.augmentations import ORIENTATIONS, PLANES, augmentation_contract, draw_orientations, orientation_key, rotate_volume
from Pillar0 import frozen_cache as fc
from Pillar0 import partial_finetune as original
from Pillar0.prefetch import PrefetchedAtlasBackend, prefetched_loader, MicrobatchLookahead
from _linear_probe.common import ContractError
from _supervised_adaptation.runtime import DeterministicBatchStream, torch_load
from _supervised_adaptation.tests.test_runtime_and_resume import TinyAdapter, args, parameters, rows, Logger


def test_all_rotation_choices_and_rng_match_original_loader(monkeypatch):
    volume = torch.arange(27.).reshape(1, 3, 3, 3)
    monkeypatch.setattr(original, "load_pillar_volume", lambda _: volume.clone())
    adapter = original.PillarAtlasAdapter()
    source = [{"cache_path": "synthetic", "label": i % 5} for i in range(80)]
    first, second = [torch.Generator().manual_seed(982) for _ in range(2)]
    observed, _ = adapter.load_batch(None, source, list(range(80)), training=True, generator=first)
    selected = draw_orientations(80, second)
    inventory = {orientation_key(*p): rotate_volume(volume, *p) for p in ORIENTATIONS}
    assert all(torch.equal(actual, inventory[key]) for actual, key in zip(observed, selected))
    assert torch.equal(first.get_state(), second.get_state())
    assert len({value.numpy().tobytes() for value in inventory.values()}) == 10
    counts = Counter(orientation_key(p, t) for p in range(3) for t in range(4))
    assert counts["identity"] == 3 and sorted(counts.values()) == [1] * 9 + [3]
    before = second.get_state().clone()
    assert draw_orientations(5, second, training=False) == ["identity"] * 5
    assert torch.equal(second.get_state(), before)


def test_digest_binds_scalar_buffers_shapes_dtypes_and_values():
    assert fc.tensor_digest(torch.tensor(4)) == fc.tensor_digest(torch.tensor(4))
    assert fc.tensor_digest(torch.tensor(4)) != fc.tensor_digest(torch.tensor(5))
    assert fc.tensor_digest(torch.tensor(4)) != fc.tensor_digest(torch.tensor([4]))
    assert fc.tensor_digest(torch.tensor(4.)) != fc.tensor_digest(torch.tensor(4))


def test_prefetch_does_not_commit_unconsumed_rotations(monkeypatch):
    volume = torch.arange(27.).reshape(1, 3, 3, 3)
    monkeypatch.setattr(original, "load_pillar_volume", lambda _: volume.clone())
    adapter = original.PillarAtlasAdapter()
    source = [{"cache_path": "synthetic", "label": i % 5} for i in range(5)]
    rng, expected = [torch.Generator().manual_seed(42) for _ in range(2)]
    stream = DeterministicBatchStream(5, 8)
    lookahead = MicrobatchLookahead(DeterministicBatchStream(5, 8), 2, 2)
    indices = stream.next(2)
    reference, _ = adapter.load_batch(None, source, indices, training=True, generator=expected)
    with prefetched_loader(adapter, lookahead):
        actual, _ = adapter.load_batch(None, source, indices, training=True, generator=rng)
        assert torch.equal(actual, reference)
    assert torch.equal(rng.get_state(), expected.get_state())
    assert "load_batch" not in adapter.__dict__


@pytest.mark.parametrize("micro", [1, 5])
def test_prefetch_training_validation_and_resume_preserve_augmented_updates(tmp_path, micro):
    class AugmentedTiny(TinyAdapter):
        def load_batch(self, args, rows, indices, *, training, generator):
            inputs, labels = super().load_batch(args, rows, indices, training=training, generator=generator)
            views = draw_orientations(len(indices), generator, training)
            codes = {orientation_key(*p): i for i, p in enumerate(ORIENTATIONS)}
            return inputs + torch.tensor([codes[key] / 10 for key in views]).unsqueeze(1), labels
    run_args = args()
    run_args.micro_batch_size = micro
    run_args.prefetch = True
    def fit(folder, prefetch=True, fail=None):
        run_args.prefetch = prefetch
        backend = PrefetchedAtlasBackend(AugmentedTiny(fail))
        backend.feature_identity = lambda _: {"synthetic": True}
        return backend.fit_one(parameters(), rows("train"), rows("validation"), folder,
                               run_args, {"fold": 1}, Logger())
    fit(tmp_path / "original", False)
    fit(tmp_path / "prefetched")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        fit(tmp_path / "resumed", fail=5 // micro + 1)
    fit(tmp_path / "resumed")
    reference = torch_load(tmp_path / "original" / "last_checkpoint.pt")
    for name in ("prefetched", "resumed"):
        actual = torch_load(tmp_path / name / "last_checkpoint.pt")
        assert all(torch.equal(value, actual["model_trainable_state"][key]) for key, value in reference["model_trainable_state"].items())
        assert torch.equal(actual["augmentation_generator_state"], reference["augmentation_generator_state"])
        assert actual["history"] == reference["history"]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    run_args = SimpleNamespace(frozen_cache_root=tmp_path / "cache", results_root=tmp_path / "results",
                              percentage=100, seed=42, parameter_source="publication", device="cpu", training_precision="fp32")
    source = []
    for i in range(2):
        path = tmp_path / f"source_{i}.rave"
        path.write_bytes(f"source{i}".encode())
        source.append({"patient_id": f"synthetic{i}", "cache_path": str(path), "label": i})
    contract = {"boundaries": ["final_block", "final_stage"], "augmentation": augmentation_contract(),
                "vision_weights_sha256": "synthetic", "tensor_shapes": {"tokens": [2, 3], "early": [6]}}
    calls = []
    class Encoder(torch.nn.Module):
        pass
    def capture(encoder, volume, scopes):
        calls.append(float(volume.flatten()[0]))
        return {scope: {"tokens": torch.ones(1, 2, 3), "early": torch.zeros(1, 6)} for scope in scopes}
    monkeypatch.setattr(fc, "weights_digest", lambda _: "synthetic")
    monkeypatch.setattr(fc, "capture_prefixes", capture)
    monkeypatch.setattr(fc, "features_from_prefix", lambda *_: torch.zeros(1, 9))
    monkeypatch.setattr(fc, "extract_vision_features", lambda *_, **kw: torch.zeros(1, 9))
    loader = lambda _: torch.arange(27.).reshape(1, 3, 3, 3)
    return run_args, source, contract, calls, Encoder, loader


def test_cache_builds_all_views_reuses_and_removes_only_owned_files(synthetic):
    run_args, source, contract, calls, factory, loader = synthetic
    keep = run_args.frozen_cache_root / "permanent_keep"
    keep.parent.mkdir(parents=True)
    keep.write_text("permanent")
    store = fc.FrozenStore(run_args, contract)
    store.prepare(source, factory, loader)
    assert len(calls) == 20
    assert len(store.manifest["forward_verification"]) == 10
    assert len(list((store.root / "artifacts").glob("*.pt"))) == 40
    value_digest = store.rows_digest(source, "final_block")
    store.prepare(source, factory, loader)
    assert len(calls) == 20
    store.finish(False)
    resumed = fc.FrozenStore(run_args, contract)
    resumed.prepare(source, factory, loader)
    assert len(calls) == 20 and resumed.rows_digest(source, "final_block") == value_digest
    resumed.finish(True)
    assert not resumed.root.exists() and keep.read_text() == "permanent"


def test_interruption_reuses_complete_patients_and_regenerates_incomplete_views(synthetic, monkeypatch):
    run_args, source, contract, calls, factory, loader = synthetic
    capture = fc.capture_prefixes
    def interrupt(*a):
        if len(calls) == 13:
            raise RuntimeError("synthetic interruption")
        return capture(*a)
    monkeypatch.setattr(fc, "capture_prefixes", interrupt)
    store = fc.FrozenStore(run_args, contract)
    with pytest.raises(RuntimeError, match="interruption"):
        store.prepare(source, factory, loader)
    store.finish(False)
    monkeypatch.setattr(fc, "capture_prefixes", capture)
    resumed = fc.FrozenStore(run_args, contract)
    resumed.prepare(source, factory, loader)
    assert len(calls) == 23
    resumed.finish(True)


@pytest.mark.parametrize("damage", ["artifact", "source", "manifest", "unexpected_file"])
def test_cache_refuses_corruption_source_changes_and_unsafe_cleanup(synthetic, damage):
    run_args, source, contract, calls, factory, loader = synthetic
    store = fc.FrozenStore(run_args, contract)
    store.prepare(source, factory, loader)
    store.finish(False)
    if damage == "artifact":
        next((store.root / "artifacts").glob("*.pt")).write_bytes(b"corrupt")
    elif damage == "source":
        Path(source[0]["cache_path"]).write_bytes(b"changed source")
    elif damage == "manifest":
        (store.root / fc.MANIFEST).write_text('{}')
        with pytest.raises(ContractError, match="manifest"):
            fc.FrozenStore(run_args, contract)
        return
    else:
        (store.root / "unrelated.txt").write_text("retain")
    resumed = fc.FrozenStore(run_args, contract)
    if damage == "unexpected_file":
        with pytest.raises(ContractError, match="Unexpected"):
            resumed.finish(True)
        assert (store.root / "unrelated.txt").exists()
    else:
        try:
            with pytest.raises(ContractError):
                resumed.prepare(source, factory, loader)
        finally:
            resumed.finish(False)


def test_verification_failure_preserves_diagnostics_and_no_committed_patient(synthetic, monkeypatch):
    run_args, source, contract, calls, factory, loader = synthetic
    monkeypatch.setattr(fc, "features_from_prefix", lambda *_: torch.ones(1, 9))
    store = fc.FrozenStore(run_args, contract)
    try:
        with pytest.raises(ContractError, match="verification failed"):
            store.prepare(source, factory, loader)
        assert not store.manifest["entries"]
    finally:
        store.finish(False)
    assert 'forward_verification_failed' in store.log.read_text()


def test_boundaries_keep_identity_when_requested_together(synthetic):
    run_args, source, contract, calls, factory, loader = synthetic
    combined = fc.FrozenStore(run_args, contract)
    single = fc.FrozenStore(run_args, {**contract, "boundaries": ["final_block"]})
    try:
        assert combined.identity_for("final_block") == single.identity_for("final_block")
        assert combined.identity_for("final_stage") != single.identity_for("final_block")
    finally:
        combined.finish(False)
        single.finish(False)


def test_cache_lease_serializes_launches(synthetic):
    run_args, source, contract, calls, factory, loader = synthetic
    first = fc.FrozenStore(run_args, contract)
    first.prepare(source, factory, loader)
    entered, acquired = threading.Event(), threading.Event()
    failures = []
    def second_launch():
        entered.set()
        try:
            second = fc.FrozenStore(run_args, contract)
            acquired.set()
            try:
                second.prepare(source, factory, loader)
            finally:
                second.finish(True)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=second_launch)
    thread.start()
    try:
        assert entered.wait(2)
        assert not acquired.wait(.1)
    finally:
        first.finish(False)
    thread.join(5)
    assert not thread.is_alive() and acquired.is_set() and not failures
    assert len(calls) == 20
