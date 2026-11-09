import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Pillar0 import partial_finetune as original
from Pillar0 import frozen_cache as fc
from Pillar0 import train_atlas_partial_finetune as cli
from Pillar0.frozen_partial_finetune import FrozenPillarAtlasAdapter
from Pillar0.prefetch import PrefetchedAtlasBackend
from Pillar0.tests.test_frozen_forward import tiny_encoder
from _linear_probe.common import ContractError
from _supervised_adaptation.runtime import torch_load
from _supervised_adaptation.tests.test_runtime_and_resume import Logger, TinyAdapter, args, parameters, rows


@pytest.mark.parametrize("scope", ["final_block", "final_stage"])
def test_real_tiny_cache_adapter_training_matches_original_augmentation(tiny_encoder, tmp_path, monkeypatch, scope):
    source = []
    volumes = {}
    for i, label in enumerate(original.CLASS_NAMES):
        path = tmp_path / f"synthetic{i}.rave"
        path.write_bytes(f"synthetic{i}".encode())
        source.append({"patient_id": f"synthetic{i}", "cache_path": str(path), "image_path": str(path),
                       "label": i, "class_name": label, "label_str": label})
        volumes[str(path)] = torch.linspace(-1200 + i * 30, 1200, 48**3).reshape(1, 48, 48, 48)
    monkeypatch.setattr(original.PillarAtlasAdapter, "preflight", lambda self, args, rows: None)
    monkeypatch.setattr(original.PillarAtlasAdapter, "resolve_identity", lambda self, args: {"synthetic": True})
    monkeypatch.setattr(original.PillarAtlasAdapter, "prepare_rows", lambda self, args, rows: [dict(row) for row in rows])
    monkeypatch.setattr(original, "load_pillar_volume", lambda path: volumes[str(path)].clone())
    monkeypatch.setattr(original, "load_official_model", lambda **kw: copy.deepcopy(tiny_encoder))
    monkeypatch.setattr(fc, "load_pillar_volume", lambda path: volumes[str(path)].clone())
    monkeypatch.setattr(FrozenPillarAtlasAdapter, "_encoder", lambda self, args: copy.deepcopy(tiny_encoder))
    run_args = SimpleNamespace(frozen_cache_root=tmp_path / "frozen", results_root=tmp_path / "results",
        percentage=100, seed=42, parameter_source="publication", device="cpu", training_precision="fp32",
        frozen_scopes=[scope], frozen_stores={}, micro_batch_size=1, checkpoint_interval=1, resume=True,
        model_repo_id="synthetic", model_revision="synthetic", local_files_only=True, prefetch=True)
    profile = Path(original.__file__).with_name(f"publication_atlas_{scope}_finetune.json")
    p = {**json.loads(profile.read_text())["parameters"], "max_steps": 2, "effective_batch_size": 5,
         "validation_interval_steps": 1, "early_stopping_patience_steps": 0}
    frozen = FrozenPillarAtlasAdapter(scope)
    frozen.preflight(run_args, source)
    try:
        for label, adapter, prefetch in [("original", original.PillarAtlasAdapter(scope), False),
                                        ("frozen", frozen, True)]:
            if label == "frozen":
                def cached_encoder(self, args):
                    encoder = copy.deepcopy(tiny_encoder)
                    def forbid(*_):
                        raise AssertionError("Training repeated a frozen prefix")
                    encoder.model.visual.patch_embeds["abdomen_ct"].register_forward_pre_hook(forbid)
                    for stage in encoder.model.visual.atlas_models[:2]:
                        stage.register_forward_pre_hook(forbid)
                    if scope == "final_block":
                        for block in encoder.model.visual.atlas_models[2].blocks[:-1]:
                            block.register_forward_pre_hook(forbid)
                    return encoder
                monkeypatch.setattr(FrozenPillarAtlasAdapter, "_encoder", cached_encoder)
            run_args.prefetch = prefetch
            backend = PrefetchedAtlasBackend(adapter)
            backend.feature_identity = lambda _: {"synthetic": True}
            backend.fit_one(p, source, source, tmp_path / label, run_args, {"fold": 1}, Logger())
        reference = torch_load(tmp_path / "original" / "last_checkpoint.pt")
        actual = torch_load(tmp_path / "frozen" / "last_checkpoint.pt")
        for name, value in reference["model_trainable_state"].items():
            torch.testing.assert_close(value, actual["model_trainable_state"][name], rtol=1e-5, atol=1e-6)
        assert torch.equal(reference["augmentation_generator_state"], actual["augmentation_generator_state"])
        assert reference["history"] == actual["history"]
    finally:
        for store in run_args.frozen_stores.values():
            store.finish(True)


def test_checkpoint_refuses_changed_orientation_values(tmp_path):
    def backend(digest):
        adapter = TinyAdapter()
        adapter.frozen_rows_identity = lambda args, rows: digest
        result = PrefetchedAtlasBackend(adapter)
        result.feature_identity = lambda _: {"synthetic": True}
        return result
    run_args = args()
    run_args.prefetch = True
    backend("original").fit_one(parameters(), rows("train"), rows("val"), tmp_path, run_args, {"fold": 1}, Logger())
    with pytest.raises(ContractError):
        backend("changed").fit_one(parameters(), rows("train"), rows("val"), tmp_path, run_args, {"fold": 1}, Logger())


@pytest.mark.parametrize("source", ["frozen", "cache"])
def test_cli_selects_inputs_and_defaults_without_cache_writes(tmp_path, monkeypatch, source):
    observed = []
    def run(args, backend):
        observed.append((args, backend.adapter))
        return {"status": "dry_run"}
    monkeypatch.setattr(cli.shared_cli, "run_pipeline", run)
    assert cli.main(["--percentage", "100", "--input_source", source, "--results_root", str(tmp_path), "--dry_run"]) == 0
    args, adapter = observed[0]
    assert isinstance(adapter, FrozenPillarAtlasAdapter) == (source == "frozen")
    assert args.prefetch and args.micro_batch_size == 1
    assert args.frozen_cache_root == tmp_path / "Cache/FrozenRuns"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail_second", [False, True])
def test_cli_shares_both_boundaries_and_closes_after_all_work(monkeypatch, fail_second):
    events = []
    class Store:
        def finish(self, success):
            events.append(("finish", success))
    def run(args, backend):
        events.append(args.adaptation_scope)
        assert args.frozen_scopes == ["final_stage", "final_block"]
        if not args.frozen_stores:
            args.frozen_stores["synthetic"] = Store()
        if fail_second and args.adaptation_scope == "final_block":
            raise RuntimeError("second scope interrupted")
        return {"status": "complete"}
    monkeypatch.setattr(cli.shared_cli, "run_pipeline", run)
    assert cli.main(["--adaptation_scope", "both"]) == int(fail_second)
    assert events == ["final_stage", "final_block", ("finish", not fail_second)]
