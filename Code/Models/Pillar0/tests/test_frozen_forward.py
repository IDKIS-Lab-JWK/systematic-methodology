import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F

from Pillar0 import model as model_module
from Pillar0 import partial_finetune as partial
from Pillar0.augmentations import ORIENTATIONS, rotate_volume
from Pillar0.frozen_forward import FrozenAtlasClassifier, capture_prefixes, features_from_prefix


@pytest.fixture
def tiny_encoder(monkeypatch):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    try:
        Visual = get_class_from_dynamic_module("multimodal_atlas.MultiModalAtlas", model_module.MODEL_REPO_ID,
                                              revision=model_module.MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("Pinned Atlas custom code is not present locally")
    config = {"modalities": {"abdomen_ct": {"image_size": [48, 48, 48], "in_chans": 11,
                "local2global": [2, 2, 2], "merge_ratio": [2, 2, 2], "num_features": 6, "patch_size": [6, 6, 6]}},
              "model": {"stages": [2, 2, 8], "atlas_args": {"dim": 12, "num_heads": 3,
                        "mlp_ratio": 2., "qkv_bias": True, "drop": 0., "attn_drop": 0., "drop_path": 0.}}}
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.visual = Visual(None, embed_dim=12, model_config=config)
        def extract_vision_feats(self, images):
            return F.normalize(self.model.visual(images), dim=-1)
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    monkeypatch.setattr(model_module, "FEATURE_DIM", 36)
    monkeypatch.setattr(partial, "FEATURE_DIM", 36)
    torch.manual_seed(18)
    try:
        yield Encoder().eval()
    finally:
        torch.set_num_threads(before)


def test_all_ten_orientations_preserve_multiscale_features_after_cpu_round_trip(tiny_encoder):
    raw = torch.linspace(-1200, 1200, 48**3).reshape(1, 48, 48, 48)
    with torch.no_grad():
        for plane, turns in ORIENTATIONS:
            image = rotate_volume(raw, plane, turns).unsqueeze(0)
            reference = model_module.extract_vision_features(tiny_encoder, image)
            captured = capture_prefixes(tiny_encoder, image, ["final_block", "final_stage"])
            for scope, inputs in captured.items():
                restored = {k: torch.stack([v[0].cpu().contiguous()]) for k, v in inputs.items()}
                actual = features_from_prefix(tiny_encoder, restored, scope)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scope", ["final_block", "final_stage"])
@pytest.mark.parametrize("batch", [1, 2])
def test_split_logits_gradients_and_optimizer_update_match_released_path(tiny_encoder, scope, batch):
    reference = partial.AtlasPartialClassifier(copy.deepcopy(tiny_encoder), scope)
    cached = FrozenAtlasClassifier(copy.deepcopy(tiny_encoder), scope)
    cached.load_state_dict(reference.state_dict())
    raw = torch.randn(batch, 1, 48, 48, 48) * 400
    cached.encoder.eval()
    with torch.no_grad():
        inputs = capture_prefixes(cached.encoder, raw, [scope])[scope]
        inputs = {k: torch.stack([item.cpu().contiguous() for item in v]) for k, v in inputs.items()}
    reference.train()
    cached.train()
    original = reference(raw)
    actual = cached(inputs)
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    labels = torch.arange(batch) % 5
    F.cross_entropy(original, labels).backward()
    F.cross_entropy(actual, labels).backward()
    other = dict(cached.named_parameters())
    for name, value in reference.named_parameters():
        if value.grad is None:
            assert other[name].grad is None
        else:
            torch.testing.assert_close(value.grad, other[name].grad, rtol=1e-4, atol=1e-6)
    for model in (reference, cached):
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001).step()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, cached.state_dict()[name], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(os.environ.get("PILLAR0_TEST_RELEASED_FROZEN") != "1", reason="Opt-in full-size pinned checkpoint GPU test")
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_full_size_pinned_checkpoint_all_orientations_and_cache_round_trip(tmp_path, precision):
    """One synthetic CT, actual 384-cube model, no cohort data or training."""
    from types import SimpleNamespace
    from Pillar0.frozen_cache import FrozenStore, execution_contract
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = SimpleNamespace(device="cuda", training_precision=precision, frozen_cache_root=tmp_path / "cache",
                           results_root=tmp_path / "results", percentage=100, seed=42, parameter_source="publication")
    encoder = model_module.load_official_model(device="cuda", local_files_only=True).eval()
    contract = execution_contract(args, {"synthetic": True}, encoder, ["final_block", "final_stage"])
    source = tmp_path / "synthetic.rave"
    source.write_bytes(b"Synthetic coordinate ramp, not a patient file")
    row = {"patient_id": "synthetic", "cache_path": str(source), "label": 0}
    raw = torch.linspace(-1200, 1200, 384**3).reshape(1, 384, 384, 384)
    store = FrozenStore(args, contract)
    try:
        store.prepare([row], lambda: encoder, lambda _: raw)
        assert len(store.manifest["forward_verification"]) == 10
        assert len(list((store.root / "artifacts").glob("*.pt"))) == 20
        for plane, turns in ORIENTATIONS:
            from Pillar0.augmentations import orientation_key
            for scope in ("final_block", "final_stage"):
                saved = store.load(row, orientation_key(plane, turns), scope)
                assert saved["tokens"].shape == (64, 384)
                assert saved["early"].shape == (768,)
        (tmp_path / "full_size_verification.json").write_text(json.dumps({
            "precision": precision, "maximum_absolute_errors": store.manifest["forward_verification"],
            "tensor_shapes": contract["tensor_shapes"], "runtime": contract["runtime"],
            "vision_weights_sha256": contract["vision_weights_sha256"],
            "artifact_bytes": sum(p.stat().st_size for p in (store.root / "artifacts").glob("*.pt"))}, indent=2))
    finally:
        store.finish(store.manifest.get("status") == "ready")
    assert not store.root.exists()
