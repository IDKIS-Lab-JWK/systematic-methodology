from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

from FlexiCT import EXPECTED_PATCH_TOKENS, FEATURE_DIM, INPUT_SHAPE_DHW
from FlexiCT import model as model_module


FAKE_MODELS = '''
import torch
from torch import nn

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.linspace(0.5, 1.5, 864))

    def forward(self, images, is_training=False):
        batch = images.shape[0]
        cls = self.anchor.unsqueeze(0).expand(batch, -1)
        patches = self.anchor.view(1, 1, -1).expand(batch, 8000, -1)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }

def flexi_ct_backbone_base(**kwargs):
    expected = {
        "patch_size": 8,
        "in_chans": 1,
        "n_storage_tokens": 4,
        "qkv_bias": False,
        "mask_k_bias": True,
        "drop_path_rate": 0.2,
        "layerscale_init": 1e-5,
    }
    if kwargs != expected:
        raise AssertionError((kwargs, expected))
    return TinyBackbone()
'''


def make_fake_assets(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    root = tmp_path / "upstream"
    package = root / "flexi_ct"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('canonical upstream __init__ must not execute')\n",
        encoding="utf-8",
    )
    (package / "models.py").write_text(FAKE_MODELS, encoding="utf-8")
    monkeypatch.setattr(
        model_module,
        "validate_upstream",
        lambda _root, revision: {
            "repository": "synthetic",
            "revision": revision,
            "path": str(root),
        },
    )
    monkeypatch.setattr(model_module, "_validate_torch_runtime", lambda _torch: None)
    backbone = model_module._build_official_backbone(root)
    state = {f"vision_model.{key}": value for key, value in backbone.state_dict().items()}
    state["vlm_vision_projection.weight"] = torch.randn(FEATURE_DIM, 1728)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": state, "metadata": {"synthetic": True}}, checkpoint)
    return root, checkpoint, state


def test_vision_only_loader_is_strict_frozen_and_text_tower_free(
    tmp_path: Path, monkeypatch
) -> None:
    root, checkpoint, _ = make_fake_assets(tmp_path, monkeypatch)
    canonical_before = {name for name in sys.modules if name == "flexi_ct" or name.startswith("flexi_ct.")}
    transformers_before = {name for name in sys.modules if name.startswith("transformers")}
    encoder, details = model_module.build_vision_only_encoder(
        upstream_root=root, checkpoint_path=checkpoint, device="cpu"
    )
    canonical_after = {name for name in sys.modules if name == "flexi_ct" or name.startswith("flexi_ct.")}
    transformers_after = {name for name in sys.modules if name.startswith("transformers")}
    assert canonical_after == canonical_before
    assert transformers_after == transformers_before
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert details["text_tower_loaded"] is False
    assert details["checkpoint_load_mode"] == "torch_weights_only"
    assert details["expected_patch_tokens"] == EXPECTED_PATCH_TOKENS
    assert details["projection_dimension"] == FEATURE_DIM

    image = torch.zeros((1, *INPUT_SHAPE_DHW), dtype=torch.float32)
    embedding, extraction = model_module.extract_embedding(
        encoder, image, device="cpu", precision="auto"
    )
    assert embedding.shape == (FEATURE_DIM,)
    assert embedding.dtype == torch.float32
    assert extraction["precision"] == "fp32"
    assert extraction["l2_norm"] == pytest.approx(1.0, abs=5e-4)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state.pop("vision_model.anchor"), "vision state mismatch"),
        (lambda state: state.update({"vision_model.extra": torch.ones(1)}), "vision state mismatch"),
        (lambda state: state.pop("vlm_vision_projection.weight"), "missing"),
        (
            lambda state: state.update(
                {"vlm_vision_projection.weight": torch.ones(FEATURE_DIM, 1727)}
            ),
            "projection shape",
        ),
    ],
)
def test_checkpoint_key_and_projection_contracts_fail_closed(
    tmp_path: Path, monkeypatch, mutation, message: str
) -> None:
    root, checkpoint, state = make_fake_assets(tmp_path, monkeypatch)
    altered = dict(state)
    mutation(altered)
    torch.save({"model": altered}, checkpoint)
    with pytest.raises(RuntimeError, match=message):
        model_module.build_vision_only_encoder(
            upstream_root=root, checkpoint_path=checkpoint, device="cpu"
        )


def test_safe_load_is_attempted_and_legacy_pickle_requires_explicit_hash(
    tmp_path: Path, monkeypatch
) -> None:
    root, checkpoint, state = make_fake_assets(tmp_path, monkeypatch)
    real_load = torch.load
    calls: list[bool] = []

    def simulated_load(path, *args, **kwargs):
        mode = bool(kwargs.get("weights_only"))
        calls.append(mode)
        if mode:
            raise RuntimeError("synthetic unsupported metadata")
        return {"model": state}

    monkeypatch.setattr(torch, "load", simulated_load)
    with pytest.raises(RuntimeError, match="unsafe pickle fallback remains disabled"):
        model_module.build_vision_only_encoder(
            upstream_root=root, checkpoint_path=checkpoint, device="cpu"
        )
    assert calls == [True]

    with pytest.raises(RuntimeError, match="requires an expected SHA-256"):
        model_module.build_vision_only_encoder(
            upstream_root=root,
            checkpoint_path=checkpoint,
            device="cpu",
            allow_trusted_legacy_checkpoint=True,
        )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_module.build_vision_only_encoder(
            upstream_root=root,
            checkpoint_path=checkpoint,
            device="cpu",
            allow_trusted_legacy_checkpoint=True,
            trusted_checkpoint_sha256="0" * 64,
        )

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    encoder, details = model_module.build_vision_only_encoder(
        upstream_root=root,
        checkpoint_path=checkpoint,
        device="cpu",
        allow_trusted_legacy_checkpoint=True,
        trusted_checkpoint_sha256=digest,
    )
    assert encoder.training is False
    assert details["checkpoint_load_mode"] == "trusted_pickle_opt_in_after_sha256"
    assert calls[-2:] == [True, False]
    monkeypatch.setattr(torch, "load", real_load)


def test_malformed_checkpoint_document_fails_closed(tmp_path: Path, monkeypatch) -> None:
    root, checkpoint, _ = make_fake_assets(tmp_path, monkeypatch)
    torch.save({"wrong": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="'model' state dictionary"):
        model_module.build_vision_only_encoder(
            upstream_root=root, checkpoint_path=checkpoint, device="cpu"
        )


def test_precision_resolution_prefers_bf16_then_fp32(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert model_module.resolve_precision("cuda", "auto") == ("bf16", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert model_module.resolve_precision("cuda", "auto") == ("fp32", None)
    assert model_module.resolve_precision("cuda", "fp16") == ("fp16", torch.float16)
    with pytest.raises(RuntimeError, match="BF16"):
        model_module.resolve_precision("cuda", "bf16")


def test_precision_resolution_rejects_invalid_cpu_requests(monkeypatch) -> None:
    assert model_module.resolve_precision("cpu", "auto") == ("fp32", None)
    assert model_module.resolve_precision("cpu", "fp32") == ("fp32", None)
    with pytest.raises(RuntimeError, match="requires a CUDA"):
        model_module.resolve_precision("cpu", "bf16")
    with pytest.raises(ValueError, match="Unsupported"):
        model_module.resolve_precision("cpu", "tf32")


def test_model_loader_requires_the_pinned_torch_runtime() -> None:
    class Runtime:
        __version__ = "2.0.0+cu117"

    with pytest.raises(RuntimeError, match="pinned torch 2.8.0"):
        model_module._validate_torch_runtime(Runtime())


def test_extract_embedding_checks_input_and_eval_contract(tmp_path: Path, monkeypatch) -> None:
    root, checkpoint, _ = make_fake_assets(tmp_path, monkeypatch)
    encoder, _ = model_module.build_vision_only_encoder(
        upstream_root=root, checkpoint_path=checkpoint, device="cpu"
    )
    with pytest.raises(RuntimeError, match="shape"):
        model_module.extract_embedding(
            encoder, torch.zeros(1, 8, 8, 8), device="cpu", precision="fp32"
        )
    encoder.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model_module.extract_embedding(
            encoder,
            torch.zeros((1, *INPUT_SHAPE_DHW), dtype=torch.float32),
            device="cpu",
            precision="fp32",
        )
