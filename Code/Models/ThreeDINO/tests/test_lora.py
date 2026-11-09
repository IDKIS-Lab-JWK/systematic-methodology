"""LoRA gradient, checkpoint, resumption and workflow tests on synthetic data."""

import copy
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import CLASS_NAMES, cli, orchestrator, training
from Code.Models.ThreeDINO.lora import LoRALinear, ThreeDINOLoRAClassifier, validate_parameters
from test_cli_dry_run import write_manifest


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(torch.tanh(q + k + v))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.mlp = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(x + self.attn(x) + self.mlp(x))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        x = self.patch_embed(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return x


def model_factory(**_):
    # Simulates loading the same immutable pretrained checkpoint on every call.
    with torch.random.fork_rng():
        torch.manual_seed(123)
        encoder = TinyEncoder()
    return ThreeDINOLoRAClassifier(encoder, expected_blocks=2, feature_dim=8)


def rows():
    generator = torch.Generator().manual_seed(18)
    return [{"image": torch.randn(1, 2, 2, 2, generator=generator),
             "label": i % 5, "patient_id": f"synthetic-{i}",
             "label_str": CLASS_NAMES[i % 5]} for i in range(10)]


def dataset_factory(source_rows, **_):
    return [dict(row) for row in source_rows]


def parameters():
    profile = Path(__file__).parents[1] / "default_lora.json"
    return {**json.loads(profile.read_text())["parameters"], "mixed_precision": False,
            "max_iterations": 3, "evaluation_interval_iterations": 1}


def common():
    return dict(device=torch.device("cpu"), model_factory=model_factory, dataset_factory=dataset_factory)


def test_production_default_factory_uses_lora_aware_builder():
    expected = model_factory()
    with mock.patch("Code.Models.ThreeDINO.model.build_model", return_value=(expected, {})) as builder:
        with mock.patch.object(training, "_default_model_factory", side_effect=AssertionError("full factory used")):
            actual = training._build_lora_model(
                parameters={}, classes=list(training.DEFAULT_CLASSES), device=torch.device("cpu"),
                upstream_root="source", upstream_revision="revision", checkpoint_path="weights",
                expected_checkpoint_sha256="hash", model_factory=None,
            )
    assert actual is expected
    assert builder.call_args.kwargs["variant"] == "lora"
    assert builder.call_args.kwargs["parameters"]["checkpoint_path"] == "weights"


def test_zero_initial_update_and_gradients_reach_early_adapters_only():
    base = nn.Linear(8, 24)
    inputs = torch.randn(4, 3, 8)
    expected = base(inputs).detach()
    wrapped = LoRALinear(base)
    assert torch.equal(expected, wrapped(inputs))
    assert wrapped.scale == 0.25
    model = model_factory()
    original = copy.deepcopy(model.encoder)
    x = torch.stack([r["image"] for r in rows()[:4]])
    # All B matrices start at zero, so wrapping retains original linear outputs.
    for module in original.modules():
        if hasattr(module, "lora_B"):
            assert torch.count_nonzero(module.lora_B) == 0
    before = {k: v.clone() for k, v in model.encoder.state_dict().items() if "lora_" not in k}
    from Code.Models.ThreeDINO.lora import lora_optimizer
    optimizer = lora_optimizer(model, parameters())
    for _ in range(2):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3])).backward()
        assert model.encoder.blocks[0].attn.qkv.lora_B.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in model.encoder.named_parameters() if "lora_" not in n)
        optimizer.step()
    assert not torch.equal(model.encoder.blocks[0].attn.qkv.lora_A, original.blocks[0].attn.qkv.lora_A)
    for key, value in before.items():
        assert torch.equal(value, model.encoder.state_dict()[key]), key
    model.eval()
    state = model.adaptation_state_dict()
    assert any(k.endswith("running_mean") for k in state)
    assert not any(".linear.weight" in k for k in state)
    restored = model_factory().eval()
    restored.load_adaptation_state_dict(state)
    assert torch.equal(model(x), restored(x))
    with pytest.raises(RuntimeError, match="exactly"):
        restored.load_adaptation_state_dict({**state, "encoder.patch_embed.weight": before["patch_embed.weight"]})
    with pytest.raises(RuntimeError, match="exactly"):
        restored.load_adaptation_state_dict({k: v for k, v in state.items() if not k.endswith("running_mean")})
    bad = dict(state)
    key = next(iter(bad))
    bad[key] = torch.zeros(99)
    with pytest.raises(RuntimeError, match="shape"):
        restored.load_adaptation_state_dict(bad)


def test_attention_scope_and_double_injection_fail_closed():
    with pytest.raises(ValueError, match="Expected 24"):
        ThreeDINOLoRAClassifier(TinyEncoder())
    model = model_factory()
    with pytest.raises(ValueError, match="already"):
        ThreeDINOLoRAClassifier(model.encoder, expected_blocks=2, feature_dim=8)
    model.encoder.patch_embed.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable scope"):
        model.assert_trainable_scope()


@pytest.mark.parametrize("override", [{"batch_size": 1}, {"drop_last": False},
                                     {"optimizer": "adamw"}, {"lora_rank": 0},
                                     {"lora_alpha": float("nan")}, {"scheduler": "cosine"}])
def test_invalid_recipe_rejected(override):
    with pytest.raises(ValueError):
        validate_parameters({**parameters(), **override})


def test_training_resume_predictions_and_refit():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference, interrupted = root / "reference", root / "interrupted"
        training.train_lora(rows(), rows(), parameters(), reference, **common())
        original_save = training.atomic_torch_save

        def interrupt(payload, path):
            original_save(payload, path)
            if Path(path).name == "last_checkpoint.pt" and payload["optimizer_step"] == 1:
                raise RuntimeError("synthetic interruption")

        with mock.patch.object(training, "atomic_torch_save", side_effect=interrupt):
            with pytest.raises(RuntimeError, match="synthetic interruption"):
                training.train_lora(rows(), rows(), parameters(), interrupted, **common())
        with pytest.raises(RuntimeError, match="runtime budget"):
            training.train_lora(rows(), rows(), parameters(), interrupted, resume=True, budget=4, **common())
        result = training.train_lora(rows(), rows(), parameters(), interrupted, resume=True, **common())
        first = torch.load(reference / "last_checkpoint.pt", weights_only=False)
        second = torch.load(interrupted / "last_checkpoint.pt", weights_only=False)
        assert first["checkpoint_scope"] == "lora_and_classifier"
        assert first["method"] == "lora"
        for key, value in first["model_state_dict"].items():
            assert torch.equal(value, second["model_state_dict"][key]), key
        assert first["history"] == second["history"]
        assert all(row["encoder_lr"] == 1e-4 for row in first["history"])
        assert result["state"]["checkpoint_selection"] == "final_budget"
        assert result["state"]["best_optimizer_step"] == 3
        logits, labels = training.predict_checkpoint(rows(), Path(result["checkpoint"]), method="lora", **common())
        assert logits.shape == (10, 5)
        assert labels.tolist() == [i % 5 for i in range(10)]
        refit = training.refit_model(rows(), parameters(), root / "refit", method="lora", refit_budget=2, **common())
        assert refit["state"]["refit_full_development"] is True
        assert refit["state"]["optimizer_steps_completed"] == 2
        with pytest.raises(FileExistsError):
            training.train_lora(rows(), rows(), parameters(), reference, **common())


def test_lora_dry_run_and_dispatch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_manifest(root / "Dataset")
        args = cli.build_parser("lora").parse_args([
            "--percentage", "100", "--dataset_root", str(root / "Dataset"),
            "--results_root", str(root / "Results"), "--dry_run",
        ])
        plan = orchestrator.run_pipeline(args, "lora")
        assert plan["method"] == "lora"
        assert plan["parameters_or_search"]["lora_rank"] == 4
        assert not (root / "Results").exists()
        assert orchestrator.method_root(root / "Results", "lora").name == "LoRA"
        with mock.patch.object(orchestrator, "train_lora", return_value={}) as trainer:
            with mock.patch.object(orchestrator, "_common_training_kwargs", return_value={}):
                orchestrator._train_single("lora", [], [], parameters(), root / "out", args)
        trainer.assert_called_once()
        args.parameter_source = "publication"
        with pytest.raises(ValueError, match="fixed default/manual"):
            orchestrator.run_pipeline(args, "lora")
