"""Attention LoRA for 3DINO; no dependency on PEFT or changes to base weights.

Recipe: MFM-African-Brain-78C8/src/lora/lora.py (inspected 2026-10-16).
Rank 4, alpha 1, no adapter dropout; QKV and output projections in every block.
The study's released MLP head is adapted from one output to five CT classes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def validate_parameters(parameters: Mapping) -> None:
    """Reject unsupported settings instead of silently running another recipe."""
    for key, default, expected in (
        ("optimizer", "adam", "adam"),
        ("scheduler", "none", "none"),
        ("class_weighting", "none", "none"),
        ("loss", "cross_entropy", "cross_entropy"),
        ("head_type", "study_mlp", "study_mlp"),
    ):
        if parameters.get(key, default) != expected:
            raise ValueError(f"3DINO LoRA requires {key}={expected!r}.")
    for key, default in (("lora_rank", 4), ("batch_size", 4), ("grad_accum_steps", 1), ("epochs", 50)):
        value = parameters.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if parameters.get("batch_size", 4) < 2:
        raise ValueError("The study MLP uses BatchNorm; training batch_size must be >= 2.")
    for key, default in (("lr", 1e-4), ("lora_alpha", 1.0)):
        value = float(parameters.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    dropout = float(parameters.get("head_dropout", 0.3))
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("head_dropout must be in [0, 1).")
    for key in ("lora_dropout", "weight_decay", "warmup_fraction", "max_grad_norm", "early_stopping_patience"):
        if float(parameters.get(key, 0)) != 0:
            raise ValueError(f"The supported study-based LoRA recipe requires {key}=0.")
    if parameters.get("drop_last", True) is not True:
        raise ValueError("LoRA training requires drop_last=true for the BatchNorm head.")
    if parameters.get("gradient_checkpointing", False):
        raise ValueError("Gradient checkpointing is not validated for this LoRA path.")


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        self.linear = linear
        self.linear.requires_grad_(False)
        self.lora_A = nn.Parameter(linear.weight.new_empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank))
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class ThreeDINOLoRAClassifier(nn.Module):
    def __init__(self, encoder, *, rank=4, alpha=1.0, head_dropout=0.3,
                 num_classes=5, expected_blocks=24, feature_dim=1024):
        super().__init__()
        self.encoder = encoder
        # Discover the original modules before replacing anything, including
        # chunked-block layouts; never recursively wrap an existing adapter.
        if any(isinstance(m, LoRALinear) for m in encoder.modules()):
            raise ValueError("Encoder already has LoRA adapters.")
        targets = {
            name: module for name, module in encoder.named_modules()
            if name.endswith(("attn.qkv", "attn.proj")) and isinstance(module, nn.Linear)
        }
        qkv = {name[:-4] for name in targets if name.endswith(".qkv")}
        proj = {name[:-5] for name in targets if name.endswith(".proj")}
        if qkv != proj or len(qkv) != expected_blocks:
            raise ValueError(f"Expected {expected_blocks} attention QKV/proj pairs; found {len(qkv)} / {len(proj)}.")
        for name, module in targets.items():
            expected_out = feature_dim * (3 if name.endswith(".qkv") else 1)
            if module.in_features != feature_dim or module.out_features != expected_out:
                raise ValueError(f"Unexpected attention dimensions at {name}.")
        encoder.requires_grad_(False)
        for name, module in targets.items():
            parent_name, child_name = name.rsplit(".", 1)
            parent = encoder.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, rank, alpha))
        self.target_modules = tuple(sorted(targets))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )
        self.assert_trainable_scope()

    def forward(self, images):
        # Base weights are frozen, but autograd must traverse the whole encoder
        # to reach adapters in early blocks. Do not use no_grad or detach here.
        return self.classifier(self.encoder(images))

    def assert_trainable_scope(self):
        expected = {
            f"encoder.{name}.{suffix}"
            for name in self.target_modules for suffix in ("lora_A", "lora_B")
        } | {f"classifier.{name}" for name, _ in self.classifier.named_parameters()}
        actual = {name for name, p in self.named_parameters() if p.requires_grad}
        if actual != expected:
            raise RuntimeError(f"LoRA trainable scope mismatch: {sorted(actual ^ expected)}")

    def adaptation_state_dict(self):
        self.assert_trainable_scope()
        # Include classifier BatchNorm running statistics/counters as well as
        # parameters. Frozen pretrained tensors are reconstructed from provenance.
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name.startswith("classifier.") or name.endswith((".lora_A", ".lora_B"))}

    def load_adaptation_state_dict(self, state):
        expected = self.adaptation_state_dict()
        if set(state) != set(expected):
            raise RuntimeError("LoRA checkpoint must contain exactly the adapters and complete classifier state.")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
                raise RuntimeError(f"LoRA checkpoint shape mismatch at {name}.")
        self.load_state_dict(state, strict=False)
        self.assert_trainable_scope()


def lora_optimizer(model, parameters):
    validate_parameters(parameters)
    model.assert_trainable_scope()
    return torch.optim.Adam(
        [{"params": [p for p in model.encoder.parameters() if p.requires_grad], "name": "lora"},
         {"params": list(model.classifier.parameters()), "name": "classifier"}],
        lr=float(parameters.get("lr", 1e-4)), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
