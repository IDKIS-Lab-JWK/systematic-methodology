"""Merlin classifier construction for supervised adaptation variants."""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn

try:
    from .conv_lora import inject_conv_lora
except ImportError:  # Direct study-script imports.
    from conv_lora import inject_conv_lora


_LINEAR_PROBE_ENCODER: nn.Module | None = None


def import_merlin():
    try:
        module = importlib.import_module("merlin")
        data_module = importlib.import_module("merlin.data")
        Merlin = getattr(module, "Merlin")
        MerlinDataLoader = getattr(data_module, "DataLoader")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Could not import merlin-vlm. Run from the Systematic_Methodology "
            "directory and install: python -m pip install merlin-vlm==0.0.7"
        ) from exc
    return Merlin, MerlinDataLoader


class MerlinClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        embedding_dim: int,
        dropout_prob: float,
        frozen_encoder: bool = False,
        strict_linear_head: bool = False,
        checkpoint_input_gradients: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.frozen_encoder = bool(frozen_encoder)
        self.strict_linear_head = bool(strict_linear_head)
        self.checkpoint_input_gradients = bool(checkpoint_input_gradients)
        if self.strict_linear_head:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            # The Merlin publication specifies Kaiming initialization but not
            # its exact variant. Make PyTorch's Kaiming-uniform convention
            # explicit so it is stable and recorded in model provenance.
            nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
            if self.classifier.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.classifier.weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.classifier.bias, -bound, bound)
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(dropout_prob),
                nn.Linear(embedding_dim, num_classes),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder:
            # A frozen linear probe must not update BatchNorm buffers or enable
            # any stochastic encoder behavior when the head enters train mode.
            self.encoder.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # Merlin 0.0.7 checkpoints entire image stages with reentrant autograd.
        # With a frozen stem, ordinary image inputs otherwise leave all stage
        # adapters disconnected from the loss. A differentiable input preserves
        # that graph without unfreezing any backbone parameter. Avoid mutating
        # the caller's input or requesting input gradients during evaluation.
        if (
            self.checkpoint_input_gradients
            and self.training
            and torch.is_grad_enabled()
            and not images.requires_grad
        ):
            images = images.detach().requires_grad_(True)
        if self.frozen_encoder:
            with torch.no_grad():
                output = self.encoder(images)
        else:
            output = self.encoder(images)
        embedding = output[0] if isinstance(output, (tuple, list)) else output
        if (
            embedding.ndim == 3
            and embedding.shape[0] == 1
            and embedding.shape[1] == images.shape[0]
        ):
            embedding = embedding.squeeze(0)
        elif (
            embedding.ndim == 3
            and embedding.shape[0] == images.shape[0]
            and embedding.shape[1] == 1
        ):
            embedding = embedding.squeeze(1)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2:
            raise RuntimeError(
                f"Expected Merlin embedding (batch, dim), got {tuple(embedding.shape)}"
            )
        return embedding.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the supervised head to already materialized image features."""
        if not self.frozen_encoder:
            raise RuntimeError(
                "Precomputed feature input is only valid for a frozen encoder."
            )
        if features.ndim != 2:
            raise RuntimeError(
                f"Expected feature matrix (batch, dim), got {tuple(features.shape)}"
            )
        return self.classifier(features.float())


def _unfreeze_image_projection(encoder: nn.Module) -> list[str]:
    tokens = ("image_projection", "image_proj", "visual_projection")
    unfrozen = []
    for name, parameter in encoder.named_parameters():
        if any(token in name.lower() for token in tokens):
            parameter.requires_grad = True
            unfrozen.append(name)
    return unfrozen


def _enable_gradient_checkpointing(encoder: nn.Module) -> list[str]:
    enabled = []
    method_names = (
        "gradient_checkpointing_enable",
        "enable_gradient_checkpointing",
        "set_grad_checkpointing",
    )
    for module_name, module in encoder.named_modules():
        for method_name in method_names:
            method = getattr(module, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except TypeError:
                method(True)
            enabled.append(f"{module_name or '<root>'}.{method_name}")
            break
    return enabled


def build_model(
    variant: str,
    num_classes: int,
    parameters: dict,
    device: torch.device,
) -> tuple[MerlinClassifier, dict]:
    Merlin, _ = import_merlin()
    global _LINEAR_PROBE_ENCODER
    if variant == "linear_probe":
        if _LINEAR_PROBE_ENCODER is None:
            # Encoder construction initializes modules before loading the
            # released checkpoint and therefore consumes the CPU RNG. Preserve
            # it so the first fold receives the same seeded head initialization
            # as later folds that reuse this immutable encoder instance.
            with torch.random.fork_rng(devices=[]):
                _LINEAR_PROBE_ENCODER = Merlin(ImageEmbedding=True)
        encoder = _LINEAR_PROBE_ENCODER
    else:
        encoder = Merlin(ImageEmbedding=True)
    details: dict = {"variant": variant}

    if variant == "linear_probe":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        details["conv_lora_modules"] = []
        details["unfrozen_projection_parameters"] = []
        details["encoder_frozen"] = True
        details["frozen_encoder_eval"] = True
        details["classifier_architecture"] = (
            f"Linear({int(parameters.get('embedding_dim', 2048))}, {num_classes})"
        )
        details["classifier_initialization"] = "kaiming_uniform_a_sqrt5"
        details["adaptation_scope"] = "strict linear head only"
        details["encoder_instance_reused_within_process"] = True
    elif variant == "full_finetune":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        details["conv_lora_modules"] = []
        details["unfrozen_projection_parameters"] = []
    elif variant in {"conv_lora", "conv_lora_llrd"}:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        details["conv_lora_modules"] = inject_conv_lora(
            encoder,
            rank=int(parameters["conv_lora_rank"]),
            alpha=float(parameters["conv_lora_alpha"]),
        )
        train_image_projection = parameters.get("train_image_projection", True)
        if variant == "conv_lora_llrd" and train_image_projection:
            raise ValueError(
                "conv_lora_llrd is an image-only supervised adaptation and "
                "requires train_image_projection=false; Merlin(ImageEmbedding=True) "
                "bypasses the publication's contrastive projection."
            )
        details["unfrozen_projection_parameters"] = (
            _unfreeze_image_projection(encoder) if train_image_projection else []
        )
        if variant == "conv_lora_llrd":
            details["adaptation_scope"] = (
                "image ConvLoRA layer1-layer4 plus supervised classifier"
            )
    else:
        raise ValueError(f"Unknown Merlin variant: {variant}")

    model = MerlinClassifier(
        encoder=encoder,
        num_classes=num_classes,
        embedding_dim=int(parameters.get("embedding_dim", 2048)),
        dropout_prob=float(parameters.get("dropout_prob", 0.0)),
        frozen_encoder=variant == "linear_probe",
        strict_linear_head=variant == "linear_probe",
        checkpoint_input_gradients=variant in {"conv_lora", "conv_lora_llrd"},
    )
    details["checkpoint_input_gradient_policy"] = (
        "training_only_differentiable_images_for_reentrant_stage_checkpoints_v1"
        if model.checkpoint_input_gradients
        else "not_required"
    )
    if variant == "linear_probe":
        # Frozen features are materialized only when absent. Keep the large
        # shared encoder on CPU between grid/CV head fits and move only the new
        # linear head to the training device here.
        model.encoder.to(torch.device("cpu"))
        model.classifier.to(device)
    else:
        model.to(device)
    details["gradient_checkpointing_requested"] = bool(
        parameters.get("gradient_checkpointing", variant != "linear_probe")
    )
    details["gradient_checkpointing_enabled_on"] = (
        _enable_gradient_checkpointing(model.encoder)
        if details["gradient_checkpointing_requested"]
        else []
    )
    details["total_parameters"] = sum(p.numel() for p in model.parameters())
    details["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return model, details

