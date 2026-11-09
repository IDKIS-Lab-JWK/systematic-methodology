"""Primus-M final-block/final-stage partial fine-tuning for COLIPRI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    from . import (
        CLASS_NAMES,
        INPUT_SHAPE_DHW,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
    )
    from .model import build_image_encoder, preprocess_nifti, processor_config_identity
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import (
        CLASS_NAMES,
        INPUT_SHAPE_DHW,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
    )
    from model import build_image_encoder, preprocess_nifti, processor_config_identity
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import ContractError


STRATEGY_SPECS = {
    "final_block": {
        "method_label": "PrimusMFinalBlockFineTune",
        "method_id": "primus_m_final_block_finetune",
        "profile_stem": "primus_m_final_block_finetune",
        "trainable_scope": "block 15 plus final norm",
    },
    "final_stage": {
        "method_label": "PrimusMFinalStageFineTune",
        "method_id": "primus_m_final_stage_finetune",
        "profile_stem": "primus_m_final_stage_finetune",
        "trainable_scope": "adapted final quarter, blocks 12-15 plus final norm",
    },
}
# Default-scope import aliases; runtime adapters set a strategy-specific method
# identity in ``__init__``.
METHOD_LABEL = STRATEGY_SPECS["final_block"]["method_label"]
METHOD_ID = STRATEGY_SPECS["final_block"]["method_id"]
PRIMUS_BLOCK_COUNT = 16
FINAL_STAGE_START = 12
POOLED_DIMENSION = 768


def _primus_components(model: nn.Module):
    image_encoder = getattr(model, "image_encoder", None)
    backbone = getattr(image_encoder, "backbone", None)
    eva = getattr(backbone, "eva", None)
    blocks = getattr(eva, "blocks", None)
    norm = getattr(eva, "norm", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != PRIMUS_BLOCK_COUNT:
        observed = len(blocks) if isinstance(blocks, nn.ModuleList) else None
        raise RuntimeError(
            "Pinned COLIPRI Primus-M hierarchy changed: expected "
            f"image_encoder.backbone.eva.blocks with {PRIMUS_BLOCK_COUNT} blocks, got {observed}"
        )
    if not isinstance(norm, nn.Module):
        raise RuntimeError("Pinned COLIPRI Primus-M final norm is missing")
    return blocks, norm


class PrimusMPartialClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, scope: str, dropout: float = 0.0):
        super().__init__()
        if scope not in ("final_block", "final_stage"):
            raise ValueError(f"Unsupported Primus-M adaptation scope: {scope}")
        self.encoder = encoder
        self.scope = scope
        self.encoder.requires_grad_(False)
        blocks, norm = _primus_components(encoder)
        selected = [blocks[-1], norm] if scope == "final_block" else [*blocks[FINAL_STAGE_START:], norm]
        for module in selected:
            module.requires_grad_(True)
        object.__setattr__(self, "_selected_modules", tuple(selected))
        self.classifier = nn.Sequential(
            nn.LayerNorm(POOLED_DIMENSION),
            nn.Dropout(float(dropout)),
            nn.Linear(POOLED_DIMENSION, len(CLASS_NAMES)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        for module in self._selected_modules:
            module.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder.encode_image(
            images, pool=True, project=True, normalize=False
        )
        if embeddings.ndim != 2 or embeddings.shape[-1] != POOLED_DIMENSION:
            raise RuntimeError(
                f"Expected official pooled COLIPRI [B,{POOLED_DIMENSION}], got {tuple(embeddings.shape)}"
            )
        if torch.is_grad_enabled() and not embeddings.requires_grad:
            raise RuntimeError(
                "Selected Primus-M scope produced detached embeddings; refusing a mislabeled head-only run"
            )
        return self.classifier(embeddings)


class COLIPRIPartialAdapter:
    name = "COLIPRI"
    method_label = METHOD_LABEL
    method_id = METHOD_ID

    def __init__(self, adaptation_scope: str = "final_block") -> None:
        if adaptation_scope not in STRATEGY_SPECS:
            raise ValueError(f"Unsupported Primus-M adaptation scope: {adaptation_scope}")
        strategy = STRATEGY_SPECS[adaptation_scope]
        self.adaptation_scope = adaptation_scope
        self.method_label = str(strategy["method_label"])
        self.method_id = str(strategy["method_id"])
        package = Path(__file__).resolve().parent
        stem = str(strategy["profile_stem"])
        self.publication_profile_path = package / f"publication_{stem}.json"
        self.default_profile_path = package / f"default_{stem}.json"
        self.search_profile_path = package / f"grid_{stem}.json"
        self._source_identity: dict | None = None
        self._checkpoint_identity: dict | None = None
        self._processor_identity: dict | None = None

    def implementation_paths(self) -> list[Path]:
        package = Path(__file__).resolve().parent
        return [Path(__file__), package / "model.py", package / "provisioning.py"]

    def planned_identity(self, args: Any) -> dict:
        return {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": MODEL_SHA256,
            "upstream_root": str(Path(args.upstream_root).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "pinned Hub resolution",
            "architecture": "Primus-M flat 16-block EVA backbone",
            "adaptation_scope": self.adaptation_scope,
            "trainable_scope": STRATEGY_SPECS[self.adaptation_scope]["trainable_scope"],
            "assets_resolved": False,
        }

    def preflight(self, args: Any, _rows: Sequence[Mapping[str, Any]]) -> None:
        self._source_identity = ensure_upstream(
            args.upstream_root,
            revision=str(args.upstream_revision),
            local_files_only=bool(args.local_files_only),
            provision_log=args.provision_log,
        )
        self._checkpoint_identity = resolve_checkpoint(
            args.checkpoint,
            upstream_root=args.upstream_root,
            local_files_only=bool(args.local_files_only),
            provision_log=args.provision_log,
        )
        self._processor_identity = processor_config_identity(args.upstream_root)

    def resolve_identity(self, args: Any) -> dict:
        if self._source_identity is None or self._checkpoint_identity is None:
            self.preflight(args, [])
        return {
            "source": dict(self._source_identity),
            "checkpoint": dict(self._checkpoint_identity),
            "processor_config": dict(self._processor_identity),
        }

    def representation_contract(self, _args: Any) -> dict:
        return {
            "input_shape_bcdhw": ["B", 1, *INPUT_SHAPE_DHW],
            "orientation": "RAS",
            "spacing_mm": [2.0, 2.0, 2.0],
            "primus_m_blocks": PRIMUS_BLOCK_COUNT,
            "final_block_path": "image_encoder.backbone.eva.blocks[15] + eva.norm",
            "adapted_final_stage_path": "image_encoder.backbone.eva.blocks[12:16] + eva.norm",
            "selected_adaptation_scope": self.adaptation_scope,
            "pooling": "released 768-D projector plus released multi-head attention pooler (frozen)",
            "head": "LayerNorm(768) -> Linear(768,5)",
        }

    @staticmethod
    def prepare_rows(_args: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
        return [dict(item) for item in rows]

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        allowed = {
            "adaptation_scope",
            "optimizer",
            "encoder_learning_rate",
            "head_learning_rate",
            "weight_decay",
            "head_weight_decay",
            "betas",
            "epsilon",
            "scheduler",
            "polynomial_power",
            "warmup_steps",
            "minimum_lr_ratio",
            "max_steps",
            "effective_batch_size",
            "eval_batch_size",
            "validation_interval_steps",
            "early_stopping_patience_steps",
            "early_stopping_min_delta",
            "class_weight_beta",
            "gradient_clip_norm",
            "dropout",
            "candidate_id",
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise ContractError(f"Unsupported COLIPRI partial-FT parameters: {sorted(unknown)}")
        if parameters.get("adaptation_scope") != self.adaptation_scope:
            raise ContractError(
                "COLIPRI adaptation_scope does not match the selected method: "
                f"expected {self.adaptation_scope!r}, got {parameters.get('adaptation_scope')!r}"
            )
        if str(parameters.get("optimizer", "")).lower() != "adamw":
            raise ContractError("COLIPRI partial fine-tuning requires AdamW")
        for key in ("encoder_learning_rate", "head_learning_rate"):
            if float(parameters.get(key, 0.0)) <= 0.0:
                raise ContractError(f"{key} must be positive")
        if float(parameters.get("dropout", 0.0)) != 0.0:
            raise ContractError("Dropout must remain zero for exact resumability")

    def build_model(self, args: Any, parameters: Mapping[str, Any], device: str):
        if self._checkpoint_identity is None:
            self.preflight(args, [])
        encoder, processor, _details = build_image_encoder(
            upstream_root=args.upstream_root,
            checkpoint_path=self._checkpoint_identity["path"],
            device=device,
        )
        self._processor_instance = processor
        return PrimusMPartialClassifier(
            encoder,
            str(parameters["adaptation_scope"]),
            dropout=float(parameters.get("dropout", 0.0)),
        )

    def validate_trainable_scope(self, model: nn.Module, parameters: Mapping[str, Any]) -> None:
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        encoder_names = [name for name in names if name.startswith("encoder.")]
        head_names = [name for name in names if name.startswith("classifier.")]
        if not encoder_names or not head_names:
            raise ContractError("COLIPRI partial adaptation must train encoder scope and classifier")
        allowed_block_indices = {15} if parameters["adaptation_scope"] == "final_block" else set(range(12, 16))
        for name in encoder_names:
            if ".eva.norm." in name:
                continue
            matched = [index for index in allowed_block_indices if f".eva.blocks.{index}." in name]
            if not matched:
                raise ContractError(f"Unexpected trainable Primus-M parameter: {name}")

    def optimizer_groups(self, model: nn.Module, parameters: Mapping[str, Any]) -> list[dict]:
        encoder = [
            value for name, value in model.named_parameters()
            if value.requires_grad and name.startswith("encoder.")
        ]
        head = [
            value for name, value in model.named_parameters()
            if value.requires_grad and name.startswith("classifier.")
        ]
        return [
            {
                "group_name": "primus_m_selected_scope",
                "params": encoder,
                "lr": float(parameters["encoder_learning_rate"]),
                "weight_decay": float(parameters["weight_decay"]),
            },
            {
                "group_name": "five_class_head",
                "params": head,
                "lr": float(parameters["head_learning_rate"]),
                "weight_decay": float(parameters.get("head_weight_decay", parameters["weight_decay"])),
            },
        ]

    @staticmethod
    def set_training_mode(model: PrimusMPartialClassifier, mode: bool) -> None:
        model.train(mode)

    @staticmethod
    def move_inputs(inputs: torch.Tensor, device: str) -> torch.Tensor:
        return inputs.to(device, non_blocking=False)

    def _processor(self, args: Any):
        if self._checkpoint_identity is None:
            self.preflight(args, [])
        # Building the encoder already validates and loads the same processor;
        # a lightweight processor instance is retained on the adapter per run.
        cached = getattr(self, "_processor_instance", None)
        if cached is None:
            encoder, processor, _details = build_image_encoder(
                upstream_root=args.upstream_root,
                checkpoint_path=self._checkpoint_identity["path"],
                device="cpu",
            )
            del encoder
            self._processor_instance = processor
            cached = processor
        return cached

    def load_batch(
        self,
        args: Any,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        *,
        training: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del training, generator
        processor = self._processor(args)
        volumes = []
        for index in indices:
            tensor, _metadata = preprocess_nifti(processor, rows[index]["image_path"])
            if tuple(tensor.shape) != (1, 1, *INPUT_SHAPE_DHW):
                raise RuntimeError(f"Unexpected COLIPRI preprocessed shape: {tuple(tensor.shape)}")
            volumes.append(tensor[0])
        batch = torch.stack(volumes, dim=0)
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return batch, labels

    @staticmethod
    def architecture(model: PrimusMPartialClassifier, parameters: Mapping[str, Any]) -> dict:
        blocks, _norm = _primus_components(model.encoder)
        return {
            "encoder": "COLIPRI Primus-M",
            "block_count": len(blocks),
            "adaptation_scope": parameters["adaptation_scope"],
            "adapted_final_stage_definition": "final quarter, blocks 12-15 plus final norm",
            "pooling": "frozen released projector and attention pooler",
            "pooled_dimension": POOLED_DIMENSION,
            "num_classes": len(CLASS_NAMES),
        }


__all__ = [
    "COLIPRIPartialAdapter",
    "METHOD_ID",
    "METHOD_LABEL",
    "PrimusMPartialClassifier",
    "STRATEGY_SPECS",
]
