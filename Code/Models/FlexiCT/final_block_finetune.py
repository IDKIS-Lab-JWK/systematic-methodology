"""Final-block partial fine-tuning for the FlexiCT 3D vision encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    from . import (
        CLASS_NAMES,
        EXPECTED_PATCH_TOKENS,
        INPUT_SHAPE_DHW,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from .data import preprocess_nifti, preprocessing_contract
    from .model import build_vision_only_encoder, vision_encoder_contract
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import (
        CLASS_NAMES,
        EXPECTED_PATCH_TOKENS,
        INPUT_SHAPE_DHW,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from data import preprocess_nifti, preprocessing_contract
    from model import build_vision_only_encoder, vision_encoder_contract
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import ContractError


METHOD_LABEL = "FinalBlockFineTune"
METHOD_ID = "final_block_finetune"
VISION_DIMENSION = 864
JOINT_DIMENSION = 1728
EXPECTED_BLOCK_COUNT = 16


def _vision_components(encoder: nn.Module):
    vision = getattr(encoder, "vision", None)
    blocks = getattr(vision, "blocks", None)
    norm = getattr(vision, "norm", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != EXPECTED_BLOCK_COUNT:
        observed = len(blocks) if isinstance(blocks, nn.ModuleList) else None
        raise RuntimeError(
            f"Pinned FlexiCT hierarchy changed: expected vision.blocks length 16, got {observed}"
        )
    if not isinstance(norm, nn.Module):
        raise RuntimeError("Pinned FlexiCT final vision norm is missing")
    return vision, blocks, norm


class FlexiCTFinalBlockClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        _vision, blocks, norm = _vision_components(encoder)
        selected = (blocks[-1], norm)
        for module in selected:
            module.requires_grad_(True)
        object.__setattr__(self, "_selected_modules", selected)
        self.classifier = nn.Sequential(
            nn.LayerNorm(JOINT_DIMENSION),
            nn.Dropout(float(dropout)),
            nn.Linear(JOINT_DIMENSION, len(CLASS_NAMES)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        for module in self._selected_modules:
            module.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder.vision(images, is_training=True)
        cls = outputs["x_norm_clstoken"]
        patches = outputs["x_norm_patchtokens"]
        if (
            cls.ndim != 2
            or cls.shape[-1] != VISION_DIMENSION
            or patches.ndim != 3
            or patches.shape[1:] != (EXPECTED_PATCH_TOKENS, VISION_DIMENSION)
        ):
            raise RuntimeError(
                f"Unexpected FlexiCT token shapes: cls={tuple(cls.shape)}, patches={tuple(patches.shape)}"
            )
        joint = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        if torch.is_grad_enabled() and not joint.requires_grad:
            raise RuntimeError(
                "Selected FlexiCT final block produced detached tokens; refusing a mislabeled head-only run"
            )
        return self.classifier(joint)


class FlexiCTFinalBlockAdapter:
    name = "FlexiCT"
    method_label = METHOD_LABEL
    method_id = METHOD_ID

    def __init__(self) -> None:
        package = Path(__file__).resolve().parent
        self.publication_profile_path = package / "publication_final_block_finetune.json"
        self.default_profile_path = package / "default_final_block_finetune.json"
        self.search_profile_path = package / "grid_final_block_finetune.json"
        self._source_identity: dict | None = None
        self._checkpoint_identity: dict | None = None

    def implementation_paths(self) -> list[Path]:
        package = Path(__file__).resolve().parent
        return [Path(__file__), package / "model.py", package / "data.py", package / "provisioning.py"]

    def planned_identity(self, args: Any) -> dict:
        return {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "required at execution",
            "checkpoint_sha256": args.checkpoint_sha256 or "computed at execution",
            "architecture": "FlexiCT 3D depth-16 vision transformer",
            "adaptation_scope": "vision.blocks[15] + vision.norm + new five-class head",
            "pretrained_vlm_projection_used": False,
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
            args.checkpoint, expected_sha256=args.checkpoint_sha256
        )

    def resolve_identity(self, args: Any) -> dict:
        if self._source_identity is None or self._checkpoint_identity is None:
            self.preflight(args, [])
        return {
            "source": dict(self._source_identity),
            "checkpoint": dict(self._checkpoint_identity),
            "legacy_pickle_opt_in": bool(args.allow_trusted_legacy_checkpoint),
        }

    def representation_contract(self, _args: Any) -> dict:
        return {
            "preprocessing": preprocessing_contract(),
            "vision": vision_encoder_contract(),
            "final_block_path": "vision.blocks[15] + vision.norm",
            "classifier_input": "concatenate normalized CLS token and mean normalized patch token",
            "classifier_input_dimension": JOINT_DIMENSION,
            "released_vlm_projection": "loaded for strict checkpoint verification but frozen and bypassed",
            "head": "LayerNorm(1728) -> Linear(1728,5)",
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
            raise ContractError(f"Unsupported FlexiCT final-block parameters: {sorted(unknown)}")
        if parameters.get("adaptation_scope") != "final_block":
            raise ContractError("FlexiCT final-block workflow only accepts adaptation_scope=final_block")
        if str(parameters.get("optimizer", "")).lower() != "adamw":
            raise ContractError("FlexiCT final-block adaptation uses AdamW")
        for key in ("encoder_learning_rate", "head_learning_rate"):
            if float(parameters.get(key, 0.0)) <= 0.0:
                raise ContractError(f"{key} must be positive")
        if float(parameters.get("dropout", 0.0)) != 0.0:
            raise ContractError("Dropout must remain zero for exact resumability")

    def build_model(self, args: Any, parameters: Mapping[str, Any], device: str):
        if self._checkpoint_identity is None:
            self.preflight(args, [])
        encoder, _details = build_vision_only_encoder(
            upstream_root=args.upstream_root,
            checkpoint_path=self._checkpoint_identity["path"],
            device=device,
            allow_trusted_legacy_checkpoint=bool(args.allow_trusted_legacy_checkpoint),
            trusted_checkpoint_sha256=args.checkpoint_sha256,
        )
        return FlexiCTFinalBlockClassifier(
            encoder, dropout=float(parameters.get("dropout", 0.0))
        )

    def validate_trainable_scope(self, model: nn.Module, _parameters: Mapping[str, Any]) -> None:
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        encoder_names = [name for name in names if name.startswith("encoder.")]
        head_names = [name for name in names if name.startswith("classifier.")]
        if not encoder_names or not head_names:
            raise ContractError("FlexiCT adaptation must train final encoder block/norm and classifier")
        for name in encoder_names:
            if ".vision.norm." in name or ".vision.blocks.15." in name:
                continue
            raise ContractError(f"Unexpected trainable FlexiCT parameter: {name}")

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
                "group_name": "flexict_final_block_and_norm",
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
    def set_training_mode(model: FlexiCTFinalBlockClassifier, mode: bool) -> None:
        model.train(mode)

    @staticmethod
    def move_inputs(inputs: torch.Tensor, device: str) -> torch.Tensor:
        return inputs.to(device, non_blocking=False)

    def load_batch(
        self,
        _args: Any,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        *,
        training: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del training, generator
        volumes = []
        for index in indices:
            tensor, _metadata = preprocess_nifti(rows[index]["image_path"])
            if tuple(tensor.shape) != (1, *INPUT_SHAPE_DHW):
                raise RuntimeError(f"Unexpected FlexiCT preprocessed shape: {tuple(tensor.shape)}")
            volumes.append(tensor)
        batch = torch.stack(volumes, dim=0)
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return batch, labels

    @staticmethod
    def architecture(model: FlexiCTFinalBlockClassifier, parameters: Mapping[str, Any]) -> dict:
        _vision, blocks, _norm = _vision_components(model.encoder)
        return {
            "encoder": "FlexiCT-3D-VLM vision backbone",
            "block_count": len(blocks),
            "adaptation_scope": parameters["adaptation_scope"],
            "classifier_input": "CLS + mean patch token",
            "classifier_input_dimension": JOINT_DIMENSION,
            "pretrained_projection_used_for_logits": False,
            "num_classes": len(CLASS_NAMES),
        }


__all__ = [
    "FlexiCTFinalBlockAdapter",
    "FlexiCTFinalBlockClassifier",
    "METHOD_ID",
    "METHOD_LABEL",
]
