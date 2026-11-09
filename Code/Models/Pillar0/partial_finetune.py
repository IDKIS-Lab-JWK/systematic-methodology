"""Atlas final-block/final-stage partial fine-tuning for Pillar-0."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    from . import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from .cache import load_cache_manifest, load_cached_rows, load_pillar_volume
    from .model import (
        EXPECTED_VOLUME_SHAPE,
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        extract_vision_features,
        load_official_model,
    )
    from .windows import window_contract
except ImportError:  # pragma: no cover - direct script imports
    from __init__ import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from cache import load_cache_manifest, load_cached_rows, load_pillar_volume
    from model import (
        EXPECTED_VOLUME_SHAPE,
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        extract_vision_features,
        load_official_model,
    )
    from windows import window_contract

from _linear_probe.common import ContractError, sha256_file


STRATEGY_SPECS = {
    "final_block": {
        "method_label": "AtlasFinalBlockFineTune",
        "method_id": "atlas_final_block_finetune",
        "profile_stem": "atlas_final_block_finetune",
        "trainable_scope": "model.visual.atlas_models[2].blocks[7]",
    },
    "final_stage": {
        "method_label": "AtlasFinalStageFineTune",
        "method_id": "atlas_final_stage_finetune",
        "profile_stem": "atlas_final_stage_finetune",
        "trainable_scope": "model.visual.atlas_models[2]",
    },
}
# Default-scope import aliases; runtime adapters set a strategy-specific method
# identity in ``__init__``.
METHOD_LABEL = STRATEGY_SPECS["final_block"]["method_label"]
METHOD_ID = STRATEGY_SPECS["final_block"]["method_id"]
FINAL_STAGE_INDEX = 2
EXPECTED_STAGE_DEPTHS = (2, 2, 8)


def _atlas_stages(encoder: nn.Module):
    wrapped = getattr(encoder, "model", None)
    visual = getattr(wrapped, "visual", None)
    stages = getattr(visual, "atlas_models", None)
    if not isinstance(stages, nn.ModuleList):
        raise RuntimeError(
            "Pinned Pillar-0 hierarchy changed: expected model.visual.atlas_models ModuleList"
        )
    depths = tuple(len(getattr(stage, "blocks", ())) for stage in stages)
    if depths != EXPECTED_STAGE_DEPTHS:
        raise RuntimeError(
            f"Pinned Atlas stage depths changed: expected {EXPECTED_STAGE_DEPTHS}, got {depths}"
        )
    return stages


class AtlasPartialClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, scope: str, dropout: float = 0.0):
        super().__init__()
        if scope not in ("final_block", "final_stage"):
            raise ValueError(f"Unsupported Atlas adaptation scope: {scope}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.encoder = encoder
        self.scope = scope
        self.encoder.requires_grad_(False)
        stages = _atlas_stages(self.encoder)
        if scope == "final_block":
            selected = [stages[FINAL_STAGE_INDEX].blocks[-1]]
        else:
            selected = [stages[FINAL_STAGE_INDEX]]
        for module in selected:
            module.requires_grad_(True)
        object.__setattr__(self, "_selected_modules", tuple(selected))
        self.classifier = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(float(dropout)),
            nn.Linear(FEATURE_DIM, len(CLASS_NAMES)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen pretraining components never update running state.  Only the
        # explicitly selected Atlas module enters training mode.
        self.encoder.eval()
        for module in self._selected_modules:
            module.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, raw_hu: torch.Tensor) -> torch.Tensor:
        features = extract_vision_features(self.encoder, raw_hu, normalize=False)
        if torch.is_grad_enabled() and not features.requires_grad:
            raise RuntimeError(
                "Selected Atlas scope produced detached features; refusing a mislabeled head-only run"
            )
        return self.classifier(features)


class PillarAtlasAdapter:
    name = "Pillar0"
    method_label = METHOD_LABEL
    method_id = METHOD_ID

    def __init__(self, adaptation_scope: str = "final_block") -> None:
        if adaptation_scope not in STRATEGY_SPECS:
            raise ValueError(f"Unsupported Atlas adaptation scope: {adaptation_scope}")
        strategy = STRATEGY_SPECS[adaptation_scope]
        self.adaptation_scope = adaptation_scope
        self.method_label = str(strategy["method_label"])
        self.method_id = str(strategy["method_id"])
        package = Path(__file__).resolve().parent
        stem = str(strategy["profile_stem"])
        self.publication_profile_path = package / f"publication_{stem}.json"
        self.default_profile_path = package / f"default_{stem}.json"
        self.search_profile_path = package / f"grid_{stem}.json"
        self._cache_manifest: dict | None = None
        self._cache_manifest_path: Path | None = None

    def implementation_paths(self) -> list[Path]:
        return [
            Path(__file__),
            Path(__file__).with_name("model.py"),
            Path(__file__).with_name("windows.py"),
            Path(__file__).with_name("cache.py"),
        ]

    def planned_identity(self, args: Any) -> dict:
        return {
            "checkpoint_source": "Hugging Face immutable revision",
            "repo_id": str(args.model_repo_id),
            "revision": str(args.model_revision),
            "architecture": "Atlas stages [2,2,8], native 1152-D multiscale pooled output",
            "adaptation_scope": self.adaptation_scope,
            "trainable_scope": STRATEGY_SPECS[self.adaptation_scope]["trainable_scope"],
            "cache": str(Path(args.cache_dir).resolve()),
            "assets_resolved": False,
        }

    @staticmethod
    def _requested_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
        return [
            {
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                "label_str": str(row.get("class_name", CLASS_NAMES[int(row["label"])])),
            }
            for row in rows
        ]

    def preflight(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> None:
        path = Path(args.cache_dir)
        self._cache_manifest = load_cache_manifest(
            path, expected_namespace=str(args.cache_namespace)
        )
        self._cache_manifest_path = path / "cache_manifest.json" if path.is_dir() else path
        load_cached_rows(
            self._cache_manifest,
            self._requested_rows(rows),
            expected_namespace=str(args.cache_namespace),
        )

    def resolve_identity(self, args: Any) -> dict:
        if self._cache_manifest is None or self._cache_manifest_path is None:
            raise ContractError("Pillar-0 cache must be preflighted before identity resolution")
        return {
            "repo_id": str(args.model_repo_id),
            "revision": str(args.model_revision),
            "checkpoint_identity": "immutable_huggingface_commit",
            "cache_manifest": str(self._cache_manifest_path.resolve()),
            "cache_manifest_sha256": sha256_file(self._cache_manifest_path),
            "cache_namespace": str(args.cache_namespace),
        }

    def representation_contract(self, _args: Any) -> dict:
        return {
            "input": "raw HU from validated RAVE cache",
            "input_shape_bcdhw": ["B", 1, *EXPECTED_VOLUME_SHAPE],
            "windowing": window_contract(),
            "atlas_stage_depths": list(EXPECTED_STAGE_DEPTHS),
            "final_stage_index_zero_based": FINAL_STAGE_INDEX,
            "final_block_path": "model.visual.atlas_models[2].blocks[7]",
            "selected_adaptation_scope": self.adaptation_scope,
            "feature_dimension": FEATURE_DIM,
            "head": "LayerNorm(1152) -> Linear(1152,5)",
        }

    def prepare_rows(
        self, args: Any, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict]:
        if self._cache_manifest is None:
            self.preflight(args, rows)
        cached = load_cached_rows(
            self._cache_manifest,
            self._requested_rows(rows),
            expected_namespace=str(args.cache_namespace),
        )
        result = []
        for row in cached:
            item = dict(row)
            item["class_name"] = str(item["label_str"])
            item["image_path"] = str(item["cache_path"])
            result.append(item)
        return result

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
            "random_right_angle_rotation",
            "candidate_id",
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise ContractError(f"Unsupported Pillar-0 partial-FT parameters: {sorted(unknown)}")
        if parameters.get("adaptation_scope") != self.adaptation_scope:
            raise ContractError(
                "Pillar-0 adaptation_scope does not match the selected method: "
                f"expected {self.adaptation_scope!r}, got {parameters.get('adaptation_scope')!r}"
            )
        for key in ("encoder_learning_rate", "head_learning_rate"):
            if float(parameters.get(key, 0.0)) <= 0.0:
                raise ContractError(f"{key} must be positive")
        if str(parameters.get("optimizer", "")).lower() != "adamw":
            raise ContractError("Pillar-0 publication adaptation uses AdamW")
        if float(parameters.get("dropout", 0.0)) != 0.0:
            raise ContractError("Dropout must remain zero for exact resumability")
        if parameters.get("random_right_angle_rotation") is not True:
            raise ContractError(
                "Pillar-0 partial fine-tuning requires publication-backed right-angle rotations"
            )

    def build_model(self, args: Any, parameters: Mapping[str, Any], device: str):
        encoder = load_official_model(
            repo_id=str(args.model_repo_id),
            revision=str(args.model_revision),
            device=device,
            dtype=None,
            local_files_only=bool(args.local_files_only),
        )
        return AtlasPartialClassifier(
            encoder,
            str(parameters["adaptation_scope"]),
            dropout=float(parameters.get("dropout", 0.0)),
        )

    def validate_trainable_scope(self, model: nn.Module, parameters: Mapping[str, Any]) -> None:
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        encoder_names = [name for name in names if name.startswith("encoder.")]
        head_names = [name for name in names if name.startswith("classifier.")]
        marker = (
            "atlas_models.2.blocks.7"
            if parameters["adaptation_scope"] == "final_block"
            else "atlas_models.2"
        )
        if not encoder_names or not head_names or any(marker not in name for name in encoder_names):
            raise ContractError(
                f"Atlas trainable scope differs from {parameters['adaptation_scope']}: {encoder_names[:8]}"
            )

    def optimizer_groups(self, model: nn.Module, parameters: Mapping[str, Any]) -> list[dict]:
        encoder = [
            value
            for name, value in model.named_parameters()
            if value.requires_grad and name.startswith("encoder.")
        ]
        head = [
            value
            for name, value in model.named_parameters()
            if value.requires_grad and name.startswith("classifier.")
        ]
        return [
            {
                "group_name": "atlas_selected_scope",
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
    def set_training_mode(model: AtlasPartialClassifier, mode: bool) -> None:
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
        volumes = [load_pillar_volume(rows[index]["cache_path"]) for index in indices]
        batch = torch.stack(volumes, dim=0)
        if training and generator is not None:
            augmented = []
            for volume in batch:
                plane_index = int(torch.randint(0, 3, (1,), generator=generator).item())
                turns = int(torch.randint(0, 4, (1,), generator=generator).item())
                planes = ((1, 2), (1, 3), (2, 3))
                augmented.append(torch.rot90(volume, turns, planes[plane_index]))
            batch = torch.stack(augmented, dim=0).contiguous()
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return batch, labels

    @staticmethod
    def architecture(model: AtlasPartialClassifier, parameters: Mapping[str, Any]) -> dict:
        stages = _atlas_stages(model.encoder)
        return {
            "encoder": "Pillar-0 Atlas abdomen CT",
            "stage_depths": [len(stage.blocks) for stage in stages],
            "adaptation_scope": parameters["adaptation_scope"],
            "feature_dimension": FEATURE_DIM,
            "num_classes": len(CLASS_NAMES),
            "pooling": "released native multiscale max pooling",
        }


__all__ = [
    "AtlasPartialClassifier",
    "METHOD_ID",
    "METHOD_LABEL",
    "PillarAtlasAdapter",
    "STRATEGY_SPECS",
]
