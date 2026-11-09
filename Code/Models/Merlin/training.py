"""Reusable Merlin training and evaluation engine."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import inspect
import logging
import math
import os
import pickle
import random
import sys
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from metrics import compute_metrics, fold_selection_score, format_predictions
from model import build_model, import_merlin


# Merlin-vlm currently constructs MONAI Orientationd without an explicit
# ``labels`` argument. MONAI emits this compatibility warning during every
# loader construction, but it does not affect the transform or patient data.
warnings.filterwarnings(
    "ignore",
    message=r".*Orientationd.*labels.*Current default value of argument.*",
    category=FutureWarning,
)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@lru_cache(maxsize=8)
def _sha256_file_cached(path_string: str, size: int, modified_ns: int) -> str:
    """Hash an immutable model artifact once per process."""
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_cached(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def _merlin_checkpoint_provenance(model) -> dict:
    encoder = model.encoder
    config = getattr(encoder, "_config", None)
    checkpoint_name = config.get("checkpoint") if isinstance(config, dict) else None
    if not checkpoint_name:
        raise RuntimeError(
            "Pinned Merlin encoder does not expose its checkpoint filename."
        )
    encoder_source = Path(inspect.getfile(encoder.__class__)).resolve()
    checkpoint_path = encoder_source.parent / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Merlin checkpoint not found: {checkpoint_path}")
    try:
        package_version = importlib.metadata.version("merlin-vlm")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "package": "merlin-vlm",
        "package_version": package_version,
        "checkpoint_name": str(checkpoint_name),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }


def _comparable_encoder_provenance(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in (
            "package",
            "package_version",
            "checkpoint_name",
            "checkpoint_sha256",
        )
    }


class MerlinFeatureStore:
    """Per-CT frozen 2048-D feature cache with strict model provenance."""

    def __init__(self, cache_dir: Path, model, extraction_precision: str):
        cache_dir = Path(cache_dir).resolve()
        preprocessing_manifest = cache_dir / "cache_manifest.json"
        if not preprocessing_manifest.is_file():
            raise FileNotFoundError(
                f"Shared preprocessing manifest not found: {preprocessing_manifest}"
            )
        expected = {
            "schema_version": 1,
            "kind": "merlin_frozen_image_embedding_features",
            "feature_dim": 2048,
            "dtype": "float32",
            "extraction_precision": extraction_precision,
            "preprocessing_cache": str(cache_dir),
            "preprocessing_manifest": str(preprocessing_manifest),
            "preprocessing_manifest_sha256": _sha256_file(
                preprocessing_manifest
            ),
            "model": _merlin_checkpoint_provenance(model),
        }
        signature = hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self.root = (
            cache_dir.parent.parent
            / "Features"
            / f"merlin_image_embedding_2048_{signature}"
        )
        self.expected = expected
        manifest_path = self.root / "feature_cache_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            comparable = {key: existing.get(key) for key in expected}
            if comparable != expected:
                raise RuntimeError(
                    "Frozen-feature cache provenance mismatch at "
                    f"{manifest_path}; use a new preprocessing namespace."
                )
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            save_json(
                manifest_path,
                {
                    **expected,
                    "created_at": datetime.now().astimezone().isoformat(),
                },
            )
        self.manifest_path = manifest_path

    def path_for(self, row: dict) -> Path:
        identity = json.dumps(
            {
                "patient_id": str(row["patient_id"]),
                "image": str(Path(row["image"]).resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self.root / f"{hashlib.sha256(identity).hexdigest()}.pt"

    def load(self, row: dict) -> torch.Tensor | None:
        path = self.path_for(row)
        if not path.is_file():
            return None
        migrated_monai_metadata = False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            # Versions written before the plain-tensor normalization below
            # may contain a MONAI MetaTensor. Keep weights-only deserialization
            # enabled and narrowly allowlist only the four MONAI metadata types
            # present in that legacy payload so it can be migrated safely.
            safe_globals = getattr(torch.serialization, "safe_globals", None)
            if safe_globals is None:
                return None
            from monai.data.meta_tensor import MetaTensor
            from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

            try:
                with safe_globals(
                    [MetaTensor, MetaKeys, SpaceKeys, TraceKeys]
                ):
                    payload = torch.load(
                        path, map_location="cpu", weights_only=True
                    )
            except (pickle.UnpicklingError, EOFError, RuntimeError):
                return None
            migrated_monai_metadata = True
        except (EOFError, RuntimeError):
            # A feature cache is derivative data. Treat a truncated or
            # otherwise unreadable entry as a cache miss and regenerate it.
            return None
        if not isinstance(payload, dict):
            return None
        stored_feature = payload.get("feature")
        if not isinstance(stored_feature, torch.Tensor):
            raise RuntimeError(f"Invalid frozen-feature payload: {path}")
        feature = self._plain_feature_tensor(stored_feature)
        if not isinstance(feature, torch.Tensor):
            raise RuntimeError(f"Invalid frozen-feature payload: {path}")
        if (
            feature.shape != (2048,)
            or feature.dtype != torch.float32
            or not bool(torch.isfinite(feature).all())
        ):
            raise RuntimeError(f"Invalid cached Merlin feature: {path}")
        if migrated_monai_metadata or type(stored_feature) is not torch.Tensor:
            # Rewrite the legacy payload immediately; every subsequent load is
            # then compatible with PyTorch 2.6+ weights-only deserialization.
            self.save(row, feature)
        return feature

    @staticmethod
    def _plain_feature_tensor(feature: torch.Tensor) -> torch.Tensor:
        """Remove MONAI Tensor subclasses and metadata before serialization."""
        if type(feature) is not torch.Tensor:
            as_tensor = getattr(feature, "as_tensor", None)
            if callable(as_tensor):
                feature = as_tensor()
            else:
                feature = feature.as_subclass(torch.Tensor)
        return (
            feature.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            .contiguous()
            .clone()
        )

    def save(self, row: dict, feature: torch.Tensor) -> Path:
        feature = self._plain_feature_tensor(feature)
        if feature.shape != (2048,):
            raise ValueError(
                f"Expected 2048-D Merlin feature, got {tuple(feature.shape)}"
            )
        if not bool(torch.isfinite(feature).all()):
            raise ValueError("Frozen Merlin features must be finite.")
        path = self.path_for(row)
        atomic_torch_save(
            {
                "feature": feature,
                "patient_id": str(row["patient_id"]),
                "image": str(Path(row["image"]).resolve()),
            },
            path,
        )
        return path


class FrozenFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], features: torch.Tensor):
        if len(rows) != len(features):
            raise ValueError("Feature count does not match row count.")
        self.rows = rows
        self.features = features.float().cpu()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "image": self.features[index],
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "label_str": str(row["label_str"]),
        }


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"merlin_{abs(hash(str(output_dir.resolve())))}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    for handler in (
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grad_scaler(enabled: bool):
    """Construct GradScaler across the declared PyTorch >=2.1.2 range."""
    scaler_class = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_class is not None:
        return scaler_class("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(
    rows: list[dict],
    cache_dir: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    _, MerlinDataLoader = import_merlin()
    # MONAI or a dependency may prepend its own warning filters while being
    # imported. Reapply this one warning-specific rule at the point where
    # merlin-vlm creates Orientationd.
    warnings.filterwarnings(
        "ignore",
        message=r".*Orientationd.*labels.*Current default value of argument.*",
        category=FutureWarning,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_temp_dir = Path(
        os.environ.get(
            "MERLIN_CACHE_TEMP_DIR",
            str(cache_dir.parent / "_tmp"),
        )
    ).resolve()
    cache_temp_dir.mkdir(parents=True, exist_ok=True)
    # Merlin's CTPersistentDataset uses tempfile.TemporaryDirectory while
    # serializing a transformed volume. Keep those temporary files beside the
    # shared cache so Windows workers do not write through the user profile's
    # %TEMP% directory.
    tempfile.tempdir = str(cache_temp_dir)
    for variable in ("TMP", "TEMP", "TMPDIR"):
        os.environ[variable] = str(cache_temp_dir)
    return MerlinDataLoader(
        datalist=rows,
        cache_dir=str(cache_dir),
        batchsize=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _batch_strings(values) -> list[str]:
    if isinstance(values, str):
        return [values]
    if isinstance(values, torch.Tensor):
        return [str(value) for value in values.detach().cpu().tolist()]
    return [str(value) for value in values]


def materialize_linear_probe_features(
    rows: list[dict],
    model,
    store: MerlinFeatureStore,
    cache_dir: Path,
    extraction_batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Extract each immutable frozen feature once and return row-ordered data."""
    missing = [row for row in rows if store.load(row) is None]
    if missing:
        if logger:
            logger.info(
                "Extracting %d missing frozen Merlin features into %s",
                len(missing),
                store.root,
            )
        row_by_patient = {str(row["patient_id"]): row for row in missing}
        if len(row_by_patient) != len(missing):
            raise RuntimeError(
                "Patient IDs must be unique while caching Merlin features."
            )
        loader = make_loader(
            missing,
            cache_dir,
            int(extraction_batch_size),
            False,
            int(num_workers),
        )
        # Linear-probe model construction keeps the shared, frozen encoder on
        # CPU between runs. Move it only while missing features are extracted;
        # callers return it to CPU before fitting/evaluating the small head.
        model.encoder.to(device)
        model.eval()
        completed = 0
        for batch in loader:
            patient_ids = _batch_strings(batch["patient_id"])
            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type,
                enabled=bool(amp) and device.type == "cuda",
            ):
                features = model.encode(images)
            features = features.detach().float().cpu()
            if features.shape != (len(patient_ids), 2048):
                raise RuntimeError(
                    "Expected frozen Merlin feature matrix "
                    f"({len(patient_ids)}, 2048), got {tuple(features.shape)}"
                )
            for patient_id, feature in zip(patient_ids, features):
                try:
                    row = row_by_patient[patient_id]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Unexpected patient ID from Merlin loader: {patient_id}"
                    ) from exc
                store.save(row, feature)
                completed += 1
            if logger and (
                completed == len(patient_ids)
                or completed % 25 < len(patient_ids)
                or completed == len(missing)
            ):
                logger.info(
                    "Frozen feature cache %d/%d", completed, len(missing)
                )
    features = [store.load(row) for row in rows]
    if any(feature is None for feature in features):
        raise RuntimeError("Frozen Merlin feature cache is incomplete.")
    return torch.stack(features)


def make_feature_loader(
    rows: list[dict],
    features: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        FrozenFeatureDataset(rows, features),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        generator=generator,
    )


def compute_class_weights(rows: list[dict], num_classes: int, device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    values = np.asarray(
        [counts.get(index, 0) for index in range(num_classes)], dtype=np.float32
    )
    if np.any(values == 0):
        raise ValueError(f"Every training split must contain every class: {values}")
    weights = values.sum() / (num_classes * values)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_optimizer(model: nn.Module, variant: str, parameters: dict):
    if variant == "conv_lora_llrd":
        return _build_llrd_optimizer(model, parameters)

    head_parameters = list(model.classifier.parameters())
    if variant == "linear_probe":
        encoder_trainable = [
            name
            for name, parameter in model.encoder.named_parameters()
            if parameter.requires_grad
        ]
        if encoder_trainable:
            raise RuntimeError(
                "Linear probing requires a completely frozen encoder; "
                f"trainable={encoder_trainable[:5]}"
            )
        trainable_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        head_ids = {id(parameter) for parameter in head_parameters}
        if trainable_ids != head_ids or not head_ids:
            raise RuntimeError(
                "Linear-probe trainable parameters must be exactly the classifier."
            )
        return torch.optim.AdamW(
            [
                {
                    "params": head_parameters,
                    "lr": float(parameters["head_lr"]),
                    "weight_decay": float(parameters["head_weight_decay"]),
                    "name": "classifier",
                }
            ],
            betas=(0.9, 0.999),
        )

    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if not encoder_parameters:
        raise RuntimeError("No trainable Merlin encoder parameters were found.")
    encoder_lr_key = "encoder_lr" if variant == "full_finetune" else "adapter_lr"
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(parameters[encoder_lr_key]),
                "weight_decay": float(parameters["encoder_weight_decay"]),
                "name": "encoder" if variant == "full_finetune" else "adapters",
            },
            {
                "params": head_parameters,
                "lr": float(parameters["head_lr"]),
                "weight_decay": float(parameters["head_weight_decay"]),
                "name": "classifier",
            },
        ],
        betas=(0.9, 0.999),
    )


def _uses_weight_decay(
    qualified_name: str,
    parameter: nn.Parameter,
    owner: nn.Module | None,
    exclude_norm_bias: bool,
) -> bool:
    if not exclude_norm_bias:
        return True
    normalization_types = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.LocalResponseNorm,
        nn.modules.batchnorm._BatchNorm,
        nn.modules.instancenorm._InstanceNorm,
    )
    leaf_name = qualified_name.rsplit(".", 1)[-1]
    return not (
        leaf_name == "bias"
        or parameter.ndim <= 1
        or isinstance(owner, normalization_types)
    )


def _parameter_owners(model: nn.Module) -> dict[str, nn.Module]:
    owners = {}
    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(recurse=False):
            qualified_name = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
            owners[qualified_name] = module
    return owners


def _build_llrd_optimizer(model: nn.Module, parameters: dict):
    """Build strict layer-wise groups for the image-only ConvLoRA variant."""
    layer_decay = float(parameters["image_layer_decay"])
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("image_layer_decay must be in (0, 1].")
    if parameters.get("scheduler_step_unit") != "optimizer_update":
        raise ValueError(
            "conv_lora_llrd requires scheduler_step_unit=optimizer_update."
        )

    stage_depths = {
        "layer4": 0,
        "layer3": 1,
        "layer2": 2,
        "layer1": 3,
    }
    owners = _parameter_owners(model)
    exclude_norm_bias = bool(
        parameters["exclude_norm_bias_from_weight_decay"]
    )
    named_trainable = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not named_trainable:
        raise RuntimeError("No trainable Merlin parameters were found.")

    grouped: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {}
    errors = []
    for name, parameter in named_trainable:
        parts = name.split(".")
        if parts[0] == "encoder":
            matching_stages = [stage for stage in stage_depths if stage in parts]
            if len(matching_stages) != 1 or parts[-1] not in {"lora_a", "lora_b"}:
                errors.append(name)
                continue
            section = matching_stages[0]
        elif parts[0] == "classifier":
            section = "classifier"
        else:
            errors.append(name)
            continue
        decayed = _uses_weight_decay(
            name,
            parameter,
            owners.get(name),
            exclude_norm_bias,
        )
        grouped.setdefault((section, decayed), []).append((name, parameter))

    if errors:
        raise RuntimeError(
            "conv_lora_llrd found unsupported trainable parameters; only "
            "layer1-layer4 lora_a/lora_b matrices and classifier parameters "
            f"are allowed: {errors}"
        )

    base_adapter_lr = float(parameters["adapter_lr"])
    head_lr = float(parameters["head_lr"])
    optimizer_groups = []
    assigned_ids: list[int] = []
    manifest_groups = []
    for section in (*stage_depths, "classifier"):
        if not any(key[0] == section for key in grouped):
            raise RuntimeError(
                f"conv_lora_llrd has no trainable parameters in {section}."
            )
        if section != "classifier":
            stage_leaf_names = {
                name.rsplit(".", 1)[-1]
                for (group_section, _), members in grouped.items()
                if group_section == section
                for name, _ in members
            }
            if stage_leaf_names != {"lora_a", "lora_b"}:
                raise RuntimeError(
                    f"conv_lora_llrd requires both lora_a and lora_b in "
                    f"{section}; found {sorted(stage_leaf_names)}."
                )
        for decayed in (True, False):
            members = grouped.get((section, decayed), [])
            if not members:
                continue
            if section == "classifier":
                learning_rate = head_lr
                weight_decay = (
                    float(parameters["head_weight_decay"]) if decayed else 0.0
                )
                depth = None
                lr_scale = 1.0
            else:
                depth = stage_depths[section]
                lr_scale = layer_decay**depth
                learning_rate = base_adapter_lr * lr_scale
                weight_decay = (
                    float(parameters["encoder_weight_decay"])
                    if decayed
                    else 0.0
                )
            parameter_names = [name for name, _ in members]
            tensors = [parameter for _, parameter in members]
            assigned_ids.extend(id(parameter) for parameter in tensors)
            group_name = (
                f"adapter_{section}_{'decay' if decayed else 'no_decay'}"
                if section != "classifier"
                else f"classifier_{'decay' if decayed else 'no_decay'}"
            )
            group = {
                "params": tensors,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": group_name,
                "stage": section,
                "depth": depth,
                "lr_scale": lr_scale,
                "base_lr": head_lr if section == "classifier" else base_adapter_lr,
                "parameter_names": parameter_names,
            }
            optimizer_groups.append(group)
            manifest_group = {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            manifest_group.update(
                {
                    "tensor_count": len(tensors),
                    "element_count": sum(
                        parameter.numel() for parameter in tensors
                    ),
                }
            )
            manifest_groups.append(manifest_group)

    expected_ids = [id(parameter) for _, parameter in named_trainable]
    duplicate_ids = len(assigned_ids) != len(set(assigned_ids))
    if duplicate_ids or set(assigned_ids) != set(expected_ids):
        raise RuntimeError(
            "conv_lora_llrd optimizer parameter assignment is not exactly complete."
        )

    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))
    manifest = {
        "strategy": "image_conv_lora_llrd",
        "image_layer_decay": layer_decay,
        "exclude_norm_bias_from_weight_decay": exclude_norm_bias,
        "groups": manifest_groups,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    optimizer._merlin_group_manifest = manifest
    optimizer._merlin_group_signature = hashlib.sha256(encoded).hexdigest()[:16]
    return optimizer


def optimizer_group_manifest(optimizer) -> dict | None:
    manifest = getattr(optimizer, "_merlin_group_manifest", None)
    if manifest is None:
        return None
    return {
        **manifest,
        "signature": getattr(optimizer, "_merlin_group_signature", None),
    }


def build_scheduler(optimizer, epochs: int, parameters: dict):
    warmup_epochs = int(parameters["warmup_epochs"])
    minimum = float(parameters["min_lr_factor"])

    if parameters.get("scheduler_decay_epochs") is not None:
        decay_horizon = int(parameters["scheduler_decay_epochs"])
        if decay_horizon <= 0:
            raise ValueError("scheduler_decay_epochs must be positive.")

        def linear_probe_lr_factor(epoch_index: int) -> float:
            epoch_number = epoch_index + 1
            if warmup_epochs and epoch_number <= warmup_epochs:
                return epoch_number / warmup_epochs
            # LambdaLR evaluates index zero during construction. Preserve the
            # reported initial learning rate, then advance along the independent
            # cosine horizon after each completed epoch.
            completed_decay_epochs = max(epoch_number - warmup_epochs - 1, 0)
            progress = min(completed_decay_epochs / decay_horizon, 1.0)
            return minimum + (1.0 - minimum) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_probe_lr_factor
        )

    def lr_factor(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs and epoch_number <= warmup_epochs:
            return epoch_number / warmup_epochs
        decay_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch_number - warmup_epochs, 0) / decay_epochs, 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def build_optimizer_update_scheduler(
    optimizer,
    total_updates: int,
    warmup_updates: int,
    minimum_lr_factor: float,
):
    """Cosine schedule stepped after successful optimizer updates."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive.")
    if not 0 <= warmup_updates <= total_updates:
        raise ValueError("warmup_updates must be between zero and total_updates.")
    if not 0.0 <= minimum_lr_factor <= 1.0:
        raise ValueError("minimum_lr_factor must be in [0, 1].")

    def lr_factor(update_index: int) -> float:
        if warmup_updates and update_index < warmup_updates:
            return (update_index + 1) / warmup_updates
        decay_updates = max(total_updates - warmup_updates, 1)
        completed_decay_updates = max(update_index - warmup_updates, 0)
        progress = min(completed_decay_updates / decay_updates, 1.0)
        return minimum_lr_factor + (1.0 - minimum_lr_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_factor
    )


def _set_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    parameters: dict,
    amp: bool,
    update_scheduler=None,
    normalize_partial_accumulation: bool = False,
    return_update_stats: bool = False,
    features_only: bool = False,
):
    model.train()
    _set_frozen_batchnorm_eval(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    accumulation = int(parameters["grad_accum_steps"])
    max_grad_norm = float(parameters["max_grad_norm"])
    successful_updates = 0
    skipped_updates = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"]).long().to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = (
                model.forward_features(images) if features_only else model(images)
            )
            loss = criterion(logits, labels)
        if normalize_partial_accumulation:
            window_start = ((step - 1) // accumulation) * accumulation
            accumulation_divisor = min(
                accumulation, len(loader) - window_start
            )
        else:
            accumulation_divisor = accumulation
        scaler.scale(loss / accumulation_divisor).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if update_scheduler is None and not return_update_stats:
                # Preserve the legacy variants' optimizer/scaler path exactly.
                scaler.step(optimizer)
                scaler.update()
            else:
                scale_before_update = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after_update = float(scaler.get_scale())
                update_skipped = amp and scale_after_update < scale_before_update
                if update_skipped:
                    skipped_updates += 1
                else:
                    successful_updates += 1
                    if update_scheduler is not None:
                        update_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
    mean_loss = total_loss / total_samples
    if return_update_stats:
        return mean_loss, {
            "successful_optimizer_updates": successful_updates,
            "skipped_optimizer_updates": skipped_updates,
        }
    return mean_loss


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    row_count: int,
    classes: list[str],
    criterion,
    device,
    features_only: bool = False,
) -> tuple[dict, list[dict], pd.DataFrame, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    logits_values = []
    label_values = []
    patient_ids: list[str] = []
    label_strings: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"]).long().to(device, non_blocking=True)
        logits = model.forward_features(images) if features_only else model(images)
        total_loss += float(criterion(logits, labels).item()) * images.size(0)
        logits_values.append(logits.detach().cpu())
        label_values.append(labels.detach().cpu())
        patient_ids.extend(str(value) for value in batch["patient_id"])
        label_strings.extend(str(value) for value in batch["label_str"])

    logits = torch.cat(logits_values).float()
    labels = torch.cat(label_values).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = compute_metrics(labels, predictions, probabilities, classes)
    metrics["loss"] = total_loss / row_count
    metrics["selection_score"] = fold_selection_score(metrics)
    rows, frame = format_predictions(
        patient_ids, label_strings, probabilities, classes
    )
    return metrics, rows, frame, labels, probabilities


def _checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    variant: str,
    parameters: dict,
    runtime: dict,
    parameter_provenance: dict | None,
    classes: list[str],
    model_details: dict,
    epoch: int,
    best_epoch: int | None,
    best_score: float,
    history: list[dict],
) -> dict:
    checkpoint = {
        "model_name": f"merlin_{variant}",
        "model_family": "Merlin",
        "variant": variant,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "parameters": parameters,
        "runtime": runtime,
        "parameter_origin": (
            parameter_provenance.get("parameter_origin")
            if parameter_provenance
            else None
        ),
        "parameter_provenance": parameter_provenance,
        "publication_parameter_overrides": (
            parameter_provenance.get("publication_parameter_overrides")
            if parameter_provenance
            else None
        ),
        "classes": classes,
        "model_details": model_details,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "history": history,
    }
    if variant == "linear_probe":
        # The frozen official encoder is reconstructed from its pinned package
        # checkpoint. Saving it in every fold would duplicate gigabytes while
        # adding no learned state.
        checkpoint["classifier_state_dict"] = model.classifier.state_dict()
        checkpoint["checkpoint_scope"] = "classifier_only"
    else:
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["checkpoint_scope"] = "complete_adapted_model"
    return checkpoint


def _optimizer_learning_rates(optimizer, variant: str) -> dict[str, float]:
    by_name = {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    if variant == "linear_probe":
        return {"head_lr": by_name["classifier"]}
    if variant != "conv_lora_llrd":
        return {
            "encoder_or_adapter_lr": by_name[
                "encoder" if variant == "full_finetune" else "adapters"
            ],
            "head_lr": by_name["classifier"],
        }

    rates = {}
    for section in ("layer1", "layer2", "layer3", "layer4"):
        values = {
            float(group["lr"])
            for group in optimizer.param_groups
            if group.get("stage") == section
        }
        if len(values) != 1:
            raise RuntimeError(
                f"Expected one current learning rate for {section}, got {values}."
            )
        rates[f"adapter_{section}_lr"] = values.pop()
    classifier_values = {
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("stage") == "classifier"
    }
    if len(classifier_values) != 1:
        raise RuntimeError(
            "Expected one current classifier learning rate, got "
            f"{classifier_values}."
        )
    rates["head_lr"] = classifier_values.pop()
    rates["encoder_or_adapter_lr"] = rates["adapter_layer4_lr"]
    return rates


def train_run(
    variant: str,
    train_rows: list[dict],
    validation_rows: list[dict] | None,
    classes: list[str],
    parameters: dict,
    runtime: dict,
    output_dir: Path,
    cache_dir: Path,
    parameter_provenance: dict | None = None,
    fixed_epochs: int | None = None,
) -> dict:
    """Train one run and write the legacy-compatible artifact set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    seed_everything(int(runtime["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(runtime["amp"]) and device.type == "cuda"
    epochs = int(fixed_epochs or parameters["epochs"])
    start_time = time.time()

    logger.info("Merlin %s run | train=%d | validation=%d", variant, len(train_rows), len(validation_rows or []))
    logger.info("Parameters: %s", json.dumps(parameters, indent=2))
    if parameter_provenance:
        logger.info(
            "Parameter origin: %s",
            parameter_provenance.get("parameter_origin"),
        )
        overrides = parameter_provenance.get(
            "publication_parameter_overrides"
        )
        if overrides:
            logger.warning(
                "%s: %s",
                overrides["highlight"],
                json.dumps(overrides["changed_parameters"], indent=2),
            )
    logger.info("Shared preprocessing cache: %s", cache_dir)
    model, model_details = build_model(variant, len(classes), parameters, device)
    is_linear_probe = variant == "linear_probe"
    if is_linear_probe:
        # Cache one canonical representation independent of head-training AMP.
        # This prevents mixed FP16/FP32 features from sharing one CV identity.
        extraction_precision = "fp32"
        logger.info(
            "Resolving frozen-feature cache provenance; the first checkpoint "
            "SHA-256 pass may take a moment."
        )
        feature_store = MerlinFeatureStore(
            cache_dir, model, extraction_precision
        )
        extraction_batch_size = int(
            runtime.get("feature_extraction_batch_size", 1)
        )
        train_features = materialize_linear_probe_features(
            train_rows,
            model,
            feature_store,
            cache_dir,
            extraction_batch_size,
            int(runtime["num_workers"]),
            device,
            False,
            logger,
        )
        train_loader = make_feature_loader(
            train_rows,
            train_features,
            int(parameters["batch_size"]),
            True,
            int(runtime["seed"]),
        )
        validation_loader = None
        if validation_rows:
            validation_features = materialize_linear_probe_features(
                validation_rows,
                model,
                feature_store,
                cache_dir,
                extraction_batch_size,
                int(runtime["num_workers"]),
                device,
                False,
                logger,
            )
            validation_loader = make_feature_loader(
                validation_rows,
                validation_features,
                int(parameters["batch_size"]),
                False,
                int(runtime["seed"]),
            )
        model_details["feature_cache"] = {
            "root": str(feature_store.root),
            "manifest": str(feature_store.manifest_path),
            "extraction_precision": extraction_precision,
            "extraction_batch_size": extraction_batch_size,
            "features_materialized_once_per_ct": True,
            "preprocessing_manifest_sha256": feature_store.expected[
                "preprocessing_manifest_sha256"
            ],
        }
        model_details["frozen_encoder_provenance"] = feature_store.expected[
            "model"
        ]
        model.encoder.to(torch.device("cpu"))
        model_details["encoder_device_during_head_training"] = "cpu"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Frozen feature cache: %s", feature_store.root)
    else:
        train_loader = make_loader(
            train_rows,
            cache_dir,
            int(parameters["batch_size"]),
            True,
            int(runtime["num_workers"]),
        )
        validation_loader = None
        if validation_rows:
            validation_loader = make_loader(
                validation_rows,
                cache_dir,
                int(parameters["batch_size"]),
                False,
                int(runtime["num_workers"]),
            )

    class_weights = compute_class_weights(train_rows, len(classes), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, variant, parameters)
    optimizer_manifest = optimizer_group_manifest(optimizer)
    if optimizer_manifest is not None:
        model_details["optimizer_group_manifest"] = optimizer_manifest
    is_llrd = variant == "conv_lora_llrd"
    if is_llrd:
        updates_per_epoch = math.ceil(
            len(train_loader) / int(parameters["grad_accum_steps"])
        )
        total_updates = epochs * updates_per_epoch
        warmup_updates = int(parameters["warmup_epochs"]) * updates_per_epoch
        scheduler = build_optimizer_update_scheduler(
            optimizer,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            minimum_lr_factor=float(parameters["min_lr_factor"]),
        )
        model_details["scheduler"] = {
            "step_unit": "optimizer_update",
            "updates_per_epoch": updates_per_epoch,
            "planned_total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "minimum_lr_factor": float(parameters["min_lr_factor"]),
        }
    else:
        scheduler = build_scheduler(optimizer, epochs, parameters)
        if is_linear_probe:
            model_details["scheduler"] = {
                "type": "cosine",
                "step_unit": "epoch",
                "training_epochs": epochs,
                "decay_horizon_epochs": int(
                    parameters["scheduler_decay_epochs"]
                ),
                "warmup_epochs": int(parameters["warmup_epochs"]),
                "minimum_lr_factor": float(parameters["min_lr_factor"]),
            }
    scaler = build_grad_scaler(amp)

    start_epoch = 1
    best_epoch = None
    best_score = -float("inf")
    patience = 0
    history: list[dict] = []
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    if runtime.get("resume") and last_path.is_file():
        state = torch.load(
            last_path,
            map_location="cpu" if is_linear_probe else device,
        )
        if is_linear_probe and (
            state.get("variant") != variant
            or state.get("parameters") != parameters
        ):
            raise RuntimeError(
                "Refusing to resume linear_probe from a checkpoint with a "
                "different variant or parameter dictionary."
            )
        if is_linear_probe:
            saved_details = state.get("model_details", {})
            if _comparable_encoder_provenance(
                saved_details.get("frozen_encoder_provenance")
            ) != _comparable_encoder_provenance(
                model_details.get("frozen_encoder_provenance")
            ):
                raise RuntimeError(
                    "Refusing to resume linear_probe with a different frozen "
                    "Merlin encoder checkpoint."
                )
            saved_feature_cache = saved_details.get("feature_cache", {})
            current_feature_cache = model_details.get("feature_cache", {})
            for key in (
                "preprocessing_manifest_sha256",
                "extraction_precision",
            ):
                if saved_feature_cache.get(key) != current_feature_cache.get(key):
                    raise RuntimeError(
                        "Refusing to resume linear_probe with different frozen "
                        f"feature provenance ({key})."
                    )
        if is_llrd:
            if state.get("variant") != variant or state.get("parameters") != parameters:
                raise RuntimeError(
                    "Refusing to resume conv_lora_llrd from a checkpoint with "
                    "different variant or parameters."
                )
            saved_signature = state.get("model_details", {}).get(
                "optimizer_group_manifest", {}
            ).get("signature")
            current_signature = model_details.get(
                "optimizer_group_manifest", {}
            ).get("signature")
            if saved_signature != current_signature:
                raise RuntimeError(
                    "Refusing to resume conv_lora_llrd because the optimizer "
                    "parameter-group signature changed."
                )
            if state.get("model_details", {}).get("scheduler") != model_details.get(
                "scheduler"
            ):
                raise RuntimeError(
                    "Refusing to resume conv_lora_llrd because the planned "
                    "optimizer-update schedule changed."
                )
        if is_linear_probe:
            model.classifier.load_state_dict(
                state["classifier_state_dict"], strict=True
            )
        else:
            model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = state.get("best_epoch")
        best_score = float(state.get("best_selection_score", -float("inf")))
        history = list(state.get("history", []))
        patience = int(state.get("patience_counter", 0))
        if is_llrd and history:
            recorded_updates = history[-1].get("global_optimizer_updates")
            if recorded_updates is None or int(recorded_updates) != int(
                scheduler.last_epoch
            ):
                raise RuntimeError(
                    "Refusing to resume conv_lora_llrd because scheduler state "
                    "does not match the recorded optimizer-update count."
                )
        logger.info("Resuming at epoch %d", start_epoch)

    for epoch in range(start_epoch, epochs + 1):
        if is_llrd:
            train_loss, update_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                parameters,
                amp,
                update_scheduler=scheduler,
                normalize_partial_accumulation=True,
                return_update_stats=True,
            )
        else:
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                parameters,
                amp,
                features_only=is_linear_probe,
            )
            update_stats = None
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **_optimizer_learning_rates(optimizer, variant),
        }
        if update_stats is not None:
            history_row.update(update_stats)
            history_row["global_optimizer_updates"] = int(
                scheduler.last_epoch
            )
        improved = validation_loader is None
        validation_metrics = None
        validation_predictions = None
        validation_frame = None
        if validation_loader is not None:
            (
                validation_metrics,
                validation_predictions,
                validation_frame,
                _,
                _,
            ) = evaluate_model(
                model,
                validation_loader,
                len(validation_rows),
                classes,
                criterion,
                device,
                features_only=is_linear_probe,
            )
            score = float(validation_metrics["selection_score"])
            improved = score > best_score + float(parameters["min_delta"])
            history_row.update(
                {
                    "val_loss": validation_metrics["loss"],
                    "val_f1_macro": validation_metrics["f1_macro"],
                    "val_balanced_accuracy": validation_metrics[
                        "balanced_accuracy"
                    ],
                    "val_average_precision_macro": validation_metrics[
                        "average_precision_macro"
                    ],
                    "val_selection_score": score,
                }
            )
        else:
            score = float(epoch)

        if improved:
            best_epoch = epoch
            best_score = score
            patience = 0
        else:
            patience += 1
        history_row["is_best_epoch"] = improved
        history_row["patience_counter"] = patience
        history.append(history_row)
        if not is_llrd:
            scheduler.step()

        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            variant,
            parameters,
            runtime,
            parameter_provenance,
            classes,
            model_details,
            epoch,
            best_epoch,
            best_score,
            history,
        )
        checkpoint["patience_counter"] = patience
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, best_path)
            if validation_predictions is not None:
                save_json(
                    output_dir / "validation_patient_predictions.json",
                    validation_predictions,
                )
                validation_frame.to_csv(
                    output_dir / "validation_probabilities.csv", index=False
                )

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv", index=False
        )
        save_json(
            output_dir / "training_state.json",
            {
                "model_name": f"merlin_{variant}",
                "parameters": parameters,
                "runtime": runtime,
                "parameter_origin": (
                    parameter_provenance.get("parameter_origin")
                    if parameter_provenance
                    else None
                ),
                "parameter_provenance": parameter_provenance,
                "publication_parameter_overrides": (
                    parameter_provenance.get(
                        "publication_parameter_overrides"
                    )
                    if parameter_provenance
                    else None
                ),
                "classes": classes,
                "model_details": model_details,
                "epochs_completed": epoch,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "patience_counter": patience,
                "history": history,
            },
        )
        logger.info(
            "Epoch %d/%d | train_loss=%.4f%s",
            epoch,
            epochs,
            train_loss,
            (
                f" | val_score={validation_metrics['selection_score']:.4f}"
                if validation_metrics
                else ""
            ),
        )
        if (
            validation_loader is not None
            and int(parameters["early_stopping_patience"]) > 0
            and patience >= int(parameters["early_stopping_patience"])
        ):
            logger.info("Early stopping at epoch %d", epoch)
            break

    if best_path.is_file():
        state = torch.load(
            best_path,
            map_location="cpu" if is_linear_probe else device,
        )
        if is_linear_probe:
            model.classifier.load_state_dict(
                state["classifier_state_dict"], strict=True
            )
        else:
            model.load_state_dict(state["model_state_dict"], strict=True)

    final_validation_metrics = None
    if validation_loader is not None:
        (
            final_validation_metrics,
            predictions,
            frame,
            _,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            len(validation_rows),
            classes,
            criterion,
            device,
            features_only=is_linear_probe,
        )
        save_json(output_dir / "validation_patient_predictions.json", predictions)
        frame.to_csv(output_dir / "validation_probabilities.csv", index=False)

    summary = {
        "model_name": f"merlin_{variant}",
        "model_family": "Merlin",
        "variant": variant,
        "parameters": parameters,
        "runtime": runtime,
        "parameter_origin": (
            parameter_provenance.get("parameter_origin")
            if parameter_provenance
            else None
        ),
        "parameter_provenance": parameter_provenance,
        "publication_parameter_overrides": (
            parameter_provenance.get("publication_parameter_overrides")
            if parameter_provenance
            else None
        ),
        "classes": classes,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "validation_metrics": final_validation_metrics,
        "trainable_parameters": model_details["trainable_parameters"],
        "total_parameters": model_details["total_parameters"],
        "model_details": model_details,
        "total_runtime_seconds": time.time() - start_time,
    }
    save_json(output_dir / "metrics_summary.json", summary)
    return summary


def predict_checkpoint(
    checkpoint_path: Path,
    rows: list[dict],
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[dict, list[dict], pd.DataFrame, np.ndarray, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    variant = checkpoint["variant"]
    parameters = checkpoint["parameters"]
    classes = checkpoint["classes"]
    model, _ = build_model(variant, len(classes), parameters, device)
    if variant == "linear_probe":
        model.classifier.load_state_dict(
            checkpoint["classifier_state_dict"], strict=True
        )
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    class_weights = compute_class_weights(rows, len(classes), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    features_only = variant == "linear_probe"
    if features_only:
        runtime = checkpoint.get("runtime", {})
        extraction_precision = "fp32"
        feature_store = MerlinFeatureStore(
            cache_dir, model, extraction_precision
        )
        saved_details = checkpoint.get("model_details", {})
        saved_encoder = saved_details.get("frozen_encoder_provenance")
        saved_preprocessing = saved_details.get("feature_cache", {}).get(
            "preprocessing_manifest_sha256"
        )
        if saved_encoder is None or saved_preprocessing is None:
            raise RuntimeError(
                "Linear-probe checkpoint lacks frozen encoder/cache provenance."
            )
        if _comparable_encoder_provenance(
            saved_encoder
        ) != _comparable_encoder_provenance(feature_store.expected["model"]):
            raise RuntimeError(
                "Linear-probe checkpoint was trained with a different frozen "
                "Merlin encoder checkpoint."
            )
        if saved_preprocessing != feature_store.expected[
            "preprocessing_manifest_sha256"
        ]:
            raise RuntimeError(
                "Linear-probe checkpoint was trained with a different Merlin "
                "preprocessing manifest."
            )
        features = materialize_linear_probe_features(
            rows,
            model,
            feature_store,
            cache_dir,
            int(runtime.get("feature_extraction_batch_size", 1)),
            num_workers,
            device,
            False,
        )
        loader = make_feature_loader(
            rows,
            features,
            batch_size,
            False,
            int(runtime.get("seed", 42)),
        )
        model.encoder.to(torch.device("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        loader = make_loader(rows, cache_dir, batch_size, False, num_workers)
    metrics, predictions, frame, labels, probabilities = evaluate_model(
        model,
        loader,
        len(rows),
        classes,
        criterion,
        device,
        features_only=features_only,
    )
    return metrics, predictions, frame, labels, probabilities, checkpoint
