"""Training and evaluation engines for Pillar-0 adaptations.

The linear probe caches immutable 1152-dimensional encoder features.  The
end-to-end path always reloads raw-HU RAVE samples, applies random augmentation
after deterministic caching, constructs the eleven windows on the fly, and
verifies that encoder gradients are present.
"""

from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from .cache import load_pillar_volume
    from .data import CLASS_NAMES
    from .metrics import compute_metrics, fold_selection_score, format_predictions
    from .model import (
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        PillarClassifier,
        extract_vision_features,
        load_official_model,
        set_gradient_checkpointing,
        trainable_parameter_report,
    )
except ImportError:  # pragma: no cover - script entry points
    from cache import load_pillar_volume
    from data import CLASS_NAMES
    from metrics import compute_metrics, fold_selection_score, format_predictions
    from model import (
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        PillarClassifier,
        extract_vision_features,
        load_official_model,
        set_gradient_checkpointing,
        trainable_parameter_report,
    )


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fieldnames} for row in rows)


def atomic_torch_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"pillar0_{abs(hash(str(output_dir.resolve())))}"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(rows: list[dict], scheme: str = "inverse_frequency", beta: float = 0.999):
    counts = Counter(int(row["label"]) for row in rows)
    values = np.asarray([counts.get(i, 0) for i in range(len(CLASS_NAMES))], dtype=np.float64)
    if np.any(values == 0):
        raise ValueError(f"Every training partition must contain all classes: {values.tolist()}")
    if scheme == "inverse_frequency":
        weights = values.sum() / (len(values) * values)
    elif scheme == "effective_number":
        weights = (1.0 - beta) / (1.0 - np.power(beta, values))
    elif scheme == "none":
        weights = np.ones_like(values)
    else:
        raise ValueError(f"Unknown class weighting scheme: {scheme}")
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class CachedVolumeDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        volume = load_pillar_volume(Path(row["cache_path"]))
        return volume, int(row["label"]), str(row["patient_id"]), str(row["label_str"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureStore:
    """Strict per-patient feature cache tied to raw cache and model revision."""

    def __init__(
        self,
        root: Path,
        raw_cache_manifest: Path,
        repo_id: str = MODEL_REPO_ID,
        revision: str = MODEL_REVISION,
    ):
        self.root = Path(root)
        self.raw_cache_manifest = Path(raw_cache_manifest)
        if not self.raw_cache_manifest.is_file():
            raise FileNotFoundError(f"Raw cache manifest not found: {self.raw_cache_manifest}")
        self.expected = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pillar0_frozen_vision_features",
            "feature_dim": FEATURE_DIM,
            "dtype": "float32",
            "model_repo_id": repo_id,
            "model_revision": revision,
            "raw_cache_manifest": str(self.raw_cache_manifest.resolve()),
            "raw_cache_manifest_sha256": _sha256_file(self.raw_cache_manifest),
            "modality": "abdomen_ct",
            "windowing": "RAVE_all_11_per_sample_no_post_normalization",
        }
        manifest_path = self.root / "feature_cache_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            comparable = {key: existing.get(key) for key in self.expected}
            if comparable != self.expected:
                raise RuntimeError(
                    f"Feature cache provenance mismatch at {manifest_path}; use a new namespace."
                )
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            save_json(manifest_path, {**self.expected, "created_at_utc": utc_now()})

    def path_for(self, row: dict) -> Path:
        identity = json.dumps(
            {
                "patient_id": str(row["patient_id"]),
                "label_str": str(row["label_str"]),
                "cache_path": str(Path(row["cache_path"]).resolve()),
                "source_fingerprint": row.get("source_fingerprint"),
            },
            sort_keys=True,
        ).encode("utf-8")
        return self.root / f"{hashlib.sha256(identity).hexdigest()}.pt"

    def load(self, row: dict) -> torch.Tensor | None:
        path = self.path_for(row)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feature = payload["feature"]
        if feature.shape != (FEATURE_DIM,) or feature.dtype != torch.float32:
            raise RuntimeError(f"Invalid cached Pillar feature: {path}")
        return feature

    def save(self, row: dict, feature: torch.Tensor) -> Path:
        feature = feature.detach().cpu().float().reshape(-1)
        if feature.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected {FEATURE_DIM}-D feature, received {tuple(feature.shape)}")
        path = self.path_for(row)
        atomic_torch_save(
            {
                "feature": feature,
                "patient_id": str(row["patient_id"]),
                "label_str": str(row["label_str"]),
                "created_at_utc": utc_now(),
            },
            path,
        )
        return path


def materialize_features(
    rows: list[dict],
    store: FeatureStore,
    *,
    device: torch.device,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    logger: logging.Logger | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = [row for row in rows if store.load(row) is None]
    if missing:
        if logger:
            logger.info("Extracting %d missing frozen Pillar-0 features", len(missing))
        encoder = load_official_model(repo_id, revision, device=device)
        encoder.eval()
        for index, row in enumerate(missing, start=1):
            raw = load_pillar_volume(Path(row["cache_path"])).unsqueeze(0).to(device)
            with torch.inference_mode():
                feature = extract_vision_features(encoder, raw).squeeze(0)
            store.save(row, feature)
            if logger and (index == 1 or index % 25 == 0 or index == len(missing)):
                logger.info("Feature cache %d/%d", index, len(missing))
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    features = torch.stack([store.load(row) for row in rows])
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    return features, labels


def random_z_rotation(raw_hu: torch.Tensor, degrees: float, probability: float) -> torch.Tensor:
    """Apply differentiable per-volume rotations, padding outside with -1024 HU."""

    if degrees <= 0.0 or probability <= 0.0:
        return raw_hu
    batch = raw_hu.shape[0]
    apply = torch.rand(batch, device=raw_hu.device) < probability
    angles = (torch.rand(batch, device=raw_hu.device) * 2.0 - 1.0) * math.radians(degrees)
    angles = angles * apply
    cosine, sine = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros((batch, 3, 4), device=raw_hu.device, dtype=raw_hu.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    theta[:, 2, 2] = 1.0
    grid = F.affine_grid(theta, raw_hu.shape, align_corners=False)
    return F.grid_sample(raw_hu + 1024.0, grid, mode="bilinear", padding_mode="zeros", align_corners=False) - 1024.0


def _autocast(device: torch.device, enabled: bool):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


def _cosine_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup_steps = int(total_steps * warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _layer_key(name: str) -> tuple[int, int, int, int]:
    """Return a stable architectural order for Pillar's visual parameters."""

    if "patch_embeds." in name:
        return (0, 0, 0, 0)
    atlas = re.search(r"atlas_models\.(\d+)", name)
    if atlas:
        stage = int(atlas.group(1))
        outer = re.search(r"atlas_models\.\d+\.blocks\.(\d+)", name)
        inner = re.search(
            r"atlas_models\.\d+\.blocks\.\d+\.blocks\.(\d+)", name
        )
        return (
            1,
            stage,
            int(outer.group(1)) if outer else -1,
            int(inner.group(1)) if inner else -1,
        )
    # Pooling/readout components follow the atlas stages.
    return (2, 0, 0, 0)


def full_finetune_optimizer(model: PillarClassifier, parameters: dict):
    encoder_lr = float(parameters["encoder_lr"])
    head_lr = float(parameters["head_lr"])
    weight_decay = float(parameters.get("weight_decay", 0.0))
    layer_decay = float(parameters.get("layer_decay", 1.0))
    named = [(name, value) for name, value in model.encoder.named_parameters() if value.requires_grad]
    ordered_keys = {
        key: index
        for index, key in enumerate(sorted({_layer_key(name) for name, _ in named}))
    }
    max_depth = max(ordered_keys.values(), default=0)
    groups = {}
    for name, value in named:
        depth = ordered_keys[_layer_key(name)]
        decay = not (name.endswith("bias") or value.ndim <= 1 or "norm" in name.lower())
        key = (depth, decay)
        groups.setdefault(key, []).append(value)
    optimizer_groups = []
    for (depth, decay), tensors in sorted(groups.items()):
        optimizer_groups.append(
            {
                "params": tensors,
                "lr": encoder_lr * (layer_decay ** (max_depth - depth)),
                "weight_decay": weight_decay if decay else 0.0,
                "name": f"encoder_depth_{depth}_{'decay' if decay else 'no_decay'}",
            }
        )
    optimizer_groups.append(
        {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay, "name": "classifier"}
    )
    return torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))


def _evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict]) -> dict:
    probabilities = logits.float().softmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    metrics = compute_metrics(truth, predicted, probabilities, list(CLASS_NAMES))
    prediction_rows, frame = format_predictions(
        [str(row["patient_id"]) for row in rows],
        [str(row["label_str"]) for row in rows],
        probabilities,
        list(CLASS_NAMES),
    )
    return {"metrics": metrics, "predictions": prediction_rows, "probabilities": frame}


def _save_evaluation(output_dir: Path, prefix: str, evaluation: dict) -> None:
    save_json(output_dir / "metrics_summary.json", evaluation["metrics"])
    save_json(output_dir / f"{prefix}_patient_predictions.json", evaluation["predictions"])
    evaluation["probabilities"].to_csv(output_dir / f"{prefix}_probabilities.csv", index=False)


def train_linear_probe(
    train_rows: list[dict],
    validation_rows: list[dict],
    parameters: dict,
    output_dir: Path,
    feature_store: FeatureStore,
    *,
    device: torch.device,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    seed: int = 42,
    budget: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    logger = setup_logger(output_dir)
    seed_everything(seed)
    train_x, train_y = materialize_features(train_rows, feature_store, device=device, repo_id=repo_id, revision=revision, logger=logger)
    val_x, val_y = materialize_features(validation_rows, feature_store, device=device, repo_id=repo_id, revision=revision, logger=logger)
    head = nn.Linear(FEATURE_DIM, len(CLASS_NAMES)).to(device)
    optimizer_name = str(parameters.get("optimizer", "adamw")).lower()
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(
        head.parameters(),
        lr=float(parameters["lr"]),
        weight_decay=float(parameters.get("weight_decay", 0.0)),
    )
    weights = class_weights(
        train_rows,
        str(parameters.get("class_weighting", "inverse_frequency")),
        float(parameters.get("effective_number_beta", 0.999)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    epochs = min(int(parameters["epochs"]), int(budget)) if budget else int(parameters["epochs"])
    patience = int(parameters.get("early_stopping_patience", epochs))
    min_delta = float(parameters.get("min_delta", 0.0))
    batch_size = min(int(parameters.get("batch_size", len(train_rows))), len(train_rows))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, generator=generator)
    history, best_score, stale = [], -math.inf, 0
    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_x.to(device)).cpu()
        evaluation = _evaluate_logits(val_logits, val_y, validation_rows)
        score = fold_selection_score(evaluation["metrics"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_score": score, **{f"validation_{k}": evaluation["metrics"][k] for k in ("f1_macro", "balanced_accuracy")}})
        if score > best_score + min_delta:
            best_score, stale = score, 0
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": "linear_probe",
                    "model_repo_id": repo_id,
                    "model_revision": revision,
                    "classes": list(CLASS_NAMES),
                    "parameters": parameters,
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": evaluation["metrics"],
                },
                output_dir / "best_model.pt",
            )
            _save_evaluation(output_dir, "validation", evaluation)
        else:
            stale += 1
        if stale >= patience:
            break
    atomic_torch_save(
        {"head_state_dict": head.state_dict(), "epoch": epoch, "parameters": parameters},
        output_dir / "last_checkpoint.pt",
    )
    save_csv(output_dir / "training_history.csv", history, list(history[0]))
    best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    head.load_state_dict(best["head_state_dict"])
    state = {
        "schema_version": SCHEMA_VERSION,
        "method": "linear_probe",
        "completed": True,
        "best_epoch": int(best["epoch"]),
        "epochs_completed": epoch,
        "best_selection_score": best_score,
        "frozen_encoder": True,
        "head": f"Linear({FEATURE_DIM},{len(CLASS_NAMES)})",
        "class_weights": weights.detach().cpu().tolist(),
        "parameters": parameters,
        "seed": seed,
        "completed_at_utc": utc_now(),
    }
    save_json(output_dir / "training_state.json", state)
    return {"state": state, "metrics": best["validation_metrics"], "checkpoint": str(output_dir / "best_model.pt")}


def _full_model_dtype(
    device: torch.device,
    mixed_precision: bool,
    bf16_model_weights: bool = False,
) -> torch.dtype | None:
    """Return the explicitly requested low-precision checkpoint dtype.

    FP32 master weights plus BF16 autocast are the validated default.  Loading
    all trainable weights as BF16 reduced precision and timed out in the local
    optimizer-step smoke test, so it remains an explicit experimental option.
    """

    if (
        mixed_precision
        and bf16_model_weights
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return None


def _activation_offload(device: torch.device, enabled: bool):
    """Optionally move tensors saved for backward to pinned host memory."""

    if enabled and device.type == "cuda":
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return contextlib.nullcontext()


def _volume_logits(
    model,
    rows: list[dict],
    device: torch.device,
    num_workers: int = 0,
    *,
    mixed_precision: bool = False,
):
    loader = DataLoader(CachedVolumeDataset(rows), batch_size=1, shuffle=False, num_workers=num_workers)
    logits, labels = [], []
    encoder_parameter = next(model.encoder.parameters(), None)
    encoder_dtype = encoder_parameter.dtype if encoder_parameter is not None else torch.float32
    model.eval()
    with torch.inference_mode():
        for volume, target, _, _ in loader:
            volume = volume.to(
                device=device,
                dtype=encoder_dtype if encoder_dtype in (torch.float16, torch.bfloat16) else torch.float32,
            )
            with _autocast(device, mixed_precision):
                output = model(volume)
            logits.append(output.float().cpu())
            labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def train_full_finetune(
    train_rows: list[dict],
    validation_rows: list[dict],
    parameters: dict,
    output_dir: Path,
    *,
    device: torch.device,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    seed: int = 42,
    budget: int | None = None,
    num_workers: int = 0,
) -> dict:
    output_dir = Path(output_dir)
    logger = setup_logger(output_dir)
    seed_everything(seed)
    mixed = bool(parameters.get("mixed_precision", True))
    model_dtype = _full_model_dtype(
        device,
        mixed,
        bool(parameters.get("bf16_model_weights", False)),
    )
    encoder = load_official_model(repo_id, revision, device=device, dtype=model_dtype)
    model = PillarClassifier(
        encoder,
        len(CLASS_NAMES),
        frozen_encoder=False,
        dropout=float(parameters.get("dropout", 0.0)),
    ).to(device)
    checkpointing_active = set_gradient_checkpointing(model.encoder, bool(parameters.get("gradient_checkpointing", True)))
    report = trainable_parameter_report(model)
    if not any(name.startswith("encoder.") for name in report.trainable_names):
        raise RuntimeError("Full fine-tuning has no trainable encoder parameters.")
    optimizer = full_finetune_optimizer(model, parameters)
    accumulation = int(parameters.get("grad_accum_steps", 1))
    epochs = int(parameters.get("epochs", 50))
    maximum_steps = int(budget or parameters.get("max_steps", 10000))
    loader = DataLoader(CachedVolumeDataset(train_rows), batch_size=int(parameters.get("batch_size", 1)), shuffle=True, num_workers=num_workers)
    total_steps = min(maximum_steps, epochs * max(1, math.ceil(len(loader) / accumulation)))
    scheduler = _cosine_scheduler(optimizer, total_steps, float(parameters.get("warmup_fraction", 0.05)))
    weights = class_weights(train_rows, str(parameters.get("class_weighting", "inverse_frequency")), float(parameters.get("effective_number_beta", 0.999))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and device.type == "cuda" and not torch.cuda.is_bf16_supported())
    patience_steps = int(parameters.get("early_stopping_patience_steps", 1000))
    eval_every = int(parameters.get("evaluation_interval_steps", max(1, min(100, len(loader)))))
    history, best_score = [], -math.inf
    optimizer_step = stale_steps = 0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_index, (volume, target, _, _) in enumerate(loader, start=1):
            volume, target = volume.to(device), target.to(device)
            volume = random_z_rotation(volume, float(parameters.get("rotation_degrees", 20.0)), float(parameters.get("rotation_probability", 0.5)))
            if model_dtype is not None:
                volume = volume.to(dtype=model_dtype)
            with _activation_offload(
                device, bool(parameters.get("activation_offload", False))
            ):
                with _autocast(device, mixed):
                    loss = criterion(model(volume), target) / accumulation
            scaler.scale(loss).backward()
            if batch_index % accumulation != 0 and batch_index != len(loader):
                continue
            if float(parameters.get("max_grad_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(parameters["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            stale_steps += 1
            should_evaluate = optimizer_step % eval_every == 0 or optimizer_step >= maximum_steps
            if should_evaluate:
                val_logits, val_y = _volume_logits(
                    model,
                    validation_rows,
                    device,
                    num_workers,
                    mixed_precision=mixed,
                )
                evaluation = _evaluate_logits(val_logits, val_y, validation_rows)
                score = fold_selection_score(evaluation["metrics"])
                history.append({"epoch": epoch, "optimizer_step": optimizer_step, "train_loss": float(loss.detach().cpu()) * accumulation, "validation_score": score, "validation_f1_macro": evaluation["metrics"]["f1_macro"], "validation_balanced_accuracy": evaluation["metrics"]["balanced_accuracy"]})
                if score > best_score + float(parameters.get("min_delta", 0.0)):
                    best_score, stale_steps = score, 0
                    atomic_torch_save(
                        {"schema_version": SCHEMA_VERSION, "method": "full_finetune", "model_repo_id": repo_id, "model_revision": revision, "classes": list(CLASS_NAMES), "parameters": parameters, "model_state_dict": model.state_dict(), "epoch": epoch, "optimizer_step": optimizer_step, "validation_metrics": evaluation["metrics"]},
                        output_dir / "best_model.pt",
                    )
                    _save_evaluation(output_dir, "validation", evaluation)
                model.train()
            if optimizer_step >= maximum_steps or stale_steps >= patience_steps:
                stop = True
                break
        if stop:
            break
    atomic_torch_save({"model_state_dict": model.state_dict(), "epoch": epoch, "optimizer_step": optimizer_step, "parameters": parameters}, output_dir / "last_checkpoint.pt")
    if not (output_dir / "best_model.pt").is_file():
        val_logits, val_y = _volume_logits(
            model,
            validation_rows,
            device,
            num_workers,
            mixed_precision=mixed,
        )
        evaluation = _evaluate_logits(val_logits, val_y, validation_rows)
        atomic_torch_save({"schema_version": SCHEMA_VERSION, "method": "full_finetune", "model_repo_id": repo_id, "model_revision": revision, "classes": list(CLASS_NAMES), "parameters": parameters, "model_state_dict": model.state_dict(), "epoch": epoch, "optimizer_step": optimizer_step, "validation_metrics": evaluation["metrics"]}, output_dir / "best_model.pt")
        _save_evaluation(output_dir, "validation", evaluation)
        best_score = fold_selection_score(evaluation["metrics"])
    if history:
        save_csv(output_dir / "training_history.csv", history, list(history[0]))
    best = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    state = {"schema_version": SCHEMA_VERSION, "method": "full_finetune", "completed": True, "best_epoch": int(best["epoch"]), "best_optimizer_step": int(best["optimizer_step"]), "optimizer_steps_completed": optimizer_step, "best_selection_score": best_score, "frozen_encoder": False, "gradient_checkpointing_active": checkpointing_active, "model_weight_dtype": str(model_dtype or torch.float32).replace("torch.", ""), "trainable_parameters": report.trainable, "frozen_parameters": report.frozen, "class_weights": weights.detach().cpu().tolist(), "parameters": parameters, "seed": seed, "completed_at_utc": utc_now()}
    save_json(output_dir / "training_state.json", state)
    return {"state": state, "metrics": best["validation_metrics"], "checkpoint": str(output_dir / "best_model.pt")}


def evaluate_checkpoint(
    rows: list[dict],
    checkpoint_path: Path,
    output_dir: Path,
    *,
    method: str,
    device: torch.device,
    feature_store: FeatureStore | None = None,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    prefix: str = "test",
    num_workers: int = 0,
) -> dict:
    logits, labels = predict_checkpoint(
        rows,
        checkpoint_path,
        method=method,
        device=device,
        feature_store=feature_store,
        repo_id=repo_id,
        revision=revision,
        num_workers=num_workers,
    )
    evaluation = _evaluate_logits(logits, labels, rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_evaluation(output_dir, prefix, evaluation)
    return evaluation["metrics"]


def predict_checkpoint(
    rows: list[dict],
    checkpoint_path: Path,
    *,
    method: str,
    device: torch.device,
    feature_store: FeatureStore | None = None,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits/labels without writing artifacts (used for ensembling)."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("model_revision") not in (None, revision):
        raise RuntimeError("Checkpoint model revision differs from the requested encoder.")
    if method == "linear_probe":
        if feature_store is None:
            raise ValueError("Linear-probe evaluation requires a FeatureStore.")
        features, labels = materialize_features(rows, feature_store, device=device, repo_id=repo_id, revision=revision)
        head = nn.Linear(FEATURE_DIM, len(CLASS_NAMES)).to(device)
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        with torch.inference_mode():
            logits = head(features.to(device)).cpu()
    elif method == "full_finetune":
        mixed = bool(payload.get("parameters", {}).get("mixed_precision", True))
        model_dtype = _full_model_dtype(
            device,
            mixed,
            bool(payload.get("parameters", {}).get("bf16_model_weights", False)),
        )
        encoder = load_official_model(repo_id, revision, device=device, dtype=model_dtype)
        model = PillarClassifier(encoder, len(CLASS_NAMES), frozen_encoder=False, dropout=float(payload.get("parameters", {}).get("dropout", 0.0))).to(device)
        model.load_state_dict(payload["model_state_dict"])
        logits, labels = _volume_logits(
            model,
            rows,
            device,
            num_workers,
            mixed_precision=mixed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return logits, labels


def refit_model(
    rows: list[dict],
    parameters: dict,
    output_dir: Path,
    *,
    method: str,
    device: torch.device,
    refit_budget: int,
    feature_store: FeatureStore | None = None,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    seed: int = 42,
    num_workers: int = 0,
) -> dict:
    """Refit on all development patients for a CV-derived fixed budget.

    ``refit_budget`` is the median best epoch for a probe and median best
    optimizer step for full fine-tuning.  No held-out data and no surrogate
    validation subset are consulted.
    """

    if refit_budget <= 0:
        raise ValueError("refit_budget must be positive.")
    output_dir = Path(output_dir)
    logger = setup_logger(output_dir)
    seed_everything(seed)
    weights = class_weights(
        rows,
        str(parameters.get("class_weighting", "inverse_frequency")),
        float(parameters.get("effective_number_beta", 0.999)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict] = []

    if method == "linear_probe":
        if feature_store is None:
            raise ValueError("Linear-probe refit requires a FeatureStore.")
        features, labels = materialize_features(
            rows,
            feature_store,
            device=device,
            repo_id=repo_id,
            revision=revision,
            logger=logger,
        )
        model = nn.Linear(FEATURE_DIM, len(CLASS_NAMES)).to(device)
        optimizer_class = (
            torch.optim.Adam
            if str(parameters.get("optimizer", "adamw")).lower() == "adam"
            else torch.optim.AdamW
        )
        optimizer = optimizer_class(
            model.parameters(),
            lr=float(parameters["lr"]),
            weight_decay=float(parameters.get("weight_decay", 0.0)),
        )
        loader = DataLoader(
            TensorDataset(features, labels),
            batch_size=min(int(parameters.get("batch_size", len(rows))), len(rows)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        for epoch in range(1, refit_budget + 1):
            losses = []
            model.train()
            for batch_features, batch_labels in loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        model.eval()
        with torch.inference_mode():
            logits = model(features.to(device)).float().cpu()
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "model_repo_id": repo_id,
            "model_revision": revision,
            "classes": list(CLASS_NAMES),
            "parameters": parameters,
            "head_state_dict": model.state_dict(),
            "epoch": refit_budget,
            "refit_full_development": True,
        }
        completed_units = refit_budget
    elif method == "full_finetune":
        mixed = bool(parameters.get("mixed_precision", True))
        model_dtype = _full_model_dtype(
            device,
            mixed,
            bool(parameters.get("bf16_model_weights", False)),
        )
        encoder = load_official_model(repo_id, revision, device=device, dtype=model_dtype)
        model = PillarClassifier(
            encoder,
            len(CLASS_NAMES),
            frozen_encoder=False,
            dropout=float(parameters.get("dropout", 0.0)),
        ).to(device)
        set_gradient_checkpointing(
            model.encoder, bool(parameters.get("gradient_checkpointing", True))
        )
        optimizer = full_finetune_optimizer(model, parameters)
        accumulation = int(parameters.get("grad_accum_steps", 1))
        loader = DataLoader(
            CachedVolumeDataset(rows),
            batch_size=int(parameters.get("batch_size", 1)),
            shuffle=True,
            num_workers=num_workers,
        )
        scheduler = _cosine_scheduler(
            optimizer, refit_budget, float(parameters.get("warmup_fraction", 0.05))
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=mixed
            and device.type == "cuda"
            and not torch.cuda.is_bf16_supported(),
        )
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = epoch = 0
        while optimizer_step < refit_budget:
            epoch += 1
            model.train()
            for batch_index, (volume, target, _, _) in enumerate(loader, start=1):
                volume, target = volume.to(device), target.to(device)
                volume = random_z_rotation(
                    volume,
                    float(parameters.get("rotation_degrees", 20.0)),
                    float(parameters.get("rotation_probability", 0.5)),
                )
                if model_dtype is not None:
                    volume = volume.to(dtype=model_dtype)
                with _activation_offload(
                    device, bool(parameters.get("activation_offload", False))
                ):
                    with _autocast(device, mixed):
                        loss = criterion(model(volume), target) / accumulation
                scaler.scale(loss).backward()
                if batch_index % accumulation != 0 and batch_index != len(loader):
                    continue
                if float(parameters.get("max_grad_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(parameters["max_grad_norm"])
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "train_loss": float(loss.detach().cpu()) * accumulation,
                    }
                )
                if optimizer_step >= refit_budget:
                    break
        logits, labels = _volume_logits(
            model,
            rows,
            device,
            num_workers,
            mixed_precision=mixed,
        )
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "model_repo_id": repo_id,
            "model_revision": revision,
            "classes": list(CLASS_NAMES),
            "parameters": parameters,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "refit_full_development": True,
        }
        completed_units = optimizer_step
    else:
        raise ValueError(f"Unknown method: {method}")

    evaluation = _evaluate_logits(logits, labels, rows)
    metrics = {
        **evaluation["metrics"],
        "evaluation_partition": "full_development_training_set",
        "selection_use": "none_training_diagnostics_only",
    }
    checkpoint["development_training_metrics"] = metrics
    atomic_torch_save(checkpoint, output_dir / "best_model.pt")
    atomic_torch_save(checkpoint, output_dir / "last_checkpoint.pt")
    save_csv(output_dir / "training_history.csv", history, list(history[0]))
    save_json(output_dir / "metrics_summary.json", metrics)
    state = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "completed": True,
        "refit_full_development": True,
        "refit_budget_source": "median_best_budget_from_five_cv_folds",
        "refit_budget": refit_budget,
        "completed_units": completed_units,
        "heldout_test_used": False,
        "parameters": parameters,
        "seed": seed,
        "completed_at_utc": utc_now(),
    }
    save_json(output_dir / "training_state.json", state)
    return {"state": state, "metrics": metrics, "checkpoint": str(output_dir / "best_model.pt")}
