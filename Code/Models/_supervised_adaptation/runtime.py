"""Deterministic optimization and safe PyTorch artifact helpers."""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model environment boundary
        raise RuntimeError("PyTorch is required for supervised adaptation") from exc
    return torch


def atomic_torch_save(value: Any, destination: str | Path) -> None:
    torch = require_torch()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def torch_load(path: str | Path, *, map_location: Any = "cpu") -> dict:
    torch = require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected PyTorch artifact payload: {path}")
    return payload


class DeterministicBatchStream:
    """Serializable shuffled row stream with exact CPU-generator resume."""

    def __init__(self, length: int, seed: int, state: Mapping[str, Any] | None = None):
        torch = require_torch()
        if int(length) <= 0:
            raise ValueError("Training split must contain at least one row")
        self.length = int(length)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.epoch = 0
        self.position = 0
        self.order = torch.randperm(self.length, generator=self.generator)
        if state is not None:
            self.load_state_dict(state)

    def next(self, count: int) -> list[int]:
        torch = require_torch()
        requested = int(count)
        if requested <= 0:
            raise ValueError("Batch size must be positive")
        result: list[int] = []
        while len(result) < requested:
            available = self.length - self.position
            take = min(requested - len(result), available)
            result.extend(int(value) for value in self.order[self.position : self.position + take])
            self.position += take
            if self.position == self.length:
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(self.length, generator=self.generator)
        return result

    def state_dict(self) -> dict:
        return {
            "length": self.length,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        torch = require_torch()
        if int(state.get("length", -1)) != self.length:
            raise RuntimeError("Batch-stream row count changed across resume")
        order = torch.as_tensor(state["order"], dtype=torch.long).clone()
        if tuple(order.shape) != (self.length,):
            raise RuntimeError("Invalid resumed batch-stream order")
        position = int(state["position"])
        if not 0 <= position <= self.length:
            raise RuntimeError("Invalid resumed batch-stream position")
        self.epoch = int(state["epoch"])
        self.position = position
        self.order = order
        self.generator.set_state(
            torch.as_tensor(state["generator_state"], dtype=torch.uint8).cpu()
        )


def resolve_precision(device: str, requested: str) -> tuple[str, Any | None]:
    torch = require_torch()
    if requested not in ("auto", "bf16", "fp32"):
        raise ValueError("Training precision must be auto, bf16, or fp32")
    if not str(device).startswith("cuda"):
        if requested == "bf16":
            raise RuntimeError("Explicit BF16 adaptation requires CUDA")
        return "fp32", None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return "bf16", torch.bfloat16
    return "fp32", None


def autocast_context(device: str, dtype: Any | None):
    torch = require_torch()
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def effective_number_weights(
    labels: Sequence[int], num_classes: int, beta: float
) -> tuple[Any, Any]:
    torch = require_torch()
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("class_weight_beta must lie in [0,1)")
    counts = torch.bincount(
        torch.as_tensor(list(labels), dtype=torch.long), minlength=int(num_classes)
    ).to(dtype=torch.float64)
    if bool((counts <= 0).any()):
        raise RuntimeError(f"Every training fold must contain all classes; counts={counts.tolist()}")
    if float(beta) == 0.0:
        weights = torch.ones_like(counts)
    else:
        weights = (1.0 - float(beta)) / (1.0 - torch.pow(float(beta), counts))
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32), counts.to(dtype=torch.long)


def optimizer_from_groups(groups: list[dict], parameters: Mapping[str, Any]):
    torch = require_torch()
    name = str(parameters["optimizer"]).lower()
    if name == "adamw":
        betas = tuple(float(value) for value in parameters.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(
            groups,
            betas=betas,
            eps=float(parameters.get("epsilon", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(parameters.get("momentum", 0.9)),
            nesterov=bool(parameters.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer: {parameters['optimizer']!r}")


def scheduler_from_optimizer(optimizer: Any, parameters: Mapping[str, Any], max_steps: int):
    torch = require_torch()
    schedule = str(parameters.get("scheduler", "cosine")).lower()
    warmup = int(parameters.get("warmup_steps", 0))
    power = float(parameters.get("polynomial_power", 1.0))
    minimum_ratio = float(parameters.get("minimum_lr_ratio", 0.0))
    if warmup < 0 or warmup >= int(max_steps):
        raise ValueError("warmup_steps must be non-negative and smaller than max_steps")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must lie in [0,1]")

    def multiplier(step: int) -> float:
        current = min(max(int(step), 0), int(max_steps))
        if warmup and current < warmup:
            return max(float(current + 1) / float(warmup), 1.0 / float(warmup))
        progress = (current - warmup) / max(1, int(max_steps) - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule == "polynomial":
            value = (1.0 - progress) ** power
        elif schedule == "constant":
            value = 1.0
        else:
            raise ValueError(f"Unsupported scheduler: {schedule!r}")
        return minimum_ratio + (1.0 - minimum_ratio) * value

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def trainable_state_dict(model: Any) -> dict[str, Any]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    selected: dict[str, Any] = {}
    for key, value in state.items():
        if key in trainable or any(key.startswith(f"{name}.") for name in trainable):
            selected[key] = value.detach().cpu().clone()
    # Buffers belonging to a trainable module (for example normalization
    # counters) are retained even though they are not named parameters.
    trainable_modules = {
        name.rsplit(".", 1)[0] if "." in name else "" for name in trainable
    }
    for key, value in state.items():
        if key in selected:
            continue
        if any(module and key.startswith(f"{module}.") for module in trainable_modules):
            selected[key] = value.detach().cpu().clone()
    if not selected:
        raise RuntimeError("Adaptation model exposes no trainable state")
    return selected


def load_trainable_state_dict(model: Any, state: Mapping[str, Any]) -> None:
    expected = set(trainable_state_dict(model))
    observed = set(state)
    if expected != observed:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "Trainable-state architecture mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    full = model.state_dict()
    for key, value in state.items():
        full[key] = value
    model.load_state_dict(full, strict=True)


def torch_rng_state() -> dict:
    torch = require_torch()
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    torch = require_torch()
    torch.set_rng_state(torch.as_tensor(state["cpu"], dtype=torch.uint8).cpu())
    cuda_states = list(state.get("cuda", []))
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG topology changed across resume")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_states]
        )


__all__ = [
    "DeterministicBatchStream",
    "atomic_torch_save",
    "autocast_context",
    "effective_number_weights",
    "load_trainable_state_dict",
    "optimizer_from_groups",
    "require_torch",
    "resolve_precision",
    "restore_torch_rng_state",
    "scheduler_from_optimizer",
    "torch_load",
    "torch_rng_state",
    "trainable_state_dict",
]
