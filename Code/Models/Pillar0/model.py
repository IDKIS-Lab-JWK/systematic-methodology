"""Official Pillar-0 checkpoint adapter and five-class classifier."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Support both ``python file.py`` and package-style imports.
    from .windows import apply_pillar_ct_windows
except ImportError:  # pragma: no cover - exercised by command-line entry points
    from windows import apply_pillar_ct_windows


MODEL_REPO_ID = "YalaLab/Pillar0-AbdomenCT"
MODEL_REVISION = "d73d111bb278b3f6411f3c8e72c7e6f8a5fb1bb8"
MODEL_ACCESS_URL = "https://huggingface.co/YalaLab/Pillar0-AbdomenCT"
MODALITY = "abdomen_ct"
FEATURE_DIM = 1152
EXPECTED_VOLUME_SHAPE = (384, 384, 384)


class PillarModelError(RuntimeError):
    """Base exception for explicit Pillar-0 setup/model failures."""


class PillarAccessError(PillarModelError):
    """Raised when the user has not accepted the gated checkpoint terms."""


class ZeroShotCapabilityError(PillarModelError):
    """Raised instead of constructing an invalid zero-shot projection."""


def _gate_message(repo_id: str) -> str:
    return (
        f"Access to {repo_id} is gated. Sign in at {MODEL_ACCESS_URL}, accept "
        "the Pillar-0 terms/share-contact gate with the Hugging Face account "
        "used by this environment, then retry. MedSigLIP access is separate."
    )


def _looks_like_gate_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "gatedrepo",
            "gated repo",
            "access to model",
            "cannot access gated repo",
            "403 forbidden",
            "401 unauthorized",
        )
    )


def checkpoint_access_report(
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
) -> dict:
    """Attempt an authenticated config fetch and return a token-safe report."""

    report = {
        "repo_id": repo_id,
        "revision": revision,
        "access_url": MODEL_ACCESS_URL,
        "accessible": False,
        "status": "unknown",
    }
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )
        report.update(
            {
                "accessible": True,
                "status": "accessible",
                "resolved_config": str(Path(path).resolve()),
            }
        )
    except Exception as exc:  # Hugging Face exception classes vary by version.
        if _looks_like_gate_error(exc):
            report.update({"status": "gated_not_accepted", "action": _gate_message(repo_id)})
        else:
            report.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return report


def load_official_model(
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    local_files_only: bool = False,
):
    """Load the pinned official custom-code checkpoint.

    The caller controls device placement so zero-shot can encode text and
    images sequentially on a 24-GB GPU.  No Hugging Face token is ever read or
    logged by this module; the hub library uses its normal credential store.
    """

    try:
        from transformers import AutoModel

        kwargs = {
            "revision": revision,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if dtype is not None:
            # ``dtype`` is the current transformers spelling.  The released
            # custom class may still consume ``torch_dtype`` internally.
            kwargs["torch_dtype"] = dtype
        model = AutoModel.from_pretrained(repo_id, **kwargs)
    except Exception as exc:
        if _looks_like_gate_error(exc):
            raise PillarAccessError(_gate_message(repo_id)) from exc
        raise PillarModelError(
            f"Could not load {repo_id}@{revision}: {type(exc).__name__}: {exc}"
        ) from exc
    model.to(device)
    return model


def _model_roots(model) -> Iterable[tuple[str, object]]:
    """Yield shallow wrapper roots without recursively traversing the network."""

    queue = [("model", model)]
    seen = set()
    while queue:
        name, root = queue.pop(0)
        if id(root) in seen:
            continue
        seen.add(id(root))
        yield name, root
        if len(name.split(".")) >= 4:
            continue
        for child_name in ("model", "module", "clip"):
            child = getattr(root, child_name, None)
            if child is not None:
                queue.append((f"{name}.{child_name}", child))


def _as_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        features = output
    elif isinstance(output, dict):
        for key in ("image_features", "vision_features", "features", "hidden"):
            if isinstance(output.get(key), torch.Tensor):
                features = output[key]
                break
        else:
            raise PillarModelError(
                f"Vision output dictionary has no recognized tensor key: {sorted(output)}"
            )
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        features = output[0]
    else:
        raise PillarModelError(f"Unsupported Pillar-0 vision output: {type(output).__name__}")
    if features.ndim > 2:
        features = features.flatten(start_dim=2).mean(dim=2)
    if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
        raise PillarModelError(
            f"Expected [batch,{FEATURE_DIM}] vision features; received {tuple(features.shape)}."
        )
    return features


def extract_vision_features(model, raw_hu, *, normalize: bool = False) -> torch.Tensor:
    """Apply the official windows and call a differentiable released-model API."""

    windowed = apply_pillar_ct_windows(raw_hu)
    errors = []
    for path, root in _model_roots(model):
        method = getattr(root, "extract_vision_feats", None)
        if callable(method):
            try:
                features = _as_feature_tensor(method({MODALITY: windowed}))
                return F.normalize(features, dim=-1) if normalize else features
            except Exception as exc:
                errors.append(f"{path}.extract_vision_feats: {type(exc).__name__}: {exc}")
        method = getattr(root, "encode_image", None)
        if callable(method):
            for payload in ({MODALITY: windowed}, windowed):
                try:
                    features = _as_feature_tensor(method(payload, normalize=normalize))
                    return F.normalize(features, dim=-1) if normalize else features
                except Exception as exc:
                    errors.append(f"{path}.encode_image: {type(exc).__name__}: {exc}")
    raise PillarModelError(
        "The released model exposed no working differentiable vision feature "
        "API. Attempts: " + " | ".join(errors[-6:])
    )


def _module_shape(module) -> tuple[int | None, int | None]:
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    first = next((m for m in module.modules() if isinstance(m, nn.Linear)), None)
    last = next((m for m in reversed(list(module.modules())) if isinstance(m, nn.Linear)), None)
    if first is not None and last is not None:
        return int(first.in_features), int(last.out_features)
    if isinstance(module, torch.Tensor) and module.ndim == 2:
        return int(module.shape[0]), int(module.shape[1])
    return None, None


def zero_shot_capability_report(model) -> dict:
    """Inspect only released, trained components required for zero-shot use."""

    candidates = []
    usable = []
    for path, root in _model_roots(model):
        for name in ("text", "text_projection", "text_projector"):
            value = getattr(root, name, None)
            if value is None:
                continue
            in_dim, out_dim = _module_shape(value)
            record = {
                "path": f"{path}.{name}",
                "type": type(value).__name__,
                "input_dim": in_dim,
                "output_dim": out_dim,
            }
            candidates.append(record)
            if in_dim == 4096 and out_dim == FEATURE_DIM and callable(value):
                usable.append(record)
        if callable(getattr(root, "encode_text", None)):
            candidates.append(
                {
                    "path": f"{path}.encode_text",
                    "type": "method",
                    "input_dim": "runtime_check_required",
                    "output_dim": "runtime_check_required",
                }
            )
    parameter_names = []
    if hasattr(model, "named_parameters"):
        parameter_names = [
            name
            for name, _ in model.named_parameters()
            if any(marker in name.lower() for marker in ("text", "logit_scale", "logit_bias"))
        ]
    return {
        "supported_by_paper": False,
        "status": "candidate_projection_present" if usable else "runtime_probe_required",
        "projection_candidates": candidates,
        "dimension_verified_candidates": usable,
        "matching_parameter_names": parameter_names,
        "required_raw_text_dim": 4096,
        "required_aligned_dim": FEATURE_DIM,
        "required_checkpoint_logit_scale": True,
        "warning": (
            "Pillar-0 did not report zero-shot classification. A run is valid only "
            "if both the trained 4096-to-1152 text projection and the trained "
            "checkpoint logit scale pass runtime probes."
        ),
    }


def project_qwen_embeddings(model, raw_embeddings: torch.Tensor) -> torch.Tensor:
    """Use a released trained projection or fail; never invent a projection."""

    if raw_embeddings.ndim != 2 or raw_embeddings.shape[-1] != 4096:
        raise ZeroShotCapabilityError(
            f"Expected Qwen mean-pooled [batch,4096] embeddings, got {tuple(raw_embeddings.shape)}."
        )
    errors = []
    for path, root in _model_roots(model):
        method = getattr(root, "encode_text", None)
        if callable(method):
            try:
                output = method(raw_embeddings, normalize=True)
                if isinstance(output, torch.Tensor) and output.shape == (raw_embeddings.shape[0], FEATURE_DIM):
                    return F.normalize(output, dim=-1)
            except Exception as exc:
                errors.append(f"{path}.encode_text: {type(exc).__name__}: {exc}")
        for name in ("text", "text_projection", "text_projector"):
            projection = getattr(root, name, None)
            in_dim, out_dim = _module_shape(projection) if projection is not None else (None, None)
            if not callable(projection) or (in_dim, out_dim) != (4096, FEATURE_DIM):
                continue
            try:
                output = projection(raw_embeddings)
                if isinstance(output, torch.Tensor) and output.shape[-1] == FEATURE_DIM:
                    return F.normalize(output, dim=-1)
            except Exception as exc:
                errors.append(f"{path}.{name}: {type(exc).__name__}: {exc}")
    raise ZeroShotCapabilityError(
        "unsupported_missing_text_projection: the released checkpoint did not "
        "expose a working trained 4096-to-1152 projection. Raw Qwen and image "
        "vectors cannot be compared. Attempts: " + " | ".join(errors[-5:])
    )


def checkpoint_logit_scale(model) -> float | None:
    for _, root in _model_roots(model):
        value = getattr(root, "logit_scale", None)
        if value is not None:
            with torch.no_grad():
                scalar = value.detach().float().reshape(-1).mean().item()
            # Pretraining stores log(scale); custom wrappers sometimes expose scale.
            return float(torch.exp(torch.tensor(scalar)).item()) if scalar <= 6.0 else scalar
    return None


def set_gradient_checkpointing(model, enabled: bool) -> bool:
    if not enabled:
        # Freshly loaded official checkpoints start with checkpointing off.
        # Do not call a generic ``gradient_checkpointing_enable`` method when
        # the requested state is false.
        return False
    for _, root in _model_roots(model):
        for name in ("gradient_checkpointing_enable", "set_grad_checkpointing"):
            method = getattr(root, name, None)
            if callable(method):
                try:
                    method() if name.endswith("enable") else method(enabled)
                    return True
                except (AttributeError, TypeError, ValueError, NotImplementedError):
                    continue
    return False


class PillarClassifier(nn.Module):
    """Pillar-0 encoder plus a true five-logit linear classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 5,
        frozen_encoder: bool = True,
        dropout: float = 0.0,
        normalize_features: bool = False,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1).")
        self.encoder = encoder
        self.frozen_encoder = bool(frozen_encoder)
        self.normalize_features = bool(normalize_features)
        self.classifier = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(float(dropout)),
            nn.Linear(FEATURE_DIM, int(num_classes)),
        )
        named_encoder_parameters = list(self.encoder.named_parameters())
        has_named_vision_tower = any(
            ".visual." in f".{name}." for name, _ in named_encoder_parameters
        )
        for name, parameter in named_encoder_parameters:
            if self.frozen_encoder:
                parameter.requires_grad = False
            elif has_named_vision_tower:
                # The released wrapper also contains the frozen text projection
                # and logit scale. They are unused by supervised image
                # classification and must not be mislabeled as fine-tuned.
                parameter.requires_grad = ".visual." in f".{name}."

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self

    def encode(self, raw_hu) -> torch.Tensor:
        context = torch.no_grad() if self.frozen_encoder else contextlib.nullcontext()
        with context:
            features = extract_vision_features(
                self.encoder,
                raw_hu,
                normalize=self.normalize_features,
            )
        if not self.frozen_encoder and torch.is_grad_enabled() and not features.requires_grad:
            raise PillarModelError(
                "Full fine-tuning requested, but the checkpoint's selected vision "
                "path detached its output. Refusing a head-only run mislabeled as "
                "end-to-end fine-tuning."
            )
        return features

    def forward(self, raw_hu) -> torch.Tensor:
        return self.classifier(self.encode(raw_hu))


@dataclass(frozen=True)
class TrainableParameterReport:
    trainable: int
    frozen: int
    trainable_names: tuple[str, ...]


def trainable_parameter_report(model: nn.Module) -> TrainableParameterReport:
    trainable_names = tuple(name for name, p in model.named_parameters() if p.requires_grad)
    return TrainableParameterReport(
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        trainable_names=trainable_names,
    )
