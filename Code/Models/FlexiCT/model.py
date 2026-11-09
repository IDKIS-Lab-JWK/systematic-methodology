"""Strict vision-only loader for the FlexiCT-3D-VLM checkpoint."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from . import (
        EXPECTED_PATCH_TOKENS,
        FEATURE_DIM,
        INPUT_SHAPE_DHW,
        PATCH_SIZE,
        UPSTREAM_REVISION,
    )
    from .provisioning import validate_upstream
except ImportError:  # pragma: no cover
    from __init__ import (
        EXPECTED_PATCH_TOKENS,
        FEATURE_DIM,
        INPUT_SHAPE_DHW,
        PATCH_SIZE,
        UPSTREAM_REVISION,
    )
    from provisioning import validate_upstream


# Copied verbatim from ``flexi_ct/flexi_ct_2d.py`` at UPSTREAM_REVISION.  Keeping
# this small constructor contract locally lets the vision loader avoid importing
# ``flexi_ct.__init__``.  The latter eagerly imports Flexi_CT_VLM and its
# Transformers/Qwen text tower, which is irrelevant to a frozen image probe.
_BACKBONE_KWARGS = {
    "patch_size": PATCH_SIZE,
    "in_chans": 1,
    "n_storage_tokens": 4,
    "qkv_bias": False,
    "mask_k_bias": True,
    "drop_path_rate": 0.2,
    "layerscale_init": 1e-5,
}
_VISION_DIM = 864
_JOINT_DIM = 2 * _VISION_DIM
_PINNED_TORCH_VERSION = "2.8.0"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for FlexiCT feature extraction") from exc
    return torch, nn, functional


def _validate_torch_runtime(torch: Any) -> None:
    observed = str(torch.__version__).split("+", 1)[0]
    if observed != _PINNED_TORCH_VERSION:
        raise RuntimeError(
            "FlexiCT source loading requires the pinned torch "
            f"{_PINNED_TORCH_VERSION} runtime, observed {torch.__version__}. "
            "Use the dedicated .venv_flexict environment."
        )


def resolve_precision(device: str, requested: str) -> tuple[str, Any | None]:
    torch, _, _ = _require_torch()
    requested = str(requested).lower()
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {requested}")
    if not str(device).lower().startswith("cuda"):
        if requested not in {"auto", "fp32"}:
            raise RuntimeError(f"{requested.upper()} extraction requires a CUDA device")
        return "fp32", None
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        # BF16 is the official extraction precision.  FP32 is the conservative
        # fallback; silently falling back to FP16 would introduce an unvalidated
        # numerical recipe and can overflow on CT feature extraction.
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but torch.cuda.is_bf16_supported() is false")
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16
    if requested == "fp32":
        return "fp32", None
    raise AssertionError("precision validation fell through")


def vision_encoder_contract() -> dict:
    """Return the immutable, checkpoint-independent vision recipe."""

    return {
        "vision_dimension": _VISION_DIM,
        "joint_dimension": _JOINT_DIM,
        "projection_dimension": FEATURE_DIM,
        "input_shape_dhw": list(INPUT_SHAPE_DHW),
        "patch_size": PATCH_SIZE,
        "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
        "backbone_kwargs": dict(_BACKBONE_KWARGS),
        "representation": "normalized CLS + mean normalized patch -> VLM projection -> L2",
        "text_tower_loaded": False,
    }


def _official_models_module(root: Path):
    """Load only the pinned ``models.py`` under an isolated package alias.

    Importing ``flexi_ct.models`` normally executes the upstream package's
    eager ``__init__`` and imports the Qwen/Transformers text tower.  An
    isolated package with the official directory as ``__path__`` preserves
    all relative imports used by ``models.py`` without executing that
    initializer or mutating the canonical ``flexi_ct`` package.
    """

    package_dir = root / "flexi_ct"
    models_path = package_dir / "models.py"
    if not models_path.is_file():
        raise RuntimeError(f"Pinned FlexiCT models.py is missing: {models_path}")
    root_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    package_name = f"_systematic_flexict_{root_hash}"
    module_name = f"{package_name}.models"

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(package_dir / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_dir)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
    elif list(getattr(package, "__path__", [])) != [str(package_dir)]:
        raise RuntimeError(f"Isolated FlexiCT package alias collision for {root}")

    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, models_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create an import spec for {models_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    observed = Path(module.__file__).resolve()
    if observed != models_path.resolve():
        raise RuntimeError(f"Loaded FlexiCT models from an unexpected location: {observed}")
    return module


def _build_official_backbone(root: Path):
    models_module = _official_models_module(root)
    factory = getattr(models_module, "flexi_ct_backbone_base", None)
    if not callable(factory):
        raise RuntimeError("Pinned FlexiCT models.py has no flexi_ct_backbone_base factory")
    return factory(**_BACKBONE_KWARGS)


def build_vision_only_encoder(
    *,
    upstream_root: str | Path,
    checkpoint_path: str | Path,
    device: str,
    allow_trusted_legacy_checkpoint: bool = False,
    trusted_checkpoint_sha256: str | None = None,
) -> tuple[Any, dict]:
    torch, nn, functional = _require_torch()
    _validate_torch_runtime(torch)
    source_identity = validate_upstream(upstream_root, UPSTREAM_REVISION)
    root = Path(upstream_root).resolve()
    backbone = _build_official_backbone(root)
    projection = nn.Linear(_JOINT_DIM, FEATURE_DIM, bias=False)
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_trusted_legacy_checkpoint:
            raise RuntimeError(
                "FlexiCT checkpoint could not be loaded in PyTorch weights-only mode. "
                "The official release's safe-mode compatibility is not yet validated; "
                "unsafe pickle fallback remains disabled"
            ) from exc
        if not trusted_checkpoint_sha256:
            raise RuntimeError(
                "Trusted legacy checkpoint loading requires an expected SHA-256"
            ) from exc
        observed_sha256 = _sha256_file(checkpoint)
        if observed_sha256.lower() != trusted_checkpoint_sha256.lower():
            raise RuntimeError(
                "Trusted legacy checkpoint SHA-256 mismatch: "
                f"expected {trusted_checkpoint_sha256}, observed {observed_sha256}"
            ) from exc
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_load_mode = "trusted_pickle_opt_in_after_sha256"
    else:
        checkpoint_load_mode = "torch_weights_only"
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("FlexiCT VLM checkpoint must contain a 'model' state dictionary")
    state = payload["model"]
    vision_state = {
        key[len("vision_model.") :]: value
        for key, value in state.items()
        if key.startswith("vision_model.")
    }
    if not vision_state:
        raise RuntimeError(
            "FlexiCT vision state mismatch: checkpoint contains no vision_model.* tensors"
        )
    expected_vision_keys = set(backbone.state_dict())
    observed_vision_keys = set(vision_state)
    if observed_vision_keys != expected_vision_keys:
        missing = sorted(expected_vision_keys - observed_vision_keys)
        unexpected = sorted(observed_vision_keys - expected_vision_keys)
        raise RuntimeError(
            "FlexiCT vision state mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    backbone.load_state_dict(vision_state, strict=True)
    projection_key = "vlm_vision_projection.weight"
    if projection_key not in state:
        raise RuntimeError(f"FlexiCT checkpoint is missing {projection_key}")
    weight = state[projection_key]
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"FlexiCT {projection_key} is not a tensor")
    if tuple(weight.shape) != (FEATURE_DIM, _JOINT_DIM):
        raise RuntimeError(
            "Unexpected FlexiCT projection shape "
            f"{tuple(weight.shape)}; expected {(FEATURE_DIM, _JOINT_DIM)}"
        )
    projection.load_state_dict({"weight": weight}, strict=True)

    class VisionOnly(nn.Module):
        def __init__(self, vision, project):
            super().__init__()
            self.vision = vision
            self.project = project

        def forward(self, images):
            outputs = self.vision(images, is_training=True)
            if not isinstance(outputs, Mapping):
                raise RuntimeError("FlexiCT backbone did not return the training-output mapping")
            try:
                cls = outputs["x_norm_clstoken"]
                patches = outputs["x_norm_patchtokens"]
            except KeyError as exc:
                raise RuntimeError(f"FlexiCT backbone output is missing {exc.args[0]}") from exc
            if cls.ndim != 2 or cls.shape[-1] != _VISION_DIM:
                raise RuntimeError(f"Unexpected FlexiCT CLS-token shape: {tuple(cls.shape)}")
            if patches.ndim != 3 or patches.shape[-1] != _VISION_DIM:
                raise RuntimeError(f"Unexpected FlexiCT patch-token shape: {tuple(patches.shape)}")
            if cls.shape[0] != patches.shape[0]:
                raise RuntimeError("FlexiCT CLS and patch-token batch dimensions differ")
            if patches.shape[1] != EXPECTED_PATCH_TOKENS:
                raise RuntimeError(
                    f"Expected {EXPECTED_PATCH_TOKENS} patch tokens, got {patches.shape[1]}"
                )
            joint = torch.cat([cls, patches.mean(dim=1)], dim=-1)
            return functional.normalize(self.project(joint), dim=-1)

    model = VisionOnly(backbone, projection).to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("FlexiCT encoder was not fully frozen")
    return model, {
        "source": source_identity,
        "checkpoint_load_mode": checkpoint_load_mode,
        **vision_encoder_contract(),
    }


def extract_embedding(model: Any, image: Any, *, device: str, precision: str):
    torch, _, _ = _require_torch()
    resolved_name, dtype = resolve_precision(device, precision)
    if not isinstance(image, torch.Tensor):
        raise TypeError("FlexiCT preprocessed image must be a torch.Tensor")
    if tuple(image.shape) != (1, *INPUT_SHAPE_DHW):
        raise RuntimeError(
            "FlexiCT preprocessed image must have shape "
            f"{(1, *INPUT_SHAPE_DHW)}, got {tuple(image.shape)}"
        )
    if image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise RuntimeError("FlexiCT preprocessed image must be finite float32")
    if getattr(model, "training", False):
        raise RuntimeError("FlexiCT encoder must remain in eval mode during extraction")
    images = image.unsqueeze(0).to(device, non_blocking=False)
    with torch.inference_mode():
        if dtype is None:
            embedding = model(images)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                embedding = model(images)
    embedding = embedding.float().cpu().contiguous()
    if tuple(embedding.shape) != (1, FEATURE_DIM):
        raise RuntimeError(f"Unexpected FlexiCT embedding shape: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise RuntimeError("FlexiCT embedding contains non-finite values")
    norm = float(torch.linalg.vector_norm(embedding, dim=-1).item())
    if not (0.995 <= norm <= 1.005):
        raise RuntimeError(f"FlexiCT embedding is not L2-normalized: norm={norm}")
    return embedding[0], {"precision": resolved_name, "l2_norm": norm}


__all__ = [
    "build_vision_only_encoder",
    "extract_embedding",
    "resolve_precision",
    "vision_encoder_contract",
]
