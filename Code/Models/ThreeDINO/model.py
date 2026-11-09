"""Strict loading and feature extraction for provisioned official 3DINO assets.

No upstream source or checkpoint is vendored here. Automatic acquisition is
centralized in :mod:`provisioning`; this adapter accepts resolved local paths
and rejects missing or incompatible weights, avoiding the official helper's
random-initialization fallback when a path is wrong.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Torch is a core runtime dependency but remains optional for dry imports.
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in dependency preflight.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:  # Package imports.
    from . import (
        FEATURE_DIM,
        INPUT_SIZE,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import (
        FEATURE_DIM,
        INPUT_SIZE,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )


_ModuleBase = nn.Module if nn is not None else object
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_REPRESENTATIONS = (
    (1, False),
    (1, True),
    (4, False),
    (4, True),
)


class UpstreamValidationError(RuntimeError):
    """The supplied external source tree is absent or structurally incompatible."""


class CheckpointValidationError(RuntimeError):
    """The supplied pretrained checkpoint cannot be trusted or loaded."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Resolved local checkpoint provenance."""

    path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "size_bytes": int(self.size_bytes),
            "sha256": self.sha256,
            # A byte-verified local file does not by itself prove a Hub origin
            # or revision. The provisioner records those independently.
            "filename": self.path.name,
            "download_performed": False,
        }


def feature_dimension(
    n_last_blocks: int, use_mean_patch_token: bool = False
) -> int:
    """Return the publication probe dimension for one feature recipe."""

    blocks = int(n_last_blocks)
    if blocks not in (1, 4):
        raise ValueError(
            f"3DINO linear probing supports 1 or 4 last blocks, got {blocks}"
        )
    return FEATURE_DIM * (blocks + int(bool(use_mean_patch_token)))


def validate_upstream_checkout(
    upstream_root: str | Path, upstream_revision: str
) -> dict:
    """Validate an external source tree against the immutable revision argument.

    This intentionally does not execute Git.  The exact revision must be
    supplied by the caller, and the required official files must be present.
    """

    revision = str(upstream_revision).strip().lower()
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise UpstreamValidationError(
            "upstream_revision must be a full immutable 40-character commit SHA"
        )
    if revision != UPSTREAM_REVISION:
        raise UpstreamValidationError(
            "Unsupported 3DINO source revision: expected "
            f"{UPSTREAM_REVISION}, found {revision}"
        )
    root = Path(upstream_root).resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"External 3DINO source directory not found: {root}")
    required = (
        "LICENSE",
        "dinov2/__init__.py",
        "dinov2/configs/__init__.py",
        "dinov2/configs/train/vit3d_highres.yaml",
        "dinov2/models/__init__.py",
        "dinov2/models/vision_transformer.py",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise UpstreamValidationError(
            f"External 3DINO source tree is incomplete at {root}; missing {missing}"
        )
    return {
        "root": str(root),
        "repository": UPSTREAM_REPOSITORY,
        "revision": revision,
        "revision_claim_verified_from_scm": False,
        "validation_scope": (
            "caller-supplied pinned revision string, required-file presence, "
            "import origin, and runtime architecture; no Git/SCM attestation"
        ),
        "required_files_present": True,
        "git_invoked": False,
        "source_vendored": False,
    }


def resolve_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CheckpointInfo:
    """Resolve and hash a non-empty local checkpoint without downloading it."""

    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(
            "Official 3DINO checkpoint is required; random initialization is "
            f"disabled. Missing file: {path}"
        )
    size = int(path.stat().st_size)
    if size <= 0:
        raise CheckpointValidationError(f"Checkpoint is empty: {path}")
    actual = _sha256_file(path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if not _SHA256.fullmatch(expected):
            raise ValueError("expected_sha256 must be a 64-character hex digest")
        if actual != expected:
            raise CheckpointValidationError(
                f"Checkpoint SHA-256 mismatch for {path}: expected {expected}, "
                f"found {actual}"
            )
    return CheckpointInfo(path=path, size_bytes=size, sha256=actual)


def load_official_model(
    *,
    upstream_root: str | Path,
    upstream_revision: str,
    checkpoint_path: str | Path,
    device: Any = "cpu",
    config_name: str = "train/vit3d_highres",
    expected_checkpoint_sha256: str | None = None,
) -> Any:
    """Construct the official ViT-L and strictly load meaningful pretrained weights."""

    _require_torch()
    upstream = validate_upstream_checkout(upstream_root, upstream_revision)
    checkpoint = resolve_checkpoint(
        checkpoint_path, expected_sha256=expected_checkpoint_sha256
    )
    root = Path(upstream["root"])
    _activate_external_upstream(root)

    try:
        config_module = importlib.import_module("dinov2.configs")
        models_module = importlib.import_module("dinov2.models")
        config_loader = getattr(config_module, "load_and_merge_config_3d")
        model_builder = getattr(models_module, "build_model_from_cfg")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Could not import the official pinned 3DINO source tree. Install "
            "its dependencies in the isolated 3DINO environment."
        ) from exc
    _require_module_under_root(config_module, root)
    _require_module_under_root(models_module, root)

    config = config_loader(str(config_name))
    built = model_builder(config, only_teacher=True)
    model = built[0] if isinstance(built, (tuple, list)) else built
    if not isinstance(model, nn.Module):
        raise UpstreamValidationError(
            f"Official builder returned {type(model).__name__}, not torch.nn.Module"
        )
    _validate_official_architecture(model)

    load_report = _load_checkpoint_fail_closed(model, checkpoint.path)
    model.eval()
    model.to(torch.device(device))
    model._threedino_provenance = {  # type: ignore[attr-defined]
        "upstream": upstream,
        "checkpoint": checkpoint.to_dict(),
        "config_name": str(config_name),
        "load_report": load_report,
        "random_fallback_allowed": False,
    }
    return model


def extract_representation_bank(
    model: Any,
    images: Any,
    representations: Iterable[tuple[int, bool]] = DEFAULT_REPRESENTATIONS,
) -> dict[tuple[int, bool], Any]:
    """Extract any publication feature recipes with one encoder forward pass."""

    _require_torch()
    requested: list[tuple[int, bool]] = []
    for recipe in representations:
        if not isinstance(recipe, (tuple, list)) or len(recipe) != 2:
            raise ValueError(
                "Each representation must be (n_last_blocks, use_mean_patch_token)"
            )
        key = (int(recipe[0]), bool(recipe[1]))
        feature_dimension(*key)  # validates the supported block counts.
        if key not in requested:
            requested.append(key)
    if not requested:
        raise ValueError("At least one representation recipe is required")
    if not hasattr(images, "ndim") or int(images.ndim) != 5:
        shape = getattr(images, "shape", None)
        raise ValueError(
            f"Expected 3DINO images shaped (batch, 1, D, H, W), got {shape}"
        )
    if int(images.shape[1]) != 1:
        raise ValueError(f"3DINO expects one input channel, got {images.shape[1]}")

    encoder = _unwrap_encoder(model)
    method = getattr(encoder, "get_intermediate_layers", None)
    if not callable(method):
        raise AttributeError(
            "3DINO encoder lacks the official get_intermediate_layers method"
        )
    max_blocks = max(recipe[0] for recipe in requested)
    keyword_arguments = {
        "n": max_blocks,
        "reshape": False,
        "return_class_token": True,
        "norm": True,
    }
    try:
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            keyword_arguments = {
                key: value
                for key, value in keyword_arguments.items()
                if key in signature.parameters
            }
    except (TypeError, ValueError):
        pass
    output = method(images, **keyword_arguments)
    pairs = _normalise_intermediate_output(output, expected_count=max_blocks)
    class_tokens = [pair[1] for pair in pairs]
    final_patch_tokens = pairs[-1][0]
    _validate_tokens(class_tokens, final_patch_tokens, images)

    bank: dict[tuple[int, bool], Any] = {}
    pooled_patch = final_patch_tokens.mean(dim=1)
    for blocks, use_mean in requested:
        components = list(class_tokens[-blocks:])
        if use_mean:
            components.append(pooled_patch)
        feature = torch.cat(components, dim=-1).reshape(images.shape[0], -1).float()
        expected = feature_dimension(blocks, use_mean)
        if tuple(feature.shape) != (int(images.shape[0]), expected):
            raise RuntimeError(
                f"3DINO feature recipe {(blocks, use_mean)} should produce "
                f"(batch, {expected}), got {tuple(feature.shape)}"
            )
        bank[(blocks, use_mean)] = feature
    return bank


def extract_intermediate_features(
    model: Any,
    images: Any,
    *,
    n_last_blocks: int = 1,
    use_mean_patch_token: bool = False,
) -> Any:
    """Extract one frozen-probe-compatible representation."""

    key = (int(n_last_blocks), bool(use_mean_patch_token))
    return extract_representation_bank(model, images, [key])[key]


class ThreeDINOClassifier(_ModuleBase):
    """Five-class head on official 3DINO intermediate representations."""

    def __init__(
        self,
        encoder: Any,
        num_classes: int = 5,
        n_last_blocks: int = 1,
        use_mean_patch_token: bool = False,
        freeze_encoder: bool = True,
        dropout: float = 0.0,
    ) -> None:
        _require_torch()
        super().__init__()
        if int(num_classes) <= 1:
            raise ValueError(f"num_classes must be greater than one, got {num_classes}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.encoder = encoder
        self.n_last_blocks = int(n_last_blocks)
        self.use_mean_patch_token = bool(use_mean_patch_token)
        self.frozen_encoder = bool(freeze_encoder)
        dimension = feature_dimension(
            self.n_last_blocks, self.use_mean_patch_token
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not self.frozen_encoder
        if self.frozen_encoder:
            self.encoder.eval()
        if float(dropout) > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(float(dropout)), nn.Linear(dimension, int(num_classes))
            )
            linear = self.classifier[-1]
        else:
            self.classifier = nn.Linear(dimension, int(num_classes))
            linear = self.classifier
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear.bias)

    def train(self, mode: bool = True) -> Any:
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self

    def encode(self, images: Any) -> Any:
        context = torch.no_grad() if self.frozen_encoder else nullcontext()
        with context:
            return extract_intermediate_features(
                self.encoder,
                images,
                n_last_blocks=self.n_last_blocks,
                use_mean_patch_token=self.use_mean_patch_token,
            )

    def forward(self, images: Any) -> Any:
        return self.classifier(self.encode(images))

    def forward_features(self, features: Any) -> Any:
        """Apply the head to already materialized frozen-encoder features."""

        if not self.frozen_encoder:
            raise RuntimeError(
                "Precomputed features are valid only for a frozen encoder"
            )
        expected = feature_dimension(
            self.n_last_blocks, self.use_mean_patch_token
        )
        if getattr(features, "ndim", None) != 2 or int(features.shape[1]) != expected:
            raise RuntimeError(
                f"Expected features shaped (batch, {expected}), got "
                f"{getattr(features, 'shape', None)}"
            )
        return self.classifier(features.float())


def trainable_parameter_report(model: Any) -> dict:
    """Return exact trainable/frozen parameter counts and names."""

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
    }


def set_gradient_checkpointing(model: Any, enabled: bool = True) -> list[str]:
    """Enable a supported upstream checkpointing hook when one exists."""

    if not enabled:
        return []
    activated: list[str] = []
    method_names = (
        "gradient_checkpointing_enable",
        "enable_gradient_checkpointing",
        "set_grad_checkpointing",
    )
    for module_name, module in model.named_modules():
        for method_name in method_names:
            method = getattr(module, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except TypeError:
                method(True)
            activated.append(f"{module_name or '<root>'}.{method_name}")
            break
    return activated


def build_model(
    variant: str,
    num_classes: int,
    parameters: Mapping[str, Any],
    device: Any,
) -> tuple[ThreeDINOClassifier, dict]:
    """Repository-compatible classifier factory for probe or guarded full tuning."""

    _require_torch()
    method = str(variant).strip().lower()
    if method not in {"linear_probe", "full_finetune", "lora"}:
        raise ValueError(
            f"Unknown 3DINO variant {variant!r}; expected linear_probe, full_finetune or lora"
        )
    if method == "full_finetune" and not bool(
        parameters.get(
            "allow_experimental_full_finetune",
            parameters.get("allow_full_finetune", False),
        )
    ):
        raise PermissionError(
            "Experimental full fine-tuning is disabled. It is not the "
            "publication classification protocol and requires explicit licence "
            "and memory approval via allow_experimental_full_finetune=true."
        )

    encoder = parameters.get("encoder")
    if encoder is None:
        required = ("upstream_root", "upstream_revision", "checkpoint_path")
        missing = [name for name in required if not parameters.get(name)]
        if missing:
            raise ValueError(
                f"3DINO model factory is missing required parameters: {missing}"
            )
        encoder = load_official_model(
            upstream_root=parameters["upstream_root"],
            upstream_revision=str(parameters["upstream_revision"]),
            checkpoint_path=parameters["checkpoint_path"],
            device=device,
            config_name=str(parameters.get("config_name", "train/vit3d_highres")),
            expected_checkpoint_sha256=parameters.get("checkpoint_sha256"),
        )

    if method == "lora":
        try:
            from .lora import ThreeDINOLoRAClassifier, validate_parameters
        except ImportError:  # direct invocation
            from lora import ThreeDINOLoRAClassifier, validate_parameters
        validate_parameters(parameters)
        model = ThreeDINOLoRAClassifier(
            encoder, rank=int(parameters.get("lora_rank", 4)),
            alpha=float(parameters.get("lora_alpha", 1.0)),
            head_dropout=float(parameters.get("head_dropout", 0.3)),
            num_classes=int(num_classes),
        ).to(torch.device(device))
        return model, {
            "variant": method, "pretrained_encoder_weights_frozen": True,
            "adaptation_scope": "all 24 attention QKV/proj adapters and study MLP head",
            "target_modules": list(model.target_modules),
            "feature_dim": FEATURE_DIM, "input_size": list(INPUT_SIZE),
            "upstream_provenance": getattr(encoder, "_threedino_provenance", None),
            **trainable_parameter_report(model),
        }

    model = ThreeDINOClassifier(
        encoder=encoder,
        num_classes=int(num_classes),
        n_last_blocks=int(parameters.get("n_last_blocks", 1)),
        use_mean_patch_token=bool(parameters.get("use_mean_patch_token", False)),
        freeze_encoder=method == "linear_probe",
        dropout=0.0 if method == "linear_probe" else float(parameters.get("dropout", 0.0)),
    )
    model.to(torch.device(device))
    checkpointing = set_gradient_checkpointing(
        model.encoder,
        bool(parameters.get("gradient_checkpointing", False))
        and method == "full_finetune",
    )
    details = {
        "variant": method,
        "encoder_frozen": method == "linear_probe",
        "frozen_encoder_eval": method == "linear_probe",
        "n_last_blocks": model.n_last_blocks,
        "use_mean_patch_token": model.use_mean_patch_token,
        "feature_dim": feature_dimension(
            model.n_last_blocks, model.use_mean_patch_token
        ),
        "classifier_initialization": "normal_mean_0_std_0.01_bias_zero",
        "adaptation_scope": (
            "strict linear head only"
            if method == "linear_probe"
            else "experimental full encoder and supervised head"
        ),
        "gradient_checkpointing_enabled_on": checkpointing,
        "input_size": list(INPUT_SIZE),
        "upstream_provenance": getattr(encoder, "_threedino_provenance", None),
    }
    details.update(trainable_parameter_report(model))
    return model, details


def _activate_external_upstream(root: Path) -> None:
    existing = sys.modules.get("dinov2")
    if existing is not None:
        _require_module_under_root(existing, root)
    root_text = str(root)
    resolved_paths = {
        str(Path(value).resolve(strict=False)) for value in sys.path if value
    }
    if root_text not in resolved_paths:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("dinov2")
    if spec is None or spec.origin is None:
        raise UpstreamValidationError(f"Could not locate dinov2 under {root}")
    try:
        Path(spec.origin).resolve().relative_to(root)
    except ValueError as exc:
        raise UpstreamValidationError(
            "A different dinov2 package shadows the supplied official source: "
            f"{spec.origin}"
        ) from exc


def _require_module_under_root(module: Any, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise UpstreamValidationError(
            f"Cannot verify source location for module {getattr(module, '__name__', module)}"
        )
    try:
        Path(module_file).resolve().relative_to(root)
    except ValueError as exc:
        raise UpstreamValidationError(
            f"Module {module.__name__} was imported outside pinned root {root}: "
            f"{module_file}"
        ) from exc


def _load_checkpoint_fail_closed(model: Any, path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CheckpointValidationError(
            f"Could not deserialize official checkpoint {path}: {exc}"
        ) from exc
    state = _extract_state_dict(payload)
    normalised: dict[str, Any] = {}
    for key, value in state.items():
        normalised_key = _normalise_weight_key(str(key))
        if normalised_key in normalised:
            raise CheckpointValidationError(
                "Checkpoint has duplicate keys after module/backbone prefix "
                f"normalization: {normalised_key}"
            )
        normalised[normalised_key] = value
    normalised = {
        key: value
        for key, value in normalised.items()
        if hasattr(value, "shape")
    }
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in normalised.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    shape_mismatches = sorted(
        key
        for key, value in normalised.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    if shape_mismatches:
        raise CheckpointValidationError(
            "Checkpoint tensors do not match the official ViT-L architecture; "
            f"first mismatches: {shape_mismatches[:10]}"
        )
    if not compatible:
        raise CheckpointValidationError(
            "Checkpoint contains no weights compatible with the official 3DINO model"
        )
    # The mask token is not used by downstream classification inference. It is
    # the only model tensor that may be absent from a released teacher export.
    # Every other encoder tensor must be present with its exact shape; a broad
    # percentage threshold could otherwise accept whole randomly initialized
    # transformer blocks.
    allowed_missing = {"mask_token"}
    missing_before_load = sorted(set(model_state) - set(compatible))
    disallowed_missing = [
        key for key in missing_before_load if key not in allowed_missing
    ]
    if disallowed_missing:
        raise CheckpointValidationError(
            "Checkpoint does not fully cover the official 3DINO encoder; "
            f"first missing keys: {disallowed_missing[:10]}"
        )
    required_prefixes = ("patch_embed.", "blocks.", "norm.")
    missing_families = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in compatible)
    ]
    required_exact = [name for name in ("cls_token", "pos_embed") if name in model_state]
    missing_exact = [name for name in required_exact if name not in compatible]
    if missing_families or missing_exact:
        raise CheckpointValidationError(
            "Checkpoint lacks essential encoder weights: "
            f"prefixes={missing_families}, exact={missing_exact}"
        )

    result = model.load_state_dict(compatible, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    expected_allowed_missing = sorted(
        key for key in allowed_missing if key in model_state and key not in compatible
    )
    if missing != expected_allowed_missing or unexpected:
        raise CheckpointValidationError(
            "Unexpected state-dict result while loading the official encoder: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    post_coverage = (len(model_state) - len(missing)) / max(len(model_state), 1)
    return {
        "model_state_keys": len(model_state),
        "compatible_checkpoint_keys": len(compatible),
        "coverage": post_coverage,
        "missing_keys": missing,
        "allowed_missing_keys": expected_allowed_missing,
        "unexpected_checkpoint_keys": sorted(set(normalised) - set(model_state)),
        "unexpected_keys_loaded": unexpected,
        "random_fallback_used": False,
    }


def _validate_official_architecture(model: Any) -> None:
    embed_dim = getattr(model, "embed_dim", getattr(model, "num_features", None))
    block_count = getattr(model, "n_blocks", None)
    blocks = getattr(model, "blocks", None)
    if block_count is None and blocks is not None:
        if bool(getattr(model, "chunked_blocks", False)) and len(blocks):
            # Official BlockChunk objects are padded with Identity modules to
            # preserve global block indices; the final chunk length is depth.
            block_count = max(len(chunk) for chunk in blocks)
        else:
            block_count = len(blocks)
    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(patch_embed, "patch_size", None)
    patch_projection = getattr(patch_embed, "proj", None)
    input_channels = getattr(patch_projection, "in_channels", None)
    if int(embed_dim or -1) != FEATURE_DIM:
        raise UpstreamValidationError(
            f"Expected official ViT-L embedding width {FEATURE_DIM}, got {embed_dim}"
        )
    if int(block_count or -1) != 24:
        raise UpstreamValidationError(
            f"Expected official ViT-L depth 24, got {block_count}"
        )
    patch_values = (
        tuple(int(value) for value in patch_size)
        if isinstance(patch_size, (tuple, list))
        else (int(patch_size or -1),) * 3
    )
    if patch_values != (16, 16, 16):
        raise UpstreamValidationError(
            f"Expected 3D patch size (16, 16, 16), got {patch_size}"
        )
    if int(input_channels or -1) != 1:
        raise UpstreamValidationError(
            f"Expected one-channel 3D patch embedding, got {input_channels}"
        )


def _extract_state_dict(payload: Any) -> Mapping:
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError(
            f"Checkpoint root must be a mapping, got {type(payload).__name__}"
        )
    for key in ("teacher", "state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested:
            return nested
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise CheckpointValidationError("Checkpoint contains no recognizable state dictionary")


def _normalise_weight_key(key: str) -> str:
    value = key
    # Released/evaluation checkpoints normally expose a top-level ``teacher``
    # mapping; backbone-only exports may omit it.  Tolerate the equivalent
    # flattened prefix without broad string replacement.
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "teacher.", "backbone."):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
    return value


def _unwrap_encoder(model: Any) -> Any:
    candidate = model
    visited: set[int] = set()
    while not callable(getattr(candidate, "get_intermediate_layers", None)):
        if id(candidate) in visited:
            break
        visited.add(id(candidate))
        nested = getattr(candidate, "module", None)
        if nested is None:
            nested = getattr(candidate, "encoder", None)
        if nested is None:
            break
        candidate = nested
    return candidate


def _normalise_intermediate_output(
    output: Any, *, expected_count: int
) -> list[tuple[Any, Any]]:
    if not isinstance(output, (tuple, list)):
        raise RuntimeError(
            "get_intermediate_layers must return a sequence of layer outputs"
        )
    if len(output) != expected_count:
        raise RuntimeError(
            f"Requested {expected_count} intermediate blocks, received {len(output)}"
        )
    pairs: list[tuple[Any, Any]] = []
    for index, item in enumerate(output):
        if isinstance(item, Mapping):
            patches = item.get("x_norm_patchtokens")
            class_token = item.get("x_norm_clstoken")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            patches, class_token = item
        elif hasattr(item, "ndim") and int(item.ndim) == 3 and item.shape[1] >= 2:
            class_token = item[:, 0]
            patches = item[:, 1:]
        else:
            raise RuntimeError(
                f"Unrecognized intermediate layer output at index {index}: "
                f"{type(item).__name__}"
            )
        if patches is None or class_token is None:
            raise RuntimeError(f"Intermediate layer {index} lacks patch/CLS tokens")
        pairs.append((patches, class_token))
    return pairs


def _validate_tokens(
    class_tokens: Sequence[Any], patch_tokens: Any, images: Any
) -> None:
    batch = int(images.shape[0])
    for index, token in enumerate(class_tokens):
        if getattr(token, "ndim", None) != 2:
            raise RuntimeError(
                f"CLS token {index} must be (batch, dim), got {getattr(token, 'shape', None)}"
            )
        if tuple(token.shape) != (batch, FEATURE_DIM):
            raise RuntimeError(
                f"CLS token {index} must be {(batch, FEATURE_DIM)}, got {tuple(token.shape)}"
            )
    if getattr(patch_tokens, "ndim", None) != 3:
        raise RuntimeError(
            "Patch tokens must be shaped (batch, patches, 1024), got "
            f"{getattr(patch_tokens, 'shape', None)}"
        )
    if int(patch_tokens.shape[0]) != batch or int(patch_tokens.shape[-1]) != FEATURE_DIM:
        raise RuntimeError(
            f"Patch tokens must be (batch, patches, {FEATURE_DIM}), got "
            f"{tuple(patch_tokens.shape)}"
        )
    if int(patch_tokens.shape[1]) <= 0:
        raise RuntimeError("3DINO returned no patch tokens")


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "3DINO model execution requires PyTorch. Install this model's "
            "requirements.txt in its isolated environment."
        )


__all__ = [
    "CheckpointInfo",
    "CheckpointValidationError",
    "DEFAULT_REPRESENTATIONS",
    "FEATURE_DIM",
    "INPUT_SIZE",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "ThreeDINOClassifier",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_REVISION",
    "UpstreamValidationError",
    "build_model",
    "extract_intermediate_features",
    "extract_representation_bank",
    "feature_dimension",
    "load_official_model",
    "resolve_checkpoint",
    "set_gradient_checkpointing",
    "trainable_parameter_report",
    "validate_upstream_checkout",
]
