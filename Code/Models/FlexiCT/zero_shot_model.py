"""Pinned FlexiCT image/text construction for frozen zero-shot inference."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from . import FEATURE_DIM, UPSTREAM_REVISION
    from .model import build_vision_only_encoder
    from .provisioning import validate_upstream
except ImportError:  # pragma: no cover
    from __init__ import FEATURE_DIM, UPSTREAM_REVISION
    from model import build_vision_only_encoder
    from provisioning import validate_upstream

from _linear_probe.common import (
    ContractError,
    append_jsonl,
    fingerprint,
    sha256_file,
    utc_now,
)
from _supervised_adaptation.runtime import require_torch


TEXT_MODEL_REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
TEXT_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TEXT_MODEL_FILES = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "bb4e0a697380f3636723b76b175cd1d6ca2e72659a4c269be3b3e5d41c4b7403"
)
PINNED_TRANSFORMERS_VERSION = "4.56.1"
PINNED_HF_HUB_VERSION = "0.34.4"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"FlexiCT zero-shot requires {name}; rerun FlexiCT_Environment_Setup.ps1"
        ) from exc


def resolve_text_snapshot(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    provision_log: str | Path | None,
) -> dict:
    """Materialize only the pinned Qwen config/tokenizer files required here."""

    hub_version = _package_version("huggingface-hub")
    if hub_version != PINNED_HF_HUB_VERSION:
        raise RuntimeError(
            "FlexiCT zero-shot requires huggingface-hub "
            f"{PINNED_HF_HUB_VERSION}, observed {hub_version}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise RuntimeError(
            "FlexiCT zero-shot requires huggingface-hub; rerun the environment setup"
        ) from exc
    kwargs: dict[str, Any] = {
        "repo_id": TEXT_MODEL_REPO_ID,
        "revision": TEXT_MODEL_REVISION,
        "allow_patterns": list(TEXT_MODEL_FILES),
        "local_files_only": bool(local_files_only),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    try:
        snapshot = Path(snapshot_download(**kwargs)).resolve()
    except Exception as exc:
        mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub/cache"
        raise ContractError(
            f"Could not resolve pinned {TEXT_MODEL_REPO_ID}@{TEXT_MODEL_REVISION} from {mode}"
        ) from exc
    records = []
    for relative in TEXT_MODEL_FILES:
        path = snapshot / relative
        if not path.is_file():
            raise ContractError(f"Pinned FlexiCT text-model snapshot is missing {relative}")
        records.append(
            {
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    identity = {
        "repo_id": TEXT_MODEL_REPO_ID,
        "revision": TEXT_MODEL_REVISION,
        "snapshot_path": str(snapshot),
        "files": records,
        "files_sha256": fingerprint(records),
        "acquisition": "pinned_huggingface_snapshot",
    }
    if provision_log is not None:
        append_jsonl(
            provision_log,
            {
                "timestamp": utc_now(),
                "asset": "flexict_zero_shot_text_model_config_and_tokenizer",
                **identity,
            },
        )
    return identity


def _load_checkpoint(
    path: str | Path,
    *,
    allow_trusted_legacy_checkpoint: bool,
    trusted_checkpoint_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    torch = require_torch()
    checkpoint = Path(path).resolve()
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        if not allow_trusted_legacy_checkpoint:
            raise RuntimeError(
                "FlexiCT checkpoint failed PyTorch weights-only loading; trusted legacy "
                "pickle fallback remains disabled"
            ) from exc
        observed = sha256_file(checkpoint)
        if observed.lower() != trusted_checkpoint_sha256.lower():
            raise ContractError(
                "Trusted FlexiCT checkpoint SHA-256 changed before legacy deserialization"
            ) from exc
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        load_mode = "trusted_pickle_opt_in_after_sha256"
    else:
        load_mode = "torch_weights_only"
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("FlexiCT VLM checkpoint must contain a 'model' state dictionary")
    return payload["model"], load_mode


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    torch = require_torch()
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


def build_text_encoder(
    *,
    upstream_root: str | Path,
    checkpoint_path: str | Path,
    text_snapshot: Mapping[str, Any],
    device: str,
    allow_trusted_legacy_checkpoint: bool,
    trusted_checkpoint_sha256: str,
) -> tuple[Any, Any, float, dict]:
    """Build the released Qwen tower from config, then load FlexiCT text weights."""

    torch = require_torch()
    source = validate_upstream(upstream_root, UPSTREAM_REVISION)
    required_source = Path(upstream_root).resolve() / "flexi_ct" / "text_tower.py"
    if not required_source.is_file():
        raise ContractError(f"Pinned FlexiCT text-tower source is missing: {required_source}")
    transformers_version = _package_version("transformers")
    if transformers_version != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "FlexiCT zero-shot requires transformers "
            f"{PINNED_TRANSFORMERS_VERSION}, observed {transformers_version}"
        )
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise RuntimeError("FlexiCT zero-shot requires Transformers") from exc

    snapshot_path = Path(str(text_snapshot["snapshot_path"])).resolve()
    config = AutoConfig.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    hidden_size = int(getattr(config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Pinned Qwen configuration has no positive hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        padding_side="left",
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.padding_side != "left":
        raise RuntimeError("FlexiCT zero-shot requires left-padded Qwen tokenization")

    state, load_mode = _load_checkpoint(
        checkpoint_path,
        allow_trusted_legacy_checkpoint=allow_trusted_legacy_checkpoint,
        trusted_checkpoint_sha256=trusted_checkpoint_sha256,
    )
    prefix = "text_model.backbone."
    text_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not text_state:
        raise RuntimeError("FlexiCT checkpoint contains no text_model.backbone.* tensors")
    floating_dtypes = {
        value.dtype for value in text_state.values() if value.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "FlexiCT text-backbone checkpoint dtype changed: "
            f"observed={sorted(str(value) for value in floating_dtypes)}"
        )
    # The released Flexi_CT_VLM helper constructs this tower in float32 before
    # loading its checkpoint. This also keeps non-persistent Qwen buffers (for
    # example rotary-frequency buffers) materialized correctly; a meta-only
    # construction would leave those uncheckpointed buffers empty.
    backbone = AutoModel.from_config(
        config,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    expected_keys = set(backbone.state_dict())
    observed_keys = set(text_state)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            "FlexiCT text-backbone state mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    backbone.load_state_dict(text_state, strict=True)

    projection_state = {
        key[len("text_model.projection.") :]: value
        for key, value in state.items()
        if key.startswith("text_model.projection.")
    }
    if hidden_size == FEATURE_DIM:
        if projection_state:
            raise RuntimeError(
                "Unexpected FlexiCT text projection for an already 1024-dimensional tower"
            )
        projection = torch.nn.Identity()
    else:
        projection = torch.nn.Linear(hidden_size, FEATURE_DIM, bias=False)
        projection.load_state_dict(projection_state, strict=True)

    logit_scale = state.get("logit_scale")
    if not isinstance(logit_scale, torch.Tensor) or tuple(logit_scale.shape) != (1,):
        raise RuntimeError("FlexiCT checkpoint has no scalar logit_scale tensor")
    logit_multiplier = float(logit_scale.detach().float().exp().item())
    if not (0.0 < logit_multiplier < float("inf")):
        raise RuntimeError("FlexiCT checkpoint logit scale is non-finite or non-positive")

    class FrozenTextEncoder(torch.nn.Module):
        def __init__(self, text_backbone: Any, text_projection: Any):
            super().__init__()
            self.backbone = text_backbone
            self.projection = text_projection

        def forward(self, tokens: Mapping[str, Any]):
            outputs = self.backbone(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            hidden = outputs.last_hidden_state
            pooled = _last_token_pool(hidden, tokens["attention_mask"])
            return self.projection(pooled)

    model = FrozenTextEncoder(backbone, projection).to(str(device)).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("FlexiCT zero-shot text encoder was not fully frozen")
    state_signature = [
        {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(text_state.items())
    ]
    details = {
        "source": source,
        "checkpoint_load_mode": load_mode,
        "text_model_repo_id": TEXT_MODEL_REPO_ID,
        "text_model_revision": TEXT_MODEL_REVISION,
        "transformers_version": transformers_version,
        "huggingface_hub_version": _package_version("huggingface-hub"),
        "hidden_size": hidden_size,
        "embedding_dimension": FEATURE_DIM,
        "pooling": "last_non_padding_token",
        "padding_side": "left",
        "max_length": 8192,
        "text_state_signature_sha256": fingerprint(state_signature),
        "checkpoint_text_tensor_dtype": "torch.bfloat16",
        "materialized_text_parameter_dtype": "torch.float32",
        "logit_multiplier": logit_multiplier,
    }
    return model, tokenizer, logit_multiplier, details


def build_image_encoder(
    *,
    upstream_root: str | Path,
    checkpoint_path: str | Path,
    device: str,
    allow_trusted_legacy_checkpoint: bool,
    trusted_checkpoint_sha256: str,
) -> tuple[Any, dict]:
    return build_vision_only_encoder(
        upstream_root=upstream_root,
        checkpoint_path=checkpoint_path,
        device=device,
        allow_trusted_legacy_checkpoint=allow_trusted_legacy_checkpoint,
        trusted_checkpoint_sha256=trusted_checkpoint_sha256,
    )


__all__ = [
    "OFFICIAL_CHECKPOINT_SHA256",
    "PINNED_HF_HUB_VERSION",
    "PINNED_TRANSFORMERS_VERSION",
    "TEXT_MODEL_FILES",
    "TEXT_MODEL_REPO_ID",
    "TEXT_MODEL_REVISION",
    "build_image_encoder",
    "build_text_encoder",
    "resolve_text_snapshot",
]
