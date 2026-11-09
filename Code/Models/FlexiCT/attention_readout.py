"""Frozen FlexiCT encoder with a trainable single-head attention readout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    from . import CLASS_NAMES, UPSTREAM_REVISION
    from .attention_cache import (
        CACHE_SCHEMA_VERSION,
        DEFAULT_ATTENTION_CACHE_NAMESPACE,
        TOKEN_SHAPE,
        load_attention_cache,
        resolve_cached_rows,
    )
except ImportError:  # pragma: no cover
    from __init__ import CLASS_NAMES, UPSTREAM_REVISION
    from attention_cache import (
        CACHE_SCHEMA_VERSION,
        DEFAULT_ATTENTION_CACHE_NAMESPACE,
        TOKEN_SHAPE,
        load_attention_cache,
        resolve_cached_rows,
    )

from _linear_probe.common import ContractError, sha256_file
from _supervised_adaptation.runtime import torch_load


METHOD_LABEL = "FrozenSingleHeadAttentionReadout"
METHOD_ID = "frozen_single_head_attention_readout"


class SingleHeadAttentionReadout(nn.Module):
    def __init__(self, dimension: int = 864):
        super().__init__()
        self.dimension = int(dimension)
        self.query = nn.Parameter(torch.empty(1, 1, self.dimension))
        self.attention = nn.MultiheadAttention(
            self.dimension, num_heads=1, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(self.dimension)
        self.classifier = nn.Linear(self.dimension, len(CLASS_NAMES))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != TOKEN_SHAPE:
            raise RuntimeError(
                f"Expected cached depth tokens [B,{TOKEN_SHAPE[0]},{TOKEN_SHAPE[1]}], got {tuple(tokens.shape)}"
            )
        query = self.query.expand(tokens.shape[0], -1, -1)
        attended, _weights = self.attention(
            query, tokens, tokens, need_weights=False
        )
        pooled = self.norm(query + attended)[:, 0]
        return self.classifier(pooled)


class FlexiCTAttentionAdapter:
    name = "FlexiCT"
    method_label = METHOD_LABEL
    method_id = METHOD_ID

    def __init__(self) -> None:
        package = Path(__file__).resolve().parent
        self.publication_profile_path = package / "publication_attention_readout.json"
        self.default_profile_path = package / "default_attention_readout.json"
        self.search_profile_path = package / "grid_attention_readout.json"
        self._cache_manifest: dict | None = None
        self._cache_manifest_path: Path | None = None

    def implementation_paths(self) -> list[Path]:
        return [Path(__file__), Path(__file__).with_name("attention_cache.py")]

    def planned_identity(self, args: Any) -> dict:
        return {
            "encoder": "pinned FlexiCT-3D-VLM vision checkpoint",
            "upstream_revision": UPSTREAM_REVISION,
            "encoder_trainable": False,
            "readout": "one learned query, one-head cross-attention, residual LayerNorm, five-class linear layer",
            "token_cache": str(Path(args.cache_dir).resolve()),
            "exact_curia_head_boundary": (
                "published gated Curia implementation was unavailable; transparent local one-head contract is used"
            ),
            "assets_resolved": False,
        }

    def preflight(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> None:
        self._cache_manifest_path = Path(args.cache_dir) / "cache_manifest.json"
        self._cache_manifest = load_attention_cache(
            args.cache_dir,
            expected_namespace=str(args.cache_namespace),
            validate_shards=bool(args.validate_cache_shards),
        )
        resolve_cached_rows(
            args.cache_dir, rows, expected_namespace=str(args.cache_namespace)
        )

    def resolve_identity(self, args: Any) -> dict:
        if self._cache_manifest is None or self._cache_manifest_path is None:
            self.preflight(args, [])
        return {
            "attention_cache_manifest": str(self._cache_manifest_path.resolve()),
            "attention_cache_manifest_sha256": sha256_file(self._cache_manifest_path),
            "attention_cache_contract": self._cache_manifest["contract"],
            "encoder_checkpoint": self._cache_manifest["contract"]["checkpoint"],
        }

    def representation_contract(self, _args: Any) -> dict:
        return {
            "encoder": "fully frozen and materialized in sealed cache",
            "input_tokens": "20 depth-wise mean-HW final normalized patch tokens",
            "input_shape": list(TOKEN_SHAPE),
            "readout": "learned query + nn.MultiheadAttention(embed_dim=864,num_heads=1,dropout=0) + residual LayerNorm",
            "classifier": "Linear(864,5)",
        }

    def prepare_rows(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
        if self._cache_manifest is None:
            self.preflight(args, rows)
        return resolve_cached_rows(
            args.cache_dir, rows, expected_namespace=str(args.cache_namespace)
        )

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        allowed = {
            "encoder_state",
            "attention_heads",
            "query_count",
            "optimizer",
            "head_learning_rate",
            "weight_decay",
            "momentum",
            "nesterov",
            "scheduler",
            "warmup_steps",
            "minimum_lr_ratio",
            "epochs",
            "max_steps",
            "effective_batch_size",
            "eval_batch_size",
            "validation_interval_steps",
            "early_stopping_patience_steps",
            "early_stopping_min_delta",
            "class_weight_beta",
            "gradient_clip_norm",
            "candidate_id",
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise ContractError(f"Unsupported FlexiCT attention-readout parameters: {sorted(unknown)}")
        if parameters.get("encoder_state") != "frozen":
            raise ContractError("Attention readout requires encoder_state=frozen")
        if int(parameters.get("attention_heads", 0)) != 1 or int(parameters.get("query_count", 0)) != 1:
            raise ContractError("Published attention-readout contract requires one head and one query")
        if str(parameters.get("optimizer", "")).lower() != "sgd":
            raise ContractError("FlexiCT classification attention readout uses SGD")
        if float(parameters.get("head_learning_rate", 0.0)) <= 0.0:
            raise ContractError("head_learning_rate must be positive")

    @staticmethod
    def build_model(_args: Any, _parameters: Mapping[str, Any], _device: str):
        return SingleHeadAttentionReadout()

    @staticmethod
    def validate_trainable_scope(model: nn.Module, _parameters: Mapping[str, Any]) -> None:
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        if not names or any(name.startswith("encoder.") for name in names):
            raise ContractError("Frozen attention readout may train only its local query/attention/head")
        if not any(name.startswith("attention.") for name in names) or "query" not in names:
            raise ContractError("Single-head attention/query parameters are not trainable")

    @staticmethod
    def optimizer_groups(model: nn.Module, parameters: Mapping[str, Any]) -> list[dict]:
        return [
            {
                "group_name": "single_head_attention_readout",
                "params": [value for value in model.parameters() if value.requires_grad],
                "lr": float(parameters["head_learning_rate"]),
                "weight_decay": float(parameters["weight_decay"]),
            }
        ]

    @staticmethod
    def set_training_mode(model: SingleHeadAttentionReadout, mode: bool) -> None:
        model.train(mode)

    @staticmethod
    def move_inputs(inputs: torch.Tensor, device: str) -> torch.Tensor:
        return inputs.to(device, dtype=torch.float32, non_blocking=False)

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
        tokens = []
        for index in indices:
            observed_sha256 = sha256_file(rows[index]["token_path"])
            if observed_sha256 != str(rows[index]["token_sha256"]):
                raise ContractError(
                    f"Cached token SHA-256 changed: {rows[index]['token_path']}"
                )
            payload = torch_load(rows[index]["token_path"], map_location="cpu")
            if (
                payload.get("schema_version") != CACHE_SCHEMA_VERSION
                or str(payload.get("patient_id")) != str(rows[index]["patient_id"])
                or int(payload.get("label", -1)) != int(rows[index]["label"])
                or str(payload.get("class_name")) != str(rows[index]["class_name"])
                or payload.get("contract_sha256")
                != str(rows[index]["token_contract_sha256"])
                or payload.get("source_identity") != dict(rows[index]["source_identity"])
            ):
                raise ContractError(f"Cached token identity changed: {rows[index]['token_path']}")
            value = payload.get("tokens")
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != TOKEN_SHAPE
                or value.dtype != torch.float16
                or not bool(torch.isfinite(value).all())
            ):
                raise ContractError(
                    f"Cached token tensor changed: {rows[index]['token_path']}"
                )
            tokens.append(value.float())
        batch = torch.stack(tokens, dim=0)
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return batch, labels

    @staticmethod
    def architecture(_model: SingleHeadAttentionReadout, parameters: Mapping[str, Any]) -> dict:
        return {
            "encoder": "FlexiCT frozen before cached token extraction",
            "token_shape": list(TOKEN_SHAPE),
            "query_count": int(parameters["query_count"]),
            "attention_heads": int(parameters["attention_heads"]),
            "attention_dimension": TOKEN_SHAPE[1],
            "num_classes": len(CLASS_NAMES),
            "implementation_relation": "publication-backed family; transparent project adaptation to depth tokens",
        }


__all__ = [
    "FlexiCTAttentionAdapter",
    "METHOD_ID",
    "METHOD_LABEL",
    "SingleHeadAttentionReadout",
]
