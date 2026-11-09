"""Publication-described COLIPRI classification-pooling heads.

The COLIPRI paper calls the downstream experiment a linear probe even though
three of its five pooling choices contain trainable attention. The project
therefore assigns the complete searched family the explicit
``PoolingOptimisedLinearProbe`` identity while retaining a machine-readable
strict/non-strict distinction on every model artifact.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn


POOLING_NAMES: Final[tuple[str, ...]] = (
    "average",
    "max",
    "learned_attention",
    "average_attention",
    "multi_learned_attention",
)


def dense_to_tokens(dense: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,C,D,H,W]`` backbone features to ``[B,T,C]`` tokens."""

    if dense.ndim != 5:
        raise ValueError(f"Expected dense features [B,C,D,H,W], got {tuple(dense.shape)}")
    # Keep this as a view.  Materialising a contiguous [B, 13824, 864]
    # tensor costs about 45.6 MiB per patient and is unnecessary for the
    # publication pooling heads (including PyTorch MultiheadAttention).
    return dense.flatten(2).transpose(1, 2)


class AveragePool1D(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class MaxPool1D(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.amax(dim=1)


class LearnedQueryAttentionPool1D(nn.Module):
    """One learned query followed by PyTorch multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / math.sqrt(embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return pooled[:, 0]


class AverageQueryAttentionPool1D(nn.Module):
    """Official COLIPRI ``AttentionPool1D`` design with a mean-token query."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = tokens.mean(dim=1, keepdim=True)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return pooled[:, 0]


class MultiLearnedQueryAttentionPool1D(nn.Module):
    """Four learned queries, with their four outputs averaged."""

    def __init__(self, embed_dim: int, num_heads: int, num_queries: int = 4):
        super().__init__()
        if num_queries != 4:
            raise ValueError("The publication-backed COLIPRI probe uses exactly four queries")
        self.query = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) / math.sqrt(embed_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return pooled.mean(dim=1)


def build_pooling(name: str, *, embed_dim: int, num_heads: int) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "average":
        return AveragePool1D()
    if normalized == "max":
        return MaxPool1D()
    if normalized == "learned_attention":
        return LearnedQueryAttentionPool1D(embed_dim, num_heads)
    if normalized == "average_attention":
        return AverageQueryAttentionPool1D(embed_dim, num_heads)
    if normalized == "multi_learned_attention":
        return MultiLearnedQueryAttentionPool1D(embed_dim, num_heads)
    raise ValueError(f"Unsupported COLIPRI pooling method: {name!r}")


def is_strict_linear(pooling: str) -> bool:
    if pooling not in POOLING_NAMES:
        raise ValueError(f"Unsupported COLIPRI pooling method: {pooling!r}")
    return pooling in ("average", "max")


class COLIPRIPoolingOptimisedLinearProbe(nn.Module):
    """Frozen-backbone readout used by the native COLIPRI probe grid."""

    def __init__(
        self,
        *,
        pooling: str,
        embed_dim: int = 864,
        num_heads: int = 12,
        num_classes: int = 5,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.pooling_name = pooling
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_classes = int(num_classes)
        self.pool = build_pooling(pooling, embed_dim=embed_dim, num_heads=num_heads)
        self.classifier = nn.Linear(embed_dim, num_classes)

    @property
    def strict_linear(self) -> bool:
        return is_strict_linear(self.pooling_name)

    def architecture(self) -> dict:
        return {
            "pooling": self.pooling_name,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
            "strict_linear": self.strict_linear,
        }

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(dense_to_tokens(dense))

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Classify an already-tokenised frozen feature batch.

        The backend calls this entry point so all 20 publication candidates
        reuse one token view per microbatch.
        """

        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,T,C], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected token dimension {self.embed_dim}, got {tokens.shape[-1]}"
            )
        return self.classifier(self.pool(tokens))


__all__ = [
    "AveragePool1D",
    "AverageQueryAttentionPool1D",
    "COLIPRIPoolingOptimisedLinearProbe",
    "LearnedQueryAttentionPool1D",
    "MaxPool1D",
    "MultiLearnedQueryAttentionPool1D",
    "POOLING_NAMES",
    "build_pooling",
    "dense_to_tokens",
    "is_strict_linear",
]
