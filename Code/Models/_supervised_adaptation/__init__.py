"""Shared contracts for supervised encoder adaptation experiments.

This package is intentionally separate from ``_linear_probe``.  It reuses
only dependency-light split, metric, hashing, and sealing utilities so the
existing linear-probe implementations and their resumable artifacts remain
unchanged.
"""

from __future__ import annotations

METHOD_SCHEMA_VERSION = "supervised-adaptation-v1"
CHECKPOINT_SCHEMA_VERSION = "supervised-adaptation-checkpoint-v1"

__all__ = ["CHECKPOINT_SCHEMA_VERSION", "METHOD_SCHEMA_VERSION"]
