"""FlexiCT publication-grid facade kept parallel to the model hierarchy."""

from __future__ import annotations

from typing import Any, Mapping

from .backend import FlexiCTBackend


def build_candidates(profile: Mapping[str, Any]) -> list[dict]:
    """Validate a profile and return its canonical FlexiCT candidates."""

    return FlexiCTBackend().build_candidates(profile)


__all__ = ["build_candidates"]
