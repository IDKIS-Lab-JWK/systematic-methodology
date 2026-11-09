"""Pillar-0 five-class abdominal CT experiments."""

from __future__ import annotations

CLASS_NAMES = (
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
)

CACHE_SCHEMA_VERSION = "pillar0-rave-cache-manifest-v1"
DEFAULT_CACHE_NAMESPACE = (
    "pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1"
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CLASS_NAMES",
    "DEFAULT_CACHE_NAMESPACE",
]
