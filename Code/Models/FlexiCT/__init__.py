"""FlexiCT linear probing for the systematic five-class CT methodology."""

from __future__ import annotations

CLASS_NAMES = (
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
)

UPSTREAM_REPOSITORY = "https://github.com/ricklisz/FlexiCT"
UPSTREAM_REVISION = "fd74ac06a31813fe02805674e96f811e9b2a5c43"
MODEL_ACCESS_URL = "https://github.com/ricklisz/FlexiCT#model-checkpoints"
FEATURE_DIM = 1024
INPUT_SHAPE_DHW = (160, 160, 160)
PATCH_SIZE = 8
EXPECTED_PATCH_TOKENS = 8000
DEFAULT_CACHE_NAMESPACE = "flexict_vlm_lps_2mm_160cube_clsmean_proj_l2_v1"

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_CACHE_NAMESPACE",
    "EXPECTED_PATCH_TOKENS",
    "FEATURE_DIM",
    "INPUT_SHAPE_DHW",
    "MODEL_ACCESS_URL",
    "PATCH_SIZE",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_REVISION",
]
