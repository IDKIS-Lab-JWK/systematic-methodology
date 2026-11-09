"""Isolated 3DINO-ViT experiments for five-class abdominal CT."""

from __future__ import annotations

CLASS_NAMES = (
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
)

# Official upstream assets are provisioned outside this package.  Source and
# weights live in different repositories and therefore have independent,
# immutable revisions.
UPSTREAM_REPOSITORY = "https://github.com/AICONSlab/3DINO"
UPSTREAM_REVISION = "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89"
MODEL_REPO_ID = "AICONSlab/3DINO-ViT"
MODEL_FILENAME = "3dino_vit_weights.pth"
MODEL_REVISION = "8a00a2bb14becb3fbe955064837322a3217f7ac0"
MODEL_SHA256 = "b68dde2e435648bdf72edae70f970b07e4a903a8122c5be9e360bdb076cf695c"
MODEL_SIZE_BYTES = 1_417_113_974
MODEL_ACCESS_URL = "https://huggingface.co/AICONSlab/3DINO-ViT"

FEATURE_DIM = 1024
INPUT_SIZE = (112, 112, 112)
PREFIX_SIZE = (144, 144, 112)

CACHE_SCHEMA_VERSION = "threedino-deterministic-prefix-cache-v2"
DEFAULT_CACHE_NAMESPACE = (
    "threedino_ct_ras_p0005_p9995_fg_resize_144x144x112_v2"
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CLASS_NAMES",
    "DEFAULT_CACHE_NAMESPACE",
    "FEATURE_DIM",
    "INPUT_SIZE",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_SIZE_BYTES",
    "MODEL_ACCESS_URL",
    "PREFIX_SIZE",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_REVISION",
]
