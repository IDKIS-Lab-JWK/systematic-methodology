"""COLIPRI pooling-optimised probing for the five-class CT methodology."""

from __future__ import annotations

CLASS_NAMES = (
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
)

MODEL_REPO_ID = "microsoft/colipri"
MODEL_REVISION = "be872ebec5c62894cf183b09835ec6fd7be537f5"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "97d66586507d1f48577b5149777d02e12d25787435a2a295cac5563df9291c14"
MODEL_SIZE_BYTES = 1_033_541_380
FEATURE_SHAPE_CDHW = (864, 24, 24, 24)
INPUT_SHAPE_DHW = (192, 192, 192)
PATCH_SIZE = 8
DEFAULT_CACHE_NAMESPACE = "colipri_ras_2mm_192cube_raw_backbone_864x24cube_fp32_v1"
METHOD_ID = "pooling_optimised_linear_probe"
METHOD_LABEL = "PoolingOptimisedLinearProbe"
RESULT_FOLDER = METHOD_LABEL

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_CACHE_NAMESPACE",
    "FEATURE_SHAPE_CDHW",
    "INPUT_SHAPE_DHW",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_SIZE_BYTES",
    "METHOD_ID",
    "METHOD_LABEL",
    "PATCH_SIZE",
    "RESULT_FOLDER",
]
