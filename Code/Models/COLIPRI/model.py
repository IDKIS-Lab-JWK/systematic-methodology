"""Pinned official COLIPRI image-only model and processor adapters."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

try:
    from . import FEATURE_SHAPE_CDHW, INPUT_SHAPE_DHW, MODEL_REVISION
    from .provisioning import validate_upstream
except ImportError:  # pragma: no cover
    from __init__ import FEATURE_SHAPE_CDHW, INPUT_SHAPE_DHW, MODEL_REVISION
    from provisioning import validate_upstream


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for COLIPRI feature extraction") from exc
    return torch


def resolve_precision(device: str, requested: str) -> tuple[str, Any | None]:
    torch = _require_torch()
    if requested not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError(f"Unsupported precision: {requested}")
    if not str(device).startswith("cuda"):
        if requested in ("bf16", "fp16"):
            raise RuntimeError(
                f"Explicit {requested} feature extraction requires a CUDA device"
            )
        return "fp32", None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction was requested but CUDA is unavailable")
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but torch.cuda.is_bf16_supported() is false")
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16
    if requested == "fp32":
        return "fp32", None
    raise AssertionError("unreachable precision branch")


def _activate_source(root: Path) -> None:
    source = str((root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def processor_config_identity(upstream_root: str | Path) -> dict:
    try:
        from _linear_probe.common import fingerprint, sha256_file
    except ImportError:  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _linear_probe.common import fingerprint, sha256_file

    root = Path(upstream_root).resolve()
    files = [
        root / "src" / "colipri" / "configs" / "config.yaml",
        root / "src" / "colipri" / "configs" / "processor" / "default.yaml",
        root
        / "src"
        / "colipri"
        / "configs"
        / "processor"
        / "image_transform"
        / "default.yaml",
    ]
    records = [{"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in files]
    return {"files": records, "sha256": fingerprint(records)}


def build_image_encoder(
    *, upstream_root: str | Path, checkpoint_path: str | Path, device: str
) -> tuple[Any, Any, dict]:
    torch = _require_torch()
    source_identity = validate_upstream(upstream_root, MODEL_REVISION)
    root = Path(upstream_root).resolve()
    _activate_source(root)
    colipri = importlib.import_module("colipri")
    module_path = Path(colipri.__file__).resolve()
    if root not in module_path.parents:
        raise RuntimeError(f"Imported colipri from an unexpected location: {module_path}")
    model = colipri.get_model(
        checkpoint_path=Path(checkpoint_path).resolve(),
        pretrained=True,
        image_only=True,
    ).to(device)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("COLIPRI image encoder was not completely frozen")
    processor = colipri.get_processor(image_only=True)
    return model, processor, {
        "source": source_identity,
        "processor_config": processor_config_identity(root),
        "input_shape_dhw": list(INPUT_SHAPE_DHW),
        "feature_shape_cdhw": list(FEATURE_SHAPE_CDHW),
        "encode_image_arguments": {
            "pool": False,
            "project": False,
            "normalize": False,
        },
    }


def preprocess_nifti(processor: Any, path: str | Path) -> tuple[Any, dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    images = processor.process_images(source)
    if len(images) != 1:
        raise RuntimeError("COLIPRI processor returned an unexpected image count")
    image = images[0]
    batch = processor.to_images_batch(images)
    if tuple(batch.shape) != (1, 1, *INPUT_SHAPE_DHW):
        raise RuntimeError(f"Unexpected COLIPRI processed shape: {tuple(batch.shape)}")
    orientation = tuple(str(value) for value in image.orientation)
    spacing = tuple(float(value) for value in image.spacing)
    if orientation != ("R", "A", "S"):
        raise RuntimeError(f"Pinned COLIPRI processor produced {orientation}, expected RAS")
    if any(abs(value - 2.0) > 1e-4 for value in spacing):
        raise RuntimeError(f"Pinned COLIPRI processor produced spacing {spacing}, expected 2 mm")
    if not batch.isfinite().all():
        raise RuntimeError("COLIPRI processor produced non-finite values")
    return batch, {
        "source": str(source.resolve()),
        "orientation": list(orientation),
        "spacing_xyz_mm": list(spacing),
        "shape_bcdhw": list(batch.shape),
        "minimum": float(batch.min()),
        "maximum": float(batch.max()),
    }


def extract_dense_features(
    model: Any, images: Any, *, device: str, precision: str
) -> tuple[Any, dict]:
    torch = _require_torch()
    resolved_name, dtype = resolve_precision(device, precision)
    images = images.to(device)
    with torch.inference_mode():
        if dtype is None:
            dense = model.encode_image(
                images, pool=False, project=False, normalize=False
            )
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                dense = model.encode_image(
                    images, pool=False, project=False, normalize=False
                )
    dense = dense.float().cpu().contiguous()
    if tuple(dense.shape) != (1, *FEATURE_SHAPE_CDHW):
        raise RuntimeError(f"Unexpected COLIPRI dense feature shape: {tuple(dense.shape)}")
    if not torch.isfinite(dense).all():
        raise RuntimeError("COLIPRI dense features contain non-finite values")
    return dense[0], {
        "precision": resolved_name,
        "minimum": float(dense.min()),
        "maximum": float(dense.max()),
        "mean": float(dense.mean()),
        "std": float(dense.std()),
    }


__all__ = [
    "build_image_encoder",
    "extract_dense_features",
    "preprocess_nifti",
    "processor_config_identity",
    "resolve_precision",
]
