"""Official-code-backed whole-volume FlexiCT preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import INPUT_SHAPE_DHW
except ImportError:  # pragma: no cover
    from __init__ import INPUT_SHAPE_DHW


def preprocessing_contract() -> dict:
    return {
        "schema_version": 1,
        "source": {
            "basis": "FlexiCT official CT-RATE VLM evaluation code",
            "repository_revision": "fd74ac06a31813fe02805674e96f811e9b2a5c43",
            "reference_file": "downstream/vlm/ct_rate_eval.py and downstream/vlm/dataset.py",
            "task_adaptation": "explicit LPS reorientation before the official 2 mm whole-volume recipe",
        },
        "orientation": "LPS",
        "axis_convention": "SimpleITK XYZ metadata; model tensor CDHW with array order ZYX",
        "target_spacing_xyz_mm": [2.0, 2.0, 2.0],
        "resampling": (
            "SimpleITK linear; identity physical transform; default background 0 HU; "
            "output pixel type retained from input"
        ),
        "intensity": "clip [-1000,1000], then whole-volume z-score",
        "std_definition": "torch.std correction=1, matching the official dataset implementation",
        "padding": "symmetric constant padding with normalized volume minimum",
        "crop": "center crop",
        "output_shape_cdhw": [1, *INPUT_SHAPE_DHW],
        "dtype": "float32",
    }


def _require_runtime():
    try:
        import SimpleITK as sitk
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - runtime environment check
        raise RuntimeError(
            "FlexiCT preprocessing requires SimpleITK and PyTorch; install requirements.txt"
        ) from exc
    return sitk, torch, functional


def _center_pad_crop(tensor, target: tuple[int, int, int], functional):
    if len(target) != 3 or any(int(value) <= 0 for value in target):
        raise ValueError(f"target must contain three positive spatial sizes, got {target}")
    if tensor.ndim < 3:
        raise ValueError(f"tensor must have at least three spatial dimensions, got {tensor.ndim}")
    current = tuple(int(value) for value in tensor.shape[-3:])
    padding = []
    for size, wanted in reversed(list(zip(current, target, strict=True))):
        missing = max(0, wanted - size)
        before = missing // 2
        after = missing - before
        padding.extend([before, after])
    if any(padding):
        tensor = functional.pad(tensor, tuple(padding), value=float(tensor.amin().item()))
    starts = [
        (int(size) - wanted) // 2
        for size, wanted in zip(tensor.shape[-3:], target, strict=True)
    ]
    d, h, w = starts
    td, th, tw = target
    return tensor[..., d : d + td, h : h + th, w : w + tw]


def preprocess_nifti(path: str | Path) -> tuple[Any, dict]:
    sitk, torch, functional = _require_runtime()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    image = sitk.ReadImage(str(source))
    if int(image.GetDimension()) != 3:
        raise RuntimeError(f"FlexiCT requires a scalar 3D image, got dimension {image.GetDimension()}")
    if int(image.GetNumberOfComponentsPerPixel()) != 1:
        raise RuntimeError("FlexiCT requires a single-component CT volume")
    oriented = sitk.DICOMOrient(image, "LPS")
    original_spacing = tuple(float(value) for value in oriented.GetSpacing())
    original_size = tuple(int(value) for value in oriented.GetSize())
    target_spacing = (2.0, 2.0, 2.0)
    target_size = tuple(
        max(1, int(round(size * spacing / wanted)))
        for size, spacing, wanted in zip(original_size, original_spacing, target_spacing, strict=True)
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(oriented.GetOrigin())
    resampler.SetOutputDirection(oriented.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    # Match downstream/vlm/dataset.py::resample_to_isotropic. SimpleITK's
    # outside value is 0; setting it explicitly makes that behavior auditable.
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(oriented.GetPixelID())
    resampled = resampler.Execute(oriented)
    array = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)
    tensor = torch.from_numpy(array).unsqueeze(0).to(dtype=torch.float32)
    tensor = tensor.clamp_(-1000.0, 1000.0)
    mean = tensor.mean()
    std = tensor.std(correction=1)
    if not torch.isfinite(std) or float(std) < 1e-6:
        tensor = tensor - mean
        std_value = 0.0
    else:
        tensor = (tensor - mean) / std
        std_value = float(std)
    tensor = _center_pad_crop(tensor, INPUT_SHAPE_DHW, functional).contiguous()
    if tuple(tensor.shape) != (1, *INPUT_SHAPE_DHW):
        raise RuntimeError(f"Unexpected FlexiCT input shape: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite FlexiCT input generated for {source}")
    metadata = {
        "source": str(source.resolve()),
        "source_spacing_xyz_mm": list(original_spacing),
        "source_size_xyz": list(original_size),
        "resampled_spacing_xyz_mm": [float(value) for value in resampled.GetSpacing()],
        "resampled_size_xyz": [int(value) for value in resampled.GetSize()],
        "resampled_direction": [float(value) for value in resampled.GetDirection()],
        "orientation": "LPS",
        "output_shape_cdhw": list(tensor.shape),
        "clipped_mean": float(mean),
        "clipped_std_correction_1": std_value,
        "output_min": float(tensor.min()),
        "output_max": float(tensor.max()),
    }
    return tensor, metadata


__all__ = ["preprocess_nifti", "preprocessing_contract"]
