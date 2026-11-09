"""Train-only physical-space planning, with an explicit whole-volume policy."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence, Mapping, Any

import nibabel as nib
import numpy as np

from _linear_probe.common import ContractError, fingerprint, rows_fingerprint

RATIONALE = {
    "input": "Existing unsplit NIfTI and patient manifests; no new split or DICOM series selection.",
    "orientation": "Axis-aligned RAS physical coordinates for consistent anatomy; use the full affine, including obliquity.",
    "reference_spacing": "Per-axis median of training-patient native spacings (Med3D section 3.1). Validation/test do not fit statistics.",
    "working_spacing": "Per-axis max(reference median, largest training scan world-coordinate span/(canvas-1)). Fit whole training volumes without organ/lesion crops; coarsening is a task adaptation to memory constraints.",
    "canvas": "Fixed, GPU-benchmarked voxel box; dimensions are a project resource choice, not a published CT optimum.",
    "oversize": "Keep the fitted spacing unchanged for every validation/test patient. Expand the centered canvas to multiples of 32 when needed for full coverage; evaluate one patient at a time and log dimensions. Never silently coarsen or crop on memory failure.",
    "interpolation": "Physical-space linear interpolation, preceded by Gaussian anti-alias filtering on axes being downsampled. Single geometric interpolation avoids a second arbitrary resize.",
    "hu_window": "Clip to [-200,250] HU, the Med3D LiTS abdominal precedent, then map to [0,1]. Mapping and transfer to five-class classification/DenseNet are adaptations; other HU ranges may contain useful information.",
    "padding": "Center the full scan in the physical canvas; outside-acquisition fill -1000 HU becomes zero after windowing. No anatomy crop or mask.",
    "augmentation": "No geometric augmentation in the baseline. Preserve laterality and complete field of view without lesion boxes; this differs from Med3D pretraining and is explicitly adapted.",
    "limits": "Coverage is geometric, not proof that small diagnostic findings survive downsampling. Resolution sensitivity uses development folds only.",
}
SOURCES = {
    "med3d": "https://arxiv.org/pdf/1904.00625",
    "nnunet": "https://doi.org/10.1038/s41592-020-01008-z",
    "leakage": "https://doi.org/10.1016/j.patter.2023.100804",
    "antialias": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize",
}


def source_stat(path: str | Path) -> dict:
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def image_geometry(path: str | Path) -> dict:
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ContractError(f"Expected a non-degenerate 3D NIfTI: {path}")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ContractError(f"NIfTI spatial units must explicitly be mm: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise ContractError(f"Invalid NIfTI affine: {path}")
    native = nib.affines.voxel_sizes(affine)
    order = np.argsort(nib.orientations.io_orientation(affine)[:, 0].astype(int))
    # Boundary corners, not just voxel centers: the output box covers the entire acquired support.
    corners = np.asarray(list(itertools.product(*[(-0.5, n - 0.5) for n in image.shape])))
    world = nib.affines.apply_affine(affine, corners)
    low, high = world.min(axis=0), world.max(axis=0)
    return {
        "source": source_stat(path), "native_shape": list(image.shape),
        "native_spacing_ras_mm": native[order].tolist(),
        "orientation": list(nib.aff2axcodes(affine)),
        "world_low_mm": low.tolist(), "world_high_mm": high.tolist(),
        "world_span_mm": (high - low).tolist(), "affine": affine.tolist(),
    }


def fit_plan(rows: Sequence[Mapping[str, Any]], canvas: Sequence[int], *, geometries=None) -> dict:
    shape = np.asarray(canvas, dtype=int)
    if shape.shape != (3,) or np.any(shape < 32) or np.any(shape % 32):
        raise ContractError("Canvas must contain three multiples of 32, each at least 32")
    if not rows:
        raise ContractError("Cannot fit preprocessing without training patients")
    records = [image_geometry(r["image_path"]) for r in rows] if geometries is None else list(geometries)
    if len(records) != len(rows):
        raise ContractError("Geometry count does not match training rows")
    spacing = np.asarray([r["native_spacing_ras_mm"] for r in records])
    spans = np.asarray([r["world_span_mm"] for r in records])
    reference = np.median(spacing, axis=0)
    working = np.maximum(reference, spans.max(axis=0) / (shape - 1))
    numeric = {
        "schema_version": 1, "canvas_xyz": shape.tolist(),
        "working_spacing_xyz_mm": working.tolist(), "hu_window": [-200.0, 250.0],
        "outside_hu": -1000.0, "interpolation_order": 1,
        "antialias": "native-axis Gaussian sigma=max((target/native-1)/2,0)",
        "orientation": "axis_aligned_RAS", "oversize_policy": "expand_canvas_keep_spacing",
    }
    return {
        "schema_version": 1, "training_patient_count": len(rows),
        "training_rows_sha256": rows_fingerprint(rows),
        "training_geometry_sha256": fingerprint(records),
        "reference_spacing_xyz_mm": reference.tolist(),
        "training_max_span_xyz_mm": spans.max(axis=0).tolist(),
        "spacing_quantiles_mm": {str(q): np.percentile(spacing, q, axis=0).tolist() for q in (0, 10, 50, 90, 100)},
        "span_quantiles_mm": {str(q): np.percentile(spans, q, axis=0).tolist() for q in (0, 50, 95, 100)},
        "numeric": numeric, "numeric_sha256": fingerprint(numeric),
        "rationale": RATIONALE, "sources": SOURCES,
    }


def grid_for_image(geometry: Mapping[str, Any], plan: Mapping[str, Any]) -> dict:
    numeric = plan["numeric"]
    shape = np.asarray(numeric["canvas_xyz"], dtype=int)
    reference = np.asarray(numeric["working_spacing_xyz_mm"], dtype=float)
    span = np.asarray(geometry["world_span_mm"], dtype=float)
    required = np.ceil(span / reference - 1e-8).astype(int) + 1
    expanded = np.maximum(shape, ((required + 31) // 32) * 32)
    enlarged = bool(np.any(expanded > shape))
    shape = expanded
    spacing = reference
    center = (np.asarray(geometry["world_low_mm"]) + np.asarray(geometry["world_high_mm"])) / 2
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = center - (shape - 1) * spacing / 2
    return {"shape_xyz": shape.tolist(), "affine": affine.tolist(),
            "actual_spacing_xyz_mm": spacing.tolist(), "expanded_canvas": enlarged,
            "geometric_coverage": "full_acquired_world_bounding_box"}
