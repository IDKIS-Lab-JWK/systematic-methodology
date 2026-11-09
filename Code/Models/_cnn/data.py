"""Deterministic, physically located volumes and validated disk-cache shards."""
from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter
from filelock import FileLock

from _linear_probe.common import atomic_write_json, fingerprint, read_json, sha256_file, ContractError
from .geometry import image_geometry, grid_for_image, source_stat


def preprocess(path, plan):
    started = time.perf_counter()
    geometry = image_geometry(path)
    grid = grid_for_image(geometry, plan)
    image = nib.load(str(path))
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ContractError(f"Non-finite input intensities: {path}")
    # Project target-grid resolution onto each native voxel direction. For an
    # orthogonal oblique acquisition this avoids treating array axes as RAS axes.
    linear = np.asarray(image.affine[:3, :3])
    native = nib.affines.voxel_sizes(image.affine)
    directions = linear / native
    target_spacing = np.asarray(grid["actual_spacing_xyz_mm"])
    projected = np.sqrt((directions ** 2).T @ (target_spacing ** 2))
    sigma = np.maximum((projected / native - 1.0) / 2.0, 0.0)
    if np.any(sigma > 1e-6):
        values = gaussian_filter(values, sigma=sigma, mode="nearest", output=np.float32)
    source = nib.Nifti1Image(values, image.affine)
    target = resample_from_to(source, (tuple(grid["shape_xyz"]), np.asarray(grid["affine"])),
                              order=1, mode="constant", cval=-1000.0)
    result = np.asarray(target.dataobj, dtype=np.float32)
    lo, hi = plan["numeric"]["hu_window"]
    result = (np.clip(result, lo, hi) - lo) / (hi - lo)
    # PyTorch [C,D,H,W] = [1,S,A,R]; world-space planning is XYZ=(R,A,S).
    result = np.ascontiguousarray(result.transpose(2, 1, 0)[None])
    metadata = {"grid": grid, "antialias_sigma_native_voxels": sigma.tolist(),
                "source_geometry": geometry, "preprocessing_seconds": time.perf_counter() - started}
    return result, metadata


class VolumeCache:
    def __init__(self, root, implementation_identity):
        self.root = Path(root)
        self.implementation_identity = implementation_identity
        self.verified = set()
        self.source_hashes = {}
        self.hits = 0
        self.misses = 0
        self.preprocessing_seconds = 0.0
        self.namespaces = set()

    def get(self, path, plan):
        import torch
        from _supervised_adaptation.runtime import atomic_torch_save
        stat = source_stat(path)
        namespace = fingerprint({"numeric": plan["numeric"], "implementation": self.implementation_identity})
        key = fingerprint(stat)
        # Short path components stay below common Windows path limits. Full
        # digests remain in receipts/owner markers; a collision fails closed.
        folder = self.root / namespace[:24]
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "cnn_cache_owner.json"
        if not marker.exists():
            atomic_write_json(marker, {"namespace": namespace, "kind": "cnn_preprocessed_volumes_v1"})
        elif read_json(marker) != {"namespace": namespace, "kind": "cnn_preprocessed_volumes_v1"}:
            raise ContractError(f"Cache ownership mismatch: {folder}")
        self.namespaces.add(namespace)
        artifact, receipt = folder / f"{key[:24]}.pt", folder / f"{key[:24]}.json"
        with FileLock(str(folder / f"{key[:24]}.lock"), timeout=600):
            # Source content is hashed once per process even on cache hits.
            if key not in self.source_hashes:
                self.source_hashes[key] = sha256_file(path)
            identity = {"source": stat, "source_sha256": self.source_hashes[key],
                        "namespace": namespace, "numeric_sha256": fingerprint(plan["numeric"])}
            if artifact.is_file() and receipt.is_file():
                saved = read_json(receipt)
                if saved["identity"] != identity:
                    raise ContractError(f"Source/cache identity changed: {receipt}")
                check_key = (str(artifact), artifact.stat().st_mtime_ns, artifact.stat().st_size)
                if check_key not in self.verified:
                    if sha256_file(artifact) != saved["artifact_sha256"]:
                        raise ContractError(f"Corrupt CNN cache: {artifact}")
                    self.verified.add(check_key)
                payload = torch.load(artifact, weights_only=True, map_location="cpu")
                tensor = payload["image"]
                expected_grid = grid_for_image(image_geometry(path), plan)
                if saved["metadata"]["grid"] != expected_grid:
                    raise ContractError(f"Invalid cached geometry: {artifact}")
                if list(tensor.shape) != [1, *reversed(expected_grid["shape_xyz"])] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ContractError(f"Invalid cached CNN volume: {artifact}")
                self.hits += 1
                return tensor, saved["metadata"]
            # Incomplete/orphan shards are recomputed, never adopted.
            array, metadata = preprocess(path, plan)
            if source_stat(path) != stat:
                raise ContractError(f"Input changed while preprocessing: {path}")
            tensor = torch.from_numpy(array)
            atomic_torch_save({"image": tensor}, artifact)
            atomic_write_json(receipt, {"identity": identity, "artifact_sha256": sha256_file(artifact), "metadata": metadata})
            self.misses += 1
            self.preprocessing_seconds += metadata["preprocessing_seconds"]
            return tensor, metadata

    def release(self):
        """Only remove owned numeric namespaces after sealed work; caller holds cache lease."""
        import shutil
        root = self.root.resolve()
        for namespace in sorted(self.namespaces):
            folder = (root / namespace[:24]).resolve()
            if folder.parent != root or folder.name != namespace[:24] or len(namespace) != 64:
                raise ContractError(f"Unsafe CNN cache cleanup path: {folder}")
            marker = folder / "cnn_cache_owner.json"
            if not folder.exists():
                continue
            if not marker.is_file() or read_json(marker) != {"namespace": namespace, "kind": "cnn_preprocessed_volumes_v1"}:
                raise ContractError(f"Refusing to delete unowned cache: {folder}")
            shutil.rmtree(folder)
        self.namespaces.clear()
        self.verified.clear()
