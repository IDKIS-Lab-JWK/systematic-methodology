"""Versioned, provenance-rich RAVE cache preparation for Pillar-0.

The cache is intentionally separate from Merlin's tensors.  It stores one
RAVE LZ4 archive per full 3-D NIfTI volume in raw Hounsfield units; Pillar-0's
eleven CT windows are applied later, at model input time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

try:  # Package imports.
    from . import CACHE_SCHEMA_VERSION, CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from .data import collect_required_rows, load_manifest_file, require_nifti_path
except ImportError:  # Direct imports from this directory.
    from __init__ import CACHE_SCHEMA_VERSION, CLASS_NAMES, DEFAULT_CACHE_NAMESPACE
    from data import collect_required_rows, load_manifest_file, require_nifti_path


DEFAULT_RAVE_REPOSITORY = "https://github.com/YalaLab/rave"
DEFAULT_RAVE_REVISION = "20adeb873021c864e9410fae8898b0c8da309769"
RATE_EVALS_PAD_VALUE = -1.0
_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CacheConfigurationError(ValueError):
    """The RAVE config does not describe a safe Pillar-0 cache contract."""


class CacheCompatibilityError(RuntimeError):
    """An existing namespace contains a different preprocessing plan."""


class CacheValidationError(RuntimeError):
    """A generated cache is incomplete or violates its declared contract."""


class RaveExecutionError(RuntimeError):
    """The external RAVE preprocessing process failed."""


@dataclass(frozen=True)
class RaveCacheContract:
    """Relevant preprocessing settings extracted from the supplied RAVE YAML."""

    modality: str
    anatomy: str
    output_shape_dhw: tuple[int, int, int]
    target_spacing_xyz_mm: tuple[float, float, float]
    conversion_backend: str
    compression: str
    exporter_profile: str | None
    value_representation: str = "raw_hounsfield_units"

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "anatomy": self.anatomy,
            "output_shape_dhw": list(self.output_shape_dhw),
            "target_spacing_xyz_mm": list(self.target_spacing_xyz_mm),
            "conversion_backend": self.conversion_backend,
            "compression": self.compression,
            "exporter_profile": self.exporter_profile,
            "value_representation": self.value_representation,
            "raw_archive_shape_contract": {
                "depth": f"1..{self.output_shape_dhw[0]}",
                "height": self.output_shape_dhw[1],
                "width": self.output_shape_dhw[2],
            },
            "post_cache_model_loader": {
                "center_pad_or_crop_to_dhw": list(self.output_shape_dhw),
                "pad_value_hu": RATE_EVALS_PAD_VALUE,
                "flip_depth_axis_after_pad_crop": True,
                "source": "YalaLab/rate-evals rve_dataset.py and rve_abd_ct.yaml",
            },
        }


@dataclass(frozen=True)
class CachePlan:
    """A deterministic preprocessing plan; constructing it never writes files."""

    namespace: str
    cache_directory: Path
    config_path: Path
    config_sha256: str
    split_manifest_path: Path
    split_manifest_sha256: str
    split_percentage: int | None
    contract: RaveCacheContract
    sources: tuple[dict, ...]
    source_fingerprint_mode: str
    provenance: Mapping[str, str]
    created_at_utc: str
    plan_fingerprint: str

    @property
    def source_csv_path(self) -> Path:
        return self.cache_directory / "source_series.csv"

    @property
    def artifact_directory(self) -> Path:
        return self.cache_directory / "artifacts"

    @property
    def plan_path(self) -> Path:
        return self.cache_directory / "cache_plan.json"

    @property
    def manifest_path(self) -> Path:
        return self.cache_directory / "cache_manifest.json"

    def to_dict(self) -> dict:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": "pillar0_rave_cache_plan",
            "created_at_utc": self.created_at_utc,
            "namespace": self.namespace,
            "cache_directory": str(self.cache_directory),
            "cache_format": "rave_lz4_raw_hu",
            "primary_input_format": "nifti_3d",
            "source_fingerprint_mode": self.source_fingerprint_mode,
            "source_count": len(self.sources),
            "split_manifest": {
                "path": str(self.split_manifest_path),
                "sha256": self.split_manifest_sha256,
                "percentage": self.split_percentage,
            },
            "rave_config": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
                "contract": self.contract.to_dict(),
            },
            "provenance": dict(self.provenance),
            "sources": [dict(row) for row in self.sources],
            "plan_fingerprint": self.plan_fingerprint,
        }


def load_cache_config(path: str | Path) -> dict:
    """Load a RAVE YAML config without making PyYAML a module dependency."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"RAVE config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Reading a RAVE config requires PyYAML (`pip install pyyaml`)."
        ) from exc
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CacheConfigurationError(
            f"Invalid RAVE YAML in {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise CacheConfigurationError(f"RAVE config must be a YAML mapping: {config_path}")
    return config


def cache_contract_from_config(config: Mapping) -> RaveCacheContract:
    """Derive shape/spacing/storage assumptions solely from the given config."""

    modality = str(config.get("modality", "")).upper()
    anatomy = str(config.get("anatomy", "")).lower()
    if modality != "CT":
        raise CacheConfigurationError(
            f"Pillar-0 abdomen cache requires CT modality, found {modality!r}"
        )
    if anatomy != "abdomen":
        raise CacheConfigurationError(
            f"Pillar-0 abdomen cache requires anatomy='abdomen', found {anatomy!r}"
        )

    processing = _mapping(config.get("processing"), "processing")
    crop_pad = _mapping(processing.get("crop_pad"), "processing.crop_pad")
    size_hw = _positive_int_tuple(crop_pad.get("size"), 2, "crop_pad.size")
    resampling = _mapping(
        processing.get("resampling"), "processing.resampling"
    )
    spacing_xyz = _positive_float_tuple(
        resampling.get("target_spacing"), 3, "resampling.target_spacing"
    )
    slice_selection = _mapping(
        processing.get("slice_selection"), "processing.slice_selection"
    )
    if slice_selection.get("enabled") is not True:
        raise CacheConfigurationError(
            "Pillar-0 cache config must enable deterministic slice selection"
        )
    depth = _positive_int_tuple(
        [slice_selection.get("slices")], 1, "slice_selection.slices"
    )[0]

    exporter = config.get("exporter")
    compression = None
    if isinstance(exporter, Mapping):
        compression = exporter.get("compression")
    if compression is None:
        compression = config.get("compression")
    exporter_profile_value = config.get("exporter_config")
    exporter_profile = (
        str(exporter_profile_value) if exporter_profile_value is not None else None
    )
    if compression is None and exporter_profile and "lz4" in exporter_profile.lower():
        compression = "lz4"
    if str(compression).lower() != "lz4":
        raise CacheConfigurationError(
            "The practical Pillar-0 cache must use RAVE LZ4 raw-volume storage; "
            f"resolved compression={compression!r}, exporter_config={exporter_profile!r}"
        )

    backend = str(processing.get("conversion_backend", "sitk")).lower()
    if backend not in {"sitk", "simpleitk"}:
        raise CacheConfigurationError(
            f"Expected the audited SimpleITK conversion backend, found {backend!r}"
        )
    return RaveCacheContract(
        modality=modality,
        anatomy=anatomy,
        output_shape_dhw=(depth, size_hw[0], size_hw[1]),
        target_spacing_xyz_mm=spacing_xyz,
        conversion_backend=backend,
        compression="lz4",
        exporter_profile=exporter_profile,
    )


def cache_directory(cache_root: str | Path, namespace: str) -> Path:
    """Return ``<cache_root>/Shared/<namespace>`` after traversal checks."""

    if not _NAMESPACE_PATTERN.fullmatch(str(namespace)) or namespace in {".", ".."}:
        raise ValueError(
            "Cache namespace must be 1-128 safe filename characters "
            f"([A-Za-z0-9._-]) and may not contain path separators: {namespace!r}"
        )
    shared_root = (Path(cache_root).resolve(strict=False) / "Shared").resolve(
        strict=False
    )
    target = (shared_root / namespace).resolve(strict=False)
    if shared_root not in target.parents:
        raise ValueError(f"Unsafe cache namespace resolved outside Shared: {namespace!r}")
    return target


def build_cache_plan(
    *,
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    cache_root: str | Path,
    namespace: str = DEFAULT_CACHE_NAMESPACE,
    config_path: str | Path,
    source_fingerprint_mode: str = "stat",
    expected_shape_dhw: Sequence[int] | None = None,
    expected_spacing_xyz_mm: Sequence[float] | None = None,
    provenance: Mapping[str, str] | None = None,
    require_source_files: bool = True,
) -> CachePlan:
    """Build a deterministic shared-cache plan without touching the filesystem."""

    fingerprint_mode = str(source_fingerprint_mode).lower()
    if fingerprint_mode not in {"stat", "sha256"}:
        raise ValueError("source_fingerprint_mode must be 'stat' or 'sha256'")

    dataset = Path(dataset_root).resolve(strict=False)
    split_path = Path(split_manifest_path).resolve(strict=False)
    rave_config_path = Path(config_path).resolve(strict=False)
    config = load_cache_config(rave_config_path)
    contract = cache_contract_from_config(config)
    if expected_shape_dhw is not None:
        expected_shape = tuple(int(value) for value in expected_shape_dhw)
        if contract.output_shape_dhw != expected_shape:
            raise CacheConfigurationError(
                f"RAVE config shape {contract.output_shape_dhw} does not match "
                f"requested Pillar-0 shape {expected_shape}"
            )
    if expected_spacing_xyz_mm is not None:
        expected_spacing = tuple(float(value) for value in expected_spacing_xyz_mm)
        if not _float_tuples_close(contract.target_spacing_xyz_mm, expected_spacing):
            raise CacheConfigurationError(
                f"RAVE config spacing {contract.target_spacing_xyz_mm} does not match "
                f"requested Pillar-0 spacing {expected_spacing}"
            )

    manifest = load_manifest_file(split_path)
    rows = collect_required_rows(
        dataset, manifest, require_files=require_source_files
    )
    sources: list[dict] = []
    for row in rows:
        source_path = require_nifti_path(
            row["source_path"], must_exist=require_source_files
        ).resolve(strict=False)
        source = dict(row)
        source["image"] = str(source_path)
        source["source_path"] = str(source_path)
        source["source_fingerprint"] = fingerprint_file(
            source_path,
            mode=fingerprint_mode,
            require_file=require_source_files,
        )
        sources.append(source)

    recorded_percentage = manifest.get("metadata", {}).get("percentage")
    percentage = int(recorded_percentage) if recorded_percentage is not None else None
    provenance_values = {
        "rave_repository": DEFAULT_RAVE_REPOSITORY,
        "rave_revision": DEFAULT_RAVE_REVISION,
    }
    if provenance:
        provenance_values.update(
            {str(key): str(value) for key, value in provenance.items()}
        )

    immutable_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "config_sha256": sha256_file(rave_config_path),
        "split_manifest_sha256": sha256_file(split_path),
        "split_percentage": percentage,
        "contract": contract.to_dict(),
        "source_fingerprint_mode": fingerprint_mode,
        "sources": sources,
        "provenance": provenance_values,
    }
    plan_fingerprint = sha256_json(immutable_payload)
    return CachePlan(
        namespace=namespace,
        cache_directory=cache_directory(cache_root, namespace),
        config_path=rave_config_path,
        config_sha256=immutable_payload["config_sha256"],
        split_manifest_path=split_path,
        split_manifest_sha256=immutable_payload["split_manifest_sha256"],
        split_percentage=percentage,
        contract=contract,
        sources=tuple(sources),
        source_fingerprint_mode=fingerprint_mode,
        provenance=provenance_values,
        created_at_utc=_utc_now(),
        plan_fingerprint=plan_fingerprint,
    )


def stratified_smoke_plan(
    plan: CachePlan,
    *,
    patients_per_class: int,
    namespace: str,
) -> CachePlan:
    """Derive an explicitly non-production, class-stratified smoke-test plan."""

    count = int(patients_per_class)
    if count <= 0:
        raise ValueError("patients_per_class must be positive.")
    if "smoke" not in namespace.lower():
        raise ValueError("A smoke-test cache namespace must contain 'smoke'.")
    selected = []
    for class_name in CLASS_NAMES:
        members = [row for row in plan.sources if row["label_str"] == class_name]
        if len(members) < count:
            raise ValueError(
                f"Smoke plan requested {count} {class_name} patients but found {len(members)}."
            )
        selected.extend(members[:count])
    provenance = {
        **dict(plan.provenance),
        "scope": "class_stratified_runtime_smoke_test_not_for_experiment_results",
        "smoke_patients_per_class": str(count),
    }
    immutable_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "config_sha256": plan.config_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "split_percentage": plan.split_percentage,
        "contract": plan.contract.to_dict(),
        "source_fingerprint_mode": plan.source_fingerprint_mode,
        "sources": selected,
        "provenance": provenance,
    }
    return replace(
        plan,
        namespace=namespace,
        cache_directory=cache_directory(plan.cache_directory.parents[1], namespace),
        sources=tuple(selected),
        provenance=provenance,
        created_at_utc=_utc_now(),
        plan_fingerprint=sha256_json(immutable_payload),
    )


def prepare_cache(
    plan: CachePlan,
    *,
    python_executable: str | Path | None = None,
    rave_module: str = "vision_engine",
    workers: int = 4,
    dry_run: bool = False,
    validate_volumes: bool = True,
    stop_on_error: bool = True,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    """Run a plan via RAVE, reconcile outputs, validate, and write a manifest.

    ``dry_run=True`` is strictly read-only: it returns the planned command and
    does not create the namespace, source CSV, logs, or manifest.
    """

    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    if not rave_module or any(character.isspace() for character in rave_module):
        raise ValueError(f"Unsafe/invalid RAVE module name: {rave_module!r}")
    command = rave_command(
        plan,
        python_executable=python_executable,
        rave_module=rave_module,
        workers=int(workers),
        stop_on_error=stop_on_error,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "filesystem_mutated": False,
            "plan": plan.to_dict(),
            "command": command,
        }

    existing_manifest = _load_json_if_present(plan.manifest_path)
    if existing_manifest and existing_manifest.get("status") == "complete":
        _require_matching_fingerprint(existing_manifest, plan)
        validation = validate_cache(
            plan.manifest_path, verify_volumes=validate_volumes
        )
        return {
            **existing_manifest,
            "reused_existing_cache": True,
            "validation": validation,
        }

    _create_cache_scaffold(plan)
    started_at = _utc_now()
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = _utc_now()
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    log_paths = _write_rave_logs(plan, started_at, stdout, stderr)
    invocation = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_log": str(log_paths[0]),
        "stderr_log": str(log_paths[1]),
    }
    _append_run_history(plan, invocation)

    if completed.returncode != 0:
        failure_manifest = {
            **plan.to_dict(),
            "kind": "pillar0_rave_cache_manifest",
            "status": "failed",
            "completed_at_utc": finished_at,
            "rave_invocation": invocation,
            "entries": [],
        }
        _atomic_write_json(plan.manifest_path, failure_manifest)
        tail = stderr.strip().splitlines()[-1] if stderr.strip() else "no stderr"
        raise RaveExecutionError(
            f"RAVE exited with code {completed.returncode}; last error: {tail}. "
            f"See {log_paths[1]}"
        )

    try:
        entries = reconcile_rave_mapping(
            plan, plan.artifact_directory / "mapping.csv"
        )
        validation = _validate_entries(
            entries,
            expected_shape_dhw=plan.contract.output_shape_dhw,
            verify_volumes=validate_volumes,
        )
    except Exception as exc:
        failure_manifest = {
            **plan.to_dict(),
            "kind": "pillar0_rave_cache_manifest",
            "status": "failed_validation",
            "completed_at_utc": _utc_now(),
            "rave_invocation": invocation,
            "failure": str(exc),
            "entries": [],
        }
        _atomic_write_json(plan.manifest_path, failure_manifest)
        raise

    complete_manifest = {
        **plan.to_dict(),
        "kind": "pillar0_rave_cache_manifest",
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "rave_invocation": invocation,
        "entry_count": len(entries),
        "entries": entries,
        "validation": validation,
    }
    _atomic_write_json(plan.manifest_path, complete_manifest)
    return complete_manifest


def rave_command(
    plan: CachePlan,
    *,
    python_executable: str | Path | None = None,
    rave_module: str = "vision_engine",
    workers: int = 4,
    stop_on_error: bool = True,
) -> list[str]:
    """Build the argument-vector invocation; no shell is used."""

    executable = str(python_executable or sys.executable)
    command = [
        executable,
        "-m",
        rave_module,
        "process",
        "--config",
        str(plan.config_path),
        "--input-series-csv",
        str(plan.source_csv_path),
        "--path-column",
        "series_path",
        "--output",
        str(plan.artifact_directory),
        "--workers",
        str(int(workers)),
    ]
    if stop_on_error:
        command.append("--stop-on-error")
    return command


def reconcile_rave_mapping(
    plan: CachePlan, mapping_path: str | Path
) -> list[dict]:
    """Join RAVE's mapping.csv to source labels without relying on row order."""

    path = Path(mapping_path)
    if not path.is_file():
        raise CacheValidationError(f"RAVE mapping file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CacheValidationError(f"RAVE mapping is empty: {path}")

    outputs: dict[str, Path] = {}
    for index, row in enumerate(rows, start=2):
        if not row.get("source_path") or not row.get("output_path"):
            raise CacheValidationError(
                f"RAVE mapping row {index} lacks source_path/output_path"
            )
        source_key = _normalised_path_key(row["source_path"])
        if source_key in outputs:
            raise CacheValidationError(
                f"Duplicate source_path in RAVE mapping: {row['source_path']}"
            )
        output = Path(row["output_path"])
        if not output.is_absolute():
            output = path.parent / output
        outputs[source_key] = output.resolve(strict=False)

    expected_keys = {
        _normalised_path_key(source["source_path"]) for source in plan.sources
    }
    unexpected = set(outputs) - expected_keys
    if unexpected:
        raise CacheValidationError(
            f"RAVE mapping contains {len(unexpected)} unexpected source(s)"
        )

    entries: list[dict] = []
    missing_sources: list[str] = []
    for source in plan.sources:
        source_key = _normalised_path_key(source["source_path"])
        cached_path = outputs.get(source_key)
        if cached_path is None:
            missing_sources.append(source["source_path"])
            continue
        entry = {
            "cache_path": str(cached_path),
            "patient_id": source["patient_id"],
            "label": int(source["label"]),
            "label_str": source["label_str"],
            "source_path": source["source_path"],
            "accession": source["accession"],
            "series_number": int(source["series_number"]),
            "source_fingerprint": source["source_fingerprint"],
            "shape": None,
            "dtype": None,
        }
        entries.append(entry)
    if missing_sources:
        raise CacheValidationError(
            f"RAVE mapping is missing {len(missing_sources)} expected source(s); "
            f"first={missing_sources[0]}"
        )
    return entries


def load_cache_manifest(
    path: str | Path,
    *,
    expected_namespace: str | None = None,
    require_complete: bool = True,
) -> dict:
    """Load a cache manifest with schema/status/namespace checks."""

    candidate = Path(path)
    manifest_path = candidate / "cache_manifest.json" if candidate.is_dir() else candidate
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pillar-0 cache manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheCompatibilityError(
            f"Unsupported cache schema {manifest.get('schema_version')!r}; "
            f"expected {CACHE_SCHEMA_VERSION!r}"
        )
    if require_complete and manifest.get("status") != "complete":
        raise CacheValidationError(
            f"Cache manifest status is {manifest.get('status')!r}, not 'complete'"
        )
    if expected_namespace and manifest.get("namespace") != expected_namespace:
        raise CacheCompatibilityError(
            f"Cache namespace mismatch: expected {expected_namespace!r}, "
            f"found {manifest.get('namespace')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CacheValidationError("Cache manifest has no entries list")
    return manifest


def rows_from_cache_manifest(
    path_or_manifest: str | Path | Mapping,
    *,
    expected_namespace: str | None = None,
) -> list[dict]:
    """Return training-ready cached rows with stable identity field names."""

    if isinstance(path_or_manifest, Mapping):
        manifest = dict(path_or_manifest)
        if manifest.get("status") != "complete":
            raise CacheValidationError("Cannot load rows from an incomplete cache")
    else:
        manifest = load_cache_manifest(
            path_or_manifest, expected_namespace=expected_namespace
        )
    required = {"cache_path", "patient_id", "label", "label_str"}
    result: list[dict] = []
    for index, entry in enumerate(manifest.get("entries", [])):
        missing = required - set(entry)
        if missing:
            raise CacheValidationError(
                f"Cache entry {index} lacks required fields: {sorted(missing)}"
            )
        result.append(dict(entry))
    return result


def load_cached_rows(
    cache_directory_or_manifest: str | Path | Mapping,
    requested_rows: Sequence[Mapping] | None = None,
    *,
    expected_namespace: str | None = None,
) -> list[dict]:
    """Load cache rows, optionally resolving a requested split in its row order.

    Identity is patient-level.  Every cache entry is checked for duplicate IDs
    and for the canonical class/index relationship.  When ``requested_rows`` is
    supplied, all requested patients must exist exactly once and their labels
    must match the cache manifest.
    """

    cached = rows_from_cache_manifest(
        cache_directory_or_manifest, expected_namespace=expected_namespace
    )
    by_patient: dict[str, dict] = {}
    for row in cached:
        patient_id = str(row["patient_id"])
        if patient_id in by_patient:
            raise CacheValidationError(
                f"Duplicate patient_id in cache manifest: {patient_id}"
            )
        label = int(row["label"])
        label_str = str(row["label_str"])
        if label not in range(len(CLASS_NAMES)) or CLASS_NAMES[label] != label_str:
            raise CacheValidationError(
                f"Invalid class/index pair for patient {patient_id}: "
                f"label={label}, label_str={label_str!r}"
            )
        by_patient[patient_id] = row
    if requested_rows is None:
        selected = cached
        _require_cached_paths(selected)
        return selected

    selected: list[dict] = []
    requested_ids: set[str] = set()
    for requested in requested_rows:
        missing_identity = {"patient_id", "label", "label_str"} - set(requested)
        if missing_identity:
            raise CacheValidationError(
                f"Requested row lacks identity fields: {sorted(missing_identity)}"
            )
        patient_id = str(requested["patient_id"])
        if patient_id in requested_ids:
            raise CacheValidationError(
                f"Duplicate patient_id in requested rows: {patient_id}"
            )
        requested_ids.add(patient_id)
        cached_row = by_patient.get(patient_id)
        if cached_row is None:
            raise CacheValidationError(
                f"Requested patient is absent from the Pillar-0 cache: {patient_id}"
            )
        expected_pair = (int(requested["label"]), str(requested["label_str"]))
        cached_pair = (int(cached_row["label"]), str(cached_row["label_str"]))
        if expected_pair != cached_pair:
            raise CacheValidationError(
                f"Cache label mismatch for patient {patient_id}: requested "
                f"{expected_pair}, cached {cached_pair}"
            )
        selected.append(dict(cached_row))
    _require_cached_paths(selected)
    return selected


def load_cached_volume(path: str | Path):
    """Load one RAVE archive as raw-HU ``float32 [1,D,H,W]``.

    RAVE and PyTorch are lazy imports so data/config commands remain usable in
    lightweight environments.  Windowing is intentionally not performed here.
    """

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cached RAVE volume not found: {cache_path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Loading Pillar-0 cache volumes requires PyTorch.") from exc
    try:
        import rve
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Loading Pillar-0 cache volumes requires the pinned YalaLab RAVE "
            "package. Install it in the Pillar-0 environment; Merlin's cache is "
            "not a substitute."
        ) from exc
    volume = rve.load_sample(str(cache_path), device="cpu")
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)
    if volume.ndim != 4 or int(volume.shape[0]) != 1:
        raise CacheValidationError(
            "Expected a raw RAVE volume shaped [D,H,W] (returned as [1,D,H,W]); "
            f"received {tuple(volume.shape)} from {cache_path}"
        )
    return volume.to(dtype=torch.float32).contiguous()


def load_pillar_volume(
    path: str | Path,
    *,
    target_shape_dhw: Sequence[int] = (384, 384, 384),
    pad_value: float = RATE_EVALS_PAD_VALUE,
    flip_depth: bool = True,
):
    """Decode raw RAVE HU data and reproduce RATE-Evals model-input geometry."""

    import torch
    import torch.nn.functional as torch_f

    target = tuple(int(value) for value in target_shape_dhw)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError(f"Invalid target_shape_dhw: {target_shape_dhw}")
    volume = load_cached_volume(path)
    # Center crop any excessive dimension first.
    for dimension, size in zip((1, 2, 3), target):
        current = int(volume.shape[dimension])
        if current > size:
            start = (current - size) // 2
            slices = [slice(None)] * 4
            slices[dimension] = slice(start, start + size)
            volume = volume[tuple(slices)]
    depth, height, width = (int(value) for value in volume.shape[1:])
    differences = (target[0] - depth, target[1] - height, target[2] - width)
    d_before, h_before, w_before = (value // 2 for value in differences)
    padding = (
        w_before,
        differences[2] - w_before,
        h_before,
        differences[1] - h_before,
        d_before,
        differences[0] - d_before,
    )
    if any(padding):
        volume = torch_f.pad(volume, padding, mode="constant", value=float(pad_value))
    if tuple(volume.shape[1:]) != target:
        raise CacheValidationError(
            f"RATE-Evals pad/crop failed: expected {target}, got {tuple(volume.shape[1:])}."
        )
    if flip_depth:
        volume = torch.flip(volume, dims=[1])
    return volume.contiguous()


def validate_cache(
    path_or_manifest: str | Path | Mapping,
    *,
    verify_volumes: bool = True,
) -> dict:
    """Validate entry paths and, by default, every declared output shape."""

    if isinstance(path_or_manifest, Mapping):
        manifest = dict(path_or_manifest)
    else:
        manifest = load_cache_manifest(path_or_manifest)
    contract = manifest.get("rave_config", {}).get("contract", {})
    shape = contract.get("output_shape_dhw")
    if not isinstance(shape, list) or len(shape) != 3:
        raise CacheValidationError("Cache manifest lacks output_shape_dhw contract")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CacheValidationError("Cache manifest lacks entries")
    validation = _validate_entries(
        entries,
        expected_shape_dhw=tuple(int(value) for value in shape),
        verify_volumes=verify_volumes,
    )
    declared_count = manifest.get("entry_count", manifest.get("source_count"))
    if declared_count is not None and int(declared_count) != len(entries):
        raise CacheValidationError(
            f"Manifest entry-count mismatch: declared {declared_count}, found {len(entries)}"
        )
    return validation


def fingerprint_file(
    path: str | Path,
    *,
    mode: str = "stat",
    require_file: bool = True,
) -> dict:
    """Return a reproducible source fingerprint (stat, optionally content SHA)."""

    candidate = Path(path)
    if not candidate.is_file():
        if require_file:
            raise FileNotFoundError(f"Source file not found: {candidate}")
        return {"exists": False}
    stat = candidate.stat()
    result = {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if mode == "sha256":
        result["sha256"] = sha256_file(candidate)
    elif mode != "stat":
        raise ValueError("Fingerprint mode must be 'stat' or 'sha256'")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _create_cache_scaffold(plan: CachePlan) -> None:
    existing_plan = _load_json_if_present(plan.plan_path)
    if existing_plan is not None:
        _require_matching_fingerprint(existing_plan, plan)
    elif plan.cache_directory.exists() and any(plan.cache_directory.iterdir()):
        raise CacheCompatibilityError(
            f"Refusing to adopt non-empty cache namespace without a plan: "
            f"{plan.cache_directory}. Use a new namespace."
        )
    plan.cache_directory.mkdir(parents=True, exist_ok=True)
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(plan.plan_path, plan.to_dict())
    _write_source_csv(plan.source_csv_path, plan.sources)


def _write_source_csv(path: Path, sources: Iterable[Mapping]) -> None:
    rows = list(sources)
    fieldnames = [
        "series_path",
        "accession",
        "series_number",
        "series_description",
    ]
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for source in rows:
        writer.writerow(
            {
                "series_path": source["source_path"],
                "accession": source["accession"],
                "series_number": int(source["series_number"]),
                "series_description": (
                    f"Pillar0 five-class source: {source['label_str']}"
                ),
            }
        )
    lines.append(buffer.getvalue())
    _atomic_write_text(path, "".join(lines))


def _validate_entries(
    entries: Sequence[dict],
    *,
    expected_shape_dhw: tuple[int, int, int],
    verify_volumes: bool,
) -> dict:
    if not entries:
        raise CacheValidationError("Cache contains no entries")
    seen_patients: set[str] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        patient_id = str(entry.get("patient_id", ""))
        if not patient_id or patient_id in seen_patients:
            raise CacheValidationError(
                f"Empty or duplicate patient_id in cache: {patient_id!r}"
            )
        seen_patients.add(patient_id)
        cache_path = Path(str(entry.get("cache_path", "")))
        path_key = _normalised_path_key(cache_path)
        if path_key in seen_paths:
            raise CacheValidationError(f"Duplicate cache_path: {cache_path}")
        seen_paths.add(path_key)
        if not cache_path.is_file():
            raise CacheValidationError(f"Cached RAVE output missing: {cache_path}")
        if verify_volumes:
            volume = load_cached_volume(cache_path)
            actual_shape = tuple(int(value) for value in volume.shape[1:])
            expected_depth, expected_height, expected_width = expected_shape_dhw
            valid_raw_shape = (
                0 < actual_shape[0] <= expected_depth
                and actual_shape[1] == expected_height
                and actual_shape[2] == expected_width
            )
            if not valid_raw_shape:
                raise CacheValidationError(
                    f"Cache shape mismatch for {patient_id}: expected raw archive "
                    f"depth in [1,{expected_depth}] and H/W "
                    f"{(expected_height, expected_width)}, found {actual_shape} "
                    f"({cache_path})"
                )
            entry["shape"] = list(actual_shape)
            entry["dtype"] = str(volume.dtype).replace("torch.", "")
            del volume
    return {
        "status": "valid",
        "level": "volume_shape_and_paths" if verify_volumes else "paths_only",
        "validated_at_utc": _utc_now(),
        "entry_count": len(entries),
        "expected_model_input_shape_dhw": list(expected_shape_dhw),
        "raw_archive_depth_range": [1, int(expected_shape_dhw[0])],
        "post_cache_pad_value_hu": RATE_EVALS_PAD_VALUE,
        "post_cache_flip_depth_axis": True,
        "raw_value_representation": "hounsfield_units",
    }


def _write_rave_logs(
    plan: CachePlan, started_at: str, stdout: str, stderr: str
) -> tuple[Path, Path]:
    token = started_at.replace(":", "-").replace("+", "_")
    log_dir = plan.cache_directory / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"rave_{token}.stdout.log"
    stderr_path = log_dir / f"rave_{token}.stderr.log"
    _atomic_write_text(stdout_path, stdout)
    _atomic_write_text(stderr_path, stderr)
    return stdout_path, stderr_path


def _append_run_history(plan: CachePlan, invocation: Mapping) -> None:
    history = plan.cache_directory / "cache_runs.jsonl"
    with history.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(invocation), sort_keys=True) + "\n")


def _require_matching_fingerprint(existing: Mapping, plan: CachePlan) -> None:
    recorded = existing.get("plan_fingerprint")
    if recorded != plan.plan_fingerprint:
        raise CacheCompatibilityError(
            f"Cache namespace {plan.namespace!r} already belongs to a different "
            "split/config/source fingerprint. Preserve it and choose a new versioned "
            f"namespace instead of overwriting {plan.cache_directory}."
        )


def _load_json_if_present(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CacheCompatibilityError(f"Expected JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise CacheConfigurationError(f"RAVE config needs a {name} mapping")
    return value


def _positive_int_tuple(value: object, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CacheConfigurationError(f"{name} must contain {length} values")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CacheConfigurationError(f"{name} contains a non-integer value") from exc
    if any(item <= 0 for item in result):
        raise CacheConfigurationError(f"{name} values must be positive: {result}")
    return result


def _positive_float_tuple(
    value: object, length: int, name: str
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CacheConfigurationError(f"{name} must contain {length} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CacheConfigurationError(f"{name} contains a non-numeric value") from exc
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise CacheConfigurationError(
            f"{name} values must be finite and positive: {result}"
        )
    return result


def _float_tuples_close(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-6
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _normalised_path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _require_cached_paths(rows: Sequence[Mapping]) -> None:
    for row in rows:
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.is_file():
            raise CacheValidationError(
                f"Cached RAVE output for patient {row['patient_id']} is missing: "
                f"{cache_path}"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_NAMESPACE",
    "DEFAULT_RAVE_REPOSITORY",
    "DEFAULT_RAVE_REVISION",
    "RATE_EVALS_PAD_VALUE",
    "CacheCompatibilityError",
    "CacheConfigurationError",
    "CachePlan",
    "CacheValidationError",
    "RaveCacheContract",
    "RaveExecutionError",
    "build_cache_plan",
    "cache_contract_from_config",
    "cache_directory",
    "fingerprint_file",
    "load_cache_config",
    "load_cache_manifest",
    "load_cached_rows",
    "load_cached_volume",
    "load_pillar_volume",
    "prepare_cache",
    "rave_command",
    "reconcile_rave_mapping",
    "rows_from_cache_manifest",
    "sha256_file",
    "sha256_json",
    "stratified_smoke_plan",
    "validate_cache",
]
