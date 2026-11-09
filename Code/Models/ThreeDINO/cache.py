"""Dedicated persistent cache for 3DINO's deterministic CT prefix.

This cache never reuses Merlin or Pillar-0 artifacts.  A plan binds the split
manifest, source file metadata, preprocessing contract, and namespace before
MONAI is allowed to materialize any tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:  # Package imports.
    from . import (
        CACHE_SCHEMA_VERSION,
        CLASS_NAMES,
        DEFAULT_CACHE_NAMESPACE,
        PREFIX_SIZE,
    )
    from .data import (
        collect_required_rows,
        load_manifest_file,
        make_dataset,
        make_prefix_dataset,
        persistent_cache_key,
        preprocessing_contract,
    )
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import (
        CACHE_SCHEMA_VERSION,
        CLASS_NAMES,
        DEFAULT_CACHE_NAMESPACE,
        PREFIX_SIZE,
    )
    from data import (
        collect_required_rows,
        load_manifest_file,
        make_dataset,
        make_prefix_dataset,
        persistent_cache_key,
        preprocessing_contract,
    )


_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_INTEGRITY_SCHEMA = "threedino-cache-artifact-integrity-v1"
_ARTIFACT_INTEGRITY_KIND = "threedino_deterministic_prefix_artifact_integrity"
_PAYLOAD_VALIDATION_CACHE_MAX_ENTRIES = 4096
_PAYLOAD_VALIDATION_CACHE: dict[tuple[Any, ...], None] = {}
_PAYLOAD_VALIDATION_CACHE_LOCK = threading.Lock()


class CacheCompatibilityError(RuntimeError):
    """An existing namespace is bound to another immutable cache plan."""


class CacheValidationError(RuntimeError):
    """A cache manifest or its artifacts are incomplete or inconsistent."""


@dataclass(frozen=True)
class CachePlan:
    """A deterministic preprocessing plan; construction never writes files."""

    namespace: str
    cache_directory: Path
    dataset_root: Path
    split_manifest_path: Path
    split_manifest_sha256: str
    split_percentage: int | None
    sources: tuple[dict, ...]
    contract: Mapping[str, Any]
    created_at_utc: str
    plan_fingerprint: str

    @property
    def data_directory(self) -> Path:
        return self.cache_directory / "persistent"

    @property
    def plan_path(self) -> Path:
        return self.cache_directory / "cache_plan.json"

    @property
    def manifest_path(self) -> Path:
        return self.cache_directory / "cache_manifest.json"

    def to_dict(self) -> dict:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": "threedino_deterministic_prefix_cache_plan",
            "created_at_utc": self.created_at_utc,
            "namespace": self.namespace,
            "cache_directory": str(self.cache_directory),
            "data_directory": str(self.data_directory),
            "dataset_root": str(self.dataset_root),
            "split_manifest_path": str(self.split_manifest_path),
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_percentage": self.split_percentage,
            "contract": dict(self.contract),
            "source_count": len(self.sources),
            "sources": [dict(source) for source in self.sources],
            "plan_fingerprint": self.plan_fingerprint,
            "isolation": {
                "model_family": "3DINO",
                "reuses_merlin_cache": False,
                "reuses_pillar0_cache": False,
                "cached_scope": "deterministic prefix only",
                "stochastic_training_transforms_cached": False,
            },
        }


def cache_directory(cache_root: str | Path, namespace: str) -> Path:
    """Resolve a safe 3DINO shared-cache namespace."""

    value = str(namespace).strip()
    if not _NAMESPACE_PATTERN.fullmatch(value):
        raise ValueError(
            "Cache namespace must contain only letters, numbers, dot, dash, "
            f"or underscore and be at most 128 characters: {namespace!r}"
        )
    root = Path(cache_root).resolve(strict=False)
    candidate = (root / "Shared" / value).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # Defensive even though the regex excludes separators.
        raise ValueError(f"Cache namespace escapes cache root: {namespace!r}") from exc
    return candidate


def create_cache_plan(
    *,
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    cache_root: str | Path,
    namespace: str = DEFAULT_CACHE_NAMESPACE,
    require_files: bool = True,
) -> CachePlan:
    """Build an immutable full-development-plus-held-out cache plan."""

    dataset = Path(dataset_root).resolve(strict=False)
    manifest_path = Path(split_manifest_path).resolve(strict=False)
    manifest = load_manifest_file(manifest_path)
    rows = collect_required_rows(dataset, manifest, require_files=require_files)
    directory = cache_directory(cache_root, namespace)

    sources: list[dict] = []
    for row in rows:
        source = Path(str(row["source_path"]))
        entry = {
            "image": str(source),
            "source_path": str(source),
            "label": int(row["label"]),
            "label_str": str(row["label_str"]),
            "patient_id": str(row["patient_id"]),
            "accession": str(row["accession"]),
            "source_size_bytes": row.get("source_size_bytes"),
            "source_mtime_ns": row.get("source_mtime_ns"),
            "source_sha256": _sha256_file(source) if source.is_file() else None,
            "cache_key": persistent_cache_key(row),
        }
        entry["cache_path"] = str(
            directory / "persistent" / f"{entry['cache_key']}.pt"
        )
        sources.append(entry)

    _validate_source_entries(sources)
    split_sha = _sha256_file(manifest_path)
    contract = preprocessing_contract()
    fingerprint_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "split_manifest_sha256": split_sha,
        "contract": contract,
        "sources": sources,
    }
    fingerprint = _canonical_sha256(fingerprint_payload)
    percentage = manifest.get("metadata", {}).get("percentage")
    return CachePlan(
        namespace=str(namespace),
        cache_directory=directory,
        dataset_root=dataset,
        split_manifest_path=manifest_path,
        split_manifest_sha256=split_sha,
        split_percentage=int(percentage) if percentage is not None else None,
        sources=tuple(sources),
        contract=contract,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_fingerprint=fingerprint,
    )


# Pillar-0 calls the analogous operation build_cache_plan.  Keep the alias
# local to this package so generic orchestration can use either spelling.
build_cache_plan = create_cache_plan


def prepare_cache(
    plan: CachePlan,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    progress_callback: Callable[[int, int, Mapping], None] | None = None,
) -> dict:
    """Materialize every deterministic-prefix item and seal the manifest.

    Existing compatible partial work can resume.  An incompatible or orphaned
    namespace is rejected rather than deleted or silently overwritten.
    """

    if not isinstance(plan, CachePlan):
        raise TypeError("prepare_cache requires a CachePlan")
    _validate_plan_document(
        plan.to_dict(), path=plan.plan_path, file_backed=False
    )
    for entry in plan.sources:
        _verify_source_provenance(entry)
    plan.cache_directory.mkdir(parents=True, exist_ok=True)

    if plan.manifest_path.is_file():
        existing = load_cache_manifest(plan.manifest_path)
        _require_matching_fingerprint(existing, plan.plan_fingerprint)
        result = validate_cache(plan.manifest_path, verify_artifacts=True)
        result["status"] = "already_prepared"
        return result

    if plan.plan_path.is_file():
        existing_plan = _load_json(plan.plan_path)
        _validate_plan_document(
            existing_plan, path=plan.plan_path, file_backed=True
        )
        _require_matching_fingerprint(existing_plan, plan.plan_fingerprint)
    else:
        unexpected = [
            child
            for child in plan.cache_directory.iterdir()
            if child.name not in {plan.plan_path.name, plan.manifest_path.name}
        ]
        if unexpected:
            raise CacheCompatibilityError(
                "Cache namespace contains orphaned files without a matching "
                f"plan: {plan.cache_directory}. Use a new namespace."
            )
        _atomic_write_json(plan.plan_path, plan.to_dict())

    factory = dataset_factory or make_prefix_dataset
    dataset = factory(plan.sources, cache_dir=plan.data_directory)
    total = len(plan.sources)
    if len(dataset) != total:
        raise CacheValidationError(
            f"Prefix dataset length mismatch: expected {total}, found {len(dataset)}"
        )
    for index, row in enumerate(plan.sources):
        dataset[index]
        if progress_callback is not None:
            progress_callback(index + 1, total, row)

    missing = [
        str(Path(entry["cache_path"]))
        for entry in plan.sources
        if not Path(entry["cache_path"]).is_file()
    ]
    if missing:
        raise CacheValidationError(
            "MONAI did not materialize all planned cache artifacts; first "
            f"missing paths: {missing[:5]}"
        )

    _reject_undeclared_artifacts(plan.sources, plan.data_directory)

    manifest = plan.to_dict()
    manifest.update(
        {
            "kind": "threedino_deterministic_prefix_cache_manifest",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "entries": manifest.pop("sources"),
        }
    )
    # Artifact integrity is deliberately sealed after materialization and kept
    # outside ``plan_fingerprint``.  The immutable plan describes what should
    # be cached; this section attests to the exact bytes and payloads that were
    # actually materialized for that plan.
    manifest["artifact_integrity"] = _build_artifact_integrity(
        manifest["entries"], plan_fingerprint=plan.plan_fingerprint
    )
    _atomic_write_json(plan.manifest_path, manifest)
    result = validate_cache(plan.manifest_path, verify_artifacts=True)
    result["status"] = "prepared"
    return result


def load_cache_manifest(path_or_directory: str | Path) -> dict:
    """Load and structurally validate a sealed 3DINO cache manifest."""

    path = Path(path_or_directory)
    if path.is_dir() or not path.suffix:
        path = path / "cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"3DINO cache manifest not found: {path}")
    path = path.resolve(strict=True)
    manifest = _load_json(path)
    _validate_manifest_document(manifest, path=path, file_backed=True)
    return manifest


def validate_cache(
    plan_or_manifest: CachePlan | Mapping | str | Path,
    *,
    verify_artifacts: bool = True,
) -> dict:
    """Validate a sealed cache and optionally every cached tensor payload.

    ``verify_artifacts=False`` is structural-only validation.  It validates the
    immutable plan fingerprint and the artifact-integrity metadata seal, but it
    intentionally does not read, hash, or deserialize artifact files and
    therefore returns ``artifacts_verified=False``.  Execution readiness
    requires the default exhaustive validation.
    """

    expected_fingerprint: str | None = None
    if isinstance(plan_or_manifest, CachePlan):
        expected_fingerprint = plan_or_manifest.plan_fingerprint
        manifest = load_cache_manifest(plan_or_manifest.manifest_path)
        manifest_path = plan_or_manifest.manifest_path
    elif isinstance(plan_or_manifest, Mapping):
        manifest = dict(plan_or_manifest)
        manifest_path = Path(str(manifest.get("cache_directory", "."))) / "cache_manifest.json"
        _validate_manifest_document(
            manifest, path=manifest_path, file_backed=False
        )
    else:
        path = Path(plan_or_manifest)
        manifest_path = path / "cache_manifest.json" if path.is_dir() else path
        manifest = load_cache_manifest(manifest_path)

    if expected_fingerprint is not None:
        _require_matching_fingerprint(manifest, expected_fingerprint)
    entries = manifest["entries"]
    integrity_records = _validate_artifact_integrity(manifest)
    if verify_artifacts:
        _reject_undeclared_artifacts(
            entries, Path(str(manifest["data_directory"]))
        )
        for entry in entries:
            _verify_source_provenance(entry)
            key = str(entry["cache_key"])
            _verify_artifact(entry, integrity_records[key])
        # Recheck after the potentially long exhaustive pass so an artifact
        # added concurrently cannot be hidden by the initial inventory.
        _reject_undeclared_artifacts(
            entries, Path(str(manifest["data_directory"]))
        )

    return {
        "status": "valid",
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "namespace": manifest["namespace"],
        "plan_fingerprint": manifest["plan_fingerprint"],
        "entry_count": len(entries),
        "artifacts_verified": bool(verify_artifacts),
        "validation_scope": (
            "exhaustive_hash_and_payload"
            if verify_artifacts
            else "structural_only"
        ),
        "cached_scope": "deterministic prefix only",
    }


def cached_path_for_row(cache_directory_path: str | Path, row: Mapping) -> Path:
    """Resolve the stable ``.pt`` path MONAI uses for one source row."""

    root = Path(cache_directory_path)
    if root.name != "persistent":
        root = root / "persistent"
    return root / f"{persistent_cache_key(row)}.pt"


def make_cached_dataset(
    plan_or_manifest: CachePlan | Mapping | str | Path,
    rows: Sequence[Mapping],
    *,
    split: str,
    seed: int | None = None,
) -> Any:
    """Create a train/validation/test dataset backed by a sealed shared cache."""

    if isinstance(plan_or_manifest, CachePlan):
        manifest = load_cache_manifest(plan_or_manifest.manifest_path)
        validate_cache(plan_or_manifest, verify_artifacts=True)
    elif isinstance(plan_or_manifest, Mapping):
        manifest = dict(plan_or_manifest)
        validate_cache(manifest, verify_artifacts=True)
    else:
        manifest = load_cache_manifest(plan_or_manifest)
        validate_cache(manifest, verify_artifacts=True)

    by_key = {str(entry["cache_key"]): entry for entry in manifest["entries"]}
    missing_rows = [
        str(row.get("patient_id", persistent_cache_key(row)))
        for row in rows
        if persistent_cache_key(row) not in by_key
    ]
    if missing_rows:
        raise CacheCompatibilityError(
            "Requested rows are absent from the sealed cache manifest: "
            f"{missing_rows[:5]}"
        )
    identity_mismatches: list[str] = []
    for row in rows:
        key = persistent_cache_key(row)
        entry = by_key[key]
        requested_identity = (
            str(row.get("patient_id", "")),
            int(row.get("label", -1)),
            str(row.get("label_str", "")),
        )
        cached_identity = (
            str(entry["patient_id"]),
            int(entry["label"]),
            str(entry["label_str"]),
        )
        if requested_identity != cached_identity:
            identity_mismatches.append(
                f"{row.get('patient_id', '<missing>')} ({key})"
            )
    if identity_mismatches:
        raise CacheCompatibilityError(
            "Requested rows reuse a cache key under a different patient/class "
            f"identity: {identity_mismatches[:5]}"
        )
    data_directory = Path(str(manifest["data_directory"]))
    return make_dataset(rows, split=split, cache_dir=data_directory, seed=seed)


def _payload_contract() -> dict:
    """Return the exact deterministic-prefix payload contract."""

    return {
        "serialization": "torch.save",
        "container": "mapping",
        "required_fields": ["image", "patient_id", "label", "label_str"],
        "image_shape": [1, *[int(value) for value in PREFIX_SIZE]],
        "image_dtype": "floating_point",
        "image_values": "all_finite",
        "matching_entry_fields": ["patient_id", "label", "label_str"],
    }


def _build_artifact_integrity(
    entries: Sequence[Mapping], *, plan_fingerprint: str
) -> dict:
    """Hash and validate materialized payloads, then seal their metadata."""

    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry["cache_key"])
        records[key] = _inspect_artifact(entry)
    section: dict[str, Any] = {
        "schema_version": _ARTIFACT_INTEGRITY_SCHEMA,
        "kind": _ARTIFACT_INTEGRITY_KIND,
        "payload_contract": _payload_contract(),
        "artifact_count": len(records),
        "artifacts": records,
    }
    section["seal_sha256"] = _artifact_integrity_seal(
        section, plan_fingerprint=plan_fingerprint
    )
    return section


def _artifact_integrity_seal(
    section: Mapping, *, plan_fingerprint: str
) -> str:
    payload = {key: value for key, value in section.items() if key != "seal_sha256"}
    return _canonical_sha256(
        {
            "plan_fingerprint": str(plan_fingerprint),
            "artifact_integrity": payload,
        }
    )


def _validate_cache_location(
    document: Mapping,
    *,
    path: Path,
    file_backed: bool,
    document_name: str,
) -> tuple[Path, Path]:
    """Validate namespace layout and optionally bind a document to its file."""

    namespace = document.get("namespace")
    if not isinstance(namespace, str) or not _NAMESPACE_PATTERN.fullmatch(namespace):
        raise CacheCompatibilityError(
            f"Invalid namespace embedded in {document_name}: {namespace!r}"
        )
    cache_value = document.get("cache_directory")
    if not isinstance(cache_value, str) or not cache_value.strip():
        raise CacheValidationError(
            f"Embedded cache_directory is missing in {document_name}: {path}"
        )
    embedded_cache = Path(cache_value).resolve(strict=False)
    if embedded_cache.name != namespace:
        raise CacheCompatibilityError(
            f"Embedded namespace {namespace!r} does not match cache_directory "
            f"name {embedded_cache.name!r}"
        )
    if embedded_cache.parent.name != "Shared":
        raise CacheCompatibilityError(
            f"Embedded cache_directory is not under CacheRoot/Shared: {embedded_cache}"
        )
    derived_root = embedded_cache.parent.parent
    safe_cache = cache_directory(derived_root, namespace)
    if embedded_cache != safe_cache:
        raise CacheCompatibilityError(
            "Embedded cache_directory does not match the safe "
            f"CacheRoot/Shared/namespace layout: {embedded_cache}"
        )
    data_value = document.get("data_directory")
    if not isinstance(data_value, str) or not data_value.strip():
        raise CacheValidationError(
            f"Embedded data_directory is missing in {document_name}: {path}"
        )
    embedded_data = Path(data_value).resolve(strict=False)
    expected_data = (embedded_cache / "persistent").resolve(strict=False)
    if embedded_data != expected_data:
        raise CacheCompatibilityError(
            "Embedded data_directory must equal cache_directory/persistent: "
            f"expected {expected_data}, found {embedded_data}"
        )
    if file_backed:
        actual_parent = path.resolve(strict=False).parent
        if actual_parent != embedded_cache:
            raise CacheCompatibilityError(
                f"File-backed {document_name} is outside its embedded "
                f"cache_directory: file parent={actual_parent}, "
                f"embedded={embedded_cache}"
            )
    return embedded_cache, embedded_data


def _validate_plan_document(
    document: Mapping, *, path: Path, file_backed: bool
) -> None:
    """Validate a resumable cache plan instead of trusting its fingerprint field."""

    if document.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheCompatibilityError(
            f"Unsupported cache-plan schema in {path}: "
            f"{document.get('schema_version')!r}"
        )
    if document.get("kind") != "threedino_deterministic_prefix_cache_plan":
        raise CacheValidationError(f"Not a 3DINO cache plan: {path}")
    _, data_directory = _validate_cache_location(
        document,
        path=path,
        file_backed=file_backed,
        document_name="cache plan",
    )
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise CacheValidationError(f"Cache-plan sources must be a list: {path}")
    _validate_source_entries(sources)
    try:
        source_count = int(document.get("source_count", -1))
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(f"Invalid cache-plan source_count in {path}") from exc
    if source_count != len(sources):
        raise CacheValidationError(
            f"Cache-plan source_count mismatch in {path}: expected "
            f"{document.get('source_count')}, found {len(sources)}"
        )
    recorded_fingerprint = str(document.get("plan_fingerprint", ""))
    computed_fingerprint = _document_fingerprint(document)
    if recorded_fingerprint != computed_fingerprint:
        raise CacheCompatibilityError(
            "Cache-plan content does not match its fingerprint: "
            f"recorded={recorded_fingerprint or '<missing>'}, "
            f"computed={computed_fingerprint}"
        )
    for entry in sources:
        expected_path = data_directory / f"{entry['cache_key']}.pt"
        actual_path = Path(str(entry["cache_path"])).resolve(strict=False)
        if actual_path != expected_path:
            raise CacheCompatibilityError(
                f"Cache-plan entry path escapes or disagrees with its namespace: {actual_path}"
            )
        source_sha = entry.get("source_sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", source_sha
        ):
            raise CacheCompatibilityError(
                "Cache preparation requires a content-hashed source plan; "
                f"rebuild the plan after making the source available: {entry['source_path']}"
            )


def _validate_manifest_document(
    manifest: Mapping, *, path: Path, file_backed: bool
) -> None:
    """Validate structure, binding file-backed manifests to their location."""

    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheCompatibilityError(
            f"Unsupported cache schema in {path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("kind") != "threedino_deterministic_prefix_cache_manifest":
        raise CacheValidationError(f"Not a sealed 3DINO cache manifest: {path}")
    if manifest.get("complete") is not True:
        raise CacheValidationError(f"3DINO cache is not marked complete: {path}")
    _, data_directory = _validate_cache_location(
        manifest,
        path=path,
        file_backed=file_backed,
        document_name="cache manifest",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CacheValidationError(f"Cache entries must be a list: {path}")
    _validate_source_entries(entries)
    try:
        source_count = int(manifest.get("source_count", -1))
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(f"Invalid cache source_count in {path}") from exc
    if source_count != len(entries):
        raise CacheValidationError(
            f"Cache source_count mismatch in {path}: expected "
            f"{manifest.get('source_count')}, found {len(entries)}"
        )
    recorded_fingerprint = str(manifest.get("plan_fingerprint", ""))
    computed_fingerprint = _document_fingerprint(manifest)
    if recorded_fingerprint != computed_fingerprint:
        raise CacheCompatibilityError(
            "Cache manifest content does not match its plan fingerprint: "
            f"recorded={recorded_fingerprint or '<missing>'}, "
            f"computed={computed_fingerprint}"
        )
    for entry in entries:
        expected_path = data_directory / f"{entry['cache_key']}.pt"
        actual_path = Path(str(entry["cache_path"])).resolve(strict=False)
        if actual_path != expected_path:
            raise CacheCompatibilityError(
                f"Cache entry path escapes or disagrees with its namespace: {actual_path}"
            )
        source_sha = entry.get("source_sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", source_sha
        ):
            raise CacheCompatibilityError(
                "Sealed cache lacks source-content SHA-256 provenance; rebuild it: "
                f"{entry['source_path']}"
            )
    _validate_artifact_integrity(manifest)


def _validate_artifact_integrity(manifest: Mapping) -> dict[str, Mapping]:
    section = manifest.get("artifact_integrity")
    if not isinstance(section, Mapping):
        raise CacheCompatibilityError(
            "Sealed cache lacks artifact-integrity metadata; rebuild it with "
            "the current 3DINO cache preparation code."
        )
    expected_fields = {
        "schema_version",
        "kind",
        "payload_contract",
        "artifact_count",
        "artifacts",
        "seal_sha256",
    }
    if set(section) != expected_fields:
        raise CacheValidationError(
            "Artifact-integrity section fields do not match the sealed schema"
        )
    if section.get("schema_version") != _ARTIFACT_INTEGRITY_SCHEMA:
        raise CacheCompatibilityError(
            "Unsupported artifact-integrity schema; rebuild the 3DINO cache"
        )
    if section.get("kind") != _ARTIFACT_INTEGRITY_KIND:
        raise CacheValidationError("Invalid artifact-integrity section kind")
    if section.get("payload_contract") != _payload_contract():
        raise CacheCompatibilityError(
            "Cached payload contract differs from the deterministic 3DINO prefix"
        )
    artifacts = section.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CacheValidationError("Artifact-integrity artifacts must be a mapping")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CacheValidationError("Cache entries must be a list")
    expected_keys = {str(entry["cache_key"]) for entry in entries}
    actual_keys = {str(key) for key in artifacts}
    if actual_keys != expected_keys:
        raise CacheValidationError(
            "Artifact-integrity keys do not exactly match cache entries: "
            f"missing={sorted(expected_keys - actual_keys)[:5]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:5]}"
        )
    count = section.get("artifact_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(entries):
        raise CacheValidationError(
            "Artifact-integrity artifact_count does not match cache entries"
        )
    records: dict[str, Mapping] = {}
    for key in expected_keys:
        record = artifacts[key]
        if not isinstance(record, Mapping) or set(record) != {
            "sha256",
            "size_bytes",
        }:
            raise CacheValidationError(
                f"Invalid artifact-integrity record for cache key {key}"
            )
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CacheValidationError(f"Invalid artifact SHA-256 for cache key {key}")
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CacheValidationError(f"Invalid artifact size for cache key {key}")
        records[key] = record
    recorded_seal = section.get("seal_sha256")
    expected_seal = _artifact_integrity_seal(
        section, plan_fingerprint=str(manifest.get("plan_fingerprint", ""))
    )
    if recorded_seal != expected_seal:
        raise CacheValidationError(
            "Artifact-integrity metadata seal mismatch; the manifest was changed"
        )
    return records


def _verify_source_provenance(entry: Mapping) -> None:
    source = Path(str(entry["source_path"]))
    if not source.is_file():
        raise CacheValidationError(f"Cache source is missing: {source}")
    stat = source.stat()
    recorded_size = entry.get("source_size_bytes")
    recorded_mtime = entry.get("source_mtime_ns")
    if recorded_size is not None and int(recorded_size) != int(stat.st_size):
        raise CacheCompatibilityError(
            f"Source size changed after cache planning: {source}"
        )
    if recorded_mtime is not None and int(recorded_mtime) != int(stat.st_mtime_ns):
        raise CacheCompatibilityError(
            f"Source modification time changed after cache planning: {source}"
        )
    recorded_source_sha = entry.get("source_sha256")
    if (
        not isinstance(recorded_source_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_source_sha)
    ):
        raise CacheCompatibilityError(
            f"Cache entry lacks a valid source SHA-256; rebuild it: {source}"
        )
    actual_source_sha = _sha256_file(source)
    if actual_source_sha != recorded_source_sha:
        raise CacheCompatibilityError(
            f"Source SHA-256 changed after cache planning: {source}"
        )


def _inspect_artifact(
    entry: Mapping, expected_record: Mapping | None = None
) -> dict[str, Any]:
    """Hash and deserialize one stable file handle, validating its payload."""

    artifact = Path(str(entry["cache_path"]))
    if not artifact.is_file():
        raise CacheValidationError(f"Cache artifact is missing: {artifact}")
    try:
        with artifact.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size <= 0:
                raise CacheValidationError(f"Cache artifact is empty: {artifact}")
            if expected_record is not None and int(expected_record["size_bytes"]) != int(
                before.st_size
            ):
                raise CacheValidationError(
                    f"Cache artifact size mismatch for {artifact}: expected "
                    f"{expected_record['size_bytes']}, found {before.st_size}"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if expected_record is not None and actual_sha256 != str(
                expected_record["sha256"]
            ):
                raise CacheValidationError(
                    f"Cache artifact SHA-256 mismatch for {artifact}"
                )
            validation_key = _payload_validation_key(
                artifact, actual_sha256, before, entry
            )
            payload_is_memoized = (
                expected_record is not None
                and _payload_validation_is_memoized(validation_key)
            )
            payload = None
            if not payload_is_memoized:
                handle.seek(0)
                payload = _load_torch_payload(handle, artifact)
            after = os.fstat(handle.fileno())
    except CacheValidationError:
        raise
    except OSError as exc:
        raise CacheValidationError(f"Cannot read cache artifact {artifact}: {exc}") from exc
    if _stat_signature(before) != _stat_signature(after):
        raise CacheValidationError(
            f"Cache artifact changed while it was being validated: {artifact}"
        )
    try:
        path_after = artifact.stat()
    except OSError as exc:
        raise CacheValidationError(
            f"Cache artifact disappeared during validation: {artifact}"
        ) from exc
    if _stat_signature(before) != _stat_signature(path_after):
        raise CacheValidationError(
            f"Cache artifact changed while it was being validated: {artifact}"
        )
    if not payload_is_memoized:
        _validate_cached_payload(payload, entry, artifact)
        if expected_record is not None:
            _memoize_payload_validation(validation_key)
    return {"sha256": actual_sha256, "size_bytes": int(before.st_size)}


def _verify_artifact(entry: Mapping, record: Mapping) -> None:
    _inspect_artifact(entry, expected_record=record)


def _load_torch_payload(handle: Any, artifact: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cache preparation needs torch.
        raise CacheValidationError(
            "PyTorch is required to validate 3DINO cache payloads"
        ) from exc
    try:
        # Cache artifacts are created locally by MONAI and may contain its
        # MetaTensor type, so the restricted weights-only loader is not
        # compatible with the declared payload contract.
        return torch.load(handle, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CacheValidationError(
            f"Cache artifact is not a deserializable torch payload: {artifact}: {exc}"
        ) from exc


def _validate_cached_payload(payload: Any, entry: Mapping, artifact: Path) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cache preparation needs torch.
        raise CacheValidationError(
            "PyTorch is required to validate 3DINO cache payloads"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CacheValidationError(
            f"Cache payload must be a mapping: {artifact}"
        )
    required = set(_payload_contract()["required_fields"])
    missing = required.difference(payload)
    if missing:
        raise CacheValidationError(
            f"Cache payload is missing fields {sorted(missing)}: {artifact}"
        )
    image = payload["image"]
    if not isinstance(image, torch.Tensor):
        raise CacheValidationError(f"Cached image is not a torch tensor: {artifact}")
    expected_shape = tuple(_payload_contract()["image_shape"])
    if tuple(int(value) for value in image.shape) != expected_shape:
        raise CacheValidationError(
            f"Cached image shape mismatch for {artifact}: expected "
            f"{expected_shape}, found {tuple(image.shape)}"
        )
    if not torch.is_floating_point(image):
        raise CacheValidationError(f"Cached image is not floating point: {artifact}")
    if not bool(torch.isfinite(image).all().item()):
        raise CacheValidationError(f"Cached image contains non-finite values: {artifact}")
    expected_patient = str(entry["patient_id"])
    if not isinstance(payload["patient_id"], str) or payload["patient_id"] != expected_patient:
        raise CacheValidationError(
            f"Cached patient_id does not match manifest entry: {artifact}"
        )
    actual_label = payload["label"]
    if isinstance(actual_label, torch.Tensor):
        if actual_label.numel() != 1:
            raise CacheValidationError(f"Cached label is not scalar: {artifact}")
        actual_label = actual_label.detach().cpu().item()
    if (
        isinstance(actual_label, bool)
        or not isinstance(actual_label, int)
        or actual_label != int(entry["label"])
    ):
        raise CacheValidationError(
            f"Cached label does not match manifest entry: {artifact}"
        )
    if (
        not isinstance(payload["label_str"], str)
        or payload["label_str"] != str(entry["label_str"])
    ):
        raise CacheValidationError(
            f"Cached label_str does not match manifest entry: {artifact}"
        )


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


def _payload_validation_key(
    artifact: Path, digest: str, stat: os.stat_result, entry: Mapping
) -> tuple[Any, ...]:
    return (
        str(artifact.resolve(strict=False)),
        str(digest),
        *_stat_signature(stat),
        str(entry["patient_id"]),
        int(entry["label"]),
        str(entry["label_str"]),
        _canonical_sha256(_payload_contract()),
    )


def _payload_validation_is_memoized(key: tuple[Any, ...]) -> bool:
    with _PAYLOAD_VALIDATION_CACHE_LOCK:
        return key in _PAYLOAD_VALIDATION_CACHE


def _memoize_payload_validation(key: tuple[Any, ...]) -> None:
    with _PAYLOAD_VALIDATION_CACHE_LOCK:
        if key in _PAYLOAD_VALIDATION_CACHE:
            return
        while len(_PAYLOAD_VALIDATION_CACHE) >= _PAYLOAD_VALIDATION_CACHE_MAX_ENTRIES:
            oldest = next(iter(_PAYLOAD_VALIDATION_CACHE))
            _PAYLOAD_VALIDATION_CACHE.pop(oldest, None)
        _PAYLOAD_VALIDATION_CACHE[key] = None


def _reject_undeclared_artifacts(
    entries: Sequence[Mapping], data_directory: Path
) -> None:
    def lexical_absolute(path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    declared = {
        lexical_absolute(Path(str(entry["cache_path"]))) for entry in entries
    }
    actual: set[Path] = set()
    invalid_links: list[str] = []
    if data_directory.is_dir():
        for current_root, directories, files in os.walk(
            data_directory, followlinks=False
        ):
            for name in [*directories, *files]:
                if Path(name).suffix.lower() != ".pt":
                    continue
                path = Path(current_root) / name
                lexical = lexical_absolute(path)
                if path.is_symlink() or not path.is_file():
                    invalid_links.append(str(lexical))
                else:
                    actual.add(lexical)
    extras = sorted(str(path) for path in actual.difference(declared))
    if extras or invalid_links:
        raise CacheValidationError(
            "Cache namespace contains undeclared .pt artifacts: "
            f"{(extras + sorted(invalid_links))[:5]}"
        )


def _validate_source_entries(entries: Sequence[Mapping]) -> None:
    if not isinstance(entries, (list, tuple)):
        raise CacheValidationError("Cache sources/entries must be a sequence")
    patient_ids: set[str] = set()
    cache_keys: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise CacheValidationError(f"Cache entry {index} is not an object")
        required = {
            "source_path",
            "label",
            "label_str",
            "patient_id",
            "cache_key",
            "cache_path",
        }
        missing = required.difference(entry)
        if missing:
            raise CacheValidationError(
                f"Cache entry {index} is missing fields {sorted(missing)}"
            )
        patient = str(entry["patient_id"])
        key = str(entry["cache_key"])
        if patient in patient_ids:
            raise CacheValidationError(f"Duplicate cache patient ID: {patient}")
        if key in cache_keys:
            raise CacheValidationError(f"Duplicate cache key: {key}")
        patient_ids.add(patient)
        cache_keys.add(key)
        label = int(entry["label"])
        label_str = str(entry["label_str"])
        if label not in range(len(CLASS_NAMES)) or CLASS_NAMES[label] != label_str:
            raise CacheValidationError(
                f"Cache class mismatch for patient {patient}: {label}/{label_str!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise CacheValidationError(f"Invalid cache key for patient {patient}: {key}")


def _require_matching_fingerprint(document: Mapping, expected: str) -> None:
    actual = str(document.get("plan_fingerprint", ""))
    if actual != str(expected):
        raise CacheCompatibilityError(
            "Cache plan fingerprint mismatch; refusing to mix preprocessing "
            f"artifacts (expected {expected}, found {actual or '<missing>'})."
        )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _document_fingerprint(document: Mapping) -> str:
    entries = document.get("entries", document.get("sources"))
    return _canonical_sha256(
        {
            "schema_version": document.get("schema_version"),
            "namespace": document.get("namespace"),
            "split_manifest_sha256": document.get("split_manifest_sha256"),
            "contract": document.get("contract"),
            "sources": entries,
        }
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CacheValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


__all__ = [
    "CacheCompatibilityError",
    "CachePlan",
    "CacheValidationError",
    "build_cache_plan",
    "cache_directory",
    "cached_path_for_row",
    "create_cache_plan",
    "load_cache_manifest",
    "make_cached_dataset",
    "prepare_cache",
    "validate_cache",
]
