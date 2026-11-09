"""Resumable, directly materialized FlexiCT embedding cache."""

from __future__ import annotations

import os
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import (
        DEFAULT_CACHE_NAMESPACE,
        EXPECTED_PATCH_TOKENS,
        FEATURE_DIM,
        INPUT_SHAPE_DHW,
        PATCH_SIZE,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from .data import preprocess_nifti, preprocessing_contract
    from .model import (
        build_vision_only_encoder,
        extract_embedding,
        resolve_precision,
        vision_encoder_contract,
    )
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import (
        DEFAULT_CACHE_NAMESPACE,
        EXPECTED_PATCH_TOKENS,
        FEATURE_DIM,
        INPUT_SHAPE_DHW,
        PATCH_SIZE,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from data import preprocess_nifti, preprocessing_contract
    from model import (
        build_vision_only_encoder,
        extract_embedding,
        resolve_precision,
        vision_encoder_contract,
    )
    from provisioning import ensure_upstream, resolve_checkpoint

try:
    from _linear_probe.common import (
        ContractError,
        atomic_write_json,
        development_mapping,
        fingerprint,
        heldout_mapping,
        load_manifest,
        merge_mappings,
        read_json,
        rows_from_mapping,
        sha256_file,
        tree_completion_seal,
        utc_now,
        validate_tree_completion,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _linear_probe.common import (
        ContractError,
        atomic_write_json,
        development_mapping,
        fingerprint,
        heldout_mapping,
        load_manifest,
        merge_mappings,
        read_json,
        rows_from_mapping,
        sha256_file,
        tree_completion_seal,
        utc_now,
        validate_tree_completion,
    )


def all_rows(dataset_root: str | Path, *, require_files: bool) -> tuple[list[dict], dict]:
    manifest = load_manifest(dataset_root, 100)
    mapping = merge_mappings([development_mapping(manifest), heldout_mapping(manifest)])
    return rows_from_mapping(dataset_root, mapping, require_files=require_files), manifest


def _source_identity(
    path: Path,
    mode: str,
    *,
    prior: Mapping[str, Any] | None = None,
    force_hash: bool = False,
) -> dict:
    stat = path.stat()
    identity = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
    if mode == "sha256":
        reusable = (
            not force_hash
            and prior is not None
            and prior.get("path") == identity["path"]
            and int(prior.get("bytes", -1)) == identity["bytes"]
            and int(prior.get("modified_ns", -1)) == identity["modified_ns"]
            and isinstance(prior.get("sha256"), str)
        )
        identity["sha256"] = prior["sha256"] if reusable else sha256_file(path)
    elif mode != "stat":
        raise ValueError(f"Unsupported source fingerprint mode: {mode}")
    return identity


def _entry_key(row: Mapping[str, Any]) -> str:
    return fingerprint(
        {"patient_id": str(row["patient_id"]), "source": str(row["image_path"])},
        length=32,
    )


def _inventory_binding(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict:
    records = [
        {
            "entry_key": _entry_key(row),
            "patient_id": str(row["patient_id"]),
            "source": str(Path(row["image_path"]).resolve()),
        }
        for row in rows
    ]
    keys = [record["entry_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ContractError("FlexiCT cache inventory contains duplicate patient/source entries")
    return {
        "inventory_source_percentage": 100,
        "row_count": len(records),
        "rows_sha256": fingerprint(records),
        "split_manifest": str(manifest["_path"]),
        "split_manifest_sha256": str(manifest["_sha256"]),
    }


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _extraction_runtime(device: str) -> dict:
    """Bind numerical/runtime details that can change frozen embeddings."""
    import torch

    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": str(device),
        "dependencies": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "simpleitk": _package_version("SimpleITK"),
        },
    }
    parsed_device = torch.device(device)
    if parsed_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ContractError("CUDA was requested for extraction but is unavailable")
        index = (
            parsed_device.index
            if parsed_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        runtime["cuda_device_index"] = int(index)
        runtime["cuda_device_name"] = str(properties.name)
        runtime["cuda_compute_capability"] = [int(properties.major), int(properties.minor)]
    return runtime


def _implementation_identity() -> dict:
    package = Path(__file__).resolve().parent
    files = ("__init__.py", "cache.py", "data.py", "model.py")
    return {
        "schema_version": 1,
        "files": {
            name: sha256_file(package / name)
            for name in files
        },
    }


def _cache_contract(
    args: Any,
    checkpoint: Mapping[str, Any],
    precision: str,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    extraction_runtime: Mapping[str, Any] | None = None,
) -> dict:
    return {
        "schema_version": 4,
        "model": "FlexiCT",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "dataset_inventory": _inventory_binding(rows, manifest),
        "implementation_identity": _implementation_identity(),
        "preprocessing": preprocessing_contract(),
        "representation": "vlm_projected_cls_mean_l2_1024",
        "input_shape_dhw": list(INPUT_SHAPE_DHW),
        "patch_size": PATCH_SIZE,
        "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
        "feature_shape": [FEATURE_DIM],
        "storage_dtype": "float32",
        "extraction_precision": precision,
        "extraction_runtime": dict(
            extraction_runtime
            if extraction_runtime is not None
            else _extraction_runtime(str(args.device))
        ),
        "checkpoint_deserialization_policy": (
            "weights_only_then_explicit_sha256_trusted_pickle"
            if bool(getattr(args, "allow_trusted_legacy_checkpoint", False))
            else "weights_only_only"
        ),
        "source_fingerprint": args.source_fingerprint,
    }


def _cache_directory(
    args: Any,
    checkpoint: Mapping[str, Any],
    precision: str,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    extraction_runtime: Mapping[str, Any] | None = None,
) -> Path:
    contract = _cache_contract(
        args, checkpoint, precision, rows, manifest, extraction_runtime
    )
    suffix = fingerprint(contract, length=12)
    namespace = args.cache_namespace or DEFAULT_CACHE_NAMESPACE
    return Path(args.cache_root) / "Features" / f"{namespace}_{suffix}"


def cache_plan(args: Any, *, require_assets: bool) -> dict:
    rows, manifest = all_rows(args.dataset_root, require_files=require_assets)
    if require_assets:
        checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
        precision, _ = resolve_precision(args.device, args.feature_precision)
        runtime = _extraction_runtime(str(args.device))
        cache_dir = _cache_directory(
            args, checkpoint, precision, rows, manifest, runtime
        )
        contract = _cache_contract(
            args, checkpoint, precision, rows, manifest, runtime
        )
    else:
        checkpoint = {
            "path": str(args.checkpoint) if args.checkpoint else None,
            "sha256": args.checkpoint_sha256 or "<resolved-at-execution>",
            "bytes": "<resolved-at-execution>",
        }
        precision = f"requested:{args.feature_precision}; resolved-at-execution"
        contract = _cache_contract(
            args,
            checkpoint,
            precision,
            rows,
            manifest,
            {
                "status": "resolved-at-execution",
                "requested_device": str(args.device),
            },
        )
        namespace = args.cache_namespace or DEFAULT_CACHE_NAMESPACE
        cache_dir = Path(args.cache_root) / "Features" / (
            f"{namespace}_<resolved-contract-hash>"
        )
    return {
        "schema_version": 1,
        "model": "FlexiCT",
        "status": "plan",
        "side_effects": False,
        "row_count": len(rows),
        "inventory_percentage": 100,
        "requested_percentage": int(getattr(args, "percentage", 100)),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "cache_directory": str(cache_dir),
        "estimated_raw_feature_bytes": len(rows) * FEATURE_DIM * 4,
        "checkpoint": checkpoint,
        "contract": contract,
        "blockers": []
        if args.checkpoint
        or os.environ.get("FLEXICT_CHECKPOINT")
        or os.environ.get("FLEXICT_VLM_CHECKPOINT")
        else ["official FlexiCT-3D-VLM checkpoint path is not configured"],
    }


def _atomic_torch_save(value: Any, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_payload(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ContractError(f"Invalid FlexiCT feature artifact: {path}")
    return payload


def _validate_payload(
    path: Path,
    row: Mapping[str, Any],
    contract_sha: str,
    source_identity: Mapping[str, Any] | None = None,
) -> dict:
    import torch

    payload = _load_payload(path)
    if payload.get("patient_id") != str(row["patient_id"]):
        raise ContractError(f"Patient identity mismatch in {path}")
    if payload.get("contract_sha256") != contract_sha:
        raise ContractError(f"Feature contract mismatch in {path}")
    if source_identity is not None and payload.get("source_identity") != dict(source_identity):
        raise ContractError(f"Source identity mismatch in {path}")
    embedding = payload.get("embedding")
    if not isinstance(embedding, torch.Tensor):
        raise ContractError(f"Missing tensor embedding in {path}")
    if embedding.dtype != torch.float32 or tuple(embedding.shape) != (FEATURE_DIM,):
        raise ContractError(f"Unexpected embedding dtype/shape in {path}")
    if not torch.isfinite(embedding).all():
        raise ContractError(f"Non-finite embedding in {path}")
    norm = float(torch.linalg.vector_norm(embedding).item())
    if not (0.995 <= norm <= 1.005):
        raise ContractError(f"Embedding is not L2-normalized in {path}: norm={norm}")
    return payload


def prepare_feature_cache(args: Any) -> dict:
    import torch

    rows, manifest = all_rows(args.dataset_root, require_files=True)
    checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
    precision, _ = resolve_precision(args.device, args.feature_precision)
    runtime = _extraction_runtime(str(args.device))
    contract = _cache_contract(args, checkpoint, precision, rows, manifest, runtime)
    contract_sha = fingerprint(contract)
    root = _cache_directory(args, checkpoint, precision, rows, manifest, runtime)
    completion_path = root / "feature_cache_completion.json"
    if completion_path.is_file():
        validate_feature_cache(args, required_rows=rows)
        return {"status": "reused", "cache_directory": str(root), "row_count": len(rows)}
    source_identity = ensure_upstream(
        args.upstream_root,
        revision=args.upstream_revision,
        local_files_only=args.local_files_only,
        provision_log=args.provision_log,
    )
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    progress_path = root / "feature_cache_progress.json"
    progress = (
        read_json(progress_path)
        if progress_path.exists()
        else {
            "schema_version": 2,
            "status": "partial",
            "contract": contract,
            "contract_sha256": contract_sha,
            "inventory": contract["dataset_inventory"],
            "entries": {},
        }
    )
    if progress.get("contract_sha256") != contract_sha:
        raise ContractError(f"Existing partial cache has a different contract: {root}")
    if progress.get("inventory") != contract["dataset_inventory"]:
        raise ContractError(f"Existing partial cache has a different row inventory: {root}")
    entries = progress.get("entries")
    if not isinstance(entries, dict):
        raise ContractError(f"Existing partial cache has malformed entries: {root}")
    expected_keys = {_entry_key(row) for row in rows}
    unexpected_keys = set(entries) - expected_keys
    if unexpected_keys:
        raise ContractError(
            f"Existing partial cache contains stale entries: {sorted(unexpected_keys)[:5]}"
        )

    model = None
    previous_model_details = progress.get("model_details")
    if isinstance(previous_model_details, Mapping):
        model_details = dict(previous_model_details)
    else:
        model_details = {
            "source": source_identity,
            "checkpoint_load_mode": "not_reloaded_from_complete_partial_artifacts",
            "checkpoint_deserialization_policy": contract[
                "checkpoint_deserialization_policy"
            ],
            **vision_encoder_contract(),
        }
    for row in rows:
        key = _entry_key(row)
        artifact = artifacts / f"{key}.pt"
        source = Path(row["image_path"])
        has_prior_entry = key in entries
        prior_entry = entries.get(key, {})
        if not isinstance(prior_entry, Mapping):
            raise ContractError(f"Malformed partial-cache entry for {row['patient_id']}")
        prior_source = prior_entry.get("source_identity")
        source_info = _source_identity(
            source, args.source_fingerprint, prior=prior_source, force_hash=False
        )
        if artifact.is_file() and has_prior_entry:
            expected_bytes = int(prior_entry.get("bytes", -1))
            expected_sha = prior_entry.get("sha256")
            if (
                artifact.stat().st_size != expected_bytes
                or not isinstance(expected_sha, str)
                or sha256_file(artifact) != expected_sha
            ):
                raise ContractError(
                    "Partial FlexiCT artifact changed since its progress entry: "
                    f"{artifact}. Move the incompatible partial cache directory "
                    "aside and rerun preparation; changed features are never adopted."
                )
            payload = _validate_payload(artifact, row, contract_sha, source_info)
            extraction = payload.get("extraction", {})
        else:
            # An artifact without an atomic progress entry is an orphan from an
            # interrupted write boundary.  Its bytes have never been sealed, so
            # regenerate it through the official extractor instead of adopting
            # a merely well-shaped tensor and assigning it a fresh trusted hash.
            if model is None:
                model, loaded_details = build_vision_only_encoder(
                    upstream_root=args.upstream_root,
                    checkpoint_path=checkpoint["path"],
                    device=args.device,
                    allow_trusted_legacy_checkpoint=bool(
                        getattr(args, "allow_trusted_legacy_checkpoint", False)
                    ),
                    trusted_checkpoint_sha256=args.checkpoint_sha256,
                )
                model_details = loaded_details
            volume, preprocess_metadata = preprocess_nifti(source)
            embedding, extraction = extract_embedding(
                model,
                volume,
                device=args.device,
                precision=args.feature_precision,
            )
            payload = {
                "schema_version": 1,
                "patient_id": str(row["patient_id"]),
                "source_identity": source_info,
                "contract_sha256": contract_sha,
                "preprocessing_metadata": preprocess_metadata,
                "extraction": extraction,
                "embedding": embedding,
            }
            _atomic_torch_save(payload, artifact)
            _validate_payload(artifact, row, contract_sha, source_info)
        entries[key] = {
            "patient_id": str(row["patient_id"]),
            "artifact": f"artifacts/{artifact.name}",
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "source_identity": source_info,
            "extraction": extraction,
        }
        progress["model_details"] = model_details
        progress["completed_rows"] = len(entries)
        progress["updated_at"] = utc_now()
        atomic_write_json(progress_path, progress)
    if set(entries) != expected_keys:
        raise ContractError("FlexiCT cache preparation did not materialize the exact row inventory")
    manifest_document = {
        **progress,
        "status": "complete",
        "completed_at": utc_now(),
        "row_count": len(rows),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "checkpoint": checkpoint,
        "source": source_identity,
        "model_details": model_details,
        "torch_version": torch.__version__,
    }
    manifest_path = root / "feature_cache_manifest.json"
    atomic_write_json(manifest_path, manifest_document)
    seal_paths = [manifest_path]
    seal_paths.extend(
        root / entries[_entry_key(row)]["artifact"]
        for row in rows
    )
    atomic_write_json(
        completion_path,
        tree_completion_seal(
            root,
            seal_paths,
            context={
                "model": "FlexiCT",
                "contract_sha256": contract_sha,
                "row_count": len(rows),
                "entries_sha256": fingerprint(manifest_document["entries"]),
                "inventory_sha256": contract["dataset_inventory"]["rows_sha256"],
            },
        ),
    )
    return {"status": "complete", "cache_directory": str(root), "row_count": len(rows)}


def _safe_artifact_path(root: Path, relative_value: Any) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"Unsafe FlexiCT feature artifact path: {relative_value}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:  # pragma: no cover - belt-and-suspenders path check
        raise ContractError(f"FlexiCT feature artifact escapes cache root: {relative_value}") from exc
    return path


def _resolve_existing_cache(args: Any) -> tuple[Path, dict, list[dict]]:
    inventory_rows, inventory_manifest = all_rows(args.dataset_root, require_files=False)
    checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
    precision, _ = resolve_precision(args.device, args.feature_precision)
    runtime = _extraction_runtime(str(args.device))
    root = _cache_directory(
        args, checkpoint, precision, inventory_rows, inventory_manifest, runtime
    )
    seal = validate_tree_completion(root, "feature_cache_completion.json")
    manifest = read_json(root / "feature_cache_manifest.json")
    expected_contract = _cache_contract(
        args, checkpoint, precision, inventory_rows, inventory_manifest, runtime
    )
    if manifest.get("contract_sha256") != fingerprint(expected_contract):
        raise ContractError(f"FlexiCT feature cache contract mismatch: {root}")
    if manifest.get("contract") != expected_contract:
        raise ContractError(f"FlexiCT feature cache contract document changed: {root}")
    if manifest.get("status") != "complete":
        raise ContractError(f"FlexiCT feature cache is incomplete: {root}")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ContractError(f"FlexiCT feature cache entries are malformed: {root}")
    expected_keys = {_entry_key(row) for row in inventory_rows}
    if set(entries) != expected_keys:
        raise ContractError("FlexiCT cache entry set does not match the sealed 100% inventory")
    declared_paths = {
        _safe_artifact_path(root, entry.get("artifact")).resolve()
        for entry in entries.values()
        if isinstance(entry, Mapping)
    }
    if len(declared_paths) != len(entries):
        raise ContractError("FlexiCT cache contains malformed or duplicate artifact declarations")
    actual_paths = {path.resolve() for path in (root / "artifacts").glob("*.pt")}
    if actual_paths != declared_paths:
        raise ContractError("FlexiCT cache contains undeclared or missing feature artifacts")
    sealed_records = seal.get("artifacts", [])
    sealed_relative_paths = {str(record.get("path")) for record in sealed_records}
    expected_relative_paths = {"feature_cache_manifest.json"}
    expected_relative_paths.update(
        path.relative_to(root.resolve()).as_posix() for path in declared_paths
    )
    if (
        len(sealed_records) != len(expected_relative_paths)
        or sealed_relative_paths != expected_relative_paths
    ):
        raise ContractError("FlexiCT completion seal does not bind the exact artifact inventory")
    expected_seal_context = {
        "model": "FlexiCT",
        "contract_sha256": manifest["contract_sha256"],
        "row_count": len(inventory_rows),
        "entries_sha256": fingerprint(entries),
        "inventory_sha256": expected_contract["dataset_inventory"]["rows_sha256"],
    }
    if seal.get("context") != expected_seal_context:
        raise ContractError("FlexiCT completion-seal context does not match the cache manifest")
    return root, manifest, inventory_rows


def _validate_rows(
    args: Any,
    root: Path,
    manifest: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    force_source_hashes: bool,
) -> None:
    entries = manifest.get("entries", {})
    expected_keys = {_entry_key(row) for row in inventory_rows}
    for row in rows:
        key = _entry_key(row)
        if key not in expected_keys:
            raise ContractError(
                f"Requested patient {row['patient_id']} is outside the sealed cache inventory"
            )
        if key not in entries:
            raise ContractError(f"FlexiCT feature cache is missing patient {row['patient_id']}")
        entry = entries[key]
        if not isinstance(entry, Mapping):
            raise ContractError(f"Malformed FlexiCT feature entry for {row['patient_id']}")
        path = _safe_artifact_path(root, entry.get("artifact"))
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ContractError(f"FlexiCT feature artifact is missing or truncated: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ContractError(f"FlexiCT feature artifact hash mismatch: {path}")
        source_info = _source_identity(
            Path(row["image_path"]),
            args.source_fingerprint,
            prior=entry.get("source_identity"),
            force_hash=force_source_hashes,
        )
        _validate_payload(path, row, manifest["contract_sha256"], source_info)


def validate_feature_cache(
    args: Any, *, required_rows: Sequence[Mapping[str, Any]] | None = None
) -> dict:
    root, manifest, inventory_rows = _resolve_existing_cache(args)
    rows = (
        list(required_rows)
        if required_rows is not None
        else all_rows(args.dataset_root, require_files=True)[0]
    )
    _validate_rows(
        args,
        root,
        manifest,
        inventory_rows,
        rows,
        force_source_hashes=required_rows is None,
    )
    return {
        "status": "valid",
        "cache_directory": str(root),
        "validated_rows": len(rows),
        "contract_sha256": manifest["contract_sha256"],
    }


class FeatureStore:
    def __init__(
        self,
        args: Any,
        *,
        required_rows: Sequence[Mapping[str, Any]] | None = None,
    ):
        self._args = args
        self.root, self.manifest, self._inventory_rows = _resolve_existing_cache(args)
        self.contract_sha256 = self.manifest["contract_sha256"]
        if required_rows is not None:
            self.validate_rows(required_rows)

    def validate_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        _validate_rows(
            self._args,
            self.root,
            self.manifest,
            self._inventory_rows,
            rows,
            force_source_hashes=False,
        )

    @property
    def identity(self) -> dict:
        manifest_path = self.root / "feature_cache_manifest.json"
        return {
            "feature_cache": str(self.root.resolve()),
            "feature_contract_sha256": self.contract_sha256,
            "feature_cache_manifest_sha256": sha256_file(manifest_path),
            "feature_entries_sha256": fingerprint(self.manifest["entries"]),
            "checkpoint": self.manifest["checkpoint"],
            "upstream_revision": self.manifest["source"]["revision"],
            "representation": self.manifest["contract"]["representation"],
            "storage_dtype": self.manifest["contract"]["storage_dtype"],
            "extraction_precision": self.manifest["contract"]["extraction_precision"],
        }

    def load(self, row: Mapping[str, Any]):
        key = _entry_key(row)
        try:
            entry = self.manifest["entries"][key]
        except KeyError as exc:
            raise ContractError(f"Feature cache has no row for patient {row['patient_id']}") from exc
        path = _safe_artifact_path(self.root, entry["artifact"])
        return _validate_payload(path, row, self.contract_sha256)["embedding"].clone()


__all__ = [
    "FeatureStore",
    "all_rows",
    "cache_plan",
    "prepare_feature_cache",
    "validate_feature_cache",
]
