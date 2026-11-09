"""Direct NIfTI-to-dense-feature cache for the native COLIPRI probe."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import (
        DEFAULT_CACHE_NAMESPACE,
        FEATURE_SHAPE_CDHW,
        MODEL_REVISION,
    )
    from .model import (
        build_image_encoder,
        extract_dense_features,
        preprocess_nifti,
        processor_config_identity,
        resolve_precision,
    )
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE, FEATURE_SHAPE_CDHW, MODEL_REVISION
    from model import (
        build_image_encoder,
        extract_dense_features,
        preprocess_nifti,
        processor_config_identity,
        resolve_precision,
    )
    from provisioning import ensure_upstream, resolve_checkpoint

try:
    from _linear_probe.common import (
        ContractError,
        atomic_write_json,
        completion_seal,
        development_mapping,
        fingerprint,
        heldout_mapping,
        load_manifest,
        merge_mappings,
        read_json,
        rows_from_mapping,
        sha256_file,
        utc_now,
        validate_completion,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _linear_probe.common import (
        ContractError,
        atomic_write_json,
        completion_seal,
        development_mapping,
        fingerprint,
        heldout_mapping,
        load_manifest,
        merge_mappings,
        read_json,
        rows_from_mapping,
        sha256_file,
        utc_now,
        validate_completion,
    )


FEATURE_BYTES_PER_PATIENT = math.prod(FEATURE_SHAPE_CDHW) * 4
EXECUTION_RECEIPT_FILENAME = "feature_cache_execution_ready.json"
CANONICAL_METHOD_LABEL = "PoolingOptimisedLinearProbe"
LEGACY_METHOD_LABEL = "LinearProbe"
MIGRATION_MANIFEST = "METHOD_IDENTITY_MIGRATION.json"

EXTRACTION_RUNTIME_DISTRIBUTIONS = (
    "colipri",
    "accelerate",
    "dynamic-network-architectures",
    "einops",
    "huggingface-hub",
    "hydra-core",
    "numpy",
    "safetensors",
    "timm",
    "torchio",
    "torchvision",
    "transformers",
)


def all_rows(dataset_root: str | Path, *, require_files: bool) -> tuple[list[dict], dict]:
    manifest = load_manifest(dataset_root, 100)
    mapping = merge_mappings([development_mapping(manifest), heldout_mapping(manifest)])
    return rows_from_mapping(dataset_root, mapping, require_files=require_files), manifest


def extraction_runtime_identity(device: str, requested_precision: str) -> dict:
    """Resolve the exact software and accelerator identity used for extraction."""

    import torch

    resolved_precision, _ = resolve_precision(device, requested_precision)
    torch_device = torch.device(str(device))
    if torch_device.type not in ("cpu", "cuda"):
        raise ValueError("COLIPRI feature extraction supports CPU or CUDA devices")
    package_versions = {}
    for distribution in EXTRACTION_RUNTIME_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required COLIPRI extraction distribution is missing: {distribution}"
            ) from exc
    runtime = {
        "schema_version": 1,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_version_full": sys.version,
        # Keep every value JSON-native because this document is compared with
        # the copy reloaded from the sealed manifest on later processes.
        "python_build": list(platform.python_build()),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "precision_requested": str(requested_precision),
        "precision_resolved": resolved_precision,
        "device_requested": str(device),
        "device_type": torch_device.type,
        "device_resolved": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "cuda_runtime": None,
        "cudnn_version": None,
        "cuda_device_capability": None,
        "package_versions": package_versions,
    }
    if torch_device.type == "cuda":
        index = (
            torch.cuda.current_device()
            if torch_device.index is None
            else torch_device.index
        )
        actual_device = torch.device("cuda", index)
        runtime.update(
            {
                "device_resolved": str(actual_device),
                "device_name": torch.cuda.get_device_name(actual_device),
                "cuda_runtime": str(torch.version.cuda),
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(actual_device)
                ),
            }
        )
    return runtime


def planned_extraction_runtime(device: str, requested_precision: str) -> dict:
    return {
        "status": "resolved_at_execution",
        "device_requested": str(device),
        "precision_requested": str(requested_precision),
        "bound_fields": [
            "Python implementation/version/build and machine",
            "PyTorch version and requested/resolved precision",
            "device type/resolved index/name/capability",
            "CUDA runtime and cuDNN version when applicable",
            "direct extraction package versions",
        ],
    }


def implementation_identity() -> dict:
    package = Path(__file__).resolve().parent
    files = ("__init__.py", "cache.py", "model.py")
    return {
        "schema_version": 1,
        "files": {name: sha256_file(package / name) for name in files},
    }


def preprocessing_contract(processor_identity: Mapping[str, Any] | None = None) -> dict:
    return {
        "schema_version": 1,
        "source": "released official COLIPRI processor at pinned revision",
        "orientation": "RAS",
        "target_spacing_xyz_mm": [2.0, 2.0, 2.0],
        "resampling": "TorchIO linear, antialias false",
        "intensity": "clamp [-1000,1000], linear rescale to [-1,1]",
        "crop_or_pad": "centered 192 cubed, padding_mode minimum",
        "output_shape_bcdhw": [1, 1, 192, 192, 192],
        "processor_config": dict(processor_identity) if processor_identity else "resolved at execution",
        "paper_release_note": (
            "This shipped inference processor is reproducible but is not bit-identical to "
            "the paper's unshipped precomputed multi-resolution pyramid."
        ),
    }


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
        raise ContractError("COLIPRI cache inventory contains duplicate patient/source entries")
    return {
        "inventory_source_percentage": 100,
        "row_count": len(records),
        "rows_sha256": fingerprint(records),
        "split_manifest": str(manifest["_path"]),
        "split_manifest_sha256": str(manifest["_sha256"]),
    }


def _source_identity(
    path: Path,
    mode: str,
    *,
    prior: Mapping[str, Any] | None = None,
    force_hash: bool = False,
) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
    if mode == "sha256":
        reusable = (
            not force_hash
            and prior is not None
            and prior.get("path") == result["path"]
            and int(prior.get("bytes", -1)) == result["bytes"]
            and int(prior.get("modified_ns", -1)) == result["modified_ns"]
            and isinstance(prior.get("sha256"), str)
            and len(prior["sha256"]) == 64
        )
        result["sha256"] = prior["sha256"] if reusable else sha256_file(path)
    elif mode != "stat":
        raise ValueError(f"Unsupported source fingerprint mode: {mode}")
    return result


def _contract(
    args: Any,
    checkpoint: Mapping[str, Any],
    precision: str,
    processor_identity: Mapping[str, Any],
    extraction_runtime: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict:
    runtime = dict(extraction_runtime)
    if runtime.get("status") != "resolved_at_execution":
        if runtime.get("precision_resolved") != precision:
            raise ContractError(
                "COLIPRI extraction runtime precision does not match the cache contract"
            )
    return {
        "schema_version": 4,
        "model": "COLIPRI",
        "upstream_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "dataset_inventory": _inventory_binding(rows, manifest),
        "implementation_identity": implementation_identity(),
        "preprocessing": preprocessing_contract(processor_identity),
        "representation": "raw_unprojected_unpooled_primus_m_backbone",
        "encode_image": {"pool": False, "project": False, "normalize": False},
        "feature_shape_cdhw": list(FEATURE_SHAPE_CDHW),
        "storage_dtype": "float32",
        "extraction_precision": precision,
        "extraction_runtime": runtime,
        "source_fingerprint": args.source_fingerprint,
    }


def _cache_directory(args: Any, contract: Mapping[str, Any]) -> Path:
    return (
        Path(args.cache_root)
        / "Features"
        / f"{args.cache_namespace or DEFAULT_CACHE_NAMESPACE}_{fingerprint(contract, length=12)}"
    )


def _legacy_cache_contract(
    args: Any, current_contract: Mapping[str, Any]
) -> tuple[Path, dict] | None:
    """Resolve the sealed pre-rename cache without changing any cache payload."""

    if not bool(getattr(args, "resume", True)):
        return None
    migration_path = (
        Path(args.results_root)
        / CANONICAL_METHOD_LABEL
        / MIGRATION_MANIFEST
    )
    if not migration_path.is_file():
        return None
    migration = read_json(migration_path)
    required = {
        "schema_version": 1,
        "migration_type": "metadata_only_result_namespace",
        "model": "COLIPRI",
        "legacy_method_label": LEGACY_METHOD_LABEL,
        "canonical_method_id": "pooling_optimised_linear_probe",
        "canonical_method_label": CANONICAL_METHOD_LABEL,
    }
    if any(migration.get(key) != value for key, value in required.items()):
        raise ContractError(f"Incompatible COLIPRI migration manifest: {migration_path}")
    declared_implementation = migration.get(
        "legacy_cache_implementation_identity"
    )
    if not isinstance(declared_implementation, dict):
        raise ContractError(
            f"Migration manifest has no exact legacy cache identity: {migration_path}"
        )
    current_comparable = dict(current_contract)
    current_comparable.pop("implementation_identity", None)
    cache_base = Path(args.cache_root) / "Features"
    prefix = f"{args.cache_namespace or DEFAULT_CACHE_NAMESPACE}_"
    matches: list[tuple[Path, dict]] = []
    if cache_base.is_dir():
        for candidate_root in cache_base.iterdir():
            if not candidate_root.is_dir() or not candidate_root.name.startswith(prefix):
                continue
            manifest_path = candidate_root / "feature_cache_manifest.json"
            completion_path = candidate_root / "feature_cache_completion.json"
            if not manifest_path.is_file() or not completion_path.is_file():
                continue
            candidate_manifest = read_json(manifest_path)
            candidate_contract = candidate_manifest.get("contract")
            if not isinstance(candidate_contract, dict):
                continue
            candidate_comparable = dict(candidate_contract)
            candidate_comparable.pop("implementation_identity", None)
            expected_name = (
                f"{prefix}{fingerprint(candidate_contract, length=12)}"
            )
            if (
                candidate_manifest.get("status") == "complete"
                and candidate_contract.get("implementation_identity")
                == declared_implementation
                and candidate_comparable == current_comparable
                and candidate_root.name == expected_name
            ):
                matches.append((candidate_root, candidate_contract))
    if len(matches) > 1:
        raise ContractError(
            "Multiple caches match the declared COLIPRI legacy cache identity"
        )
    return matches[0] if matches else None


def cache_plan(args: Any, *, require_assets: bool) -> dict:
    rows, manifest = all_rows(args.dataset_root, require_files=require_assets)
    if require_assets:
        source = ensure_upstream(
            args.upstream_root,
            revision=args.upstream_revision,
            local_files_only=args.local_files_only,
            provision_log=args.provision_log,
        )
        checkpoint = resolve_checkpoint(
            args.checkpoint,
            upstream_root=args.upstream_root,
            local_files_only=args.local_files_only,
            provision_log=args.provision_log,
        )
        extraction_runtime = extraction_runtime_identity(
            args.device, args.feature_precision
        )
        precision = extraction_runtime["precision_resolved"]
        processor_identity = processor_config_identity(args.upstream_root)
        contract = _contract(
            args,
            checkpoint,
            precision,
            processor_identity,
            extraction_runtime,
            rows,
            manifest,
        )
        directory = _cache_directory(args, contract)
    else:
        source = {"revision": args.upstream_revision, "path": str(args.upstream_root)}
        checkpoint = {
            "repo_id": "microsoft/colipri",
            "revision": MODEL_REVISION,
            "sha256": "97d66586507d1f48577b5149777d02e12d25787435a2a295cac5563df9291c14",
            "bytes": 1_033_541_380,
        }
        contract = _contract(
            args,
            checkpoint,
            "resolved_at_execution",
            {},
            planned_extraction_runtime(args.device, args.feature_precision),
            rows,
            manifest,
        )
        directory = Path(args.cache_root) / "Features" / (
            f"{args.cache_namespace}_<resolved-contract-hash>"
        )
    estimated = len(rows) * FEATURE_BYTES_PER_PATIENT
    return {
        "schema_version": 1,
        "status": "plan",
        "side_effects": False,
        "model": "COLIPRI",
        "row_count": len(rows),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "cache_directory": str(directory),
        "bytes_per_patient_raw_float32": FEATURE_BYTES_PER_PATIENT,
        "estimated_raw_feature_bytes": estimated,
        "estimated_raw_feature_gib": estimated / (1024**3),
        "streaming_contract": "one patient shard at a time; cohort is never stacked in RAM",
        "source": source,
        "checkpoint": checkpoint,
        "contract": contract,
    }


def _entry_key(row: Mapping[str, Any]) -> str:
    return fingerprint(
        {"patient_id": str(row["patient_id"]), "source": str(row["image_path"])}, length=32
    )


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
        raise ContractError(f"Invalid COLIPRI feature artifact: {path}")
    return payload


def _validate_payload(
    path: Path,
    row: Mapping[str, Any],
    contract_sha: str,
    source_identity: Mapping[str, Any] | None = None,
    *,
    verify_finite: bool = True,
) -> dict:
    import torch

    payload = _load_payload(path)
    if payload.get("patient_id") != str(row["patient_id"]):
        raise ContractError(f"Patient identity mismatch in {path}")
    if payload.get("contract_sha256") != contract_sha:
        raise ContractError(f"Feature contract mismatch in {path}")
    if source_identity is not None and payload.get("source_identity") != dict(source_identity):
        raise ContractError(f"Source identity mismatch in {path}")
    dense = payload.get("dense_features")
    if not isinstance(dense, torch.Tensor):
        raise ContractError(f"Missing dense feature tensor in {path}")
    if dense.dtype != torch.float32 or tuple(dense.shape) != FEATURE_SHAPE_CDHW:
        raise ContractError(f"Unexpected COLIPRI feature dtype/shape in {path}")
    if verify_finite and not torch.isfinite(dense).all():
        raise ContractError(f"Non-finite COLIPRI features in {path}")
    return payload


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise ContractError(f"Could not find an existing parent for {path}")
        current = current.parent
    return current


def _validate_prior_artifact(
    artifact: Path, row: Mapping[str, Any], prior_entry: Mapping[str, Any]
) -> None:
    expected_relative = f"artifacts/{artifact.name}"
    if prior_entry.get("artifact") != expected_relative:
        raise ContractError(
            f"Partial COLIPRI artifact declaration changed for {row['patient_id']}"
        )
    if prior_entry.get("patient_id") != str(row["patient_id"]):
        raise ContractError(
            f"Partial COLIPRI patient declaration changed for {row['patient_id']}"
        )
    try:
        expected_bytes = int(prior_entry["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"Partial COLIPRI artifact size is missing for {row['patient_id']}"
        ) from exc
    if artifact.stat().st_size != expected_bytes:
        raise ContractError(
            f"Partial COLIPRI artifact size changed for {row['patient_id']}"
        )
    prior_sha256 = prior_entry.get("sha256")
    if (
        not isinstance(prior_sha256, str)
        or len(prior_sha256) != 64
        or sha256_file(artifact) != prior_sha256
    ):
        raise ContractError(
            f"Partial COLIPRI artifact SHA-256 changed for {row['patient_id']}"
        )


def _recorded_artifact_reusable(
    artifact: Path, row: Mapping[str, Any], prior_entry: Mapping[str, Any]
) -> bool:
    """Return true only for an artifact already sealed in partial progress.

    An orphan produced by a crash between the atomic tensor write and progress
    commit is deliberately regenerated.  It must never be silently adopted
    into a new progress record.
    """

    if not artifact.is_file() or not prior_entry:
        return False
    _validate_prior_artifact(artifact, row, prior_entry)
    return True


def prepare_feature_cache(args: Any) -> dict:
    import torch

    rows, manifest = all_rows(args.dataset_root, require_files=True)
    source = ensure_upstream(
        args.upstream_root,
        revision=args.upstream_revision,
        local_files_only=args.local_files_only,
        provision_log=args.provision_log,
    )
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        upstream_root=args.upstream_root,
        local_files_only=args.local_files_only,
        provision_log=args.provision_log,
    )
    extraction_runtime = extraction_runtime_identity(args.device, args.feature_precision)
    precision = extraction_runtime["precision_resolved"]
    processor_identity = processor_config_identity(args.upstream_root)
    contract = _contract(
        args,
        checkpoint,
        precision,
        processor_identity,
        extraction_runtime,
        rows,
        manifest,
    )
    root = _cache_directory(args, contract)
    if not (root / "feature_cache_completion.json").is_file():
        legacy = _legacy_cache_contract(args, contract)
        if legacy is not None:
            root, contract = legacy
    contract_sha = fingerprint(contract)
    completion_path = root / "feature_cache_completion.json"
    if completion_path.is_file():
        validate_feature_cache(args, required_rows=rows, exhaustive=False)
        receipt_path = root / EXECUTION_RECEIPT_FILENAME
        execution_ready = False
        receipt = None
        if receipt_path.is_file():
            completion = validate_completion(root, "feature_cache_completion.json")
            cached_manifest = read_json(root / "feature_cache_manifest.json")
            receipt = _validate_execution_ready_receipt(
                root, cached_manifest, completion
            )
            execution_ready = True
        return {
            "status": "reused",
            "cache_directory": str(root),
            "row_count": len(rows),
            "execution_ready": execution_ready,
            "execution_receipt": str(receipt_path.resolve())
            if receipt is not None
            else None,
            "next_action": None
            if execution_ready
            else (
                "run prepare_cache.py --action validate with "
                "--source_fingerprint sha256 before training"
                if args.source_fingerprint == "sha256"
                else "prepare a scientific cache with --source_fingerprint sha256"
            ),
        }
    progress_path = root / "feature_cache_progress.json"
    fresh_preparation = not progress_path.exists()
    progress = read_json(progress_path) if progress_path.exists() else {
        "schema_version": 1,
        "status": "partial",
        "contract": contract,
        "contract_sha256": contract_sha,
        "entries": {},
    }
    if progress.get("contract_sha256") != contract_sha:
        raise ContractError(f"Existing partial COLIPRI cache has a different contract: {root}")
    if progress.get("contract") != contract:
        raise ContractError(
            f"Existing partial COLIPRI cache contract document changed: {root}"
        )
    if not isinstance(progress.get("entries"), dict):
        raise ContractError(f"Existing partial COLIPRI cache has malformed entries: {root}")
    expected_keys = {_entry_key(row) for row in rows}
    unexpected_keys = set(progress["entries"]).difference(expected_keys)
    if unexpected_keys:
        raise ContractError(
            f"Existing partial COLIPRI cache has entries outside the sealed inventory: {root}"
        )
    estimated = len(rows) * FEATURE_BYTES_PER_PATIENT
    completed_recorded_rows = 0
    for key, entry in progress["entries"].items():
        if not isinstance(entry, Mapping):
            raise ContractError(f"Malformed partial COLIPRI cache entry {key}")
        expected_relative = f"artifacts/{key}.pt"
        if entry.get("artifact") != expected_relative:
            raise ContractError(f"Unexpected partial COLIPRI artifact path for entry {key}")
        artifact = root / expected_relative
        try:
            recorded_bytes = int(entry["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Missing partial COLIPRI artifact size for entry {key}") from exc
        if artifact.is_file() and artifact.stat().st_size == recorded_bytes:
            completed_recorded_rows += 1
    remaining = max(0, estimated - completed_recorded_rows * FEATURE_BYTES_PER_PATIENT)
    free = shutil.disk_usage(_existing_parent(root)).free
    minimum_free = int(remaining * 1.08) + 2 * 1024**3
    if free < minimum_free:
        raise ContractError(
            f"COLIPRI dense cache needs approximately {minimum_free / 1024**3:.1f} GiB "
            f"additional free space for {len(rows) - completed_recorded_rows} remaining "
            f"patients, but only {free / 1024**3:.1f} GiB is available. "
            "Use --cache_root on a larger drive."
        )
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    model = None
    processor = None
    model_details = {
        "source": source,
        "processor_config": processor_identity,
        "input_shape_dhw": [192, 192, 192],
        "feature_shape_cdhw": list(FEATURE_SHAPE_CDHW),
        "encode_image_arguments": {
            "pool": False,
            "project": False,
            "normalize": False,
        },
        "checkpoint_load_mode": "not_reloaded_from_complete_partial_artifacts",
    }
    for index, row in enumerate(rows, start=1):
        key = _entry_key(row)
        artifact = artifacts / f"{key}.pt"
        source_path = Path(row["image_path"])
        prior_entry = progress["entries"].get(key, {})
        if not isinstance(prior_entry, Mapping):
            raise ContractError(f"Malformed partial-cache entry for {row['patient_id']}")
        source_identity = _source_identity(
            source_path,
            args.source_fingerprint,
            prior=prior_entry.get("source_identity"),
            force_hash=False,
        )
        if _recorded_artifact_reusable(artifact, row, prior_entry):
            payload = _validate_payload(artifact, row, contract_sha, source_identity)
            extraction = payload.get("extraction", {})
        else:
            if model is None or processor is None:
                model, processor, model_details = build_image_encoder(
                    upstream_root=args.upstream_root,
                    checkpoint_path=checkpoint["path"],
                    device=args.device,
                )
            images, preprocess_metadata = preprocess_nifti(processor, source_path)
            dense, extraction = extract_dense_features(
                model, images, device=args.device, precision=args.feature_precision
            )
            payload = {
                "schema_version": 1,
                "patient_id": str(row["patient_id"]),
                "source_identity": source_identity,
                "contract_sha256": contract_sha,
                "preprocessing_metadata": preprocess_metadata,
                "extraction": extraction,
                "dense_features": dense,
            }
            _atomic_torch_save(payload, artifact)
            _validate_payload(artifact, row, contract_sha, source_identity)
            del images, dense
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        progress["entries"][key] = {
            "patient_id": str(row["patient_id"]),
            "artifact": f"artifacts/{artifact.name}",
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "source_identity": source_identity,
            "extraction": extraction,
        }
        progress["completed_rows"] = index
        progress["updated_at"] = utc_now()
        atomic_write_json(progress_path, progress)
    manifest_document = {
        **progress,
        "status": "complete",
        "completed_at": utc_now(),
        "row_count": len(rows),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "checkpoint": checkpoint,
        "source": source,
        "model_details": model_details,
        "extraction_runtime": extraction_runtime,
        "torch_version": torch.__version__,
        "raw_feature_bytes": len(rows) * FEATURE_BYTES_PER_PATIENT,
    }
    manifest_path = root / "feature_cache_manifest.json"
    declared = {
        (root / item["artifact"]).resolve()
        for item in manifest_document["entries"].values()
    }
    actual = {path.resolve() for path in artifacts.rglob("*.pt")}
    if actual != declared:
        raise ContractError(
            "COLIPRI cache contains undeclared or missing feature artifacts before sealing"
        )
    atomic_write_json(manifest_path, manifest_document)
    atomic_write_json(
        completion_path,
        completion_seal(
            [manifest_path],
            context={
                "model": "COLIPRI",
                "contract_sha256": contract_sha,
                "row_count": len(rows),
                "entries_sha256": fingerprint(manifest_document["entries"]),
                "inventory_sha256": contract["dataset_inventory"]["rows_sha256"],
            },
        ),
    )
    execution_receipt = None
    if fresh_preparation and args.source_fingerprint == "sha256":
        execution_receipt = _write_execution_ready_receipt(
            root,
            read_json(manifest_path),
            validation_source="fresh_sha256_preparation",
        )
    return {
        "status": "complete",
        "cache_directory": str(root),
        "row_count": len(rows),
        "execution_ready": execution_receipt is not None,
        "execution_receipt": str((root / EXECUTION_RECEIPT_FILENAME).resolve())
        if execution_receipt is not None
        else None,
        "next_action": None
        if execution_receipt is not None
        else "run prepare_cache.py --action validate with source_fingerprint=sha256",
    }


def _safe_artifact_path(root: Path, relative_value: Any) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"Unsafe COLIPRI feature artifact path: {relative_value}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:  # pragma: no cover
        raise ContractError(
            f"COLIPRI feature artifact escapes cache root: {relative_value}"
        ) from exc
    return path


def _artifact_stat_binding(root: Path, entries: Mapping[str, Any]) -> dict:
    records = []
    total_bytes = 0
    for key in sorted(entries):
        entry = entries[key]
        if not isinstance(entry, Mapping):
            raise ContractError(f"Malformed COLIPRI feature entry {key}")
        path = _safe_artifact_path(root, entry.get("artifact"))
        if not path.is_file():
            raise ContractError(f"COLIPRI feature artifact is missing: {path}")
        stat = path.stat()
        relative = path.relative_to(root.resolve()).as_posix()
        records.append(
            {
                "entry_key": str(key),
                "path": relative,
                "bytes": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
            }
        )
        total_bytes += int(stat.st_size)
    return {
        "artifact_count": len(records),
        "artifact_bytes": total_bytes,
        "artifact_stat_inventory_sha256": fingerprint(records),
    }


def _execution_ready_binding(
    root: Path,
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict:
    manifest_path = root / "feature_cache_manifest.json"
    completion_path = root / "feature_cache_completion.json"
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping):
        raise ContractError("COLIPRI execution receipt cannot bind malformed entries")
    if manifest.get("contract", {}).get("source_fingerprint") != "sha256":
        raise ContractError(
            "COLIPRI execution readiness requires source_fingerprint=sha256"
        )
    return {
        "contract_sha256": manifest["contract_sha256"],
        "completion_file_sha256": sha256_file(completion_path),
        "completion_seal_sha256": completion["seal_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "entries_sha256": fingerprint(entries),
        **_artifact_stat_binding(root, entries),
    }


def _write_execution_ready_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    validation_source: str,
) -> dict:
    if validation_source not in (
        "fresh_sha256_preparation",
        "exhaustive_sha256_validation",
    ):
        raise ValueError(f"Unsupported execution-receipt source: {validation_source}")
    completion = validate_completion(root, "feature_cache_completion.json")
    binding = _execution_ready_binding(root, manifest, completion)
    document = {
        "schema_version": 1,
        "status": "execution_ready",
        "validated_at": utc_now(),
        "validation_source": validation_source,
        "source_fingerprint": "sha256",
        "binding": binding,
        "binding_sha256": fingerprint(binding),
    }
    document["receipt_sha256"] = fingerprint(document)
    atomic_write_json(root / EXECUTION_RECEIPT_FILENAME, document)
    return document


def _validate_execution_ready_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict:
    path = root / EXECUTION_RECEIPT_FILENAME
    if not path.is_file():
        raise ContractError(
            "COLIPRI cache is not execution-ready: the exhaustive-validation receipt "
            f"is missing ({path}). Run prepare_cache.py --action validate with "
            "source_fingerprint=sha256."
        )
    receipt = read_json(path)
    observed_receipt_sha = receipt.get("receipt_sha256")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if observed_receipt_sha != fingerprint(receipt_body):
        raise ContractError(f"COLIPRI execution-readiness receipt was modified: {path}")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "execution_ready"
        or receipt.get("source_fingerprint") != "sha256"
        or receipt.get("validation_source")
        not in ("fresh_sha256_preparation", "exhaustive_sha256_validation")
    ):
        raise ContractError(f"Invalid COLIPRI execution-readiness receipt: {path}")
    expected_binding = _execution_ready_binding(root, manifest, completion)
    if (
        receipt.get("binding") != expected_binding
        or receipt.get("binding_sha256") != fingerprint(expected_binding)
    ):
        raise ContractError(
            "COLIPRI execution-readiness receipt no longer matches the cache; "
            "rerun exhaustive SHA-256 validation"
        )
    return receipt


def _resolve_existing_cache(
    args: Any, *, require_execution_ready: bool = False
) -> tuple[Path, dict]:
    inventory_rows, inventory_manifest = all_rows(
        args.dataset_root, require_files=False
    )
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        upstream_root=args.upstream_root,
        local_files_only=args.local_files_only,
        provision_log=args.provision_log,
    )
    extraction_runtime = extraction_runtime_identity(args.device, args.feature_precision)
    precision = extraction_runtime["precision_resolved"]
    processor_identity = processor_config_identity(args.upstream_root)
    contract = _contract(
        args,
        checkpoint,
        precision,
        processor_identity,
        extraction_runtime,
        inventory_rows,
        inventory_manifest,
    )
    root = _cache_directory(args, contract)
    if not (root / "feature_cache_completion.json").is_file():
        legacy = _legacy_cache_contract(args, contract)
        if legacy is not None:
            root, contract = legacy
    completion = validate_completion(root, "feature_cache_completion.json")
    manifest_path = root / "feature_cache_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("contract_sha256") != fingerprint(contract) or manifest.get("status") != "complete":
        raise ContractError(f"COLIPRI feature cache contract/status mismatch: {root}")
    if manifest.get("contract") != contract:
        raise ContractError(f"COLIPRI feature cache contract document changed: {root}")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ContractError(f"COLIPRI feature cache entries are malformed: {root}")
    expected_keys = {_entry_key(row) for row in inventory_rows}
    if set(entries) != expected_keys:
        raise ContractError(
            "COLIPRI cache entry set does not match the sealed 100% inventory"
        )
    declared_paths = {
        _safe_artifact_path(root, entry.get("artifact")).resolve()
        for entry in entries.values()
        if isinstance(entry, Mapping)
    }
    if len(declared_paths) != len(entries):
        raise ContractError(
            "COLIPRI cache contains malformed or duplicate artifact declarations"
        )
    actual_paths = {path.resolve() for path in (root / "artifacts").rglob("*.pt")}
    if actual_paths != declared_paths:
        raise ContractError("COLIPRI cache contains undeclared or missing feature artifacts")
    entries_sha256 = fingerprint(manifest.get("entries", {}))
    if completion.get("context", {}).get("entries_sha256") != entries_sha256:
        raise ContractError(f"COLIPRI feature-cache entry fingerprint mismatch: {root}")
    if (
        completion.get("context", {}).get("inventory_sha256")
        != contract["dataset_inventory"]["rows_sha256"]
    ):
        raise ContractError(f"COLIPRI feature-cache inventory fingerprint mismatch: {root}")
    manifest["_sealed_manifest_sha256"] = sha256_file(manifest_path)
    manifest["_entries_sha256"] = entries_sha256
    manifest["_completion_seal_sha256"] = completion["seal_sha256"]
    if require_execution_ready:
        manifest["_execution_ready_receipt"] = _validate_execution_ready_receipt(
            root, manifest, completion
        )
    return root, manifest


def _validate_rows(
    args: Any,
    root: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    exhaustive: bool,
) -> None:
    entries = manifest.get("entries", {})
    for row in rows:
        key = _entry_key(row)
        if key not in entries:
            raise ContractError(f"COLIPRI feature cache is missing patient {row['patient_id']}")
        entry = entries[key]
        path = _safe_artifact_path(root, entry["artifact"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ContractError(f"COLIPRI feature artifact is missing or truncated: {path}")
        stored_source = entry.get("source_identity")
        if not isinstance(stored_source, dict):
            raise ContractError(f"COLIPRI source identity is missing for {row['patient_id']}")
        if args.source_fingerprint == "sha256" and not stored_source.get("sha256"):
            raise ContractError(f"COLIPRI source SHA-256 is missing for {row['patient_id']}")
        source_identity = _source_identity(
            Path(row["image_path"]),
            args.source_fingerprint,
            prior=stored_source,
            force_hash=exhaustive,
        )
        if stored_source != source_identity:
            changed = "content changed" if exhaustive else "changed"
            raise ContractError(
                f"COLIPRI source {changed} after caching: {row['image_path']}"
            )
        stored_artifact_sha = entry.get("sha256")
        if not isinstance(stored_artifact_sha, str) or len(stored_artifact_sha) != 64:
            raise ContractError(f"COLIPRI artifact SHA-256 is missing for {row['patient_id']}")
        if exhaustive:
            if sha256_file(path) != stored_artifact_sha:
                raise ContractError(f"COLIPRI feature artifact hash mismatch: {path}")
            _validate_payload(path, row, manifest["contract_sha256"], source_identity)


def validate_feature_cache(
    args: Any,
    *,
    required_rows: Sequence[Mapping[str, Any]] | None = None,
    exhaustive: bool = False,
) -> dict:
    root, manifest = _resolve_existing_cache(args)
    rows = list(required_rows) if required_rows is not None else all_rows(
        args.dataset_root, require_files=True
    )[0]
    _validate_rows(args, root, manifest, rows, exhaustive=exhaustive)
    execution_receipt = None
    execution_ready = bool(
        exhaustive
        and required_rows is None
        and args.source_fingerprint == "sha256"
    )
    if execution_ready:
        execution_receipt = _write_execution_ready_receipt(
            root,
            manifest,
            validation_source="exhaustive_sha256_validation",
        )
    if execution_ready:
        validation_level = "source_and_artifact_sha256_and_payload"
    elif exhaustive:
        validation_level = "artifact_sha256_and_payload_source_stat_only"
    else:
        validation_level = "sealed_manifest_and_structure"
    return {
        "status": "valid" if execution_ready else "diagnostic_validation_only",
        "execution_ready": execution_ready,
        "validation_level": validation_level,
        "cache_directory": str(root),
        "validated_rows": len(rows),
        "contract_sha256": manifest["contract_sha256"],
        "execution_receipt": str((root / EXECUTION_RECEIPT_FILENAME).resolve())
        if execution_receipt is not None
        else None,
        "execution_receipt_binding_sha256": execution_receipt[
            "binding_sha256"
        ]
        if execution_receipt is not None
        else None,
    }


class FeatureStore:
    def __init__(self, args: Any):
        self._args = args
        self.root, self.manifest = _resolve_existing_cache(
            args, require_execution_ready=True
        )
        self.contract_sha256 = self.manifest["contract_sha256"]
        self.execution_receipt = self.manifest["_execution_ready_receipt"]
        self._finite_validated: set[str] = set()

    def validate_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        _validate_rows(
            self._args,
            self.root,
            self.manifest,
            list(rows),
            exhaustive=False,
        )

    @property
    def identity(self) -> dict:
        return {
            "feature_contract_sha256": self.contract_sha256,
            "feature_cache_manifest_sha256": self.manifest[
                "_sealed_manifest_sha256"
            ],
            "feature_entries_sha256": self.manifest["_entries_sha256"],
            "feature_cache_completion_seal_sha256": self.manifest[
                "_completion_seal_sha256"
            ],
            "execution_receipt_binding_sha256": self.execution_receipt[
                "binding_sha256"
            ],
            "checkpoint": self.manifest["checkpoint"],
            "upstream_revision": self.manifest["source"]["revision"],
            "representation": self.manifest["contract"]["representation"],
            "feature_shape_cdhw": self.manifest["contract"]["feature_shape_cdhw"],
            "storage_dtype": self.manifest["contract"]["storage_dtype"],
            "extraction_precision": self.manifest["contract"]["extraction_precision"],
            "extraction_runtime": self.manifest["contract"]["extraction_runtime"],
        }

    def load(self, row: Mapping[str, Any]):
        key = _entry_key(row)
        try:
            entry = self.manifest["entries"][key]
        except KeyError as exc:
            raise ContractError(f"Feature cache has no row for patient {row['patient_id']}") from exc
        path = _safe_artifact_path(self.root, entry["artifact"])
        payload = _validate_payload(
            path,
            row,
            self.contract_sha256,
            entry.get("source_identity"),
            verify_finite=key not in self._finite_validated,
        )
        self._finite_validated.add(key)
        # torch.load owns this tensor; no clone is needed.  Probe heads never
        # mutate cached features.
        return payload["dense_features"]


__all__ = [
    "EXECUTION_RECEIPT_FILENAME",
    "FEATURE_BYTES_PER_PATIENT",
    "EXTRACTION_RUNTIME_DISTRIBUTIONS",
    "FeatureStore",
    "all_rows",
    "cache_plan",
    "extraction_runtime_identity",
    "planned_extraction_runtime",
    "prepare_feature_cache",
    "preprocessing_contract",
    "validate_feature_cache",
]
