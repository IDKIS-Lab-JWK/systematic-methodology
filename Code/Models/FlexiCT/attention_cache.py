"""Sealed depth-token cache for the frozen FlexiCT attention readout."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import (
        CLASS_NAMES,
        EXPECTED_PATCH_TOKENS,
        UPSTREAM_REVISION,
    )
    from .data import preprocess_nifti, preprocessing_contract
    from .model import build_vision_only_encoder, vision_encoder_contract
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import CLASS_NAMES, EXPECTED_PATCH_TOKENS, UPSTREAM_REVISION
    from data import preprocess_nifti, preprocessing_contract
    from model import build_vision_only_encoder, vision_encoder_contract
    from provisioning import ensure_upstream, resolve_checkpoint

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
    rows_fingerprint,
    rows_from_mapping,
    ensure_same_or_write,
    sha256_file,
    utc_now,
    validate_completion,
)
from _supervised_adaptation.runtime import atomic_torch_save, require_torch, torch_load
from _supervised_adaptation.safe_io import same_or_write_json


CACHE_SCHEMA_VERSION = "flexict-attention-token-cache-v1"
DEFAULT_ATTENTION_CACHE_NAMESPACE = (
    "flexict_vlm_lps_2mm_160cube_depth20_meanhw_864_fp16_v1"
)
TOKEN_SHAPE = (20, 864)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _extraction_runtime(device: str) -> dict:
    torch = require_torch()
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": str(device),
        "dependencies": {
            "numpy": _package_version("numpy"),
            "simpleitk": _package_version("SimpleITK"),
        },
    }
    parsed = torch.device(device)
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise ContractError("CUDA was requested for attention-token extraction but is unavailable")
        index = parsed.index if parsed.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        runtime["cuda_device_index"] = int(index)
        runtime["cuda_device_name"] = str(properties.name)
        runtime["cuda_compute_capability"] = [
            int(properties.major),
            int(properties.minor),
        ]
    return runtime


def _source_identity(
    path: str | Path, *, prior: Mapping[str, Any] | None = None
) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stat = source.stat()
    current = {
        "path": str(source),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if (
        isinstance(prior, Mapping)
        and all(prior.get(key) == value for key, value in current.items())
        and isinstance(prior.get("sha256"), str)
    ):
        return {**current, "sha256": str(prior["sha256"])}
    return {**current, "sha256": sha256_file(source)}


def _implementation_identity() -> dict:
    package = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        package / "model.py",
        package / "data.py",
        package / "provisioning.py",
        package.parent / "_supervised_adaptation" / "runtime.py",
    ]
    records = [
        {
            "relative_path": path.relative_to(package.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"files": records, "sha256": fingerprint(records)}


def cache_contract(
    args: Any,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    extraction_runtime: Mapping[str, Any] | None = None,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": "FlexiCT",
        "purpose": "FrozenSingleHeadAttentionReadout",
        "namespace": str(args.cache_namespace),
        "split_manifest": manifest["_path"],
        "split_manifest_sha256": manifest["_sha256"],
        "rows_sha256": rows_fingerprint(rows),
        "row_count": len(rows),
        "source": dict(source),
        "checkpoint": dict(checkpoint),
        "preprocessing": preprocessing_contract(),
        "vision": vision_encoder_contract(),
        "implementation": _implementation_identity(),
        "extraction_runtime": dict(
            extraction_runtime
            if extraction_runtime is not None
            else _extraction_runtime(str(args.device))
        ),
        "source_identity_policy": (
            "per-patient SHA-256; a recorded digest is reused on resume only when "
            "absolute path, byte size, and nanosecond mtime are unchanged"
        ),
        "extraction": {
            "patch_grid_dhw": [20, 20, 20],
            "patch_tokens": EXPECTED_PATCH_TOKENS,
            "patch_dimension": 864,
            "depth_tokens": 20,
            "aggregation": "reshape final normalized patch tokens to D,H,W and mean over H,W",
            "stored_shape": list(TOKEN_SHAPE),
            "stored_dtype": "float16",
            "encoder_state": "fully frozen eval/inference mode",
        },
    }


def _entry_path(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "Patients" / str(row["class_name"]) / f"{row['patient_id']}.pt"


def _manifest_shard_path(root: Path, relative_path: Any) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"Unsafe attention-cache shard path: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ContractError(f"Attention-cache shard escapes its cache root: {relative_path!r}")
    return resolved


def _validate_entry(
    path: Path,
    row: Mapping[str, Any],
    expected_sha256: str | None = None,
    expected_contract_sha256: str | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
) -> dict:
    torch = require_torch()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ContractError(f"FlexiCT attention-cache shard hash changed: {path}")
    payload = torch_load(path, map_location="cpu")
    if (
        payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or str(payload.get("patient_id")) != str(row["patient_id"])
        or int(payload.get("label", -1)) != int(row["label"])
        or str(payload.get("class_name")) != str(row["class_name"])
    ):
        raise ContractError(f"FlexiCT attention-cache shard identity changed: {path}")
    if expected_contract_sha256 and payload.get("contract_sha256") != expected_contract_sha256:
        raise ContractError(f"FlexiCT attention-cache shard contract changed: {path}")
    if expected_source_identity is not None and payload.get("source_identity") != dict(
        expected_source_identity
    ):
        raise ContractError(f"FlexiCT attention-cache source identity changed: {path}")
    tokens = payload.get("tokens")
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != TOKEN_SHAPE:
        raise ContractError(f"Unexpected attention-token shape in {path}")
    if tokens.dtype != torch.float16 or not bool(torch.isfinite(tokens).all()):
        raise ContractError(f"Attention tokens must be finite float16 in {path}")
    return {"sha256": digest, "bytes": path.stat().st_size}


def load_attention_cache(
    root: str | Path,
    *,
    expected_namespace: str | None = None,
    validate_shards: bool = False,
) -> dict:
    directory = Path(root)
    manifest_path = directory / "cache_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ContractError(f"Incomplete or incompatible FlexiCT attention cache: {directory}")
    if expected_namespace and manifest.get("namespace") != expected_namespace:
        raise ContractError(
            f"Attention-cache namespace mismatch: expected {expected_namespace!r}, got {manifest.get('namespace')!r}"
        )
    seal = validate_completion(directory, "cache_completion.json")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != int(manifest.get("entry_count", -1)):
        raise ContractError("Attention-cache manifest has a malformed entry inventory")
    contract_sha256 = fingerprint(manifest.get("contract"))
    if manifest.get("contract_sha256") != contract_sha256:
        raise ContractError("Attention-cache manifest contract fingerprint changed")
    expected_completion_context = {
        "contract_sha256": contract_sha256,
        "entry_inventory_sha256": fingerprint(entries),
    }
    if seal.get("context") != expected_completion_context:
        raise ContractError("Attention-cache completion context changed")
    patient_ids: set[str] = set()
    shard_paths: set[Path] = set()
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("patient_id"), str)
            or not isinstance(entry.get("relative_path"), str)
            or not isinstance(entry.get("source_identity"), Mapping)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("bytes"), int)
        ):
            raise ContractError("Attention-cache manifest contains a malformed entry")
        shard = _manifest_shard_path(directory, entry["relative_path"])
        if str(entry["patient_id"]) in patient_ids or shard in shard_paths:
            raise ContractError("Attention-cache manifest contains duplicate patients or shards")
        patient_ids.add(str(entry["patient_id"]))
        shard_paths.add(shard)
        if (
            not shard.is_file()
            or shard.stat().st_size != int(entry["bytes"])
            or sha256_file(shard) != str(entry["sha256"])
        ):
            raise ContractError(f"Attention-cache shard inventory changed: {shard}")
        if validate_shards:
            _validate_entry(
                shard,
                entry,
                expected_sha256=entry["sha256"],
                expected_contract_sha256=contract_sha256,
                expected_source_identity=entry.get("source_identity"),
            )
    return manifest


def resolve_cached_rows(
    root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_namespace: str,
) -> list[dict]:
    directory = Path(root)
    manifest = load_attention_cache(directory, expected_namespace=expected_namespace)
    by_patient = {str(item["patient_id"]): item for item in manifest["entries"]}
    if len(by_patient) != len(manifest["entries"]):
        raise ContractError("Duplicate patient IDs in FlexiCT attention cache")
    result = []
    for row in rows:
        entry = by_patient.get(str(row["patient_id"]))
        if entry is None:
            raise ContractError(f"Patient is absent from FlexiCT attention cache: {row['patient_id']}")
        if int(entry["label"]) != int(row["label"]) or str(entry["class_name"]) != str(row["class_name"]):
            raise ContractError(f"Attention-cache label changed for patient {row['patient_id']}")
        item = dict(row)
        item["token_path"] = str(
            _manifest_shard_path(directory, entry["relative_path"])
        )
        item["token_sha256"] = str(entry["sha256"])
        item["token_contract_sha256"] = str(manifest["contract_sha256"])
        item["source_identity"] = dict(entry["source_identity"])
        result.append(item)
    return result


def prepare_cache(args: Any) -> dict:
    torch = require_torch()
    manifest = load_manifest(args.dataset_root, int(args.percentage))
    mapping = merge_mappings([development_mapping(manifest), heldout_mapping(manifest)])
    rows = rows_from_mapping(args.dataset_root, mapping, require_files=not args.dry_run)
    root = Path(args.cache_dir)
    planned_source = {"revision": str(args.upstream_revision), "path": str(Path(args.upstream_root).resolve())}
    planned_checkpoint = {
        "path": str(Path(args.checkpoint).resolve()) if args.checkpoint else "required_at_execution",
        "sha256": args.checkpoint_sha256 or "computed_at_execution",
    }
    if args.dry_run:
        return {
            "status": "dry_run",
            "side_effects": False,
            "cache_dir": str(root.resolve()),
            "row_count": len(rows),
            "contract": cache_contract(
                args,
                manifest,
                rows,
                planned_source,
                planned_checkpoint,
                {
                    "status": "resolved_at_execution",
                    "requested_device": str(args.device),
                },
            ),
        }
    source = ensure_upstream(
        args.upstream_root,
        revision=str(args.upstream_revision),
        local_files_only=bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
    runtime = _extraction_runtime(str(args.device))
    contract = cache_contract(args, manifest, rows, source, checkpoint, runtime)
    contract_sha256 = fingerprint(contract)
    if (root / "cache_completion.json").is_file():
        if not args.resume:
            raise ContractError(f"Completed attention cache exists and resume is disabled: {root}")
        completed = load_attention_cache(
            root,
            expected_namespace=str(args.cache_namespace),
            validate_shards=bool(args.validate_shards),
        )
        if completed.get("contract") != contract:
            raise ContractError(f"Completed attention-cache contract changed: {root}")
        completed_by_patient = {
            str(entry["patient_id"]): entry for entry in completed["entries"]
        }
        for row in rows:
            entry = completed_by_patient.get(str(row["patient_id"]))
            if entry is None:
                raise ContractError(
                    f"Completed attention cache is missing patient {row['patient_id']}"
                )
            observed_source = _source_identity(
                row["image_path"], prior=entry.get("source_identity")
            )
            if observed_source != entry.get("source_identity"):
                raise ContractError(
                    f"Completed attention-cache source changed: {row['image_path']}"
                )
        return {"status": "reused", "cache_dir": str(root.resolve()), "entry_count": completed["entry_count"]}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ContractError(f"Partial attention cache exists and resume is disabled: {root}")
    if root.exists():
        expected_shards = {
            _entry_path(root, row).resolve() for row in rows
        }
        allowed_metadata = {
            (root / "cache_context.json").resolve(),
            (root / "cache_progress.json").resolve(),
            (root / "cache_manifest.json").resolve(),
        }
        unknown = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.resolve() not in expected_shards
            and path.resolve() not in allowed_metadata
        ]
        if unknown:
            raise ContractError(
                f"Unknown unsealed attention-cache artifact; refusing resume: {unknown[0]}"
            )
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "cache_context.json"
    if context_path.is_file() and read_json(context_path) != contract:
        raise ContractError(f"Partial attention-cache contract changed: {root}")
    ensure_same_or_write(context_path, contract)
    progress_path = root / "cache_progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {
        "schema_version": CACHE_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "completed": {},
    }
    if (
        progress.get("schema_version") != CACHE_SCHEMA_VERSION
        or progress.get("contract_sha256") != contract_sha256
        or not isinstance(progress.get("completed"), Mapping)
    ):
        raise ContractError("Attention-cache progress belongs to a different contract")
    expected_patient_ids = {str(row["patient_id"]) for row in rows}
    unexpected_patient_ids = set(progress["completed"]) - expected_patient_ids
    if unexpected_patient_ids:
        raise ContractError(
            "Attention-cache progress contains patients outside the sealed inventory: "
            f"{sorted(unexpected_patient_ids)[:5]}"
        )
    encoder = None
    for row in rows:
        patient_id = str(row["patient_id"])
        path = _entry_path(root, row)
        recorded = progress["completed"].get(patient_id)
        prior_source = recorded.get("source_identity") if isinstance(recorded, Mapping) else None
        source_identity = _source_identity(row["image_path"], prior=prior_source)
        if recorded is not None:
            if not isinstance(recorded, Mapping) or not isinstance(
                recorded.get("sha256"), str
            ):
                raise ContractError(
                    f"Malformed attention-cache progress record: {patient_id}"
                )
            relative_path = path.relative_to(root).as_posix()
            if recorded.get("relative_path") != relative_path:
                raise ContractError(f"Attention-cache progress path changed: {path}")
            verified = _validate_entry(
                path,
                row,
                expected_sha256=recorded["sha256"],
                expected_contract_sha256=contract_sha256,
                expected_source_identity=source_identity,
            )
            if recorded.get("source_identity") != source_identity:
                raise ContractError(f"Attention-cache source changed: {row['image_path']}")
            if int(recorded.get("bytes", -1)) != int(verified["bytes"]):
                raise ContractError(f"Attention-cache progress size changed: {path}")
            continue
        if path.is_file():
            # Recovery boundary for interruption after the atomic shard rename
            # but before progress.json was updated.
            verified = _validate_entry(
                path,
                row,
                expected_contract_sha256=contract_sha256,
                expected_source_identity=source_identity,
            )
            progress["completed"][patient_id] = {
                **verified,
                "relative_path": path.relative_to(root).as_posix(),
                "source_identity": source_identity,
                "completed_at": utc_now(),
            }
            atomic_write_json(progress_path, progress)
            continue
        if encoder is None:
            encoder, _details = build_vision_only_encoder(
                upstream_root=args.upstream_root,
                checkpoint_path=checkpoint["path"],
                device=str(args.device),
                allow_trusted_legacy_checkpoint=bool(
                    args.allow_trusted_legacy_checkpoint
                ),
                trusted_checkpoint_sha256=args.checkpoint_sha256,
            )
            encoder.eval().requires_grad_(False)
        image, metadata = preprocess_nifti(row["image_path"])
        images = image.unsqueeze(0).to(str(args.device))
        with torch.inference_mode():
            outputs = encoder.vision(images, is_training=True)
            patches = outputs["x_norm_patchtokens"]
        if tuple(patches.shape) != (1, EXPECTED_PATCH_TOKENS, 864):
            raise RuntimeError(f"Unexpected FlexiCT patch-token shape: {tuple(patches.shape)}")
        tokens = (
            patches.float()
            .reshape(1, 20, 20, 20, 864)
            .mean(dim=(2, 3))[0]
            .to(dtype=torch.float16, device="cpu")
            .contiguous()
        )
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "patient_id": patient_id,
            "class_name": str(row["class_name"]),
            "label": int(row["label"]),
            "tokens": tokens,
            "preprocessing_metadata": metadata,
            "contract_sha256": contract_sha256,
            "source_identity": source_identity,
        }
        atomic_torch_save(payload, path)
        verified = _validate_entry(
            path,
            row,
            expected_contract_sha256=contract_sha256,
            expected_source_identity=source_identity,
        )
        progress["completed"][patient_id] = {
            **verified,
            "relative_path": path.relative_to(root).as_posix(),
            "source_identity": source_identity,
            "completed_at": utc_now(),
        }
        atomic_write_json(progress_path, progress)

    entries = []
    for row in rows:
        patient_id = str(row["patient_id"])
        record = progress["completed"].get(patient_id)
        if record is None:
            raise ContractError(f"Missing completed attention-cache record: {patient_id}")
        entries.append(
            {
                "patient_id": patient_id,
                "class_name": str(row["class_name"]),
                "label": int(row["label"]),
                "relative_path": str(record["relative_path"]),
                "sha256": str(record["sha256"]),
                "bytes": int(record["bytes"]),
                "source_identity": dict(record["source_identity"]),
            }
        )
    final_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "namespace": str(args.cache_namespace),
        "entry_count": len(entries),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "entries": entries,
    }
    manifest_path = root / "cache_manifest.json"
    if manifest_path.is_file():
        existing_manifest = read_json(manifest_path)
        final_manifest["completed_at"] = existing_manifest.get("completed_at")
    else:
        final_manifest["completed_at"] = utc_now()
    same_or_write_json(manifest_path, final_manifest)
    atomic_write_json(
        root / "cache_completion.json",
        completion_seal(
            [root / "cache_context.json", root / "cache_progress.json", root / "cache_manifest.json"],
            context={
                "contract_sha256": contract_sha256,
                "entry_inventory_sha256": fingerprint(entries),
            },
        ),
    )
    return {"status": "complete", "cache_dir": str(root.resolve()), "entry_count": len(entries)}


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_ATTENTION_CACHE_NAMESPACE",
    "TOKEN_SHAPE",
    "load_attention_cache",
    "prepare_cache",
    "resolve_cached_rows",
]
