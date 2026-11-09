"""Sealed FP32 CT inputs shared by both Primus-M fine-tuning scopes."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import platform
import shutil
import sys
from pathlib import Path

import torch

from . import INPUT_SHAPE_DHW, MODEL_REVISION
from .cache import (
    _atomic_torch_save, _entry_key, _inventory_binding, _safe_artifact_path,
    _source_identity, all_rows, preprocessing_contract,
)
from .model import _activate_source, preprocess_nifti, processor_config_identity
from .provisioning import ensure_upstream
from _linear_probe.common import (
    ContractError, atomic_write_json, fingerprint, read_json, sha256_file, utc_now,
)


VOLUME_SHAPE = (1, *INPUT_SHAPE_DHW)
MANIFEST = "volume_cache_manifest.json"
PROGRESS = "volume_cache_progress.json"
RECEIPT = "volume_cache_execution_ready.json"


def _cache_base(args):
    return Path(args.cache_root or (Path(args.results_root) / "Cache"))


def _write_sealed(path, document):
    document = dict(document)
    document["seal_sha256"] = fingerprint(document)
    atomic_write_json(path, document)


def _read_sealed(path):
    document = read_json(path)
    digest = document.pop("seal_sha256", None)
    if digest != fingerprint(document):
        raise ContractError(f"Invalid COLIPRI volume-cache seal: {path}")
    return document


def _runtime():
    return {
        "python": platform.python_version(), "machine": platform.machine(),
        "torch": str(torch.__version__), "device": "cpu", "dtype": "float32",
        "packages": {name: importlib.metadata.version(name) for name in (
            "colipri", "torchio", "numpy", "scipy", "SimpleITK", "nibabel",
        )},
    }


def _context(args, *, provision=False):
    rows, split = all_rows(args.dataset_root, require_files=True)
    ensure_upstream(
        args.upstream_root, revision=MODEL_REVISION,
        local_files_only=not provision or bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    package = Path(__file__).resolve().parent
    contract = {
        "schema_version": 1, "model": "COLIPRI", "upstream_revision": MODEL_REVISION,
        "representation": "preprocessed_ct", "shape_cdhw": list(VOLUME_SHAPE),
        "storage_dtype": "float32", "source_fingerprint": getattr(args, "source_fingerprint", "sha256"),
        "preprocessing": preprocessing_contract(processor_config_identity(args.upstream_root)),
        "runtime": _runtime(), "inventory": _inventory_binding(rows, split),
        "implementation": {name: sha256_file(package / name) for name in (
            "volume_cache.py", "model.py", "cache.py", "provisioning.py", "__init__.py",
        )},
    }
    root = _cache_base(args) / "Volumes" / f"colipri_ct_fp32_{fingerprint(contract)[:16]}"
    return rows, contract, root


def volume_cache_plan(args):
    rows, _split = all_rows(args.dataset_root, require_files=False)
    size = math.prod(VOLUME_SHAPE) * 4
    return {
        "status": "plan", "side_effects": False, "cache_target": "finetune",
        "row_count": len(rows), "shape_cdhw": list(VOLUME_SHAPE), "storage_dtype": "float32",
        "cache_directory": str(_cache_base(args) / "Volumes" / "colipri_ct_fp32_<resolved-contract-hash>"),
        "bytes_per_patient_raw_float32": size, "estimated_raw_volume_bytes": len(rows) * size,
        "estimated_raw_volume_gib": len(rows) * size / 1024**3,
        "shared_by": ["PrimusMFinalBlockFineTune", "PrimusMFinalStageFineTune"],
        "preprocessing": preprocessing_contract(), "runtime": "resolved_at_execution",
        "encoder_checkpoint_required": False, "processing_device": "cpu",
    }


def _processor(args):
    _activate_source(Path(args.upstream_root))
    module = importlib.import_module("colipri")
    if not Path(module.__file__).resolve().is_relative_to(Path(args.upstream_root).resolve()):
        raise ContractError("Imported COLIPRI processor is outside the pinned source tree")
    return module.get_processor(image_only=True)


def _payload(path, row, contract, source, *, finite=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict)
            or payload.get("contract_sha256") != fingerprint(contract)
            or payload.get("patient_id") != str(row["patient_id"])
            or payload.get("source_identity") != source):
        raise ContractError(f"Invalid COLIPRI volume payload identity: {path}")
    volume = payload.get("volume")
    if not isinstance(volume, torch.Tensor) or tuple(volume.shape) != VOLUME_SHAPE or volume.dtype != torch.float32:
        raise ContractError(f"Invalid COLIPRI volume shape/dtype: {path}")
    if finite and not bool(torch.isfinite(volume).all()):
        raise ContractError(f"Non-finite COLIPRI volume: {path}")
    return payload


def _load_manifest(root, contract, rows):
    if not (root / MANIFEST).is_file():
        raise ContractError(
            "COLIPRI fine-tuning CT cache is missing. Run "
            "COLIPRI_Cache_Run_Script.ps1 -Action prepare -CacheTarget finetune "
            "with the same -CacheRoot, or use -InputSource nifti for a legacy run."
        )
    manifest = _read_sealed(root / MANIFEST)
    if manifest.get("contract") != contract or manifest.get("status") != "complete":
        raise ContractError("COLIPRI volume-cache contract/status changed")
    entries = manifest.get("entries", {})
    if not isinstance(entries, dict) or set(entries) != {_entry_key(row) for row in rows}:
        raise ContractError("COLIPRI volume-cache patient inventory changed")
    declared = set()
    for key, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("artifact") != f"artifacts/{key}.pt":
            raise ContractError("Invalid COLIPRI volume-cache artifact declaration")
        declared.add(_safe_artifact_path(root, entry["artifact"]))
    if declared != {path.resolve() for path in (root / "artifacts").glob("*.pt")}:
        raise ContractError("COLIPRI volume-cache artifact inventory changed")
    return manifest


def _check_source(row, entry, mode, *, exhaustive=False):
    stored = entry["source_identity"]
    current = _source_identity(Path(row["image_path"]), mode if exhaustive else "stat", force_hash=exhaustive)
    if any(stored.get(key) != value for key, value in current.items()):
        raise ContractError(f"Source changed for COLIPRI patient {row['patient_id']}; rebuild the CT cache in a new CacheRoot")
    if mode == "sha256" and not stored.get("sha256"):
        raise ContractError("COLIPRI CT cache lacks source SHA-256 identity")


def _artifact_stats(root, entries):
    records = []
    for key in sorted(entries):
        entry = entries[key]
        path = _safe_artifact_path(root, entry["artifact"])
        stat = path.stat()
        if stat.st_size != entry["bytes"]:
            raise ContractError(f"COLIPRI cached volume size changed: {path}")
        records.append([entry["artifact"], stat.st_size, stat.st_mtime_ns])
    return fingerprint(records)


def _receipt_binding(root, manifest):
    return {
        "manifest_sha256": sha256_file(root / MANIFEST),
        "artifact_stats_sha256": _artifact_stats(root, manifest["entries"]),
    }


def _write_receipt(root, manifest):
    _write_sealed(root / RECEIPT, {
        "schema_version": 1, "validated_at": utc_now(), "source_fingerprint": "sha256",
        "binding": _receipt_binding(root, manifest),
    })


def _require_receipt(root, manifest):
    message = "CT cache needs exhaustive validation: run COLIPRI_Cache_Run_Script.ps1 -Action validate -CacheTarget finetune with the same -CacheRoot"
    if not (root / RECEIPT).is_file():
        raise ContractError(message)
    receipt = _read_sealed(root / RECEIPT)
    if receipt.get("source_fingerprint") != "sha256" or receipt.get("binding") != _receipt_binding(root, manifest):
        raise ContractError(message)
    return receipt


def validate_volume_cache(args, *, exhaustive=True):
    rows, contract, root = _context(args)
    manifest = _load_manifest(root, contract, rows)
    mode = contract["source_fingerprint"]
    _artifact_stats(root, manifest["entries"])
    for row in rows:
        entry = manifest["entries"][_entry_key(row)]
        _check_source(row, entry, mode, exhaustive=exhaustive)
        path = _safe_artifact_path(root, entry["artifact"])
        if exhaustive:
            if sha256_file(path) != entry["sha256"]:
                raise ContractError(f"COLIPRI cached volume hash changed: {path}")
            _payload(path, row, contract, entry["source_identity"])
    ready = exhaustive and mode == "sha256"
    if ready:
        _write_receipt(root, manifest)
    return {"status": "valid", "cache_target": "finetune", "cache_directory": str(root),
            "row_count": len(rows), "exhaustive": exhaustive, "execution_ready": ready}


def prepare_volume_cache(args):
    rows, contract, root = _context(args, provision=True)
    mode = contract["source_fingerprint"]
    if (root / MANIFEST).is_file():
        manifest = _load_manifest(root, contract, rows)
        for row in rows:
            _check_source(row, manifest["entries"][_entry_key(row)], mode)
        _artifact_stats(root, manifest["entries"])
        try:
            _require_receipt(root, manifest)
            ready = mode == "sha256"
        except ContractError:
            ready = False
        return {"status": "reused", "cache_target": "finetune", "cache_directory": str(root),
                "row_count": len(rows), "execution_ready": ready,
                "next_action": None if ready else "validate -CacheTarget finetune"}
    progress = _read_sealed(root / PROGRESS) if (root / PROGRESS).is_file() else {
        "contract": contract, "entries": {},
    }
    if progress.get("contract") != contract or not isinstance(progress.get("entries"), dict):
        raise ContractError("COLIPRI partial CT cache contract changed")
    entries = progress["entries"]
    if set(entries) - {_entry_key(row) for row in rows}:
        raise ContractError("COLIPRI partial CT cache inventory changed")
    remaining = sum(not (root / "artifacts" / f"{_entry_key(row)}.pt").is_file() for row in rows)
    parent = root
    while not parent.exists():
        parent = parent.parent
    required = int(remaining * math.prod(VOLUME_SHAPE) * 4 * 1.08) + 2 * 1024**3
    if shutil.disk_usage(parent).free < required:
        raise ContractError(f"COLIPRI CT cache needs approximately {required / 1024**3:.1f} GiB free; choose a larger -CacheRoot")
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    processor = None
    for index, row in enumerate(rows, start=1):
        key = _entry_key(row)
        path = root / "artifacts" / f"{key}.pt"
        source = _source_identity(Path(row["image_path"]), mode, force_hash=True)
        previous = entries.get(key)
        reusable = (previous is not None and previous.get("source_identity") == source
                    and path.is_file() and previous.get("sha256") == sha256_file(path))
        if not reusable:
            if processor is None:
                processor = _processor(args)
            images, metadata = preprocess_nifti(processor, row["image_path"])
            volume = images[0].detach().cpu().contiguous()
            if tuple(volume.shape) != VOLUME_SHAPE or volume.dtype != torch.float32 or not bool(torch.isfinite(volume).all()):
                raise ContractError("Released COLIPRI processor did not produce the expected FP32 CT")
            after = _source_identity(Path(row["image_path"]), "stat")
            if any(source[k] != v for k, v in after.items()):
                raise ContractError("NIfTI changed during COLIPRI preprocessing")
            _atomic_torch_save({"patient_id": str(row["patient_id"]), "contract_sha256": fingerprint(contract),
                                "source_identity": source, "preprocessing_metadata": metadata, "volume": volume}, path)
        _payload(path, row, contract, source)
        entries[key] = {"artifact": f"artifacts/{key}.pt", "bytes": path.stat().st_size,
                        "sha256": sha256_file(path), "source_identity": source}
        progress.update({"completed_rows": index, "updated_at": utc_now()})
        _write_sealed(root / PROGRESS, progress)
        if index % 25 == 0 or index == len(rows):
            print(f"COLIPRI CT cache: {index}/{len(rows)} patients ready", file=sys.stderr, flush=True)
    manifest = {"contract": contract, "entries": entries, "status": "complete", "completed_at": utc_now()}
    _write_sealed(root / MANIFEST, manifest)
    _load_manifest(root, contract, rows)
    if mode == "sha256":
        _write_receipt(root, manifest)
    return {"status": "prepared", "cache_target": "finetune", "cache_directory": str(root),
            "row_count": len(rows), "execution_ready": mode == "sha256"}


class VolumeStore:
    def __init__(self, args):
        rows, self.contract, self.root = _context(args)
        if self.contract["source_fingerprint"] != "sha256":
            raise ContractError("Fine-tuning requires a SHA-256-validated CT cache")
        self.manifest = _load_manifest(self.root, self.contract, rows)
        receipt = _require_receipt(self.root, self.manifest)
        for row in rows:
            _check_source(row, self.manifest["entries"][_entry_key(row)], "sha256")
        self.identity = {"contract_sha256": fingerprint(self.contract), **receipt["binding"]}
        self._checked = set()

    def prepare_rows(self, rows):
        prepared = []
        for row in rows:
            entry = self.manifest["entries"].get(_entry_key(row))
            if entry is None:
                raise ContractError(f"CT cache is missing patient {row['patient_id']}")
            _check_source(row, entry, "sha256")
            prepared.append({**row, "cache_path": str(_safe_artifact_path(self.root, entry["artifact"])),
                             "cache_sha256": entry["sha256"]})
        return prepared

    def load(self, row):
        key = _entry_key(row)
        entry = self.manifest["entries"][key]
        path = _safe_artifact_path(self.root, entry["artifact"])
        payload = _payload(path, row, self.contract, entry["source_identity"], finite=key not in self._checked)
        self._checked.add(key)
        return payload["volume"]
