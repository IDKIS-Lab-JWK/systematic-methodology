"""Small, dependency-light utilities shared by FlexiCT and COLIPRI.

This module deliberately contains no model imports.  Dry-run planning, split
validation, provenance inspection, and CLI help therefore remain available
before either model environment has been provisioned.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


CLASS_NAMES = (
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
)

PERCENTAGES = (20, 40, 60, 80, 100)
MODEL_SOURCES = ("refit_full_development", "cv_ensemble", "best_cv_fold")


class ContractError(RuntimeError):
    """Raised when an artifact, split, or experiment identity is incompatible."""


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    # ``timezone.utc`` is built into Python and works in a clean Windows venv;
    # ``ZoneInfo('UTC')`` otherwise requires the optional tzdata wheel there.
    return datetime.now(tz=timezone.utc).isoformat()


def berlin_now() -> str:
    return datetime.now(tz=ZoneInfo("Europe/Berlin")).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any, length: int = 64) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(dict(value)) + "\n")
        handle.flush()


class ReleaseAfterEmitFileHandler(logging.FileHandler):
    """Close the Windows file handle after every record while retaining logging."""

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None:
            self.stream = self._open()
        try:
            super().emit(record)
        finally:
            if self.stream is not None:
                self.stream.close()
                self.stream = None


def setup_logger(output_dir: str | Path, name: str) -> logging.Logger:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"{name}.{directory}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = ReleaseAfterEmitFileHandler(
        directory / f"run_{stamp}.log", encoding="utf-8", delay=True
    )
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def split_manifest_path(dataset_root: str | Path, percentage: int) -> Path:
    if percentage not in PERCENTAGES:
        raise ContractError(f"Unsupported data percentage: {percentage}")
    return Path(dataset_root) / "Splits" / f"{percentage}_percent.json"


def load_manifest(dataset_root: str | Path, percentage: int) -> dict:
    path = split_manifest_path(dataset_root, percentage)
    manifest = read_json(path)
    validate_manifest(manifest)
    manifest["_path"] = str(path.resolve())
    manifest["_sha256"] = sha256_file(path)
    return manifest


def _normalise_mapping(value: Mapping[str, Any], name: str) -> dict[str, list[str]]:
    if set(value) != set(CLASS_NAMES):
        raise ContractError(
            f"{name} class keys must exactly match the canonical class order; "
            f"got {sorted(value)}"
        )
    result: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        ids = [str(item) for item in value[class_name]]
        if len(ids) != len(set(ids)):
            raise ContractError(f"Duplicate patient ID inside {name}/{class_name}")
        result[class_name] = ids
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    try:
        heldout = _normalise_mapping(
            manifest["heldout_test"]["patients_by_class"], "heldout_test"
        )
        folds = manifest["cross_validation"]["folds"]
    except (KeyError, TypeError) as exc:
        raise ContractError("Split manifest has an unsupported schema") from exc
    if len(folds) != 5:
        raise ContractError(f"Exactly five CV folds are required, got {len(folds)}")
    heldout_sequence = [patient for ids in heldout.values() for patient in ids]
    heldout_ids = set(heldout_sequence)
    if len(heldout_sequence) != len(heldout_ids):
        raise ContractError("A held-out patient ID occurs under more than one class")
    development_seen: set[str] = set()
    for expected_fold, fold in enumerate(folds, start=1):
        if int(fold.get("fold", -1)) != expected_fold:
            raise ContractError("CV folds must be numbered 1 through 5 in order")
        mapping = _normalise_mapping(fold["patients_by_class"], f"fold_{expected_fold:02d}")
        current_sequence = [patient for ids in mapping.values() for patient in ids]
        current = set(current_sequence)
        if len(current_sequence) != len(current):
            raise ContractError(
                f"A patient ID occurs under more than one class in fold {expected_fold}"
            )
        if development_seen.intersection(current):
            raise ContractError("A patient ID occurs in more than one development fold")
        development_seen.update(current)
    overlap = development_seen.intersection(heldout_ids)
    if overlap:
        raise ContractError("Development and held-out patient IDs overlap")


def fold_mapping(manifest: Mapping[str, Any], fold_number: int) -> dict[str, list[str]]:
    folds = manifest["cross_validation"]["folds"]
    for fold in folds:
        if int(fold["fold"]) == int(fold_number):
            return _normalise_mapping(
                fold["patients_by_class"], f"fold_{fold_number:02d}"
            )
    raise ContractError(f"Fold {fold_number} not found")


def merge_mappings(mappings: Iterable[Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    merged = {class_name: [] for class_name in CLASS_NAMES}
    for mapping in mappings:
        for class_name in CLASS_NAMES:
            merged[class_name].extend(str(value) for value in mapping[class_name])
    return merged


def cross_validation_split(
    manifest: Mapping[str, Any], fold_number: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    validation = fold_mapping(manifest, fold_number)
    training = merge_mappings(
        fold_mapping(manifest, current)
        for current in range(1, 6)
        if current != fold_number
    )
    return training, validation


def development_mapping(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    return merge_mappings(fold_mapping(manifest, fold) for fold in range(1, 6))


def heldout_mapping(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    return _normalise_mapping(
        manifest["heldout_test"]["patients_by_class"], "heldout_test"
    )


def nifti_path(dataset_root: str | Path, class_name: str, patient_id: str) -> Path:
    return Path(dataset_root) / "NIFTI" / class_name / f"{patient_id}.nii.gz"


def rows_from_mapping(
    dataset_root: str | Path,
    mapping: Mapping[str, Sequence[str]],
    *,
    require_files: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, class_name in enumerate(CLASS_NAMES):
        for patient_id in mapping[class_name]:
            patient_id = str(patient_id)
            if patient_id in seen:
                raise ContractError(f"Patient ID {patient_id} occurs more than once")
            seen.add(patient_id)
            path = nifti_path(dataset_root, class_name, patient_id)
            if require_files and not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "patient_id": patient_id,
                    "class_name": class_name,
                    "label": label,
                    "image_path": str(path.resolve()),
                }
            )
    return rows


def rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    compact = [
        {
            "patient_id": str(row["patient_id"]),
            "class_name": str(row["class_name"]),
            "label": int(row["label"]),
        }
        for row in rows
    ]
    return fingerprint(compact)


def candidate_id(parameters: Mapping[str, Any]) -> str:
    return fingerprint(dict(parameters), length=12)


def materialize_file(source: str | Path, destination: str | Path) -> dict[str, Any]:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_hash = sha256_file(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if not destination_path.is_file() or sha256_file(destination_path) != source_hash:
            raise ContractError(f"Existing destination differs: {destination_path}")
        return {"mode": "existing", "sha256": source_hash}
    try:
        os.link(source_path, destination_path)
        mode = "hardlink"
    except OSError:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source_path, temporary)
            if sha256_file(temporary) != source_hash:
                raise ContractError("Copied artifact hash differs from source")
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        mode = "verified_copy"
    if sha256_file(destination_path) != source_hash:
        raise ContractError(f"Materialized artifact failed verification: {destination_path}")
    return {"mode": mode, "sha256": source_hash}


def completion_seal(paths: Sequence[str | Path], *, context: Mapping[str, Any]) -> dict:
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ContractError(f"Cannot seal missing artifact: {path}")
        records.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "schema_version": 1,
        "completed_at": utc_now(),
        "context": dict(context),
        "artifacts": records,
        "seal_sha256": fingerprint({"context": dict(context), "artifacts": records}),
    }


def validate_completion(directory: str | Path, filename: str) -> dict:
    root = Path(directory)
    seal_path = root / filename
    if not seal_path.is_file():
        raise ContractError(f"Completion seal is missing: {seal_path}")
    seal = read_json(seal_path)
    records = seal.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ContractError(f"Invalid completion seal: {root / filename}")
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            raise ContractError(f"Sealed artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ContractError(f"Sealed artifact size changed: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ContractError(f"Sealed artifact hash changed: {path}")
    expected = fingerprint({"context": seal["context"], "artifacts": records})
    if seal.get("seal_sha256") != expected:
        raise ContractError(f"Completion seal fingerprint changed: {root / filename}")
    return seal


def tree_completion_seal(
    root: str | Path, paths: Sequence[str | Path], *, context: Mapping[str, Any]
) -> dict:
    base = Path(root).resolve()
    records = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ContractError(f"Artifact is outside seal root {base}: {path}") from exc
        if not path.is_file():
            raise ContractError(f"Cannot seal missing artifact: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "completed_at": utc_now(),
        "context": dict(context),
        "artifacts": records,
        "seal_sha256": fingerprint({"context": dict(context), "artifacts": records}),
    }


def validate_tree_completion(root: str | Path, filename: str) -> dict:
    base = Path(root)
    seal_path = base / filename
    if not seal_path.is_file():
        raise ContractError(f"Completion seal is missing: {seal_path}")
    seal = read_json(seal_path)
    records = seal.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ContractError(f"Invalid completion seal: {base / filename}")
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("Completion seal contains an unsafe relative path")
        path = base / relative
        if not path.is_file():
            raise ContractError(f"Sealed artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ContractError(f"Sealed artifact changed: {path}")
    expected = fingerprint({"context": seal["context"], "artifacts": records})
    if expected != seal.get("seal_sha256"):
        raise ContractError(f"Completion seal fingerprint changed: {base / filename}")
    return seal


def ensure_same_or_write(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        existing = read_json(destination)
        if fingerprint(existing) != fingerprint(dict(document)):
            raise ContractError(f"Existing identity document is incompatible: {destination}")
        return
    atomic_write_json(destination, dict(document))


def class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for row in rows:
        counts[str(row["class_name"])] += 1
    return counts


__all__ = [
    "CLASS_NAMES",
    "ContractError",
    "MODEL_SOURCES",
    "PERCENTAGES",
    "append_jsonl",
    "atomic_write_csv",
    "atomic_write_json",
    "berlin_now",
    "candidate_id",
    "canonical_json",
    "class_counts",
    "close_logger",
    "completion_seal",
    "cross_validation_split",
    "development_mapping",
    "ensure_same_or_write",
    "fingerprint",
    "heldout_mapping",
    "load_manifest",
    "materialize_file",
    "merge_mappings",
    "methodology_root",
    "nifti_path",
    "read_json",
    "rows_fingerprint",
    "rows_from_mapping",
    "setup_logger",
    "sha256_file",
    "split_manifest_path",
    "utc_now",
    "validate_completion",
    "tree_completion_seal",
    "validate_tree_completion",
]
