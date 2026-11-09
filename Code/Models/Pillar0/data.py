"""Patient-level NIfTI discovery and split handling for Pillar-0.

Pillar-0 consumes full 3-D CT volumes.  This module deliberately accepts only
NIfTI sources for this repository's pipeline; PNG slices and derived 2.5-D
stacks are not interchangeable with the volumetric Pillar-0 input contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # Package import.
    from . import CLASS_NAMES
except ImportError:  # Direct script/module import from this directory.
    from __init__ import CLASS_NAMES


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti_path(path: str | Path) -> bool:
    """Return whether *path* has an accepted NIfTI filename suffix."""

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def require_nifti_path(path: str | Path, *, must_exist: bool = True) -> Path:
    """Validate the repository's NIfTI-only Pillar-0 source contract."""

    candidate = Path(path)
    if not is_nifti_path(candidate):
        raise ValueError(
            "Pillar-0 primary inputs must be full 3-D NIfTI volumes "
            f"(.nii or .nii.gz), not PNG/2.5-D/DICOM inputs: {candidate}"
        )
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"Missing NIfTI volume: {candidate}")
    return candidate


def nifti_path(dataset_root: Path, class_name: str, patient_id: str) -> Path:
    """Resolve a patient NIfTI, preferring the repository's ``.nii.gz`` form."""

    _require_class_name(class_name)
    patient = _normalise_patient_id(patient_id)
    base = Path(dataset_root) / "NIFTI" / class_name
    compressed = base / f"{patient}.nii.gz"
    if compressed.is_file():
        return compressed
    uncompressed = base / f"{patient}.nii"
    if uncompressed.is_file():
        return uncompressed
    return compressed


def split_manifest_path(dataset_root: Path, percentage: int) -> Path:
    percentage = int(percentage)
    if not 0 < percentage <= 100:
        raise ValueError(f"Percentage must be in [1, 100], got {percentage}")
    return Path(dataset_root) / "Splits" / f"{percentage}_percent.json"


def load_manifest(
    dataset_root: Path,
    percentage: int,
    *,
    expected_folds: int = 5,
) -> dict:
    """Load and validate an existing percentage split manifest."""

    return load_manifest_file(
        split_manifest_path(dataset_root, percentage),
        expected_percentage=int(percentage),
        expected_folds=expected_folds,
    )


def load_manifest_file(
    path: str | Path,
    *,
    expected_percentage: int | None = None,
    expected_folds: int = 5,
) -> dict:
    """Load either supported split layout (``cross_validation``/``nested_cv``)."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid split JSON in {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Split manifest must contain a JSON object: {manifest_path}")

    if expected_percentage is not None:
        recorded = manifest.get("metadata", {}).get("percentage")
        if recorded is not None and int(recorded) != int(expected_percentage):
            raise ValueError(
                f"Manifest percentage mismatch in {manifest_path}: "
                f"expected {expected_percentage}, found {recorded}"
            )
    validate_manifest(manifest, expected_folds=expected_folds)
    return manifest


def classes_from_manifest(manifest: Mapping) -> list[str]:
    """Validate manifest labels and return the one canonical class order."""

    mappings: list[Mapping] = []
    heldout = manifest.get("heldout_test")
    if heldout is not None:
        mappings.append(_partition_mapping(heldout, "heldout_test"))
    cv = _cv_block(manifest)
    folds = cv.get("folds")
    if isinstance(folds, list) and folds:
        mappings.append(_partition_mapping(folds[0], "first CV fold"))
    if not mappings:
        raise ValueError("Manifest contains no class mappings")
    for mapping in mappings:
        _validate_class_keys(mapping)
    return list(CLASS_NAMES)


def fold_mapping(manifest: Mapping, fold_number: int) -> dict[str, list[str]]:
    """Return one fold's patient mapping in canonical class order."""

    fold_number = int(fold_number)
    for position, fold in enumerate(_folds(manifest), start=1):
        recorded = fold.get("fold", fold.get("fold_number", position))
        if int(recorded) == fold_number:
            return _normalise_mapping(
                _partition_mapping(fold, f"CV fold {fold_number}"),
                f"CV fold {fold_number}",
            )
    raise ValueError(f"Fold {fold_number} not found")


def merge_mappings(
    mappings: Iterable[Mapping[str, Iterable[str]]],
    classes: Sequence[str] = CLASS_NAMES,
) -> dict[str, list[str]]:
    """Merge patient mappings while preserving class and input order."""

    merged = {class_name: [] for class_name in classes}
    for mapping in mappings:
        normalised = _normalise_mapping(mapping, "mapping")
        for class_name in classes:
            merged[class_name].extend(normalised[class_name])
    return merged


def cross_validation_split(
    manifest: Mapping, validation_fold: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Mirror Merlin's one-fold validation / remaining-fold training workflow."""

    fold_numbers = _fold_numbers(manifest)
    if int(validation_fold) not in fold_numbers:
        raise ValueError(
            f"Validation fold {validation_fold} not in available folds {fold_numbers}"
        )
    validation = fold_mapping(manifest, int(validation_fold))
    training = merge_mappings(
        [
            fold_mapping(manifest, fold)
            for fold in fold_numbers
            if fold != int(validation_fold)
        ]
    )
    return training, validation


def development_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return all development patients from all five folds."""

    return merge_mappings(
        [fold_mapping(manifest, fold) for fold in _fold_numbers(manifest)]
    )


def heldout_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return the untouched held-out-test mapping."""

    partition = manifest.get("heldout_test")
    if partition is None:
        raise ValueError("Manifest has no heldout_test partition")
    return _normalise_mapping(
        _partition_mapping(partition, "heldout_test"), "heldout_test"
    )


def all_patient_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return development plus held-out patients for shared cache preparation."""

    return merge_mappings([development_mapping(manifest), heldout_mapping(manifest)])


def rows_from_mapping(
    dataset_root: Path,
    mapping: Mapping[str, Iterable[str]],
    classes: Sequence[str] = CLASS_NAMES,
    *,
    require_files: bool = True,
) -> list[dict]:
    """Create model/cache rows for a patient mapping using NIfTI sources only."""

    if tuple(classes) != CLASS_NAMES:
        raise ValueError(
            f"Pillar-0 class order must be {list(CLASS_NAMES)}, got {list(classes)}"
        )
    normalised = _normalise_mapping(mapping, "patient mapping")
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    rows: list[dict] = []
    for class_name in CLASS_NAMES:
        for patient_id in normalised[class_name]:
            source = nifti_path(Path(dataset_root), class_name, patient_id)
            require_nifti_path(source, must_exist=require_files)
            accession = f"{class_name}__{patient_id}"
            rows.append(
                {
                    "image": str(source.resolve(strict=False)),
                    "source_path": str(source.resolve(strict=False)),
                    "label": class_to_index[class_name],
                    "patient_id": patient_id,
                    "label_str": class_name,
                    "accession": accession,
                    "series_number": 1,
                }
            )
    return rows


def collect_required_rows(
    dataset_root: Path,
    manifest: Mapping,
    *,
    require_files: bool = True,
) -> list[dict]:
    """Collect the full development+held-out set for one shared cache."""

    rows = rows_from_mapping(
        dataset_root,
        all_patient_mapping(manifest),
        require_files=require_files,
    )
    identifiers = [row["patient_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate patient IDs found while collecting cache inputs")
    return rows


def validate_no_overlap(*row_groups: Sequence[Mapping]) -> None:
    """Raise if any patient appears in more than one partition."""

    seen: set[str] = set()
    for rows in row_groups:
        identifiers = {str(row["patient_id"]) for row in rows}
        overlap = seen.intersection(identifiers)
        if overlap:
            raise RuntimeError(
                f"Patient overlap across partitions: {sorted(overlap)[:5]}"
            )
        seen.update(identifiers)


def validate_manifest(manifest: Mapping, *, expected_folds: int = 5) -> None:
    """Validate labels, folds, counts, duplicates, and held-out isolation."""

    classes_from_manifest(manifest)
    folds = _folds(manifest)
    recorded_count = _cv_block(manifest).get("number_of_folds")
    if recorded_count is not None and int(recorded_count) != len(folds):
        raise ValueError(
            f"CV fold-count mismatch: metadata says {recorded_count}, found {len(folds)}"
        )
    if expected_folds is not None and len(folds) != int(expected_folds):
        raise ValueError(f"Expected {expected_folds} CV folds, found {len(folds)}")

    fold_numbers = _fold_numbers(manifest)
    if len(fold_numbers) != len(set(fold_numbers)):
        raise ValueError(f"Duplicate fold identifiers: {fold_numbers}")

    groups: list[set[str]] = []
    for fold_number in fold_numbers:
        mapping = fold_mapping(manifest, fold_number)
        groups.append(_mapping_patient_set(mapping, f"CV fold {fold_number}"))
    heldout = heldout_mapping(manifest)
    groups.append(_mapping_patient_set(heldout, "heldout_test"))

    seen: set[str] = set()
    for group in groups:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(
                "Patient leakage/duplication across folds or held-out test: "
                f"{sorted(overlap)[:5]}"
            )
        seen.update(group)

    _validate_recorded_counts(manifest, heldout)


def _cv_block(manifest: Mapping) -> Mapping:
    block = manifest.get("cross_validation") or manifest.get("nested_cv")
    if not isinstance(block, Mapping):
        raise ValueError("Manifest needs a cross_validation or nested_cv object")
    return block


def _folds(manifest: Mapping) -> list[Mapping]:
    folds = _cv_block(manifest).get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Cross-validation block needs a non-empty folds list")
    if not all(isinstance(fold, Mapping) for fold in folds):
        raise ValueError("Every CV fold must be a JSON object")
    return folds


def _fold_numbers(manifest: Mapping) -> list[int]:
    numbers: list[int] = []
    for position, fold in enumerate(_folds(manifest), start=1):
        value = fold.get("fold", fold.get("fold_number", position))
        try:
            numbers.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold identifier: {value!r}") from exc
    return numbers


def _partition_mapping(partition: Mapping, name: str) -> Mapping:
    mapping = partition.get("patients_by_class", partition)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} needs a patients_by_class mapping")
    return mapping


def _normalise_mapping(mapping: Mapping, name: str) -> dict[str, list[str]]:
    _validate_class_keys(mapping)
    result: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        values = mapping[class_name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name}.{class_name} must be a patient list")
        patients = [_normalise_patient_id(value) for value in values]
        if len(patients) != len(set(patients)):
            raise ValueError(f"Duplicate patients within {name}.{class_name}")
        result[class_name] = patients
    return result


def _validate_class_keys(mapping: Mapping) -> None:
    expected = set(CLASS_NAMES)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            "Class mismatch: expected exactly "
            f"{list(CLASS_NAMES)}; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _mapping_patient_set(mapping: Mapping, name: str) -> set[str]:
    normalised = _normalise_mapping(mapping, name)
    ownership: dict[str, str] = {}
    for class_name in CLASS_NAMES:
        for patient_id in normalised[class_name]:
            previous = ownership.get(patient_id)
            if previous is not None:
                raise ValueError(
                    f"Patient {patient_id} occurs in both {previous} and {class_name} ({name})"
                )
            ownership[patient_id] = class_name
    return set(ownership)


def _validate_recorded_counts(manifest: Mapping, heldout: Mapping) -> None:
    heldout_block = manifest.get("heldout_test", {})
    recorded = heldout_block.get("counts_by_class")
    if recorded is not None:
        actual = {name: len(heldout[name]) for name in CLASS_NAMES}
        normalised_recorded = {name: int(recorded[name]) for name in CLASS_NAMES}
        if normalised_recorded != actual:
            raise ValueError(
                f"Held-out class-count mismatch: recorded={normalised_recorded}, actual={actual}"
            )
    total = heldout_block.get("total")
    actual_total = sum(len(heldout[name]) for name in CLASS_NAMES)
    if total is not None and int(total) != actual_total:
        raise ValueError(
            f"Held-out total mismatch: recorded={total}, actual={actual_total}"
        )


def _normalise_patient_id(value: object) -> str:
    patient_id = str(value).strip()
    if not patient_id:
        raise ValueError("Patient IDs cannot be empty")
    if patient_id in {".", ".."} or Path(patient_id).name != patient_id:
        raise ValueError(f"Unsafe patient ID: {patient_id!r}")
    return patient_id


def _require_class_name(class_name: str) -> None:
    if class_name not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class {class_name!r}; expected one of {list(CLASS_NAMES)}"
        )


__all__ = [
    "CLASS_NAMES",
    "NIFTI_SUFFIXES",
    "all_patient_mapping",
    "classes_from_manifest",
    "collect_required_rows",
    "cross_validation_split",
    "development_mapping",
    "fold_mapping",
    "heldout_mapping",
    "is_nifti_path",
    "load_manifest",
    "load_manifest_file",
    "merge_mappings",
    "nifti_path",
    "require_nifti_path",
    "rows_from_mapping",
    "split_manifest_path",
    "validate_manifest",
    "validate_no_overlap",
]
