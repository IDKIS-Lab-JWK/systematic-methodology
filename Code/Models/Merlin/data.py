"""Patient-level dataset manifest loading for Merlin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def nifti_path(dataset_root: Path, class_name: str, patient_id: str) -> Path:
    return dataset_root / "NIFTI" / class_name / f"{patient_id}.nii.gz"


def load_manifest(dataset_root: Path, percentage: int) -> dict:
    path = dataset_root / "Splits" / f"{percentage}_percent.json"
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("metadata", {}).get("percentage") != percentage:
        raise ValueError(f"Manifest percentage mismatch in {path}")
    cv = manifest.get("cross_validation") or manifest.get("nested_cv")
    if not cv or cv.get("number_of_folds") != 5:
        raise ValueError(f"Expected five CV folds in {path}")
    return manifest


def classes_from_manifest(manifest: dict) -> list[str]:
    return list(manifest["heldout_test"]["patients_by_class"])


def rows_from_mapping(
    dataset_root: Path,
    mapping: dict[str, Iterable[str]],
    classes: list[str],
) -> list[dict]:
    class_to_index = {name: index for index, name in enumerate(classes)}
    rows = []
    for class_name in classes:
        for patient_id in mapping[class_name]:
            image_path = nifti_path(dataset_root, class_name, str(patient_id))
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing NIfTI volume: {image_path}")
            rows.append(
                {
                    "image": str(image_path),
                    "label": class_to_index[class_name],
                    "patient_id": str(patient_id),
                    "label_str": class_name,
                }
            )
    return rows


def fold_mapping(manifest: dict, fold_number: int) -> dict[str, list[str]]:
    cv = manifest.get("cross_validation") or manifest["nested_cv"]
    for fold in cv["folds"]:
        if int(fold["fold"]) == fold_number:
            return fold["patients_by_class"]
    raise ValueError(f"Fold {fold_number} not found")


def merge_mappings(
    mappings: Iterable[dict[str, Iterable[str]]], classes: list[str]
) -> dict[str, list[str]]:
    merged = {class_name: [] for class_name in classes}
    for mapping in mappings:
        for class_name in classes:
            merged[class_name].extend(str(value) for value in mapping[class_name])
    return merged


def cross_validation_split(
    manifest: dict, validation_fold: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    classes = classes_from_manifest(manifest)
    validation = fold_mapping(manifest, validation_fold)
    training = merge_mappings(
        [
            fold_mapping(manifest, fold)
            for fold in range(1, 6)
            if fold != validation_fold
        ],
        classes,
    )
    return training, validation


def development_mapping(manifest: dict) -> dict[str, list[str]]:
    classes = classes_from_manifest(manifest)
    return merge_mappings(
        [fold_mapping(manifest, fold) for fold in range(1, 6)],
        classes,
    )


def heldout_mapping(manifest: dict) -> dict[str, list[str]]:
    return manifest["heldout_test"]["patients_by_class"]


def validate_no_overlap(*row_groups: list[dict]) -> None:
    seen: set[str] = set()
    for rows in row_groups:
        identifiers = {row["patient_id"] for row in rows}
        overlap = seen.intersection(identifiers)
        if overlap:
            raise RuntimeError(f"Patient overlap across partitions: {sorted(overlap)[:5]}")
        seen.update(identifiers)
