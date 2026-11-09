#!/usr/bin/env python
"""Generate deterministic held-out-test and five-fold CV manifests.

For iteration i, fold i is validation and the other four folds are training.
The fixed held-out test set remains separate from all cross-validation folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SEED = 42
N_FOLDS = 5

# These names match the actual Dataset directories.
CLASS_TOTALS = {
    "Appendicitis": 237,
    "Cholezystitis": 185,
    "Normal": 450,
    "Perforation": 58,
    "Sigmoid": 428,
}

HELDOUT_TEST_COUNTS = {
    "Appendicitis": 47,
    "Cholezystitis": 37,
    "Normal": 90,
    "Perforation": 18,
    "Sigmoid": 86,
}

DEVELOPMENT_COUNTS = {
    20: {
        "Appendicitis": 38,
        "Cholezystitis": 30,
        "Normal": 72,
        "Perforation": 40,
        "Sigmoid": 68,
    },
    40: {
        "Appendicitis": 76,
        "Cholezystitis": 59,
        "Normal": 144,
        "Perforation": 40,
        "Sigmoid": 137,
    },
    60: {
        "Appendicitis": 114,
        "Cholezystitis": 89,
        "Normal": 216,
        "Perforation": 40,
        "Sigmoid": 205,
    },
    80: {
        "Appendicitis": 152,
        "Cholezystitis": 118,
        "Normal": 288,
        "Perforation": 40,
        "Sigmoid": 274,
    },
    100: {
        "Appendicitis": 190,
        "Cholezystitis": 148,
        "Normal": 360,
        "Perforation": 40,
        "Sigmoid": 342,
    },
}


def stable_class_seed(seed: int, class_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def read_patient_ids(dataset_root: Path) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for format_name in ("DICOM", "NIFTI", "PNG"):
        format_root = dataset_root / format_name
        if not format_root.is_dir():
            raise FileNotFoundError(f"Missing dataset format folder: {format_root}")
        result[format_name] = {}
        for class_name in CLASS_TOTALS:
            class_root = format_root / class_name
            if format_name == "NIFTI":
                ids = {
                    path.name[:-7]
                    for path in class_root.iterdir()
                    if path.is_file() and path.name.endswith(".nii.gz")
                }
            else:
                ids = {path.name for path in class_root.iterdir() if path.is_dir()}
            result[format_name][class_name] = ids
    return result


def validate_formats(format_ids: dict[str, dict[str, set[str]]]) -> dict[str, list[str]]:
    canonical: dict[str, list[str]] = {}
    for class_name, expected_total in CLASS_TOTALS.items():
        dicom = format_ids["DICOM"][class_name]
        for format_name in ("NIFTI", "PNG"):
            if format_ids[format_name][class_name] != dicom:
                missing = sorted(dicom - format_ids[format_name][class_name])
                extra = sorted(format_ids[format_name][class_name] - dicom)
                raise RuntimeError(
                    f"{format_name}/{class_name} differs from DICOM: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
        if len(dicom) != expected_total:
            raise RuntimeError(
                f"{class_name}: expected {expected_total}, found {len(dicom)}"
            )
        canonical[class_name] = sorted(dicom)
    return canonical


def assign_patients(
    patients_by_class: dict[str, list[str]], seed: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    heldout: dict[str, list[str]] = {}
    remaining: dict[str, list[str]] = {}
    for class_name, patient_ids in patients_by_class.items():
        shuffled = patient_ids.copy()
        random.Random(stable_class_seed(seed, class_name)).shuffle(shuffled)
        test_count = HELDOUT_TEST_COUNTS[class_name]
        heldout[class_name] = sorted(shuffled[:test_count])
        # Keeping this order makes every smaller percentage a strict subset of
        # each larger percentage.
        remaining[class_name] = shuffled[test_count:]
    return heldout, remaining


def create_manifest(
    percentage: int,
    heldout: dict[str, list[str]],
    remaining: dict[str, list[str]],
    seed: int,
) -> dict:
    selected = {
        class_name: remaining[class_name][:count]
        for class_name, count in DEVELOPMENT_COUNTS[percentage].items()
    }
    folds = [
        {
            "fold": fold_index + 1,
            "patients_by_class": {
                class_name: sorted(patient_ids[fold_index::N_FOLDS])
                for class_name, patient_ids in selected.items()
            },
        }
        for fold_index in range(N_FOLDS)
    ]

    # Key order is intentional: the fixed held-out test assignment comes first.
    return {
        "heldout_test": {
            "patients_by_class": heldout,
            "counts_by_class": {
                class_name: len(patient_ids)
                for class_name, patient_ids in heldout.items()
            },
            "total": sum(len(patient_ids) for patient_ids in heldout.values()),
        },
        "cross_validation": {
            "number_of_folds": N_FOLDS,
            "folds": folds,
            "development_counts_by_class": DEVELOPMENT_COUNTS[percentage],
            "development_total": sum(DEVELOPMENT_COUNTS[percentage].values()),
            "usage": {
                "single_loop": (
                    "For iteration i, use fold i as validation and the other "
                    "four folds as training."
                ),
                "final_test": (
                    "Use heldout_test only after model and hyperparameter selection."
                ),
            },
        },
        "metadata": {
            "percentage": percentage,
            "seed": seed,
            "stratified_by": "class",
            "patient_level_split": True,
            "percentage_patient_pools_are_cumulative": True,
            "perforation_development_pool_is_fixed_at_40": True,
            "class_name_note": (
                "Cholezystitis matches the dataset directory spelling; the "
                "plan calls the same class Cholecystitis."
            ),
        },
    }


def validate_manifest(manifest: dict) -> None:
    heldout_by_class = manifest["heldout_test"]["patients_by_class"]
    heldout_ids = set().union(*(set(ids) for ids in heldout_by_class.values()))
    if len(heldout_ids) != manifest["heldout_test"]["total"]:
        raise RuntimeError("Duplicate patient ID in held-out test assignment.")

    development_ids: set[str] = set()
    observed_counts = {class_name: 0 for class_name in CLASS_TOTALS}
    for fold in manifest["cross_validation"]["folds"]:
        fold_ids: set[str] = set()
        for class_name, patient_ids in fold["patients_by_class"].items():
            if len(patient_ids) != len(set(patient_ids)):
                raise RuntimeError(f"Duplicate in fold {fold['fold']}/{class_name}.")
            observed_counts[class_name] += len(patient_ids)
            fold_ids.update(patient_ids)
        if development_ids.intersection(fold_ids):
            raise RuntimeError(f"Patient overlap involving fold {fold['fold']}.")
        development_ids.update(fold_ids)

    if heldout_ids.intersection(development_ids):
        raise RuntimeError("Held-out test and development assignments overlap.")
    expected = manifest["cross_validation"]["development_counts_by_class"]
    if observed_counts != expected:
        raise RuntimeError(f"Development counts differ: {observed_counts} != {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_root = Path(__file__).resolve().parents[2] / "Dataset"
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_root / "Splits",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patients = validate_formats(read_patient_ids(args.dataset_root.resolve()))
    heldout, remaining = assign_patients(patients, args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for percentage in DEVELOPMENT_COUNTS:
        manifest = create_manifest(percentage, heldout, remaining, args.seed)
        validate_manifest(manifest)
        output_path = output_dir / f"{percentage}_percent.json"
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {output_path.name}: heldout={manifest['heldout_test']['total']}, "
            f"development={manifest['cross_validation']['development_total']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
