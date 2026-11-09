#!/usr/bin/env python
"""Audit split isolation using the repository's authoritative patient IDs.

The identity key is the exact ``patient_id`` stored in each split manifest and
used in the corresponding NIfTI filename. DICOM header tags are intentionally
outside this audit's scope. No patient identifiers are serialized in the
summary report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

try:
    from .data import (
        collect_required_rows,
        cross_validation_split,
        development_mapping,
        heldout_mapping,
        load_manifest,
        split_manifest_path,
    )
except ImportError:  # pragma: no cover - direct script execution
    from data import (
        collect_required_rows,
        cross_validation_split,
        development_mapping,
        heldout_mapping,
        load_manifest,
        split_manifest_path,
    )


DEFAULT_PERCENTAGES = (20, 40, 60, 80, 100)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_ids(mapping: Mapping) -> set[str]:
    return {
        str(patient_id)
        for patients in mapping.values()
        for patient_id in patients
    }


def run_audit(dataset_root: Path, percentages: tuple[int, ...]) -> dict:
    """Validate manifest/NIfTI patient IDs and return a non-identifying summary."""

    dataset_root = Path(dataset_root).resolve()
    summaries: dict[str, dict] = {}
    reference_heldout: set[str] | None = None
    heldout_consistent = True

    for percentage in percentages:
        manifest_path = split_manifest_path(dataset_root, percentage).resolve()
        manifest = load_manifest(dataset_root, percentage)
        development = _patient_ids(development_mapping(manifest))
        heldout = _patient_ids(heldout_mapping(manifest))
        overlap = development.intersection(heldout)
        if overlap:  # ``load_manifest`` already enforces this; keep the report explicit.
            raise ValueError(
                f"Patient-ID overlap in {percentage}% split: {sorted(overlap)[:5]}"
            )

        cv_overlap_counts: dict[str, int] = {}
        for fold in range(1, 6):
            training_mapping, validation_mapping = cross_validation_split(
                manifest, fold
            )
            training_ids = _patient_ids(training_mapping)
            validation_ids = _patient_ids(validation_mapping)
            fold_overlap = training_ids.intersection(validation_ids)
            if fold_overlap:
                raise ValueError(
                    f"Patient-ID overlap in {percentage}% fold {fold}: "
                    f"{sorted(fold_overlap)[:5]}"
                )
            cv_overlap_counts[str(fold)] = 0

        # Also prove that every selected manifest ID resolves to exactly one
        # full-volume NIfTI input under the same patient ID.
        selected_rows = collect_required_rows(dataset_root, manifest)
        selected_ids = {str(row["patient_id"]) for row in selected_rows}
        if selected_ids != development.union(heldout):
            raise RuntimeError(
                f"Patient-ID/NIfTI inventory mismatch in {percentage}% split"
            )

        if reference_heldout is None:
            reference_heldout = heldout
        else:
            heldout_consistent = heldout_consistent and heldout == reference_heldout

        summaries[str(percentage)] = {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "development_patient_count": len(development),
            "heldout_patient_count": len(heldout),
            "development_heldout_overlap_count": 0,
            "cv_train_validation_overlap_count_by_fold": cv_overlap_counts,
            "duplicate_patient_id_count": 0,
            "selected_nifti_patient_count": len(selected_ids),
            "all_selected_nifti_sources_present": True,
        }

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "status": "passed",
        "identity_key": "dataset_patient_id",
        "identity_source": "split_manifest_and_nifti_filename",
        "dicom_header_identity_used": False,
        "raw_patient_ids_serialized": False,
        "percentages": list(percentages),
        "heldout_membership_consistent_across_percentages": heldout_consistent,
        "splits": summaries,
    }


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    systematic_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=systematic_root / "Dataset"
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        choices=DEFAULT_PERCENTAGES,
        default=list(DEFAULT_PERCENTAGES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional non-identifying JSON summary path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    percentages = tuple(dict.fromkeys(int(value) for value in args.percentages))
    try:
        report = run_audit(args.dataset_root, percentages)
        if args.output is not None:
            _atomic_write_json(args.output.resolve(), report)
    except Exception as exc:
        print(
            f"Patient-ID split audit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    print(
        "Pillar-0 patient-ID split audit: "
        f"status={report['status']} | "
        f"percentages={','.join(map(str, percentages))} | "
        "development_heldout_overlap_count=0 | "
        "dicom_header_identity_used=false"
    )
    if args.output is not None:
        print(f"Summary: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
