"""Focused tests for Pillar-0 split and source handling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import data  # noqa: E402


def make_manifest(*, cv_key: str = "cross_validation") -> dict:
    folds = []
    for fold_number in range(1, 6):
        mapping = {
            class_name: [f"{class_index}{fold_number:03d}"]
            for class_index, class_name in enumerate(data.CLASS_NAMES, start=1)
        }
        # Deliberately reverse JSON insertion order; the loader must not inherit it.
        folds.append(
            {
                "fold": fold_number,
                "patients_by_class": dict(reversed(list(mapping.items()))),
            }
        )
    heldout = {
        class_name: [f"{class_index}900"]
        for class_index, class_name in enumerate(data.CLASS_NAMES, start=1)
    }
    return {
        "heldout_test": {
            "patients_by_class": dict(reversed(list(heldout.items()))),
            "counts_by_class": {name: 1 for name in data.CLASS_NAMES},
            "total": len(data.CLASS_NAMES),
        },
        cv_key: {"number_of_folds": 5, "folds": folds},
        "metadata": {"percentage": 100},
    }


def populate_nifti(dataset_root: Path, manifest: dict) -> None:
    mapping = data.all_patient_mapping(manifest)
    for class_name in data.CLASS_NAMES:
        class_dir = dataset_root / "NIFTI" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for patient_id in mapping[class_name]:
            (class_dir / f"{patient_id}.nii.gz").write_bytes(b"nifti-test")


class DataTests(unittest.TestCase):
    def test_loads_both_existing_cv_layout_names_and_forces_class_order(self):
        for key in ("cross_validation", "nested_cv"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                split_dir = root / "Splits"
                split_dir.mkdir()
                (split_dir / "100_percent.json").write_text(
                    json.dumps(make_manifest(cv_key=key)), encoding="utf-8"
                )
                manifest = data.load_manifest(root, 100)
                self.assertEqual(data.classes_from_manifest(manifest), list(data.CLASS_NAMES))
                training, validation = data.cross_validation_split(manifest, 3)
                self.assertEqual(sum(map(len, validation.values())), 5)
                self.assertEqual(sum(map(len, training.values())), 20)

    def test_rows_use_only_full_nifti_and_preserve_leading_zero_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest()
            manifest["cross_validation"]["folds"][0]["patients_by_class"][
                "Normal"
            ] = ["0001"]
            populate_nifti(root, manifest)
            rows = data.collect_required_rows(root, manifest)
            row = next(item for item in rows if item["patient_id"] == "0001")
            self.assertEqual(row["label_str"], "Normal")
            self.assertEqual(row["label"], 2)
            self.assertTrue(row["source_path"].endswith("0001.nii.gz"))
            self.assertEqual(row["accession"], "Normal__0001")
            self.assertTrue(all(data.is_nifti_path(item["image"]) for item in rows))

    def test_png_and_2p5d_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            png = Path(temporary) / "slice.png"
            png.write_bytes(b"png")
            with self.assertRaisesRegex(ValueError, "full 3-D NIfTI"):
                data.require_nifti_path(png)
            with self.assertRaisesRegex(ValueError, "full 3-D NIfTI"):
                data.require_nifti_path(Path(temporary) / "2.5d")

    def test_manifest_rejects_heldout_leakage(self):
        manifest = make_manifest()
        leaked = manifest["cross_validation"]["folds"][0]["patients_by_class"][
            "Appendicitis"
        ][0]
        manifest["heldout_test"]["patients_by_class"]["Appendicitis"] = [leaked]
        with self.assertRaisesRegex(ValueError, "leakage"):
            data.validate_manifest(manifest)

    def test_manifest_patient_ids_are_global_across_classes(self):
        manifest = make_manifest()
        leaked = manifest["cross_validation"]["folds"][0]["patients_by_class"][
            "Appendicitis"
        ][0]
        manifest["heldout_test"]["patients_by_class"]["Normal"] = [leaked]
        with self.assertRaisesRegex(ValueError, "leakage"):
            data.validate_manifest(manifest)

    def test_runtime_overlap_check_uses_patient_id_only(self):
        training = [{"patient_id": "0001", "label_str": "Appendicitis"}]
        validation = [{"patient_id": "0001", "label_str": "Normal"}]
        with self.assertRaisesRegex(RuntimeError, "Patient overlap"):
            data.validate_no_overlap(training, validation)

    def test_missing_nifti_fails_before_cache_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                data.rows_from_mapping(
                    Path(temporary),
                    {name: (["001"] if name == "Normal" else []) for name in data.CLASS_NAMES},
                )


if __name__ == "__main__":
    unittest.main()
