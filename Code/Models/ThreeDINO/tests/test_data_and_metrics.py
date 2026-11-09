"""Synthetic checks for 3DINO split safety, cache identity, and metrics."""

from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import CLASS_NAMES, INPUT_SIZE, PREFIX_SIZE  # noqa: E402
from Code.Models.ThreeDINO import data, metrics  # noqa: E402


def make_manifest() -> dict:
    folds = []
    for fold_number in range(1, 6):
        mapping = {
            class_name: [f"{class_index}{fold_number:03d}"]
            for class_index, class_name in enumerate(CLASS_NAMES, start=1)
        }
        folds.append(
            {
                "fold": fold_number,
                "patients_by_class": dict(reversed(list(mapping.items()))),
            }
        )
    heldout = {
        class_name: [f"{class_index}900"]
        for class_index, class_name in enumerate(CLASS_NAMES, start=1)
    }
    return {
        "heldout_test": {
            "patients_by_class": dict(reversed(list(heldout.items()))),
            "counts_by_class": {name: 1 for name in CLASS_NAMES},
            "total": len(CLASS_NAMES),
        },
        "cross_validation": {
            "number_of_folds": 5,
            "folds": folds,
            "development_counts_by_class": {name: 5 for name in CLASS_NAMES},
            "development_total": 5 * len(CLASS_NAMES),
        },
        "metadata": {"percentage": 100},
    }


def populate_fake_nifti(dataset_root: Path, manifest: dict) -> None:
    for class_name, patient_ids in data.all_patient_mapping(manifest).items():
        class_dir = dataset_root / "NIFTI" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for patient_id in patient_ids:
            (class_dir / f"{patient_id}.nii.gz").write_bytes(b"synthetic-nifti")


class RecordingPostTransform:
    """Small MONAI-like transform for draw-protocol tests."""

    def __init__(self):
        self.seed = None
        self.state = None
        self.events: list[tuple[str, int | None]] = []

    def set_random_state(self, seed=None, state=None):
        self.seed = seed
        self.state = state
        self.events.append(("seed", seed))
        return self

    def __call__(self, value):
        self.events.append(("call", self.seed))
        value["image"] = np.asarray(value["image"]) + int(self.seed or 0)
        value["post_transform_applied"] = True
        return value


class DataContractTests(unittest.TestCase):
    def test_manifest_keeps_canonical_order_and_isolates_all_partitions(self):
        manifest = make_manifest()
        data.validate_manifest(manifest)
        self.assertEqual(data.classes_from_manifest(manifest), list(CLASS_NAMES))
        training, validation = data.cross_validation_split(manifest, 3)
        self.assertEqual(sum(map(len, training.values())), 20)
        self.assertEqual(sum(map(len, validation.values())), 5)
        self.assertFalse(
            set().union(*map(set, training.values())).intersection(
                set().union(*map(set, validation.values()))
            )
        )

    def test_manifest_rejects_heldout_leakage(self):
        manifest = make_manifest()
        leaked = manifest["cross_validation"]["folds"][0]["patients_by_class"][
            "Appendicitis"
        ][0]
        manifest["heldout_test"]["patients_by_class"]["Appendicitis"] = [leaked]
        with self.assertRaisesRegex(ValueError, "leakage|duplication"):
            data.validate_manifest(manifest)

    def test_manifest_requires_exact_fold_identifiers(self):
        manifest = make_manifest()
        manifest["cross_validation"]["folds"][-1]["fold"] = 6
        with self.assertRaisesRegex(ValueError, "exact CV fold identifiers"):
            data.validate_manifest(manifest)

        missing_identifier = copy.deepcopy(make_manifest())
        del missing_identifier["cross_validation"]["folds"][0]["fold"]
        with self.assertRaisesRegex(ValueError, "has no fold identifier"):
            data.validate_manifest(missing_identifier)

    def test_manifest_validates_recorded_development_counts_and_total(self):
        count_mismatch = copy.deepcopy(make_manifest())
        count_mismatch["cross_validation"]["development_counts_by_class"][
            "Appendicitis"
        ] = 4
        with self.assertRaisesRegex(ValueError, "Development class-count mismatch"):
            data.validate_manifest(count_mismatch)

        total_mismatch = copy.deepcopy(make_manifest())
        total_mismatch["cross_validation"]["development_total"] = 24
        with self.assertRaisesRegex(ValueError, "Development total mismatch"):
            data.validate_manifest(total_mismatch)

    def test_rows_are_nifti_only_and_cache_key_is_source_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest()
            populate_fake_nifti(root, manifest)
            rows = data.collect_required_rows(root, manifest)
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(data.is_nifti_path(row["image"]) for row in rows))
            first_key = data.persistent_cache_key(rows[0])
            second_key = data.persistent_cache_key(rows[1])
            self.assertEqual(len(first_key), 64)
            self.assertNotEqual(first_key, second_key)
            with self.assertRaisesRegex(ValueError, "NIfTI"):
                data.require_nifti_path(root / "slice.png", must_exist=False)

    def test_manifest_file_loading_does_not_regenerate_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "Splits"
            split_dir.mkdir()
            path = split_dir / "100_percent.json"
            path.write_text(json.dumps(make_manifest()), encoding="utf-8")
            loaded = data.load_manifest(root, 100)
            self.assertEqual(loaded["metadata"]["percentage"], 100)
            self.assertEqual(path.read_text(encoding="utf-8"), json.dumps(make_manifest()))

    def test_preprocessing_contract_keeps_random_work_after_cache(self):
        contract = data.preprocessing_contract()
        self.assertEqual(contract["deterministic_resize_xyz"], list(PREFIX_SIZE))
        self.assertEqual(contract["model_crop_xyz"], list(INPUT_SIZE))
        self.assertEqual(contract["percentiles"], [0.05, 99.95])
        self.assertEqual(contract["intensity_range"], [-1.0, 1.0])
        self.assertEqual(contract["training"]["crop"], "random_spatial_crop")
        self.assertEqual(contract["validation"]["crop"], "center_spatial_crop")

    def test_post_cache_draw_request_reseeds_and_attaches_fresh_metadata(self):
        base = [
            {
                "image": np.asarray([1.0]),
                "patient_id": "patient-1",
                data.DRAW_INDEX_KEY: "stale",
                data.AUGMENTATION_SEED_KEY: "stale",
            }
        ]
        transform = RecordingPostTransform()
        dataset = data.PostCacheTransformDataset(base, transform)
        request = data.make_draw_request(0, 1234, 9876)

        first = dataset[request]
        second = dataset[request]

        self.assertEqual(request, (0, 1234, 9876))
        self.assertEqual(transform.events, [
            ("seed", 9876),
            ("call", 9876),
            ("seed", 9876),
            ("call", 9876),
        ])
        self.assertEqual(first[data.DRAW_INDEX_KEY], 1234)
        self.assertEqual(first[data.AUGMENTATION_SEED_KEY], 9876)
        np.testing.assert_array_equal(first["image"], second["image"])
        np.testing.assert_array_equal(base[0]["image"], np.asarray([1.0]))
        self.assertNotIn("post_transform_applied", base[0])
        self.assertIsNot(first, base[0])

    def test_post_cache_integer_index_remains_unannotated_and_seed_delegates(self):
        base = [
            {
                "image": np.asarray([2.0]),
                data.DRAW_INDEX_KEY: "stale",
                data.AUGMENTATION_SEED_KEY: "stale",
            }
        ]
        transform = RecordingPostTransform()
        dataset = data.PostCacheTransformDataset(base, transform)

        self.assertIs(dataset.set_random_state(seed=7), dataset)
        result = dataset[0]

        self.assertEqual(transform.events, [("seed", 7), ("call", 7)])
        self.assertNotIn(data.DRAW_INDEX_KEY, result)
        self.assertNotIn(data.AUGMENTATION_SEED_KEY, result)
        np.testing.assert_array_equal(result["image"], np.asarray([9.0]))

    def test_draw_request_validation_fails_closed(self):
        dataset = data.PostCacheTransformDataset(
            [{"image": np.asarray([1.0])}], RecordingPostTransform()
        )
        with self.assertRaisesRegex(ValueError, "sample_index"):
            data.make_draw_request(-1, 0, 0)
        with self.assertRaisesRegex(ValueError, "draw_index"):
            data.make_draw_request(0, -1, 0)
        with self.assertRaisesRegex(ValueError, "uint32"):
            data.make_draw_request(0, 0, 1 << 32)
        with self.assertRaisesRegex(TypeError, "not bool"):
            data.make_draw_request(True, 0, 0)
        with self.assertRaisesRegex(TypeError, "must be"):
            _ = dataset[(0, 1)]
        with self.assertRaisesRegex(IndexError, "outside dataset size"):
            _ = dataset[data.make_draw_request(1, 0, 0)]


class MetricContractTests(unittest.TestCase):
    def test_five_class_metrics_and_prediction_exports_are_patient_level(self):
        truth = np.arange(5, dtype=np.int64)
        probability = np.full((5, 5), 0.025, dtype=np.float64)
        probability[np.arange(5), truth] = 0.9
        predicted = probability.argmax(axis=1)
        result = metrics.compute_metrics(
            truth, predicted, probability, list(CLASS_NAMES)
        )
        self.assertEqual(result["class_order"], list(CLASS_NAMES))
        self.assertEqual(result["confusion_matrix"], np.eye(5, dtype=int).tolist())
        self.assertAlmostEqual(result["accuracy"], 1.0)
        rows, frame = metrics.format_predictions(
            [f"patient-{index}" for index in range(5)],
            list(CLASS_NAMES),
            probability,
            list(CLASS_NAMES),
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(frame.columns[:5]), list(CLASS_NAMES))
        self.assertEqual(rows[3]["patient_id"], "patient-3")

    def test_metrics_fail_closed_on_non_softmax_or_wrong_shape(self):
        truth = np.arange(5, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            metrics.compute_metrics(
                truth,
                truth,
                np.full((5, 5), 0.1),
                list(CLASS_NAMES),
            )
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            metrics.compute_metrics(
                truth[:2],
                truth[:2],
                np.full((2, 4), 0.25),
                list(CLASS_NAMES[:4]),
            )


if __name__ == "__main__":
    unittest.main()
