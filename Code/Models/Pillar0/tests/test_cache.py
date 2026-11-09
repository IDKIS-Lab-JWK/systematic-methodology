"""Focused tests for the separate Pillar-0 RAVE cache."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import cache  # noqa: E402
import data  # noqa: E402


def make_manifest() -> dict:
    folds = []
    for fold_number in range(1, 6):
        folds.append(
            {
                "fold": fold_number,
                "patients_by_class": {
                    class_name: [f"{class_index}{fold_number:03d}"]
                    for class_index, class_name in enumerate(data.CLASS_NAMES, start=1)
                },
            }
        )
    heldout = {
        class_name: [f"{class_index}900"]
        for class_index, class_name in enumerate(data.CLASS_NAMES, start=1)
    }
    return {
        "heldout_test": {
            "patients_by_class": heldout,
            "counts_by_class": {name: 1 for name in data.CLASS_NAMES},
            "total": 5,
        },
        "cross_validation": {"number_of_folds": 5, "folds": folds},
        "metadata": {"percentage": 100},
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "Dataset"
    split_dir = dataset / "Splits"
    split_dir.mkdir(parents=True)
    manifest = make_manifest()
    split_path = split_dir / "100_percent.json"
    split_path.write_text(json.dumps(manifest), encoding="utf-8")
    for class_name, patients in data.all_patient_mapping(manifest).items():
        class_dir = dataset / "NIFTI" / class_name
        class_dir.mkdir(parents=True)
        for patient_id in patients:
            (class_dir / f"{patient_id}.nii.gz").write_bytes(b"raw-nifti")
    config_path = root / "rave.yaml"
    config_path.write_text(
        "\n".join(
            [
                "modality: CT",
                "anatomy: abdomen",
                "processing:",
                "  crop_pad:",
                "    size: [12, 10]",
                "  resampling:",
                "    target_spacing: [1.2, 1.3, 1.4]",
                "  conversion_backend: sitk",
                "  slice_selection:",
                "    enabled: true",
                "    slices: 8",
                "exporter_config: lz4_fast",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset, config_path


def make_plan(root: Path) -> cache.CachePlan:
    dataset, config_path = write_fixture(root)
    return cache.build_cache_plan(
        dataset_root=dataset,
        split_manifest_path=dataset / "Splits" / "100_percent.json",
        cache_root=root / "Results" / "Pillar0" / "Cache",
        namespace="test_rave_lz4_v1",
        config_path=config_path,
    )


class CacheTests(unittest.TestCase):
    def test_model_loader_pads_then_flips_depth(self):
        raw = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
        with mock.patch.object(cache, "load_cached_volume", return_value=raw):
            loaded = cache.load_pillar_volume(
                "unused.tar.lz4",
                target_shape_dhw=(4, 4, 4),
                pad_value=-1.0,
                flip_depth=True,
            )
        self.assertEqual(tuple(loaded.shape), (1, 4, 4, 4))
        self.assertTrue(torch.equal(loaded[:, 1], raw[:, 1]))
        self.assertTrue(torch.equal(loaded[:, 2], raw[:, 0]))
        self.assertTrue(torch.all(loaded[:, 0] == -1.0))
        self.assertTrue(torch.all(loaded[:, 3] == -1.0))

    def test_smoke_plan_is_stratified_and_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            smoke = cache.stratified_smoke_plan(
                plan,
                patients_per_class=1,
                namespace="pillar0_smoke_1pc_v1",
            )
            self.assertEqual(len(smoke.sources), len(data.CLASS_NAMES))
            self.assertEqual(
                {row["label_str"] for row in smoke.sources}, set(data.CLASS_NAMES)
            )
            self.assertNotEqual(smoke.cache_directory, plan.cache_directory)
            self.assertIn("not_for_experiment_results", smoke.provenance["scope"])

    def test_contract_is_yaml_driven_and_lz4_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = write_fixture(root)
            contract = cache.cache_contract_from_config(
                cache.load_cache_config(config_path)
            )
            self.assertEqual(contract.output_shape_dhw, (8, 12, 10))
            self.assertEqual(contract.target_spacing_xyz_mm, (1.2, 1.3, 1.4))
            self.assertEqual(contract.compression, "lz4")

            config = cache.load_cache_config(config_path)
            config["exporter_config"] = "video_hevc"
            with self.assertRaises(cache.CacheConfigurationError):
                cache.cache_contract_from_config(config)

    def test_dry_run_is_strictly_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            result = cache.prepare_cache(plan, dry_run=True)
            self.assertEqual(result["status"], "dry_run")
            self.assertFalse(result["filesystem_mutated"])
            self.assertFalse(plan.cache_directory.exists())
            self.assertIn("vision_engine", result["command"])

    def test_prepare_reconciles_mapping_and_writes_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))

            def fake_runner(command, **kwargs):
                output_dir = Path(command[command.index("--output") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                mapping_path = output_dir / "mapping.csv"
                with mapping_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["source_path", "output_path"]
                    )
                    writer.writeheader()
                    for source in plan.sources:
                        output = output_dir / f"{source['accession']}.1.0.tar.lz4"
                        output.write_bytes(b"fake-rave-cache")
                        writer.writerow(
                            {
                                "source_path": source["source_path"],
                                "output_path": str(output),
                            }
                        )
                return subprocess.CompletedProcess(command, 0, "ok", "")

            manifest = cache.prepare_cache(
                plan,
                validate_volumes=False,
                runner=fake_runner,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["entry_count"], len(plan.sources))
            self.assertTrue(plan.manifest_path.is_file())
            expected_fields = {
                "cache_path",
                "patient_id",
                "label",
                "label_str",
                "source_path",
                "accession",
                "series_number",
                "source_fingerprint",
                "shape",
                "dtype",
            }
            self.assertEqual(set(manifest["entries"][0]), expected_fields)
            self.assertNotIn("Merlin", str(plan.cache_directory))

            requested = [
                {
                    "patient_id": plan.sources[3]["patient_id"],
                    "label": plan.sources[3]["label"],
                    "label_str": plan.sources[3]["label_str"],
                },
                {
                    "patient_id": plan.sources[0]["patient_id"],
                    "label": plan.sources[0]["label"],
                    "label_str": plan.sources[0]["label_str"],
                },
            ]
            selected = cache.load_cached_rows(plan.cache_directory, requested)
            self.assertEqual(
                [row["patient_id"] for row in selected],
                [row["patient_id"] for row in requested],
            )

    def test_cached_row_selection_rejects_missing_or_mismatched_patient(self):
        with tempfile.TemporaryDirectory() as temporary:
            cached_path = Path(temporary) / "001.1.0.tar.lz4"
            cached_path.write_bytes(b"cache")
            manifest = {
                "status": "complete",
                "entries": [
                    {
                        "cache_path": str(cached_path),
                        "patient_id": "001",
                        "label": 2,
                        "label_str": "Normal",
                    }
                ],
            }
            with self.assertRaisesRegex(cache.CacheValidationError, "absent"):
                cache.load_cached_rows(
                    manifest,
                    [{"patient_id": "002", "label": 2, "label_str": "Normal"}],
                )
            with self.assertRaisesRegex(cache.CacheValidationError, "label mismatch"):
                cache.load_cached_rows(
                    manifest,
                    [
                        {
                            "patient_id": "001",
                            "label": 0,
                            "label_str": "Appendicitis",
                        }
                    ],
                )

    def test_namespace_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                cache.cache_directory(Path(temporary), "../Merlin")
if __name__ == "__main__":
    unittest.main()
