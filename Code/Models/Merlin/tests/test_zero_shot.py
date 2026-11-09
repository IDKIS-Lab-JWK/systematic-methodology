from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


MERLIN_DIR = Path(__file__).resolve().parents[1]
SYSTEMATIC_ROOT = MERLIN_DIR.parents[2]
sys.path.insert(0, str(MERLIN_DIR))
sys.path.insert(0, str(SYSTEMATIC_ROOT / "Notebooks"))

import zero_shot as zero_shot_module  # noqa: E402
from _shared.result_analysis import discover_test_results  # noqa: E402
from data import classes_from_manifest, load_manifest  # noqa: E402
from run_zero_shot import build_parser  # noqa: E402
from zero_shot import (  # noqa: E402
    _author_binary_documents,
    _encode_heldout_images,
    _encode_text_prompts,
    _five_class_documents,
    _mode_output_directory,
    as_embedding_matrix,
    flatten_prompts,
    load_prompt_profile,
    mean_prompt_scores,
    normalize_embeddings,
    uncalibrated_softmax,
    validate_fixed_heldout_manifests,
    validate_merlin_package_version,
)


class FakeTextEncoder(nn.Module):
    def forward(self, prompts: list[str]) -> torch.Tensor:
        output = torch.zeros(len(prompts), 512)
        for index in range(len(prompts)):
            output[index, index] = 1.0
        return output


class FakeImageEncoder(nn.Module):
    def forward(self, images: torch.Tensor):
        ehr = torch.zeros(images.shape[0], 1692, device=images.device)
        return images, ehr


class FakeArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_text = FakeTextEncoder()
        self.encode_image = FakeImageEncoder()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


class FakeMerlin(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeArchitecture()


class ZeroShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = SYSTEMATIC_ROOT / "Dataset"
        cls.manifest = load_manifest(cls.dataset_root, 100)
        cls.classes = classes_from_manifest(cls.manifest)
        cls.prompt_path = MERLIN_DIR / "zero_shot_prompts.json"

    def test_prompt_profile_matches_manifest_and_is_hashed(self):
        profile, digest = load_prompt_profile(self.prompt_path, self.classes)

        self.assertEqual(profile["class_order"], self.classes)
        self.assertEqual(len(digest), 64)
        self.assertFalse(
            profile["prompt_provenance"]["publication_exact_prompts"]
        )
        self.assertEqual(
            profile["classes"]["Cholezystitis"]["canonical_concept"],
            "acute cholecystitis",
        )

    def test_all_percentage_manifests_share_one_heldout_set(self):
        checked = validate_fixed_heldout_manifests(
            self.dataset_root, self.manifest
        )

        self.assertEqual(checked, [20, 40, 60, 80, 100])

    def test_prompt_class_order_mismatch_is_rejected(self):
        profile = json.loads(self.prompt_path.read_text(encoding="utf-8"))
        profile["class_order"] = list(reversed(profile["class_order"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(profile), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_prompt_profile(path, self.classes)

    def test_known_merlin_singleton_embedding_layouts_are_supported(self):
        layouts = (
            torch.ones(2, 512),
            torch.ones(1, 2, 512),
            torch.ones(2, 1, 512),
        )

        for value in layouts:
            matrix = as_embedding_matrix(value, 2, "test embeddings")
            self.assertEqual(tuple(matrix.shape), (2, 512))

        single = as_embedding_matrix(
            torch.ones(512), 1, "single embedding"
        )
        self.assertEqual(tuple(single.shape), (1, 512))

    def test_prompt_scores_average_individual_cosines(self):
        profile, _ = load_prompt_profile(self.prompt_path, self.classes)
        prompts, layout = flatten_prompts(profile)
        similarities = torch.arange(
            2 * len(prompts), dtype=torch.float32
        ).reshape(2, len(prompts))

        scores = mean_prompt_scores(
            similarities, layout, self.classes, "positive"
        )

        self.assertEqual(tuple(scores.shape), (2, len(self.classes)))
        first_indices = layout[self.classes[0]]["positive"]
        self.assertAlmostEqual(
            float(scores[0, 0]),
            float(similarities[0, first_indices].mean()),
        )

    def test_normalization_and_softmax_preserve_argmax(self):
        embeddings = normalize_embeddings(
            torch.tensor([[3.0] + [4.0] + [0.0] * 510]),
            "test embeddings",
        )
        self.assertAlmostEqual(float(embeddings.norm(dim=1)[0]), 1.0)

        scores = np.asarray([[0.10, 0.50, -0.20], [0.40, 0.20, 0.10]])
        probabilities = uncalibrated_softmax(scores)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            probabilities.argmax(axis=1), scores.argmax(axis=1)
        )

    def test_fake_split_device_encoder_path(self):
        model = FakeMerlin().eval().requires_grad_(False)
        text = _encode_text_prompts(model, ["first", "second"])
        images = torch.zeros(2, 512)
        images[0, 0] = 1.0
        images[1, 1] = 1.0
        loader = [
            {
                "image": images,
                "label": torch.tensor([0, 1]),
                "patient_id": ["p0", "p1"],
                "label_str": ["Appendicitis", "Cholezystitis"],
            }
        ]

        similarities, labels, patient_ids, label_strings = (
            _encode_heldout_images(
                model,
                loader,
                2,
                text,
                torch.device("cpu"),
                False,
            )
        )

        self.assertEqual(tuple(similarities.shape), (2, 2))
        np.testing.assert_array_equal(labels, [0, 1])
        self.assertEqual(patient_ids, ["p0", "p1"])
        self.assertEqual(
            label_strings, ["Appendicitis", "Cholezystitis"]
        )
        np.testing.assert_allclose(similarities.numpy(), np.eye(2))

    def test_five_class_artifact_contract_and_author_proxy_label(self):
        profile, canonical_hash = load_prompt_profile(
            self.prompt_path, self.classes
        )
        prompts, layout = flatten_prompts(profile)
        similarities = torch.arange(
            len(self.classes) * len(prompts), dtype=torch.float32
        ).reshape(len(self.classes), len(prompts))
        similarities /= float(similarities.max())
        labels = np.arange(len(self.classes), dtype=np.int64)
        patient_ids = [f"p{index}" for index in labels]
        label_strings = list(self.classes)

        summary, predictions, probability_frame, score_frame = (
            _five_class_documents(
                similarities,
                layout,
                self.classes,
                labels,
                patient_ids,
                label_strings,
                {},
                profile,
                canonical_hash,
                "file-hash",
            )
        )

        probabilities = probability_frame[self.classes].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(len(predictions), len(self.classes))
        self.assertEqual(score_frame.shape[0], len(self.classes))
        metrics = summary["test_metrics"]
        self.assertNotIn("selection_score", metrics)
        self.assertIn("comparison_score", metrics)
        self.assertIn("raw_cosine_average_precision_macro", metrics)
        truth = (labels == 0).astype(np.int64)
        expected_ap = average_precision_score(truth, probabilities[:, 0])
        self.assertAlmostEqual(
            metrics["per_class"][self.classes[0]]["average_precision"],
            expected_ap,
        )

        binary_summary, binary_predictions, binary_frame = (
            _author_binary_documents(
                similarities,
                layout,
                self.classes,
                labels,
                patient_ids,
                label_strings,
                {},
                profile,
                canonical_hash,
                "file-hash",
            )
        )
        self.assertIn("one_vs_rest_binary_proxy", binary_summary["evaluation_task"])
        self.assertEqual(len(binary_predictions), len(self.classes))
        self.assertEqual(binary_frame.shape[0], len(self.classes))
        self.assertIn(
            "ground_truth_type", binary_summary["test_metrics"]
        )

    def test_zero_shot_result_directory_and_notebook_discovery(self):
        profile, canonical_hash = load_prompt_profile(
            self.prompt_path, self.classes
        )
        prompts, layout = flatten_prompts(profile)
        similarities = torch.eye(len(self.classes), len(prompts))
        labels = np.arange(len(self.classes), dtype=np.int64)
        summary, _, _, _ = _five_class_documents(
            similarities,
            layout,
            self.classes,
            labels,
            [f"p{index}" for index in labels],
            list(self.classes),
            {},
            profile,
            canonical_hash,
            "file-hash",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = _mode_output_directory(
                root / "Results" / "Merlin",
                "five_class_adapted",
                profile["prompt_set_id"],
            )
            result_dir.mkdir(parents=True)
            (result_dir / "metrics_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            overall, classwise, confusion, failures = discover_test_results(
                root, "Merlin"
            )

        self.assertTrue(failures.empty)
        self.assertEqual(len(overall), 1)
        self.assertEqual(overall.iloc[0]["variant"], "zero_shot")
        self.assertEqual(int(overall.iloc[0]["percentage"]), 0)
        self.assertEqual(len(classwise), len(self.classes))
        self.assertEqual(len(confusion), 1)

    def test_cli_dry_run_uses_no_training_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--dry_run",
                    "--dataset_root",
                    str(self.dataset_root),
                    "--results_root",
                    directory,
                ]
            )
            result = zero_shot_module.run_zero_shot(args)

        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["training_percentage"], 0)
        self.assertEqual(result["heldout_patient_count"], 278)

    def test_fake_complete_run_writes_both_isolated_result_schemas(self):
        def fake_loader(rows, cache_dir, batch_size, shuffle, num_workers):
            images = torch.zeros(len(rows), 512)
            images[:, 0] = 1.0
            return [
                {
                    "image": images,
                    "label": torch.tensor([row["label"] for row in rows]),
                    "patient_id": [row["patient_id"] for row in rows],
                    "label_str": [row["label_str"] for row in rows],
                }
            ]

        checkpoint = {
            "filename": "fake_merlin.pt",
            "path": None,
            "size_bytes": 123,
            "sha256": "fake-checkpoint-hash",
            "sha256_status": "computed",
        }
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            args = build_parser().parse_args(
                [
                    "--mode",
                    "both",
                    "--dataset_root",
                    str(self.dataset_root),
                    "--results_root",
                    str(temporary_root / "Results" / "Merlin"),
                    "--cache_root",
                    str(temporary_root / "Cache"),
                    "--device",
                    "cpu",
                ]
            )
            with (
                patch.object(
                    zero_shot_module,
                    "validate_merlin_package_version",
                    return_value="0.0.7",
                ),
                patch.object(
                    zero_shot_module,
                    "import_merlin",
                    return_value=(FakeMerlin, object),
                ),
                patch.object(
                    zero_shot_module, "make_loader", side_effect=fake_loader
                ),
                patch.object(
                    zero_shot_module,
                    "_checkpoint_metadata",
                    return_value=checkpoint,
                ),
                patch.object(
                    zero_shot_module,
                    "setup_logger",
                    return_value=logging.getLogger("zero_shot_test"),
                ),
            ):
                result = zero_shot_module.run_zero_shot(args)

            five_dir = Path(
                result["results"]["five_class_adapted"]["output_directory"]
            )
            binary_dir = Path(
                result["results"]["author_binary"]["output_directory"]
            )
            five_summary = json.loads(
                (five_dir / "metrics_summary.json").read_text(encoding="utf-8")
            )
            binary_summary = json.loads(
                (binary_dir / "metrics_summary.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertTrue((five_dir / "test_probabilities.csv").is_file())
            self.assertTrue(
                (five_dir / "test_similarity_scores.csv").is_file()
            )
            self.assertTrue((binary_dir / "test_binary_scores.csv").is_file())
            self.assertEqual(
                five_summary["test_metrics"]["average_precision_score_input"],
                "exported_uncalibrated_softmax_confidence",
            )
            self.assertIn(
                "one_vs_rest_binary_proxy",
                binary_summary["evaluation_task"],
            )
            self.assertFalse(
                five_summary["model_selection"]["model_selection_performed"]
            )

    def test_merlin_version_is_strictly_pinned(self):
        with patch.object(zero_shot_module, "version", return_value="0.0.7"):
            self.assertEqual(validate_merlin_package_version(), "0.0.7")
        with patch.object(zero_shot_module, "version", return_value="0.0.8"):
            with self.assertRaisesRegex(RuntimeError, "Expected merlin-vlm==0.0.7"):
                validate_merlin_package_version()


if __name__ == "__main__":
    unittest.main()
