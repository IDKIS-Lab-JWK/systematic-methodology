from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from model import (  # noqa: E402
    FEATURE_DIM,
    PillarClassifier,
    ZeroShotCapabilityError,
    project_qwen_embeddings,
    trainable_parameter_report,
)
from data import CLASS_NAMES  # noqa: E402
from run_zero_shot import build_parser as build_zero_shot_parser  # noqa: E402
from windows import WINDOW_ORDER, apply_pillar_ct_windows  # noqa: E402
from zero_shot import (  # noqa: E402
    ZERO_SHOT_VARIANT,
    aggregate_prompt_embeddings,
    five_class_margin_documents,
    load_prompt_profile,
    run_zero_shot as execute_zero_shot,
    score_five_class_prompt_margins,
)


class DummyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def extract_vision_feats(self, payload):
        value = payload["abdomen_ct"].mean(dim=(1, 2, 3, 4), keepdim=False)
        return value[:, None].repeat(1, FEATURE_DIM) * self.scale


class DummyJoint(DummyVision):
    def __init__(self):
        super().__init__()
        self.text = nn.Linear(4096, FEATURE_DIM, bias=False)

    def encode_text(self, value, normalize=True):
        output = self.text(value)
        return torch.nn.functional.normalize(output, dim=-1) if normalize else output


class DummyReleasedWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(2, 2)
        self.model.text = nn.Linear(2, 2)
        self.model.logit_scale = nn.Parameter(torch.ones(()))


class WindowAndModelTests(unittest.TestCase):
    def test_eleven_ordered_windows(self):
        raw = torch.tensor([[[-1000.0, 0.0], [1000.0, 2000.0]]])
        output = apply_pillar_ct_windows(raw)
        self.assertEqual(output.shape, (1, len(WINDOW_ORDER), 1, 2, 2))
        self.assertEqual(len(WINDOW_ORDER), 11)
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        self.assertTrue(torch.allclose(output[0, -1].flatten(), torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-5))

    def test_linear_probe_freezes_only_encoder(self):
        model = PillarClassifier(DummyVision(), frozen_encoder=True)
        report = trainable_parameter_report(model)
        self.assertTrue(all(name.startswith("classifier.") for name in report.trainable_names))
        logits = model(torch.zeros(1, 1, 2, 2, 2))
        logits.sum().backward()
        self.assertIsNone(model.encoder.scale.grad)
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_full_finetune_encoder_gets_gradient(self):
        model = PillarClassifier(DummyVision(), frozen_encoder=False)
        model(torch.zeros(1, 1, 2, 2, 2)).sum().backward()
        self.assertIsNotNone(model.encoder.scale.grad)

    def test_supervised_tuning_freezes_released_text_components(self):
        encoder = DummyReleasedWrapper()
        PillarClassifier(encoder, frozen_encoder=False)
        parameters = dict(encoder.named_parameters())
        self.assertTrue(parameters["model.visual.weight"].requires_grad)
        self.assertTrue(parameters["model.visual.bias"].requires_grad)
        self.assertFalse(parameters["model.text.weight"].requires_grad)
        self.assertFalse(parameters["model.text.bias"].requires_grad)
        self.assertFalse(parameters["model.logit_scale"].requires_grad)

    def test_zero_shot_rejects_missing_projection(self):
        with self.assertRaises(ZeroShotCapabilityError):
            project_qwen_embeddings(DummyVision(), torch.zeros(2, 4096))

    def test_zero_shot_uses_trained_projection(self):
        output = project_qwen_embeddings(DummyJoint(), torch.ones(2, 4096))
        self.assertEqual(output.shape, (2, FEATURE_DIM))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_zero_shot_margin_uses_both_prompt_polarities(self):
        image = torch.tensor([[1.0, 0.0]])
        prompts = {
            name: {
                "positive": torch.tensor([-1.0, 0.0]),
                "negative": torch.tensor([1.0, 0.0]),
            }
            for name in CLASS_NAMES
        }
        prompts[CLASS_NAMES[0]] = {
            "positive": torch.tensor([0.8, 0.6]),
            "negative": torch.tensor([-1.0, 0.0]),
        }
        prompts[CLASS_NAMES[1]] = {
            "positive": torch.tensor([1.0, 0.0]),
            "negative": torch.tensor([0.9, 0.4358899]),
        }

        scores = score_five_class_prompt_margins(image, prompts, 2.0)
        self.assertEqual(
            int(scores["positive_similarity"].argmax(dim=1).item()), 1
        )
        self.assertEqual(int(scores["class_score"].argmax(dim=1).item()), 0)
        self.assertTrue(
            torch.allclose(
                scores["class_score"],
                scores["positive_similarity"] - scores["negative_similarity"],
            )
        )
        self.assertTrue(
            torch.allclose(scores["scaled_logit"], 2.0 * scores["class_score"])
        )
        self.assertEqual(tuple(scores["probability"].shape), (1, len(CLASS_NAMES)))
        self.assertTrue(
            torch.allclose(
                scores["probability"].sum(dim=1), torch.ones(1), atol=1e-6
            )
        )
        self.assertTrue(torch.isfinite(scores["probability"]).all())
        self.assertEqual(
            int(scores["probability"].argmax(dim=1).item()),
            int(scores["class_score"].argmax(dim=1).item()),
        )

    def test_zero_shot_prompt_means_are_not_renormalized(self):
        embeddings = []
        mapping = []
        for index, class_name in enumerate(CLASS_NAMES):
            positive_values = (
                [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
                if index == 0
                else [torch.tensor([1.0, 0.0])]
            )
            for value in positive_values:
                embeddings.append(value)
                mapping.append((class_name, "positive"))
            embeddings.append(torch.tensor([-1.0, 0.0]))
            mapping.append((class_name, "negative"))

        prompts = aggregate_prompt_embeddings(torch.stack(embeddings), mapping)
        mean_cosine = torch.tensor([1.0, 0.0]) @ prompts[CLASS_NAMES[0]]["positive"]
        self.assertAlmostEqual(float(mean_cosine), 0.5, places=6)
        self.assertAlmostEqual(
            float(prompts[CLASS_NAMES[0]]["positive"].norm()),
            2 ** -0.5,
            places=6,
        )

    def test_zero_shot_margin_rejects_invalid_scale(self):
        image = torch.ones(1, 2)
        prompts = {
            name: {
                "positive": torch.ones(2),
                "negative": -torch.ones(2),
            }
            for name in CLASS_NAMES
        }
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                score_five_class_prompt_margins(image, prompts, scale)

    def test_zero_shot_documents_retain_margin_audit_scores(self):
        image = torch.eye(len(CLASS_NAMES))
        prompts = {
            class_name: {
                "positive": image[class_index],
                "negative": -image[class_index],
            }
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        rows = [
            {
                "patient_id": f"patient-{index}",
                "label": index,
                "label_str": class_name,
            }
            for index, class_name in enumerate(CLASS_NAMES)
        ]
        metrics, predictions, probabilities, score_frame = (
            five_class_margin_documents(rows, image, prompts, 1.0)
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(
            metrics["probability_interpretation"],
            "uncalibrated_relative_score",
        )
        self.assertEqual(probabilities.shape, (len(CLASS_NAMES), len(CLASS_NAMES) + 3))
        self.assertEqual(score_frame.shape, (len(CLASS_NAMES), 4 * len(CLASS_NAMES) + 3))
        self.assertIn(
            f"{CLASS_NAMES[0]}_positive_similarity", score_frame.columns
        )
        self.assertIn(f"{CLASS_NAMES[0]}_class_score", score_frame.columns)
        self.assertIn("prompt_similarity_scores", predictions[0])
        self.assertEqual(
            predictions[0]["probability_interpretation"],
            "uncalibrated_relative_score",
        )

    def test_zero_shot_uses_only_schema_v2_five_class_profile(self):
        profile = load_prompt_profile(
            MODEL_ROOT / "zero_shot_prompts_five_class_margin_v2.json"
        )
        self.assertEqual(profile["schema_version"], 2)
        self.assertEqual(profile["zero_shot_variant"], ZERO_SHOT_VARIANT)
        self.assertNotIn("author_binary_classes", profile)
        with self.assertRaises(ValueError):
            load_prompt_profile(MODEL_ROOT / "zero_shot_prompts.json")
        self.assertNotIn(
            "mode", {action.dest for action in build_zero_shot_parser()._actions}
        )

    def test_zero_shot_refuses_existing_prompt_set_before_model_load(self):
        prompt_path = MODEL_ROOT / "zero_shot_prompts_five_class_margin_v2.json"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            output = (
                results
                / "ZeroShot"
                / "Test"
                / ZERO_SHOT_VARIANT
                / "pillar0_qwen_findings_five_class_margin_v2"
            )
            output.mkdir(parents=True)
            (output / "metrics_summary.json").write_text("{}", encoding="utf-8")
            with mock.patch("zero_shot.load_official_model") as loader:
                with self.assertRaises(FileExistsError):
                    execute_zero_shot(
                        [],
                        prompt_path,
                        results / "missing_cache_manifest.json",
                        results / "features",
                        results,
                        device=torch.device("cpu"),
                    )
                loader.assert_not_called()

    def test_zero_shot_run_writes_only_canonical_v2_artifacts(self):
        prompt_path = MODEL_ROOT / "zero_shot_prompts_five_class_margin_v2.json"
        rows = [
            {
                "patient_id": f"patient-{index}",
                "label": index,
                "label_str": class_name,
            }
            for index, class_name in enumerate(CLASS_NAMES)
        ]
        image = torch.zeros(len(CLASS_NAMES), FEATURE_DIM)
        image[:, : len(CLASS_NAMES)] = torch.eye(len(CLASS_NAMES))
        prompts = {
            class_name: {
                "positive": image[class_index],
                "negative": -image[class_index],
            }
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        model = DummyJoint()

        def projected(_model, raw):
            return torch.ones(raw.shape[0], FEATURE_DIM)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "cache_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with (
                mock.patch("zero_shot.load_official_model", return_value=model),
                mock.patch(
                    "zero_shot.zero_shot_capability_report", return_value={}
                ),
                mock.patch(
                    "zero_shot.project_qwen_embeddings", side_effect=projected
                ),
                mock.patch(
                    "zero_shot.checkpoint_logit_scale", return_value=2.0
                ),
                mock.patch(
                    "zero_shot.encode_qwen_findings",
                    return_value=torch.zeros(40, 4096),
                ),
                mock.patch(
                    "zero_shot.aggregate_prompt_embeddings",
                    return_value=prompts,
                ),
                mock.patch("zero_shot.FeatureStore", return_value=mock.sentinel.store),
                mock.patch(
                    "zero_shot._materialize_image_embeddings",
                    return_value=image,
                ),
            ):
                result = execute_zero_shot(
                    rows,
                    prompt_path,
                    manifest,
                    root / "features",
                    root,
                    device=torch.device("cpu"),
                )

            output = (
                root
                / "ZeroShot"
                / "Test"
                / ZERO_SHOT_VARIANT
                / "pillar0_qwen_findings_five_class_margin_v2"
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "metrics_summary.json",
                    "model_selection.json",
                    "prompt_profile_snapshot.json",
                    "test_patient_predictions.json",
                    "test_probabilities.csv",
                    "test_similarity_scores.csv",
                },
            )
            self.assertEqual(
                result["outputs"][ZERO_SHOT_VARIANT], str(output)
            )


if __name__ == "__main__":
    unittest.main()
