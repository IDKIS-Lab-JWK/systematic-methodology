from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from monai.data import MetaTensor


MERLIN_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_SHARED = MERLIN_DIR.parents[2] / "Notebooks" / "_shared"
sys.path.insert(0, str(MERLIN_DIR))
sys.path.insert(0, str(NOTEBOOK_SHARED))

import model as model_module  # noqa: E402
import training as training_module  # noqa: E402
from cli import _explicit_parameter_names, build_parser  # noqa: E402
from model import MerlinClassifier, build_model  # noqa: E402
from orchestrator import (  # noqa: E402
    GRID_CONFIG_FILENAMES,
    VARIANT_FOLDERS,
    _validate_test_checkpoint_identity,
    grid_candidates,
    parse_bool,
    resolve_parameters,
)
from result_analysis import VARIANT_LABELS as NOTEBOOK_VARIANT_LABELS  # noqa: E402
from training import (  # noqa: E402
    MerlinFeatureStore,
    _checkpoint,
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    predict_checkpoint,
    train_epoch,
)


CLASSES = [
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
]


def linear_parameters() -> dict:
    return {
        "head_lr": 1e-4,
        "head_weight_decay": 1e-4,
        "warmup_epochs": 0,
        "min_lr_factor": 1.0,
        "max_grad_norm": 0.0,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "epochs": 2,
        "scheduler_decay_epochs": 2,
        "early_stopping_patience": 0,
        "min_delta": 0.0,
        "embedding_dim": 2048,
    }


def parsed_args(*arguments: str):
    parser = build_parser("linear_probe")
    values = list(arguments)
    args = parser.parse_args(values)
    args.explicit_parameter_names = _explicit_parameter_names(parser, values)
    args.grid = parse_bool(args.grid)
    args.percentage_specific_grid = parse_bool(args.percentage_specific_grid)
    args.train_image_projection = parse_bool(args.train_image_projection)
    args.gradient_checkpointing = parse_bool(args.gradient_checkpointing)
    return args


class FakeMerlin(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super().__init__()
        if not ImageEmbedding:
            raise ValueError("Fake probe requires ImageEmbedding=True")
        self.batch_norm = nn.BatchNorm1d(4)
        self.projection = nn.Linear(4, 2048)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.reshape(images.shape[0], -1)[:, :4]
        return self.projection(self.batch_norm(values))


def fake_built_model(device: torch.device):
    encoder = FakeMerlin(ImageEmbedding=True)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return MerlinClassifier(
        encoder,
        len(CLASSES),
        2048,
        0.0,
        frozen_encoder=True,
        strict_linear_head=True,
    ).to(device)


class LinearProbeModelTests(unittest.TestCase):
    def setUp(self):
        # Avoid leaking the process-level production encoder cache across
        # tests that replace the official Merlin class with a lightweight fake.
        model_module._LINEAR_PROBE_ENCODER = None

    def tearDown(self):
        model_module._LINEAR_PROBE_ENCODER = None

    def test_builds_strict_head_and_freezes_encoder_in_eval_mode(self):
        with patch.object(
            model_module, "import_merlin", return_value=(FakeMerlin, object)
        ):
            model, details = build_model(
                "linear_probe",
                len(CLASSES),
                linear_parameters(),
                torch.device("cpu"),
            )

        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, 2048)
        self.assertEqual(model.classifier.out_features, 5)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        model.train()
        self.assertTrue(model.classifier.training)
        self.assertFalse(model.encoder.training)
        self.assertEqual(details["adaptation_scope"], "strict linear head only")
        self.assertFalse(details["gradient_checkpointing_requested"])

    def test_encoder_reuse_does_not_perturb_seeded_head_initialization(self):
        with patch.object(
            model_module, "import_merlin", return_value=(FakeMerlin, object)
        ):
            torch.manual_seed(123)
            first, _ = build_model(
                "linear_probe",
                len(CLASSES),
                linear_parameters(),
                torch.device("cpu"),
            )
            torch.manual_seed(123)
            second, _ = build_model(
                "linear_probe",
                len(CLASSES),
                linear_parameters(),
                torch.device("cpu"),
            )

        self.assertIs(first.encoder, second.encoder)
        self.assertTrue(
            torch.equal(first.classifier.weight, second.classifier.weight)
        )
        self.assertTrue(torch.equal(first.classifier.bias, second.classifier.bias))

    def test_optimizer_contains_exactly_one_classifier_group(self):
        model = fake_built_model(torch.device("cpu"))
        optimizer = build_optimizer(model, "linear_probe", linear_parameters())

        self.assertEqual(len(optimizer.param_groups), 1)
        group = optimizer.param_groups[0]
        self.assertEqual(group["name"], "classifier")
        self.assertEqual(group["lr"], 1e-4)
        self.assertEqual(group["weight_decay"], 1e-4)
        self.assertEqual(
            {id(value) for value in group["params"]},
            {id(value) for value in model.classifier.parameters()},
        )

    def test_checkpoint_stores_only_the_learned_head(self):
        model = fake_built_model(torch.device("cpu"))
        optimizer = build_optimizer(model, "linear_probe", linear_parameters())
        scheduler = build_scheduler(optimizer, 2, linear_parameters())
        scaler = build_grad_scaler(False)
        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            "linear_probe",
            linear_parameters(),
            {"seed": 42},
            None,
            CLASSES,
            {"frozen_encoder_provenance": {}},
            1,
            1,
            0.5,
            [],
        )

        self.assertEqual(checkpoint["checkpoint_scope"], "classifier_only")
        self.assertIn("classifier_state_dict", checkpoint)
        self.assertNotIn("model_state_dict", checkpoint)

    def test_feature_training_does_not_change_encoder_parameters_or_buffers(self):
        torch.manual_seed(7)
        model = fake_built_model(torch.device("cpu"))
        before = {
            name: value.detach().clone()
            for name, value in model.encoder.state_dict().items()
        }
        head_before = model.classifier.weight.detach().clone()
        optimizer = build_optimizer(model, "linear_probe", linear_parameters())
        scaler = build_grad_scaler(False)
        loader = [
            {
                "image": torch.randn(2, 2048),
                "label": torch.tensor([0, 1]),
            },
            {
                "image": torch.randn(2, 2048),
                "label": torch.tensor([2, 3]),
            },
        ]

        train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            linear_parameters(),
            False,
            features_only=True,
        )

        self.assertFalse(model.encoder.training)
        for name, value in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(torch.equal(head_before, model.classifier.weight))

    def test_publication_cosine_horizon_is_independent_of_training_epochs(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_scheduler(
            optimizer,
            10,
            {
                "warmup_epochs": 0,
                "min_lr_factor": 0.0,
                "scheduler_decay_epochs": 300,
            },
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"],
            0.5 * (1.0 + math.cos(math.pi * 10 / 300)),
        )
        self.assertGreater(optimizer.param_groups[0]["lr"], 0.99)


class LinearProbeParameterTests(unittest.TestCase):
    def test_default_profile_is_project_grounded_and_head_only(self):
        parameters, provenance = resolve_parameters(
            parsed_args("--parameter_source", "default"), "linear_probe"
        )

        self.assertEqual(parameters["head_lr"], 1e-4)
        self.assertEqual(parameters["epochs"], 100)
        self.assertEqual(parameters["batch_size"], 1)
        self.assertEqual(parameters["grad_accum_steps"], 1)
        self.assertEqual(parameters["min_lr_factor"], 1.0)
        self.assertNotIn("dropout_prob", parameters)
        self.assertNotIn("encoder_lr", parameters)
        self.assertNotIn("gradient_checkpointing", parameters)
        self.assertEqual(provenance["parameter_origin"], "default")

    def test_publication_profile_preserves_reported_values(self):
        parameters, provenance = resolve_parameters(
            parsed_args("--parameter_source", "publication"), "linear_probe"
        )

        self.assertEqual(parameters["head_lr"], 1e-5)
        self.assertEqual(parameters["epochs"], 10)
        self.assertEqual(parameters["scheduler_decay_epochs"], 300)
        self.assertEqual(parameters["min_lr_factor"], 0.0)
        self.assertEqual(parameters["early_stopping_patience"], 0)
        self.assertEqual(provenance["parameter_origin"], "publication_adapted")
        self.assertIn(
            "five_class", provenance["publication_fidelity"]
        )

    def test_grid_and_result_registration(self):
        self.assertEqual(VARIANT_FOLDERS["linear_probe"], "LinearProbe")
        self.assertEqual(
            GRID_CONFIG_FILENAMES["linear_probe"], "grid_linear_probe.json"
        )
        candidates = grid_candidates(
            MERLIN_DIR / GRID_CONFIG_FILENAMES["linear_probe"]
        )
        self.assertEqual(len(candidates), 9)
        defaults, _ = resolve_parameters(
            parsed_args("--parameter_source", "default"), "linear_probe"
        )
        self.assertEqual(set(candidates[0]), set(defaults))
        self.assertEqual(NOTEBOOK_VARIANT_LABELS["linear_probe"], "Linear Probe")

    def test_irrelevant_encoder_option_is_rejected(self):
        args = parsed_args(
            "--parameter_source", "manual", "--encoder_lr", "2e-5"
        )
        with self.assertRaisesRegex(ValueError, "not valid for linear_probe"):
            resolve_parameters(args, "linear_probe")

    def test_heldout_checkpoint_identity_is_strict(self):
        state = {
            "variant": "linear_probe",
            "classes": CLASSES,
            "parameters": linear_parameters(),
        }
        _validate_test_checkpoint_identity(
            state, "linear_probe", CLASSES, linear_parameters()
        )

        wrong_order = {**state, "classes": list(reversed(CLASSES))}
        with self.assertRaisesRegex(ValueError, "class order"):
            _validate_test_checkpoint_identity(
                wrong_order, "linear_probe", CLASSES, linear_parameters()
            )

        wrong_parameters = {
            **state,
            "parameters": {**linear_parameters(), "head_lr": 9e-5},
        }
        with self.assertRaisesRegex(ValueError, "parameters"):
            _validate_test_checkpoint_identity(
                wrong_parameters, "linear_probe", CLASSES, linear_parameters()
            )


class LinearProbeCacheAndCheckpointTests(unittest.TestCase):
    def test_feature_store_round_trip_is_float32_and_provenance_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "Cache" / "Shared" / "preprocessing_v1"
            cache_dir.mkdir(parents=True)
            (cache_dir / "cache_manifest.json").write_text(
                json.dumps({"namespace": "preprocessing_v1"}), encoding="utf-8"
            )
            row = {
                "patient_id": "patient-1",
                "image": str(Path(directory) / "patient-1.nii.gz"),
            }
            provenance = {
                "package": "merlin-vlm",
                "package_version": "0.0.7",
                "checkpoint_name": "checkpoint.pt",
                "checkpoint_path": str(Path(directory) / "checkpoint.pt"),
                "checkpoint_sha256": "abc123",
            }
            with patch.object(
                training_module,
                "_merlin_checkpoint_provenance",
                return_value=provenance,
            ):
                store = MerlinFeatureStore(cache_dir, object(), "fp32")
                store.save(row, torch.arange(2048, dtype=torch.float64))
                loaded = store.load(row)

            self.assertEqual(loaded.shape, (2048,))
            self.assertEqual(loaded.dtype, torch.float32)
            self.assertTrue(store.manifest_path.is_file())

    def test_feature_store_migrates_metatensor_to_weights_only_plain_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "Cache" / "Shared" / "preprocessing_v1"
            cache_dir.mkdir(parents=True)
            (cache_dir / "cache_manifest.json").write_text(
                json.dumps({"namespace": "preprocessing_v1"}), encoding="utf-8"
            )
            row = {
                "patient_id": "patient-meta",
                "image": str(Path(directory) / "patient-meta.nii.gz"),
            }
            provenance = {
                "package": "merlin-vlm",
                "package_version": "0.0.7",
                "checkpoint_name": "checkpoint.pt",
                "checkpoint_path": str(Path(directory) / "checkpoint.pt"),
                "checkpoint_sha256": "abc123",
            }
            with patch.object(
                training_module,
                "_merlin_checkpoint_provenance",
                return_value=provenance,
            ):
                store = MerlinFeatureStore(cache_dir, object(), "fp32")
                path = store.path_for(row)
                training_module.atomic_torch_save(
                    {
                        "feature": MetaTensor(torch.arange(2048).float()),
                        "patient_id": row["patient_id"],
                        "image": row["image"],
                    },
                    path,
                )
                loaded = store.load(row)
                rewritten = torch.load(
                    path, map_location="cpu", weights_only=True
                )

            self.assertIs(type(loaded), torch.Tensor)
            self.assertIs(type(rewritten["feature"]), torch.Tensor)
            self.assertTrue(torch.equal(loaded, rewritten["feature"]))

    def test_predict_checkpoint_reconstructs_linear_probe_from_cached_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best_model.pt"
            source_model = fake_built_model(torch.device("cpu"))
            encoder_provenance = {
                "package": "merlin-vlm",
                "package_version": "0.0.7",
                "checkpoint_name": "fake.pt",
                "checkpoint_path": str(root / "fake.pt"),
                "checkpoint_sha256": "fake-sha256",
            }
            preprocessing_sha256 = "preprocessing-sha256"
            torch.save(
                {
                    "variant": "linear_probe",
                    "parameters": linear_parameters(),
                    "classes": CLASSES,
                    "classifier_state_dict": source_model.classifier.state_dict(),
                    "checkpoint_scope": "classifier_only",
                    "model_details": {
                        "frozen_encoder_provenance": encoder_provenance,
                        "feature_cache": {
                            "preprocessing_manifest_sha256": preprocessing_sha256
                        },
                    },
                    "runtime": {
                        "seed": 42,
                        "amp": False,
                        "feature_extraction_batch_size": 1,
                    },
                    "epoch": 1,
                    "best_epoch": 1,
                },
                checkpoint_path,
            )
            rows = [
                {
                    "patient_id": f"p{index}",
                    "image": str(root / f"p{index}.nii.gz"),
                    "label": index,
                    "label_str": class_name,
                }
                for index, class_name in enumerate(CLASSES)
            ]
            features = torch.randn(len(rows), 2048)

            def rebuilt_model(variant, num_classes, parameters, device):
                self.assertEqual(variant, "linear_probe")
                return fake_built_model(device), {}

            class DummyStore:
                expected = {
                    "model": encoder_provenance,
                    "preprocessing_manifest_sha256": preprocessing_sha256,
                }

            with (
                patch.object(training_module, "build_model", side_effect=rebuilt_model),
                patch.object(
                    training_module, "MerlinFeatureStore", return_value=DummyStore()
                ),
                patch.object(
                    training_module,
                    "materialize_linear_probe_features",
                    return_value=features,
                ),
            ):
                metrics, predictions, frame, labels, probabilities, state = (
                    predict_checkpoint(
                        checkpoint_path,
                        rows,
                        root,
                        batch_size=2,
                        num_workers=0,
                    )
                )

            self.assertEqual(state["variant"], "linear_probe")
            self.assertEqual(len(predictions), len(rows))
            self.assertEqual(frame.shape[0], len(rows))
            self.assertEqual(labels.shape, (len(rows),))
            self.assertEqual(probabilities.shape, (len(rows), len(CLASSES)))
            self.assertIn("f1_macro", metrics)


if __name__ == "__main__":
    unittest.main()
