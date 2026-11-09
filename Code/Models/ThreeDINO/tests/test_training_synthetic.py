"""One-update synthetic training checks; no NIfTI, upstream source, or weights."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent skip
    torch = None
    nn = None

from Code.Models.ThreeDINO import CLASS_NAMES  # noqa: E402
from Code.Models.ThreeDINO import orchestrator  # noqa: E402
from Code.Models.ThreeDINO import training  # noqa: E402

TorchModule = nn.Module if nn is not None else object


def rows() -> list[dict]:
    return [
        {
            "image": torch.full((1, 2, 2, 2), float(index + 1) / 10),
            "label": index,
            "patient_id": f"patient-{index}",
            "label_str": class_name,
        }
        for index, class_name in enumerate(CLASS_NAMES)
    ]


def synthetic_dataset(source_rows, **_):
    return [dict(row) for row in source_rows]


def resume_rows() -> list[dict]:
    return [
        {
            "image": torch.full((1, 2, 2, 2), float(index + 1) / 20),
            "label": index % len(CLASS_NAMES),
            "patient_id": f"resume-{index}",
            "label_str": CLASS_NAMES[index % len(CLASS_NAMES)],
        }
        for index in range(10)
    ]


class RandomizedDataset:
    """Tiny MONAI-like Randomizable dataset used to verify RNG restoration."""

    _threedino_draw_seed_protocol = training.DRAW_SEED_PROTOCOL

    def __init__(self, source_rows, seed):
        self.rows = [dict(row) for row in source_rows]
        self.R = np.random.RandomState(int(seed))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["image"] = row["image"] + float(self.R.uniform(-0.05, 0.05))
        return row


def randomized_training_dataset(source_rows, *, split, seed, **_):
    if split == "train":
        return RandomizedDataset(source_rows, seed)
    return [dict(row) for row in source_rows]


def assert_state_dict_equal(testcase, first, second):
    testcase.assertEqual(set(first), set(second))
    for key in first:
        testcase.assertTrue(torch.equal(first[key], second[key]), key)


class AssetProvisioningLogTests(unittest.TestCase):
    def test_startup_line_uses_provisioner_acquisition_fields(self):
        logger = mock.Mock()
        training._log_asset_provisioning(
            logger,
            {
                "upstream": {
                    "acquisition": "git_clone",
                    "revision": "a" * 40,
                },
                "checkpoint": {
                    "acquisition": "huggingface_download",
                    "revision": "b" * 40,
                    "sha256": "c" * 64,
                },
                "downloads_performed": True,
                "git_operations_performed": True,
            },
        )

        logger.info.assert_called_once()
        template, *values = logger.info.call_args.args
        rendered = template % tuple(values)
        self.assertIn("source_action=git_clone", rendered)
        self.assertIn("checkpoint_action=huggingface_download", rendered)
        self.assertIn("downloads_performed=True", rendered)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SyntheticTrainingTests(unittest.TestCase):
    class FrozenEncoder(TorchModule):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.calls = 0

    class TinyFullModel(TorchModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(8, 4), nn.ReLU())
            self.classifier = nn.Linear(4, 5)

        def forward(self, images):
            return self.classifier(self.encoder(images))

    def test_linear_grid_trains_head_and_never_encoder(self):
        encoder = self.FrozenEncoder()

        def extract_bank(model, images, representations):
            model.calls += 1
            base = images.flatten(1).mean(dim=1, keepdim=True)
            return {recipe: base.repeat(1, 4) * model.scale for recipe in representations}

        parameters = {
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "batch_size": 5,
            "max_iterations": 1,
            "evaluation_interval_iterations": 1,
            "checkpoint_interval_iterations": 1,
            "class_weighting": "none",
            "mixed_precision": False,
        }
        candidates = [
            {
                "name": "synthetic",
                "nominal_lr": 0.1,
                "lr": 0.1,
                "learning_rate_scale_rule": "none",
                "n_last_blocks": 1,
                "use_mean_patch_token": False,
                "feature_dim": 4,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "linear"
            result = training.train_linear_probe_grid(
                rows(),
                rows(),
                parameters,
                candidates,
                output,
                device=torch.device("cpu"),
                encoder_factory=lambda **_: encoder,
                feature_bank_extractor=extract_bank,
                feature_dimension_fn=lambda _blocks, _mean: 4,
                dataset_factory=synthetic_dataset,
            )
            self.assertTrue(result["state"]["completed"])
            self.assertTrue(result["state"]["encoder_frozen"])
            self.assertEqual(result["state"]["candidate_count"], 1)
            self.assertIsNone(encoder.scale.grad)
            self.assertGreaterEqual(encoder.calls, 2)
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "training_state.json").is_file())

            identity = result["best_candidate_id"]
            resolved = result["candidates"][identity]["candidate"]
            candidate_dir = output / "candidates" / identity
            candidate_state = json.loads(
                (candidate_dir / "training_state.json").read_text(encoding="utf-8")
            )
            candidate_checkpoint = torch.load(
                candidate_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            expected_fingerprint = training.parameter_fingerprint(resolved)
            self.assertTrue(candidate_state["completed"])
            self.assertEqual(candidate_state["candidate_id"], identity)
            self.assertEqual(candidate_state["parameters"], resolved)
            self.assertEqual(
                candidate_state["parameters_fingerprint"], expected_fingerprint
            )
            self.assertEqual(candidate_state["best_optimizer_step"], 1)
            self.assertEqual(candidate_state["classes"], list(CLASS_NAMES))
            self.assertEqual(
                candidate_state["runtime_plan"]["candidate_id"], identity
            )
            self.assertEqual(
                candidate_state["runtime_plan"][
                    "candidate_parameters_fingerprint"
                ],
                expected_fingerprint,
            )
            self.assertEqual(candidate_checkpoint["parameters"], resolved)
            self.assertEqual(
                candidate_checkpoint["parameters_fingerprint"],
                expected_fingerprint,
            )
            self.assertEqual(
                candidate_checkpoint["runtime_plan"],
                candidate_state["runtime_plan"],
            )
            self.assertEqual(
                candidate_checkpoint["runtime_plan_fingerprint"],
                candidate_state["runtime_plan_fingerprint"],
            )
            self.assertEqual(
                candidate_checkpoint["source_provenance"],
                candidate_state["source_provenance"],
            )
            for artifact in (
                "metrics_summary.json",
                "validation_patient_predictions.json",
                "validation_probabilities.csv",
            ):
                self.assertTrue((candidate_dir / artifact).is_file())
            model_identity = {
                "upstream_revision": candidate_state["source_provenance"][
                    "upstream_revision"
                ],
                "pretrained_checkpoint_sha256": candidate_state[
                    "source_provenance"
                ]["checkpoint_sha256"],
            }
            self.assertTrue(
                orchestrator._is_completed(
                    candidate_dir,
                    resolved,
                    model_identity,
                )
            )
            checkpoint_metadata = (
                orchestrator._validate_internal_classifier_checkpoint(
                    candidate_dir / "best_model.pt",
                    method="linear_probe",
                    expected_parameters=resolved,
                    model_identity=model_identity,
                    expected_runtime_budget=1,
                    expected_training_rows_fingerprint=candidate_state[
                        "runtime_plan"
                    ]["training_rows_fingerprint"],
                    expected_runtime_plan_fingerprint=candidate_state[
                        "runtime_plan_fingerprint"
                    ],
                )
            )
            self.assertTrue(checkpoint_metadata["binding_verified"])

    def test_full_finetune_is_gated_then_updates_tiny_encoder(self):
        parameters = {
            "optimizer": "adamw",
            "encoder_lr": 0.01,
            "head_lr": 0.01,
            "weight_decay": 0.0,
            "batch_size": 5,
            "grad_accum_steps": 1,
            "max_iterations": 1,
            "evaluation_interval_iterations": 1,
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "class_weighting": "none",
            "early_stopping_patience": 0,
            "max_grad_norm": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PermissionError, "disabled by default"):
                training.train_full_finetune(
                    rows(),
                    rows(),
                    parameters,
                    Path(temporary) / "blocked",
                    device=torch.device("cpu"),
                    model_factory=lambda **_: self.TinyFullModel(),
                    dataset_factory=synthetic_dataset,
                )

            tiny = self.TinyFullModel()
            before = tiny.encoder[1].weight.detach().clone()
            authorized = {
                **parameters,
                "allow_experimental_full_finetune": True,
            }
            output = Path(temporary) / "authorized"
            result = training.train_full_finetune(
                rows(),
                rows(),
                authorized,
                output,
                device=torch.device("cpu"),
                model_factory=lambda **_: tiny,
                dataset_factory=synthetic_dataset,
            )
            self.assertTrue(result["state"]["completed"])
            self.assertTrue(result["state"]["project_adapted"])
            self.assertFalse(result["state"]["publication_backed_classification_method"])
            self.assertFalse(torch.equal(before, tiny.encoder[1].weight.detach()))
            self.assertTrue((output / "best_model.pt").is_file())

    def test_probe_resume_is_exact_and_runtime_horizon_is_immutable(self):
        parameters = {
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "batch_size": 2,
            "max_iterations": 3,
            "evaluation_interval_iterations": 1,
            "checkpoint_interval_iterations": 1,
            "class_weighting": "none",
            "mixed_precision": False,
        }
        candidates = [
            {
                "name": "resume",
                "nominal_lr": 0.1,
                "lr": 0.1,
                "learning_rate_scale_rule": "none",
                "n_last_blocks": 1,
                "use_mean_patch_token": False,
                "feature_dim": 4,
            }
        ]

        def extract_bank(model, images, representations):
            base = images.flatten(1).mean(dim=1, keepdim=True)
            return {recipe: base.repeat(1, 4) * model.scale for recipe in representations}

        common = {
            "device": torch.device("cpu"),
            "encoder_factory": lambda **_: self.FrozenEncoder(),
            "feature_bank_extractor": extract_bank,
            "feature_dimension_fn": lambda _blocks, _mean: 4,
            "dataset_factory": randomized_training_dataset,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            interrupted = root / "interrupted"
            training.train_linear_probe_grid(
                resume_rows(), resume_rows(), parameters, candidates, reference, **common
            )

            original_save = training.atomic_torch_save

            def stop_after_checkpoint(value, path):
                original_save(value, path)
                if Path(path).name == "last_checkpoint.pt" and int(
                    value.get("optimizer_step", -1)
                ) == 1:
                    raise RuntimeError("synthetic interruption")

            with mock.patch.object(
                training, "atomic_torch_save", side_effect=stop_after_checkpoint
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    training.train_linear_probe_grid(
                        resume_rows(),
                        resume_rows(),
                        parameters,
                        candidates,
                        interrupted,
                        **common,
                    )

            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                training.train_linear_probe_grid(
                    resume_rows(),
                    resume_rows(),
                    parameters,
                    candidates,
                    interrupted,
                    resume=True,
                    budget=2,
                    **common,
                )

            checkpoint_path = interrupted / "last_checkpoint.pt"
            valid_partial = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            invalid = dict(valid_partial)
            invalid["optimizer_step"] = 4
            invalid["scheduler_state_dict"] = dict(
                valid_partial["scheduler_state_dict"]
            )
            invalid["scheduler_state_dict"]["last_epoch"] = 4
            original_save(invalid, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "outside scheduler horizon"):
                training.train_linear_probe_grid(
                    resume_rows(),
                    resume_rows(),
                    parameters,
                    candidates,
                    interrupted,
                    resume=True,
                    **common,
                )
            original_save(valid_partial, checkpoint_path)

            training.train_linear_probe_grid(
                resume_rows(),
                resume_rows(),
                parameters,
                candidates,
                interrupted,
                resume=True,
                **common,
            )
            expected = torch.load(
                reference / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            actual = torch.load(
                interrupted / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            assert_state_dict_equal(
                self, expected["heads_state_dict"], actual["heads_state_dict"]
            )
            self.assertEqual(expected["history"], actual["history"])
            self.assertEqual(
                expected["runtime_plan_fingerprint"],
                actual["runtime_plan_fingerprint"],
            )

    def test_full_finetune_resume_is_bit_exact(self):
        parameters = {
            "allow_experimental_full_finetune": True,
            "optimizer": "adamw",
            "encoder_lr": 0.01,
            "head_lr": 0.01,
            "weight_decay": 0.0,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "max_iterations": 3,
            "evaluation_interval_iterations": 1,
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "class_weighting": "none",
            "early_stopping_patience": 0,
            "max_grad_norm": 1.0,
        }
        common = {
            "device": torch.device("cpu"),
            "model_factory": lambda **_: self.TinyFullModel(),
            "dataset_factory": randomized_training_dataset,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            interrupted = root / "interrupted"
            training.train_full_finetune(
                resume_rows(), resume_rows(), parameters, reference, **common
            )
            original_save = training.atomic_torch_save

            def stop_after_checkpoint(value, path):
                original_save(value, path)
                if Path(path).name == "last_checkpoint.pt" and int(
                    value.get("optimizer_step", -1)
                ) == 1:
                    raise RuntimeError("synthetic interruption")

            with mock.patch.object(
                training, "atomic_torch_save", side_effect=stop_after_checkpoint
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    training.train_full_finetune(
                        resume_rows(), resume_rows(), parameters, interrupted, **common
                    )
            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                training.train_full_finetune(
                    resume_rows(),
                    resume_rows(),
                    parameters,
                    interrupted,
                    resume=True,
                    budget=2,
                    **common,
                )
            training.train_full_finetune(
                resume_rows(),
                resume_rows(),
                parameters,
                interrupted,
                resume=True,
                **common,
            )
            expected = torch.load(
                reference / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            actual = torch.load(
                interrupted / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            assert_state_dict_equal(
                self, expected["model_state_dict"], actual["model_state_dict"]
            )
            self.assertEqual(expected["history"], actual["history"])

    def test_infinite_sampler_is_full_batch_and_exactly_resumable(self):
        source = resume_rows()[:5]
        first = training._make_loader(
            source,
            split="train",
            batch_size=4,
            cache_dir=None,
            seed=17,
            num_workers=0,
            dataset_factory=synthetic_dataset,
            sampler_mode="sharded_infinite",
            drop_last=True,
        )
        iterator = iter(first)
        first_batch = next(iterator)
        self.assertEqual(training._batch_values(first_batch)[1].numel(), 4)
        training._commit_training_batch(first, first_batch)
        saved = training._capture_training_data_state(first)
        expected_ids = training._batch_values(next(iterator))[2]

        second = training._make_loader(
            source,
            split="train",
            batch_size=4,
            cache_dir=None,
            seed=17,
            num_workers=0,
            dataset_factory=synthetic_dataset,
            sampler_mode="sharded_infinite",
            drop_last=True,
        )
        mid_stream = training._restore_training_data_state(second, saved)
        resumed = training._iterator_after_data_restore(
            second, restored_mid_epoch=mid_stream
        )
        self.assertEqual(training._batch_values(next(resumed))[2], expected_ids)

    def test_publication_sampler_requires_and_uses_fp16_autocast(self):
        with mock.patch.object(torch, "autocast") as autocast:
            training._autocast(torch.device("cuda"), True, "float16")
            autocast.assert_called_once_with(
                device_type="cuda", dtype=torch.float16
            )

        parameters = {
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "batch_size": 4,
            "max_iterations": 1,
            "sampler": "sharded_infinite",
            "drop_last": True,
            "mixed_precision": False,
            "autocast_dtype": "float16",
        }
        candidate = {
            "nominal_lr": 0.1,
            "lr": 0.1,
            "learning_rate_scale_rule": "none",
            "n_last_blocks": 1,
            "use_mean_patch_token": False,
            "feature_dim": 4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "mixed_precision=true"):
                training.train_linear_probe_grid(
                    rows(),
                    rows(),
                    parameters,
                    [candidate],
                    Path(temporary),
                    device=torch.device("cpu"),
                    encoder_factory=lambda **_: self.FrozenEncoder(),
                    feature_dimension_fn=lambda _blocks, _mean: 4,
                    dataset_factory=synthetic_dataset,
                )
            parameters["mixed_precision"] = True
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
                training.train_linear_probe_grid(
                    rows(),
                    rows(),
                    parameters,
                    [candidate],
                    Path(temporary) / "cpu_rejected",
                    device=torch.device("cpu"),
                    encoder_factory=lambda **_: self.FrozenEncoder(),
                    feature_dimension_fn=lambda _blocks, _mean: 4,
                    dataset_factory=synthetic_dataset,
                )

    def test_resume_fails_closed_for_injected_loader(self):
        loader_factory = lambda *_args, **_kwargs: torch.utils.data.DataLoader(
            synthetic_dataset(resume_rows()), batch_size=2, shuffle=True
        )
        loader = training._make_loader(
            resume_rows(),
            split="train",
            batch_size=2,
            cache_dir=None,
            seed=3,
            num_workers=0,
            dataset_factory=loader_factory,
        )
        state = training._capture_training_data_state(loader)
        self.assertFalse(state["exact_resume_supported"])
        with self.assertRaisesRegex(RuntimeError, "cannot be resumed exactly"):
            training._restore_training_data_state(loader, state)


if __name__ == "__main__":
    unittest.main()
