"""Exact draw-indexed resume checks using tiny, pickleable synthetic data only."""

from __future__ import annotations

import copy
import gc
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
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

from Code.Models.ThreeDINO import training  # noqa: E402


DRAW_KEY = "_threedino_draw_index"
TorchModule = nn.Module if nn is not None else object


def draw_rows(count: int = 31) -> list[dict[str, Any]]:
    """Return enough rows for prefetch to move well beyond a committed batch."""

    return [
        {
            "image": torch.full((1, 2, 2, 2), float(index) / 100.0),
            "label": index % 5,
            "patient_id": f"draw-patient-{index:03d}",
            "label_str": f"class-{index % 5}",
        }
        for index in range(count)
    ]


class DrawRandomizedDataset:
    """Module-level MONAI-like dataset that is pickleable under Windows spawn."""

    _threedino_draw_seed_protocol = training.DRAW_SEED_PROTOCOL

    def __init__(self, source_rows: list[Mapping[str, Any]], seed: int):
        self.rows = [dict(row) for row in source_rows]
        self.set_random_state(seed=seed)

    def __len__(self) -> int:
        return len(self.rows)

    def set_random_state(self, seed: int | None = None, state=None):
        if state is not None:
            self.R = state
        else:
            self.R = np.random.RandomState(int(seed or 0) % (2**32))
        return self

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[int(index)])
        # Two RNG draws make equality sensitive to both the draw seed and call order.
        offsets = self.R.uniform(-0.25, 0.25, size=2)
        row["image"] = row["image"] + float(offsets[0]) + float(offsets[1]) / 100
        return row


def draw_randomized_dataset(
    source_rows: list[Mapping[str, Any]], *, split: str, seed: int, **_kwargs
):
    """Top-level factory so all objects crossing a Windows worker are pickleable."""

    if split == "train":
        return DrawRandomizedDataset(source_rows, seed)
    return [dict(row) for row in source_rows]


def _loader(
    workers: int,
    *,
    rows: list[dict[str, Any]] | None = None,
    batch_size: int = 3,
    seed: int = 129,
    split: str = "train",
    sampler_mode: str = "finite_shuffle",
    drop_last: bool = False,
):
    return training._make_loader(
        rows or draw_rows(),
        split=split,
        batch_size=batch_size,
        cache_dir=None,
        seed=seed,
        num_workers=workers,
        dataset_factory=draw_randomized_dataset,
        sampler_mode=sampler_mode,
        drop_last=drop_last,
    )


def _shutdown(iterator) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    del iterator
    gc.collect()


def _take(loader, batch_count: int, *, restored_mid_epoch: bool = False):
    iterator = training._iterator_after_data_restore(
        loader, restored_mid_epoch=restored_mid_epoch
    )
    images: list[torch.Tensor] = []
    draw_indices: list[torch.Tensor] = []
    patient_ids: list[str] = []
    try:
        for _ in range(batch_count):
            batch = next(iterator)
            if DRAW_KEY not in batch:
                raise AssertionError(f"Training batch omitted internal {DRAW_KEY!r}")
            images.append(torch.as_tensor(batch["image"]).detach().cpu().clone())
            draw_indices.append(
                torch.as_tensor(batch[DRAW_KEY], dtype=torch.long).cpu().clone()
            )
            patient_ids.extend(str(value) for value in batch["patient_id"])
            training._commit_training_batch(loader, batch)
        return (
            torch.cat(images),
            torch.cat(draw_indices),
            patient_ids,
            iterator,
        )
    except BaseException:
        _shutdown(iterator)
        raise


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, target) for item in value)
    return False


class TinyFrozenEncoder(TorchModule):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


class TinyFullModel(TorchModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(8, 4), nn.ReLU())
        self.classifier = nn.Linear(4, 5)

    def forward(self, images):
        return self.classifier(self.encoder(images))


def tiny_encoder_factory(**_kwargs):
    return TinyFrozenEncoder()


def tiny_full_model_factory(**_kwargs):
    return TinyFullModel()


def tiny_feature_bank_extractor(model, images, representations):
    values = images.flatten(1).mean(dim=1, keepdim=True)
    return {recipe: values.repeat(1, 4) * model.scale for recipe in representations}


def tiny_feature_dimension(_blocks: int, _use_mean: bool) -> int:
    return 4


def _assert_state_dict_equal(testcase, first, second) -> None:
    testcase.assertEqual(set(first), set(second))
    for key in first:
        testcase.assertTrue(torch.equal(first[key], second[key]), key)


def _assert_nested_equal(testcase, first, second, path: str = "root") -> None:
    if torch.is_tensor(first) or torch.is_tensor(second):
        testcase.assertTrue(torch.is_tensor(first) and torch.is_tensor(second), path)
        testcase.assertTrue(torch.equal(first, second), path)
    elif isinstance(first, Mapping) or isinstance(second, Mapping):
        testcase.assertTrue(isinstance(first, Mapping) and isinstance(second, Mapping), path)
        testcase.assertEqual(set(first), set(second), path)
        for key in first:
            _assert_nested_equal(testcase, first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        testcase.assertEqual(type(first), type(second), path)
        testcase.assertEqual(len(first), len(second), path)
        for index, (left, right) in enumerate(zip(first, second)):
            _assert_nested_equal(testcase, left, right, f"{path}[{index}]")
    else:
        testcase.assertEqual(first, second, path)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DrawIndexedWorkerResumeTests(unittest.TestCase):
    def test_full_finetune_accumulation_resume_across_workers_is_bit_exact(self):
        parameters = {
            "allow_experimental_full_finetune": True,
            "optimizer": "adamw",
            "encoder_lr": 0.01,
            "head_lr": 0.01,
            "weight_decay": 0.0,
            "batch_size": 2,
            "grad_accum_steps": 2,
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
            "model_factory": tiny_full_model_factory,
            "dataset_factory": draw_randomized_dataset,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference-full"
            interrupted = root / "interrupted-full"
            training.train_full_finetune(
                draw_rows(),
                draw_rows(),
                parameters,
                reference,
                num_workers=0,
                **common,
            )

            original_save = training.atomic_torch_save

            def stop_after_first_checkpoint(value, path):
                original_save(value, path)
                if Path(path).name == "last_checkpoint.pt" and int(
                    value.get("optimizer_step", -1)
                ) == 1:
                    raise RuntimeError("synthetic full worker interruption")

            with mock.patch.object(
                training, "atomic_torch_save", side_effect=stop_after_first_checkpoint
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic full worker interruption"
                ):
                    training.train_full_finetune(
                        draw_rows(),
                        draw_rows(),
                        parameters,
                        interrupted,
                        num_workers=2,
                        **common,
                    )

            training.train_full_finetune(
                draw_rows(),
                draw_rows(),
                parameters,
                interrupted,
                num_workers=3,
                resume=True,
                **common,
            )
            expected = torch.load(
                reference / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            actual = torch.load(
                interrupted / "last_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )

            _assert_state_dict_equal(
                self, expected["model_state_dict"], actual["model_state_dict"]
            )
            _assert_nested_equal(
                self,
                expected["optimizer_state_dict"],
                actual["optimizer_state_dict"],
                "optimizer",
            )
            self.assertEqual(expected["history"], actual["history"])
            self.assertEqual(
                actual["training_data_state"]["next_committed_draw"], 12
            )
            self.assertEqual(
                actual["training_data_state"]["worker_history"], [2, 3]
            )

    def test_linear_probe_resume_across_worker_change_is_bit_exact(self):
        parameters = {
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "batch_size": 3,
            "max_iterations": 3,
            "evaluation_interval_iterations": 1,
            "checkpoint_interval_iterations": 1,
            "class_weighting": "none",
            "mixed_precision": False,
        }
        candidates = [
            {
                "name": "worker-resume",
                "nominal_lr": 0.1,
                "lr": 0.1,
                "learning_rate_scale_rule": "none",
                "n_last_blocks": 1,
                "use_mean_patch_token": False,
                "feature_dim": 4,
            }
        ]
        common = {
            "device": torch.device("cpu"),
            "encoder_factory": tiny_encoder_factory,
            "feature_bank_extractor": tiny_feature_bank_extractor,
            "feature_dimension_fn": tiny_feature_dimension,
            "dataset_factory": draw_randomized_dataset,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            interrupted = root / "interrupted"
            training.train_linear_probe_grid(
                draw_rows(),
                draw_rows(),
                parameters,
                candidates,
                reference,
                num_workers=0,
                **common,
            )

            original_save = training.atomic_torch_save

            def stop_after_first_checkpoint(value, path):
                original_save(value, path)
                if Path(path).name == "last_checkpoint.pt" and int(
                    value.get("optimizer_step", -1)
                ) == 1:
                    raise RuntimeError("synthetic worker interruption")

            with mock.patch.object(
                training, "atomic_torch_save", side_effect=stop_after_first_checkpoint
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic worker interruption"
                ):
                    training.train_linear_probe_grid(
                        draw_rows(),
                        draw_rows(),
                        parameters,
                        candidates,
                        interrupted,
                        num_workers=2,
                        **common,
                    )

            training.train_linear_probe_grid(
                draw_rows(),
                draw_rows(),
                parameters,
                candidates,
                interrupted,
                num_workers=3,
                resume=True,
                **common,
            )
            expected = torch.load(
                reference / "last_checkpoint.pt", map_location="cpu", weights_only=False
            )
            actual = torch.load(
                interrupted / "last_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )

            _assert_state_dict_equal(
                self, expected["heads_state_dict"], actual["heads_state_dict"]
            )
            self.assertEqual(expected["history"], actual["history"])
            self.assertEqual(
                actual["training_data_state"]["worker_history"], [2, 3]
            )
            self.assertTrue(
                actual["training_data_state"]["resume_events"][-1][
                    "data_sequence_preserved"
                ]
            )

    def test_commit_rejects_missing_gapped_or_duplicate_draws(self):
        loader = _loader(0)
        iterator = iter(loader)
        try:
            batch = next(iterator)
            self.assertEqual(
                loader._threedino_stateful_sampler.next_committed_draw, 0
            )

            missing = dict(batch)
            missing.pop(DRAW_KEY)
            with self.assertRaises((KeyError, RuntimeError, ValueError)):
                training._commit_training_batch(loader, missing)

            gapped = dict(batch)
            gapped[DRAW_KEY] = torch.as_tensor(batch[DRAW_KEY]) + 1
            with self.assertRaises(RuntimeError):
                training._commit_training_batch(loader, gapped)

            training._commit_training_batch(loader, batch)
            self.assertEqual(
                loader._threedino_stateful_sampler.next_committed_draw, 3
            )
            with self.assertRaises(RuntimeError):
                training._commit_training_batch(loader, batch)
        finally:
            _shutdown(iterator)

    def test_worker_transitions_preserve_exact_draws_patients_and_tensors(self):
        reference_loader = _loader(0)
        reference_images, reference_draws, reference_patients, reference_iterator = _take(
            reference_loader, 8
        )
        _shutdown(reference_iterator)

        for first_workers, resumed_workers in ((0, 2), (2, 0), (2, 3)):
            with self.subTest(transition=f"{first_workers}->{resumed_workers}"):
                first_loader = _loader(first_workers)
                first_images, first_draws, first_patients, first_iterator = _take(
                    first_loader, 2
                )
                state = training._capture_training_data_state(first_loader)

                self.assertEqual(state["schema_version"], 2)
                self.assertEqual(state["next_committed_draw"], 6)
                self.assertEqual(
                    first_loader._threedino_stateful_sampler.next_committed_draw, 6
                )
                self.assertFalse(_contains_key(state, "issued_draw_count"))
                if first_workers > 0:
                    self.assertGreater(
                        first_loader._threedino_stateful_sampler.issued_draw_count,
                        first_loader._threedino_stateful_sampler.next_committed_draw,
                    )
                _shutdown(first_iterator)

                resumed_loader = _loader(resumed_workers)
                restored_mid_epoch = training._restore_training_data_state(
                    resumed_loader, state
                )
                resumed_images, resumed_draws, resumed_patients, resumed_iterator = _take(
                    resumed_loader, 6, restored_mid_epoch=restored_mid_epoch
                )
                _shutdown(resumed_iterator)

                self.assertTrue(
                    torch.equal(
                        reference_draws, torch.cat([first_draws, resumed_draws])
                    )
                )
                self.assertEqual(
                    reference_patients, first_patients + resumed_patients
                )
                self.assertTrue(
                    torch.equal(
                        reference_images, torch.cat([first_images, resumed_images])
                    )
                )

    def test_checkpoint_persists_only_committed_draw_not_prefetched_cursor(self):
        loader = _loader(2)
        _images, _draws, _patients, iterator = _take(loader, 1)
        sampler = loader._threedino_stateful_sampler
        state = training._capture_training_data_state(loader)
        try:
            self.assertIsInstance(sampler, training.DrawIndexedSampler)
            self.assertEqual(state["next_committed_draw"], 3)
            self.assertEqual(sampler.next_committed_draw, 3)
            self.assertGreater(sampler.issued_draw_count, sampler.next_committed_draw)
            self.assertFalse(_contains_key(state, "issued_draw_count"))
            self.assertTrue(state["sequence_schema"])
            self.assertTrue(state["augmentation_seed_schema"])
        finally:
            _shutdown(iterator)

    def test_finite_drop_last_resumes_at_clean_epoch_boundary(self):
        rows = draw_rows(5)
        first = _loader(2, rows=rows, batch_size=4, drop_last=True)
        first_batch = next(iter(first))
        training._commit_training_batch(first, first_batch)
        state = training._capture_training_data_state(first)
        self.assertEqual(state["next_committed_draw"], 4)
        self.assertFalse(first._threedino_stateful_sampler.mid_epoch)

        expected_batch = next(iter(first))
        resumed = _loader(3, rows=rows, batch_size=4, drop_last=True)
        restored_mid_epoch = training._restore_training_data_state(resumed, state)
        self.assertFalse(restored_mid_epoch)
        actual_batch = next(
            training._iterator_after_data_restore(
                resumed, restored_mid_epoch=restored_mid_epoch
            )
        )

        self.assertTrue(
            torch.equal(
                torch.as_tensor(expected_batch[DRAW_KEY]),
                torch.as_tensor(actual_batch[DRAW_KEY]),
            )
        )
        self.assertEqual(
            list(expected_batch["patient_id"]), list(actual_batch["patient_id"])
        )
        self.assertTrue(
            torch.equal(expected_batch["image"], actual_batch["image"])
        )
        self.assertEqual(
            torch.as_tensor(actual_batch[DRAW_KEY]).tolist(), [4, 5, 6, 7]
        )

    def test_resume_accepts_worker_change_but_rejects_semantic_mutations(self):
        source = _loader(0)
        _images, _draws, _patients, iterator = _take(source, 1)
        state = training._capture_training_data_state(source)
        _shutdown(iterator)

        changed_workers = _loader(2)
        self.assertTrue(training._restore_training_data_state(changed_workers, state))
        self.assertEqual(changed_workers.num_workers, 2)

        cases = {
            "legacy_schema": (_loader(0), {"schema_version": 1}),
            "dataset_size": (_loader(0, rows=draw_rows(30)), None),
            "batch_size": (_loader(0, batch_size=4), None),
            "seed": (_loader(0, seed=130), None),
            "sampler": (_loader(0, sampler_mode="sharded_infinite"), None),
            "drop_last": (_loader(0, drop_last=True), None),
            "sequence_schema": (_loader(0), {"sequence_schema": "mutated"}),
            "augmentation_seed_schema": (
                _loader(0),
                {"augmentation_seed_schema": "mutated"},
            ),
        }
        for name, (loader, mutation) in cases.items():
            with self.subTest(mutation=name):
                candidate = copy.deepcopy(state)
                if mutation:
                    candidate.update(mutation)
                with self.assertRaises(RuntimeError):
                    training._restore_training_data_state(loader, candidate)

        conflicting = copy.deepcopy(state)
        conflicting["next_committed_draw"] += 1
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            training._restore_training_data_state(_loader(0), conflicting)

        illegal = copy.deepcopy(state)
        illegal_cursor = 4
        for document in (illegal, illegal["sampler_state"]):
            document["next_committed_draw"] = illegal_cursor
            document["next_data_epoch_index"] = 0
            document["next_sample_offset_in_epoch"] = illegal_cursor
            document["last_committed_data_epoch"] = 1
        with self.assertRaisesRegex(RuntimeError, "legal batch boundary"):
            training._restore_training_data_state(_loader(0), illegal)

    def test_unattested_custom_dataset_cannot_claim_exact_resume(self):
        class HiddenPrivateRngDataset:
            def __init__(self, source_rows):
                self.rows = list(source_rows)
                self.private_rng = np.random.default_rng(7)

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = dict(self.rows[index])
                row["image"] = row["image"] + float(self.private_rng.random())
                return row

        with self.assertRaisesRegex(RuntimeError, "draw-seed-v2"):
            training._make_loader(
                draw_rows(),
                split="train",
                batch_size=3,
                cache_dir=None,
                seed=129,
                num_workers=0,
                dataset_factory=lambda source_rows, **_: HiddenPrivateRngDataset(
                    source_rows
                ),
            )

    def test_validation_and_test_workers_are_fixed_to_zero(self):
        training_loader = _loader(2)
        self.assertEqual(training_loader.num_workers, 2)
        self.assertTrue(training_loader.persistent_workers)

        default_training_loader = _loader(0)
        self.assertEqual(default_training_loader.num_workers, 0)
        self.assertFalse(default_training_loader.persistent_workers)

        for split in ("validation", "test"):
            with self.subTest(split=split):
                loader = _loader(3, split=split)
                self.assertEqual(loader.num_workers, 0)
                self.assertFalse(loader.persistent_workers)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DefaultTransformDrawSeedTests(unittest.TestCase):
    def test_independent_monai_transforms_match_for_same_draw_seed(self):
        try:
            from Code.Models.ThreeDINO.data import make_post_cache_transform
        except ImportError as exc:  # pragma: no cover - optional environment
            self.skipTest(f"MONAI transform stack is unavailable: {exc}")

        first = make_post_cache_transform("train")
        second = make_post_cache_transform("train")
        first.set_random_state(seed=987_654_321)
        second.set_random_state(seed=987_654_321)
        source = {
            "image": torch.linspace(-1.0, 1.0, 144 * 144 * 112).reshape(
                1, 144, 144, 112
            )
        }

        first_image = first(dict(source))["image"]
        second_image = second(dict(source))["image"]

        self.assertEqual(tuple(first_image.shape), (1, 112, 112, 112))
        self.assertTrue(torch.equal(first_image, second_image))


if __name__ == "__main__":
    unittest.main()
