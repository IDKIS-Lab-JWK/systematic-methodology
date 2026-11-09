"""Synthetic, download-free checks for the 3DINO adapter and cache plan."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent skip
    torch = None
    nn = None

from Code.Models.ThreeDINO import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    CLASS_NAMES,
    DEFAULT_CACHE_NAMESPACE,
    UPSTREAM_REVISION,
)
from Code.Models.ThreeDINO import cache, data, model  # noqa: E402

TorchModule = nn.Module if nn is not None else object


def make_manifest() -> dict:
    folds = []
    for fold_number in range(1, 6):
        folds.append(
            {
                "fold": fold_number,
                "patients_by_class": {
                    name: [f"{index}{fold_number:03d}"]
                    for index, name in enumerate(CLASS_NAMES, start=1)
                },
            }
        )
    return {
        "heldout_test": {
            "patients_by_class": {
                name: [f"{index}900"]
                for index, name in enumerate(CLASS_NAMES, start=1)
            },
            "counts_by_class": {name: 1 for name in CLASS_NAMES},
            "total": 5,
        },
        "cross_validation": {"number_of_folds": 5, "folds": folds},
        "metadata": {"percentage": 100},
    }


def populate_dataset(root: Path, manifest: dict) -> Path:
    for class_name, patient_ids in data.all_patient_mapping(manifest).items():
        folder = root / "NIFTI" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for patient_id in patient_ids:
            (folder / f"{patient_id}.nii.gz").write_bytes(b"fake-nifti")
    split_dir = root / "Splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "100_percent.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def make_single_entry_cache_plan(workspace: Path) -> cache.CachePlan:
    dataset_root = workspace / "Dataset"
    source = dataset_root / "NIFTI" / CLASS_NAMES[0] / "1001.nii.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic-nifti-source")
    stat = source.stat()
    manifest_path = dataset_root / "Splits" / "synthetic.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    split_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    directory = cache.cache_directory(
        workspace / "Results" / "3DINO" / "Cache", DEFAULT_CACHE_NAMESPACE
    )
    entry = {
        "image": str(source.resolve()),
        "source_path": str(source.resolve()),
        "label": 0,
        "label_str": CLASS_NAMES[0],
        "patient_id": "1001",
        "accession": "1001",
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    entry["cache_key"] = data.persistent_cache_key(entry)
    entry["cache_path"] = str(
        directory / "persistent" / f"{entry['cache_key']}.pt"
    )
    contract = data.preprocessing_contract()
    fingerprint = cache._canonical_sha256(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "namespace": DEFAULT_CACHE_NAMESPACE,
            "split_manifest_sha256": split_sha256,
            "contract": contract,
            "sources": [entry],
        }
    )
    return cache.CachePlan(
        namespace=DEFAULT_CACHE_NAMESPACE,
        cache_directory=directory,
        dataset_root=dataset_root.resolve(),
        split_manifest_path=manifest_path.resolve(),
        split_manifest_sha256=split_sha256,
        split_percentage=None,
        sources=(entry,),
        contract=contract,
        created_at_utc="2026-01-01T00:00:00+00:00",
        plan_fingerprint=fingerprint,
    )


def valid_cache_payload(entry: dict) -> dict:
    image = torch.zeros(1, 1, 1, 1, dtype=torch.float32).expand(
        1, 144, 144, 112
    )
    return {
        "image": image,
        "patient_id": entry["patient_id"],
        "label": entry["label"],
        "label_str": entry["label_str"],
    }


def synthetic_cache_factory(payload_builder=valid_cache_payload):
    class SyntheticCacheDataset:
        def __init__(self, rows, cache_dir):
            self.rows = list(rows)
            self.cache_dir = Path(cache_dir)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            entry = self.rows[index]
            artifact = Path(entry["cache_path"])
            artifact.parent.mkdir(parents=True, exist_ok=True)
            payload = payload_builder(entry)
            torch.save(payload, artifact)
            return payload

    def factory(rows, *, cache_dir):
        return SyntheticCacheDataset(rows, cache_dir)

    return factory


def reseal_artifact_manifest(manifest: dict, artifact: Path) -> None:
    key = manifest["entries"][0]["cache_key"]
    manifest["artifact_integrity"]["artifacts"][key] = {
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }
    section = manifest["artifact_integrity"]
    section["seal_sha256"] = cache._artifact_integrity_seal(
        section, plan_fingerprint=manifest["plan_fingerprint"]
    )


class CachePlanTests(unittest.TestCase):
    def test_plan_is_read_only_isolated_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            dataset_root = workspace / "Dataset"
            manifest_path = populate_dataset(dataset_root, make_manifest())
            cache_root = workspace / "Results" / "3DINO" / "Cache"
            first = cache.create_cache_plan(
                dataset_root=dataset_root,
                split_manifest_path=manifest_path,
                cache_root=cache_root,
            )
            second = cache.build_cache_plan(
                dataset_root=dataset_root,
                split_manifest_path=manifest_path,
                cache_root=cache_root,
            )
            self.assertEqual(first.namespace, DEFAULT_CACHE_NAMESPACE)
            self.assertEqual(first.plan_fingerprint, second.plan_fingerprint)
            self.assertEqual(len(first.sources), 30)
            self.assertFalse(first.cache_directory.exists())
            self.assertFalse(first.to_dict()["isolation"]["reuses_merlin_cache"])
            self.assertFalse(first.to_dict()["isolation"]["reuses_pillar0_cache"])
            self.assertFalse(
                first.to_dict()["isolation"]["stochastic_training_transforms_cached"]
            )

    def test_namespace_cannot_escape_cache_root(self):
        with self.assertRaisesRegex(ValueError, "namespace"):
            cache.cache_directory(Path("cache"), "../Merlin")


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class CacheArtifactIntegrityTests(unittest.TestCase):
    def _prepare(self, workspace: Path, payload_builder=valid_cache_payload):
        plan = make_single_entry_cache_plan(workspace)
        result = cache.prepare_cache(
            plan, dataset_factory=synthetic_cache_factory(payload_builder)
        )
        manifest = cache.load_cache_manifest(plan.manifest_path)
        return plan, result, manifest

    def test_prepare_records_hash_size_contract_and_preserves_plan_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, result, manifest = self._prepare(Path(temporary))
            entry = manifest["entries"][0]
            artifact = Path(entry["cache_path"])
            integrity = manifest["artifact_integrity"]
            record = integrity["artifacts"][entry["cache_key"]]

            self.assertEqual(manifest["plan_fingerprint"], plan.plan_fingerprint)
            self.assertEqual(record["size_bytes"], artifact.stat().st_size)
            self.assertEqual(
                record["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            self.assertEqual(
                integrity["payload_contract"]["image_shape"], [1, 144, 144, 112]
            )
            self.assertTrue(result["artifacts_verified"])
            self.assertEqual(result["validation_scope"], "exhaustive_hash_and_payload")

            structural = cache.validate_cache(
                plan.manifest_path, verify_artifacts=False
            )
            self.assertFalse(structural["artifacts_verified"])
            self.assertEqual(structural["validation_scope"], "structural_only")

    def test_same_size_corruption_and_truncation_fail_exhaustive_validation(self):
        mutations = {
            "same-size corruption": lambda raw: bytes(
                bytearray(raw[: len(raw) // 2])
                + bytearray([raw[len(raw) // 2] ^ 1])
                + bytearray(raw[len(raw) // 2 + 1 :])
            ),
            "truncation": lambda raw: raw[: max(1, len(raw) // 2)],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                plan, _, manifest = self._prepare(Path(temporary))
                artifact = Path(manifest["entries"][0]["cache_path"])
                original_stat = artifact.stat()
                artifact.write_bytes(mutate(artifact.read_bytes()))
                os.utime(
                    artifact,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                with self.assertRaisesRegex(
                    cache.CacheValidationError, "size mismatch|SHA-256 mismatch"
                ):
                    cache.validate_cache(plan.manifest_path, verify_artifacts=True)

    def test_resealed_invalid_bytes_and_identity_mismatch_still_fail_payload_check(self):
        cases = {
            "not deserializable": (
                lambda entry: b"not-a-torch-payload",
                "not a deserializable torch payload",
            ),
            "patient mismatch": (
                lambda entry: {
                    **valid_cache_payload(entry),
                    "patient_id": "different-patient",
                },
                "patient_id does not match",
            ),
            "label mismatch": (
                lambda entry: {**valid_cache_payload(entry), "label": 1},
                "label does not match",
            ),
        }
        for name, (payload_builder, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                plan, _, manifest = self._prepare(Path(temporary))
                entry = manifest["entries"][0]
                artifact = Path(entry["cache_path"])
                payload = payload_builder(entry)
                if isinstance(payload, bytes):
                    artifact.write_bytes(payload)
                else:
                    torch.save(payload, artifact)
                reseal_artifact_manifest(manifest, artifact)
                plan.manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                cache.load_cache_manifest(plan.manifest_path)
                with self.assertRaisesRegex(cache.CacheValidationError, message):
                    cache.validate_cache(plan.manifest_path, verify_artifacts=True)

    def test_shape_dtype_and_finite_payload_contract_is_enforced_while_sealing(self):
        cases = {
            "shape": (
                lambda entry: {
                    **valid_cache_payload(entry),
                    "image": torch.zeros(1, 8, 8, 8),
                },
                "image shape mismatch",
            ),
            "integer dtype": (
                lambda entry: {
                    **valid_cache_payload(entry),
                    "image": torch.zeros(1, 1, 1, 1, dtype=torch.int16).expand(
                        1, 144, 144, 112
                    ),
                },
                "not floating point",
            ),
            "non-finite": (
                lambda entry: {
                    **valid_cache_payload(entry),
                    "image": torch.full(
                        (1, 1, 1, 1), float("nan"), dtype=torch.float32
                    ).expand(1, 144, 144, 112),
                },
                "non-finite",
            ),
            "non-mapping": (lambda entry: [entry["patient_id"]], "must be a mapping"),
        }
        for name, (payload_builder, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                plan = make_single_entry_cache_plan(Path(temporary))
                with self.assertRaisesRegex(cache.CacheValidationError, message):
                    cache.prepare_cache(
                        plan,
                        dataset_factory=synthetic_cache_factory(payload_builder),
                    )
                self.assertFalse(plan.manifest_path.exists())

    def test_missing_integrity_or_tampered_integrity_seal_fails_closed(self):
        cases = ("missing", "tampered")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                plan, _, manifest = self._prepare(Path(temporary))
                if case == "missing":
                    del manifest["artifact_integrity"]
                    expected = "lacks artifact-integrity metadata"
                else:
                    key = manifest["entries"][0]["cache_key"]
                    manifest["artifact_integrity"]["artifacts"][key]["sha256"] = (
                        "0" * 64
                    )
                    expected = "metadata seal mismatch"
                plan.manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    (cache.CacheCompatibilityError, cache.CacheValidationError),
                    expected,
                ):
                    cache.load_cache_manifest(plan.manifest_path)

    def test_source_content_change_is_detected_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, manifest = self._prepare(Path(temporary))
            source = Path(manifest["entries"][0]["source_path"])
            original_stat = source.stat()
            source.write_bytes(b"x" * original_stat.st_size)
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(
                cache.CacheCompatibilityError, "Source SHA-256 changed"
            ):
                cache.validate_cache(plan.manifest_path, verify_artifacts=True)

    def test_tampered_resume_plan_is_rejected_before_dataset_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_single_entry_cache_plan(Path(temporary))
            plan.cache_directory.mkdir(parents=True)
            document = plan.to_dict()
            document["contract"]["intensity_range"] = [-2.0, 2.0]
            plan.plan_path.write_text(json.dumps(document), encoding="utf-8")
            factory = mock.Mock(side_effect=AssertionError("factory must not run"))
            with self.assertRaisesRegex(
                cache.CacheCompatibilityError,
                "Cache-plan content does not match its fingerprint",
            ):
                cache.prepare_cache(plan, dataset_factory=factory)
            factory.assert_not_called()

    def test_undeclared_tensor_and_changed_requested_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, manifest = self._prepare(Path(temporary))
            extras = (
                plan.data_directory / "undeclared.pt",
                plan.data_directory / "nested" / "undeclared.PT",
            )
            for extra in extras:
                with self.subTest(extra=str(extra)):
                    extra.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"image": torch.zeros(1)}, extra)
                    with self.assertRaisesRegex(
                        cache.CacheValidationError, "undeclared .pt artifacts"
                    ):
                        cache.validate_cache(
                            plan.manifest_path, verify_artifacts=True
                        )
                    extra.unlink()
                    if extra.parent != plan.data_directory:
                        extra.parent.rmdir()

            invalid_directory = plan.data_directory / "directory.pt"
            invalid_directory.mkdir()
            with self.assertRaisesRegex(
                cache.CacheValidationError, "undeclared .pt artifacts"
            ):
                cache.validate_cache(plan.manifest_path, verify_artifacts=True)
            invalid_directory.rmdir()

            requested = dict(manifest["entries"][0])
            requested["patient_id"] = "different-patient"
            with self.assertRaisesRegex(
                cache.CacheCompatibilityError, "different patient/class identity"
            ):
                cache.make_cached_dataset(
                    plan.manifest_path, [requested], split="validation"
                )

    def test_in_memory_manifest_gets_same_structural_checks_as_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, manifest = self._prepare(Path(temporary))
            del manifest["complete"]
            with self.assertRaisesRegex(
                cache.CacheValidationError, "not marked complete"
            ):
                cache.validate_cache(manifest, verify_artifacts=False)

    def test_copied_manifest_is_bound_to_its_embedded_cache_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            plan, _, manifest = self._prepare(workspace)
            relocated = (
                plan.cache_directory.parent
                / "relocated-cache-v2"
                / "cache_manifest.json"
            )
            relocated.parent.mkdir(parents=True)
            relocated.write_bytes(plan.manifest_path.read_bytes())

            for operation in (
                lambda: cache.load_cache_manifest(relocated),
                lambda: cache.validate_cache(
                    relocated, verify_artifacts=False
                ),
            ):
                with self.subTest(operation=operation), self.assertRaisesRegex(
                    cache.CacheCompatibilityError,
                    "outside its embedded cache_directory",
                ):
                    operation()

            # In-memory documents have no physical JSON location to bind, but
            # retain all embedded namespace/data-directory invariants.
            structural = cache.validate_cache(
                manifest, verify_artifacts=False
            )
            self.assertFalse(structural["artifacts_verified"])

    def test_in_memory_manifest_still_enforces_embedded_directory_invariants(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, manifest = self._prepare(Path(temporary))
            cases = {
                "data directory": (
                    "data_directory",
                    str(Path(manifest["cache_directory"]) / "other"),
                    "cache_directory/persistent",
                ),
                "namespace": (
                    "namespace",
                    "different-namespace-v2",
                    "does not match cache_directory name",
                ),
            }
            for name, (field, value, message) in cases.items():
                with self.subTest(name=name):
                    altered = json.loads(json.dumps(manifest))
                    altered[field] = value
                    with self.assertRaisesRegex(
                        cache.CacheCompatibilityError, message
                    ):
                        cache.validate_cache(
                            altered, verify_artifacts=False
                        )

    def test_successful_payload_validation_is_memoized_but_hash_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, manifest = self._prepare(Path(temporary))
            with mock.patch.object(
                cache,
                "_load_torch_payload",
                side_effect=AssertionError("memoized payload must not deserialize"),
            ) as loader:
                result = cache.validate_cache(
                    plan.manifest_path, verify_artifacts=True
                )
            loader.assert_not_called()
            self.assertTrue(result["artifacts_verified"])

            artifact = Path(manifest["entries"][0]["cache_path"])
            original_stat = artifact.stat()
            raw = bytearray(artifact.read_bytes())
            raw[len(raw) // 2] ^= 1
            artifact.write_bytes(raw)
            os.utime(
                artifact,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(
                cache.CacheValidationError, "SHA-256 mismatch"
            ):
                cache.validate_cache(plan.manifest_path, verify_artifacts=True)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ModelAdapterTests(unittest.TestCase):
    class DummyEncoder(TorchModule):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.forward_calls = 0

        def get_intermediate_layers(
            self, images, n, reshape=False, return_class_token=True, norm=True
        ):
            del reshape, return_class_token, norm
            self.forward_calls += 1
            outputs = []
            for index in range(int(n)):
                patch = self.scale * torch.ones(
                    images.shape[0], 3, 1024, device=images.device
                ) * (index + 1)
                cls = self.scale * torch.ones(
                    images.shape[0], 1024, device=images.device
                ) * (index + 1)
                outputs.append((patch, cls))
            return outputs

    def test_all_four_publication_representations_share_one_encoder_call(self):
        encoder = self.DummyEncoder()
        images = torch.zeros(2, 1, 4, 4, 4)
        bank = model.extract_representation_bank(encoder, images)
        self.assertEqual(encoder.forward_calls, 1)
        self.assertEqual(tuple(bank[(1, False)].shape), (2, 1024))
        self.assertEqual(tuple(bank[(1, True)].shape), (2, 2048))
        self.assertEqual(tuple(bank[(4, False)].shape), (2, 4096))
        self.assertEqual(tuple(bank[(4, True)].shape), (2, 5120))
        self.assertEqual(model.feature_dimension(4, True), 5120)

    def test_linear_classifier_freezes_encoder_and_trains_only_head(self):
        encoder = self.DummyEncoder()
        classifier = model.ThreeDINOClassifier(
            encoder,
            num_classes=5,
            n_last_blocks=4,
            use_mean_patch_token=True,
            freeze_encoder=True,
        )
        classifier.train()
        output = classifier(torch.zeros(2, 1, 4, 4, 4))
        output.sum().backward()
        report = model.trainable_parameter_report(classifier)
        self.assertEqual(tuple(output.shape), (2, 5))
        self.assertFalse(encoder.training)
        self.assertIsNone(encoder.scale.grad)
        self.assertTrue(
            all(name.startswith("classifier") for name in report["trainable_parameter_names"])
        )

    def test_full_finetune_requires_explicit_authorization(self):
        with self.assertRaisesRegex(PermissionError, "disabled"):
            model.build_model(
                "full_finetune",
                5,
                {"encoder": self.DummyEncoder()},
                "cpu",
            )
        adapted, details = model.build_model(
            "full_finetune",
            5,
            {
                "encoder": self.DummyEncoder(),
                "allow_experimental_full_finetune": True,
            },
            "cpu",
        )
        self.assertFalse(details["encoder_frozen"])
        self.assertTrue(all(parameter.requires_grad for parameter in adapted.encoder.parameters()))

    def test_single_channel_shape_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "one input channel"):
            model.extract_intermediate_features(
                self.DummyEncoder(), torch.zeros(1, 3, 4, 4, 4)
            )


class LocalSourceValidationTests(unittest.TestCase):
    def test_structural_validator_and_local_checkpoint_resolver_are_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "upstream"
            required = (
                "LICENSE",
                "dinov2/__init__.py",
                "dinov2/configs/__init__.py",
                "dinov2/configs/train/vit3d_highres.yaml",
                "dinov2/models/__init__.py",
                "dinov2/models/vision_transformer.py",
            )
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic", encoding="utf-8")
            source = model.validate_upstream_checkout(root, UPSTREAM_REVISION)
            self.assertFalse(source["git_invoked"])
            self.assertFalse(source["source_vendored"])

            checkpoint = Path(temporary) / "3dino_vit_weights.pth"
            payload = b"synthetic-checkpoint"
            checkpoint.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            resolved = model.resolve_checkpoint(
                checkpoint, expected_sha256=expected
            )
            self.assertEqual(resolved.sha256, expected)
            local_identity = resolved.to_dict()
            self.assertFalse(local_identity["download_performed"])
            self.assertEqual(local_identity["filename"], checkpoint.name)
            self.assertNotIn("repo_id", local_identity)
            self.assertNotIn("revision", local_identity)
            with self.assertRaisesRegex(model.CheckpointValidationError, "mismatch"):
                model.resolve_checkpoint(checkpoint, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
