"""Regression contracts for the Merlin-style 3DINO grid workflow.

The orchestration tests in this module are intentionally synthetic.  They mock
patient splitting and training, create only tiny text/JSON artifacts, and never
load 3DINO weights or medical volumes.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

from Code.Models.ThreeDINO import CLASS_NAMES  # noqa: E402
from Code.Models.ThreeDINO import orchestrator, training  # noqa: E402
from Code.Models.ThreeDINO.cli import build_parser  # noqa: E402


CANDIDATE_FOLD_ARTIFACTS = {
    "parameters.json",
    "best_model.pt",
    "training_state.json",
    "metrics_summary.json",
    "validation_patient_predictions.json",
    "validation_probabilities.csv",
    "shared_training_reference.json",
}


def make_args(root: Path, **overrides) -> SimpleNamespace:
    values = {
        "dataset_root": root / "Dataset",
        "results_root": root / "Results" / "3DINO",
        "cache_root": root / "Results" / "3DINO" / "Cache",
        "cache_dir": root / "Results" / "3DINO" / "Cache" / "Shared" / "synthetic",
        "cache_namespace": "synthetic",
        "upstream_root": root / "upstream",
        "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
        "pretrained_checkpoint": root / "weights" / "3dino_vit_weights.pth",
        "pretrained_checkpoint_sha256": "a" * 64,
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
        "resume": True,
        "dry_run": False,
        "stage": "grid",
        "percentage": 100,
        "parameter_source": "publication",
        "parameter_config": None,
        "grid_config": None,
        "max_grid_candidates": None,
        "model_source": "refit_full_development",
        "checkpoint": None,
        "force_test": False,
        "allow_experimental_full_finetune": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def validation_metrics(score: float) -> dict:
    return {
        "accuracy": score,
        "balanced_accuracy": score,
        "f1_macro": score,
        "average_precision_macro": score,
        "selection_score": score,
        "per_class": {
            name: {"recall": score, "sensitivity": score}
            for name in CLASS_NAMES
        },
    }


def fake_shared_grid_training(
    _train_rows,
    _validation_rows,
    _base_parameters,
    candidates,
    output_dir,
    **_kwargs,
):
    """Create the native tiny outputs produced by one shared-head fold run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_provenance = {
        "loader": "synthetic_test_factory",
        "upstream_revision": "synthetic-upstream-revision",
        "checkpoint_sha256": "f" * 64,
    }
    runtime_plan = {
        "method": "linear_probe",
        "max_iterations": 1,
        "scheduler_total_updates": 1,
        "training_patient_count": len(_train_rows),
        "training_rows_fingerprint": orchestrator._training_rows_fingerprint(
            _train_rows
        ),
    }
    best_records = {}
    for index, candidate in enumerate(candidates):
        identity = str(candidate["candidate_id"])
        score = 0.80 - index / 1000.0
        best_records[identity] = {
            "candidate_id": identity,
            "candidate": dict(candidate),
            "optimizer_step": 1,
            "epoch": 1,
            "metrics": validation_metrics(score),
            "checkpoint": str(output / "candidates" / identity / "best_model.pt"),
        }
    orchestrator.save_json(
        output / "training_state.json",
        {
            "schema_version": training.SCHEMA_VERSION,
            "model_name": "3dino_linear_probe",
            "model_family": "3DINO",
            "method": "linear_probe",
            "completed": True,
            "encoder_frozen": True,
            "shared_encoder_pass_per_candidate_bank": True,
            "candidate_count": len(candidates),
            "candidate_best_records": best_records,
            "candidates": [dict(candidate) for candidate in candidates],
            "parameters": dict(_base_parameters),
            "parameters_fingerprint": training.parameter_fingerprint(
                _base_parameters
            ),
            "runtime_plan": runtime_plan,
            "runtime_plan_fingerprint": training._runtime_plan_fingerprint(
                _base_parameters, runtime_plan
            ),
            "classes": list(CLASS_NAMES),
            "source_provenance": source_provenance,
            "seed": 42,
        },
    )
    torch.save(
        {
            "schema_version": training.SCHEMA_VERSION,
            "model_name": "3dino_linear_probe",
            "model_family": "3DINO",
            "method": "linear_probe",
            "checkpoint_scope": "linear_heads_only_frozen_encoder_reconstructed",
            "encoder_frozen": True,
            "base_parameters": dict(_base_parameters),
            "parameters": dict(_base_parameters),
            "candidates": [dict(candidate) for candidate in candidates],
            "classes": list(CLASS_NAMES),
            "source_provenance": source_provenance,
            "runtime_plan": runtime_plan,
            "best_records": best_records,
            "candidate_head_state_dicts": {
                str(candidate["candidate_id"]): {} for candidate in candidates
            },
            "optimizer_step": 1,
            "epoch": 1,
        },
        output / "last_checkpoint.pt",
    )
    (output / "best_model.pt").write_bytes(
        (output / "last_checkpoint.pt").read_bytes()
    )
    selected_metrics = next(iter(best_records.values()))["metrics"]
    orchestrator.save_json(
        output / "metrics_summary.json",
        {
            "model_family": "3DINO",
            "method": "linear_probe",
            "validation_metrics": selected_metrics,
            "candidate_best_records": best_records,
        },
    )
    orchestrator.save_json(
        output / "validation_patient_predictions.json",
        [{"patient_id": "validation", "prediction": CLASS_NAMES[0]}],
    )
    (output / "validation_probabilities.csv").write_text(
        "patient_id,Normal\nvalidation,1.0\n", encoding="utf-8"
    )
    records = {}
    for candidate in candidates:
        identity = str(candidate["candidate_id"])
        metrics = best_records[identity]["metrics"]
        candidate_dir = output / "candidates" / identity
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_parameters = dict(candidate)
        candidate_parameters_fingerprint = orchestrator.canonical_sha256(
            candidate_parameters
        )
        candidate_runtime_plan = {
            **runtime_plan,
            "candidate_id": identity,
            "candidate_parameters_fingerprint": candidate_parameters_fingerprint,
            "joint_candidate_count": len(candidates),
            "candidate_execution": "joint_shared_encoder_multi_head",
            "head_initialization_schema": training.PROBE_HEAD_INITIALIZATION_SCHEMA,
        }
        candidate_runtime_fingerprint = training._runtime_plan_fingerprint(
            candidate_parameters, candidate_runtime_plan
        )
        torch.save(
            {
                "schema_version": training.SCHEMA_VERSION,
                "model_family": "3DINO",
                "method": "linear_probe",
                "checkpoint_scope": "single_linear_head",
                "encoder_frozen": True,
                "candidate": dict(candidate),
                "candidate_id": identity,
                "parameters": candidate_parameters,
                "parameters_fingerprint": candidate_parameters_fingerprint,
                "runtime_plan": candidate_runtime_plan,
                "runtime_plan_fingerprint": candidate_runtime_fingerprint,
                "classes": list(CLASS_NAMES),
                "source_provenance": source_provenance,
                "optimizer_step": 1,
                "epoch": 1,
                "head_state_dict": {},
                "validation_metrics": metrics,
            },
            candidate_dir / "best_model.pt",
        )
        orchestrator.save_json(
            candidate_dir / "training_state.json",
            {
                "schema_version": training.SCHEMA_VERSION,
                "model_name": "3dino_linear_probe",
                "model_family": "3DINO",
                "method": "linear_probe",
                "completed": True,
                "encoder_frozen": True,
                "candidate_id": identity,
                "parameters": candidate_parameters,
                "parameters_fingerprint": candidate_parameters_fingerprint,
                "runtime_plan": candidate_runtime_plan,
                "runtime_plan_fingerprint": candidate_runtime_fingerprint,
                "classes": list(CLASS_NAMES),
                "source_provenance": source_provenance,
                "seed": 42,
                "best_optimizer_step": 1,
                "validation_metrics": metrics,
            },
        )
        orchestrator.save_json(candidate_dir / "metrics_summary.json", metrics)
        orchestrator.save_json(
            candidate_dir / "validation_patient_predictions.json",
            [{"patient_id": "validation", "prediction": CLASS_NAMES[0]}],
        )
        (candidate_dir / "validation_probabilities.csv").write_text(
            "patient_id,Normal\nvalidation,1.0\n", encoding="utf-8"
        )
        records[identity] = best_records[identity]
    return {
        "state": {"completed": True},
        "candidates": records,
        "best_candidate_id": str(candidates[0]["candidate_id"]),
    }


def write_project_grid(path: Path) -> None:
    document = {
        "schema_version": 1,
        "method": "linear_probe",
        "profile_type": "project_defined_grid",
        "strategy": "coupled_candidates",
        "fixed_parameters": {
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "loss": "cross_entropy",
            "batch_size": 32,
            "max_iterations": 1,
            "evaluation_interval_iterations": 1,
            "learning_rate_scale_rule": "none",
            "encoder_frozen": True,
        },
        "candidates": [
            {
                "nominal_lr": 0.01,
                "n_last_blocks": 1,
                "use_mean_patch_token": False,
            },
            {
                "nominal_lr": 0.02,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
            },
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def run_synthetic_grid(
    args: SimpleNamespace, execution_context: dict | None = None
):
    """Run only the orchestrator around the tiny shared-head artifact factory."""

    fold_counter = iter(range(1, 6))

    def split(_manifest, _fold):
        fold = next(fold_counter)
        return {"kind": "train", "fold": fold}, {
            "kind": "validation",
            "fold": fold,
        }

    def rows(_args, mapping):
        return [
            {
                "patient_id": f"{mapping['kind']}-{mapping['fold']}",
                "label": 0,
                "label_str": CLASS_NAMES[0],
            }
        ]

    with (
        mock.patch.object(orchestrator, "cross_validation_split", side_effect=split),
        mock.patch.object(orchestrator, "_split_rows", side_effect=rows),
        mock.patch.object(orchestrator, "validate_no_overlap"),
        mock.patch.object(orchestrator, "_common_training_kwargs", return_value={}),
        mock.patch.object(
            orchestrator,
            "train_linear_probe_grid",
            side_effect=fake_shared_grid_training,
        ) as train_mock,
    ):
        selected, provenance = orchestrator.run_search(
            args,
            "linear_probe",
            {},
            source=args.parameter_source,
            execution_context=execution_context,
        )
    return selected, provenance, train_mock


class GridCliContractTests(unittest.TestCase):
    def test_documented_names_are_canonical_and_search_names_are_aliases(self):
        parser = build_parser("linear_probe")
        canonical = parser.parse_args(
            [
                "--stage",
                "grid",
                "--parameter_source",
                "grid",
                "--grid_config",
                "grid.json",
                "--max_grid_candidates",
                "3",
            ]
        )
        legacy = parser.parse_args(
            [
                "--stage",
                "search",
                "--parameter_source",
                "search",
                "--search_config",
                "grid.json",
                "--max_search_candidates",
                "3",
            ]
        )
        self.assertEqual(canonical.stage, "grid")
        self.assertEqual(legacy.stage, "grid")
        self.assertEqual(canonical.parameter_source, "grid")
        self.assertEqual(legacy.parameter_source, "grid")
        self.assertEqual(canonical.grid_config, legacy.grid_config)
        self.assertEqual(canonical.max_grid_candidates, 3)
        self.assertEqual(legacy.max_grid_candidates, 3)

    def test_publication_rejects_custom_grid_and_candidate_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                {"grid_config": root / "replacement.json"},
                {"max_grid_candidates": 3},
            ]
            for override in cases:
                with self.subTest(override=override):
                    args = make_args(root, **override)
                    with self.assertRaisesRegex(
                        ValueError, "publication.*(grid|candidate)|complete publication"
                    ):
                        orchestrator.run_search(
                            args,
                            "linear_probe",
                            {},
                            source="publication",
                        )
            self.assertFalse((root / "Results").exists())


class MerlinStyleGridLayoutTests(unittest.TestCase):
    def _run_grid(self, args: SimpleNamespace):
        return run_synthetic_grid(args)

    def test_publication_uses_complete_shared_52_head_grid_and_merlin_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root)
            selected, provenance, train_mock = self._run_grid(args)

            grid_root = (
                args.results_root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            self.assertEqual(train_mock.call_count, 5)
            self.assertTrue(
                all(len(call.args[3]) == 52 for call in train_mock.call_args_list)
            )
            self.assertTrue(grid_root.is_dir())
            self.assertFalse(
                any(path.name.startswith("round_") for path in grid_root.rglob("*"))
            )
            for fold in range(1, 6):
                self.assertTrue(
                    (grid_root / "shared_training" / f"fold_{fold:02d}").is_dir()
                )

            candidates = sorted(grid_root.glob("candidate_*"))
            self.assertEqual(len(candidates), 52)
            for candidate_dir in candidates:
                summary = orchestrator.read_json(
                    candidate_dir / "candidate_summary.json"
                )
                self.assertEqual(summary["cv_protocol"], "single_loop_5_fold")
                self.assertEqual(len(summary["folds"]), 5)
                self.assertEqual(
                    {int(record["fold"]) for record in summary["folds"]},
                    {1, 2, 3, 4, 5},
                )
                self.assertIn("grid_selection_score", summary["aggregate"])
                for fold in range(1, 6):
                    fold_dir = candidate_dir / f"fold_{fold:02d}"
                    self.assertTrue(CANDIDATE_FOLD_ARTIFACTS.issubset(
                        {path.name for path in fold_dir.iterdir()}
                    ))

            for name in (
                "grid_index.json",
                "grid_summary.json",
                "grid_summary_by_selection_score.csv",
                "best_hyperparameters.json",
                "grid_completion.json",
                "selected_parameters.json",
            ):
                self.assertTrue((grid_root / name).is_file(), name)

            index = orchestrator.read_json(grid_root / "grid_index.json")
            completion = orchestrator.read_json(grid_root / "grid_completion.json")
            self.assertEqual(index["cv_protocol"], "single_loop_5_fold")
            self.assertEqual(index["candidate_count"], 52)
            self.assertEqual(index["parameter_source"], "publication")
            self.assertEqual(index["search_type"], "publication_grid")
            self.assertEqual(completion["candidate_count"], 52)
            self.assertTrue(completion["complete"])
            self.assertEqual(provenance["parameter_source"], "publication")
            self.assertEqual(provenance["search_type"], "publication_grid")
            self.assertEqual(provenance["profile_type"], "publication_adapted")
            self.assertTrue(provenance["publication_search_space_preserved"])
            self.assertTrue(provenance["publication_fixed_recipe_preserved"])
            self.assertFalse(provenance["publication_exact"])
            best = orchestrator.read_json(grid_root / "best_hyperparameters.json")
            self.assertEqual(
                best["best_candidate"]["candidate_id"], provenance["candidate_id"]
            )
            self.assertEqual(best["best_candidate"]["parameters"], selected)

    def test_custom_grid_accepts_config_and_limit_and_uses_grid_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom_grid.json"
            write_project_grid(config)
            args = make_args(
                root,
                parameter_source="grid",
                grid_config=config,
                max_grid_candidates=1,
            )
            selected, provenance, train_mock = self._run_grid(args)

            grid_root = args.results_root / "LinearProbe" / "Grid" / "100_percent"
            self.assertEqual(train_mock.call_count, 5)
            self.assertTrue(
                all(len(call.args[3]) == 1 for call in train_mock.call_args_list)
            )
            self.assertTrue(grid_root.is_dir())
            self.assertEqual(len(list(grid_root.glob("candidate_*"))), 1)
            grid_index = orchestrator.read_json(grid_root / "grid_index.json")
            self.assertEqual(grid_index["parameter_source"], "grid")
            self.assertEqual(grid_index["search_type"], "grid")
            self.assertFalse(
                (args.results_root / "LinearProbe" / "Publication_Grid").exists()
            )
            self.assertEqual(provenance["parameter_source"], "grid")
            self.assertEqual(provenance["search_type"], "grid")
            best = orchestrator.read_json(grid_root / "best_hyperparameters.json")
            self.assertEqual(
                best["best_candidate"]["candidate_id"], provenance["candidate_id"]
            )
            self.assertEqual(best["best_candidate"]["parameters"], selected)


class GridResumeHardeningTests(unittest.TestCase):
    def _completed_custom_grid(self, root: Path):
        config = root / "custom_grid.json"
        write_project_grid(config)
        args = make_args(
            root,
            parameter_source="grid",
            grid_config=config,
            max_grid_candidates=1,
            resume=True,
        )
        selected, provenance, train_mock = run_synthetic_grid(args)
        self.assertEqual(train_mock.call_count, 5)
        return args, selected, provenance

    @staticmethod
    def _artifact_hashes(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): orchestrator.sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_resume_reuses_completed_grid_without_training_or_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, selected, provenance = self._completed_custom_grid(root)
            grid_root = args.results_root / "LinearProbe" / "Grid" / "100_percent"
            before = self._artifact_hashes(grid_root)

            with mock.patch.object(
                orchestrator, "train_linear_probe_grid"
            ) as train_mock:
                reused_parameters, reused_provenance = orchestrator.run_search(
                    args,
                    "linear_probe",
                    {},
                    source="grid",
                )

            train_mock.assert_not_called()
            self.assertEqual(reused_parameters, selected)
            self.assertEqual(
                reused_provenance["candidate_id"], provenance["candidate_id"]
            )
            self.assertEqual(reused_provenance, provenance)
            self.assertEqual(
                provenance["selection_file_sha256"],
                orchestrator.sha256_file(grid_root / "selected_parameters.json"),
            )
            self.assertEqual(self._artifact_hashes(grid_root), before)

            args.max_grid_candidates = 2
            with (
                mock.patch.object(
                    orchestrator, "train_linear_probe_grid"
                ) as changed_train,
                self.assertRaisesRegex(RuntimeError, "profile/candidate set"),
            ):
                orchestrator.run_search(
                    args, "linear_probe", {}, source="grid"
                )
            changed_train.assert_not_called()
            self.assertEqual(self._artifact_hashes(grid_root), before)

    def test_no_resume_refuses_completed_and_partial_grid_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, _ = self._completed_custom_grid(root)
            args.resume = False
            with (
                mock.patch.object(
                    orchestrator, "train_linear_probe_grid"
                ) as train_mock,
                self.assertRaisesRegex(RuntimeError, "completed grid already exists"),
            ):
                orchestrator.run_search(
                    args, "linear_probe", {}, source="grid"
                )
            train_mock.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom_grid.json"
            write_project_grid(config)
            args = make_args(
                root,
                parameter_source="grid",
                grid_config=config,
                max_grid_candidates=1,
                resume=False,
            )
            grid_root = args.results_root / "LinearProbe" / "Grid" / "100_percent"
            grid_root.mkdir(parents=True)
            (grid_root / "interrupted_artifact.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with (
                mock.patch.object(
                    orchestrator, "train_linear_probe_grid"
                ) as train_mock,
                self.assertRaisesRegex(RuntimeError, "Partial grid artifacts"),
            ):
                orchestrator.run_search(
                    args, "linear_probe", {}, source="grid"
                )
            train_mock.assert_not_called()

    def test_completion_inventory_path_and_candidate_directory_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, provenance = self._completed_custom_grid(root)
            grid_root = args.results_root / "LinearProbe" / "Grid" / "100_percent"
            marker_path = grid_root / "grid_completion.json"
            original = orchestrator.read_json(marker_path)

            removed_inventory = json.loads(json.dumps(original))
            removed_inventory["artifacts"].pop(next(iter(removed_inventory["artifacts"])))
            removed_inventory["seal_fingerprint"] = orchestrator.canonical_sha256(
                {
                    key: value
                    for key, value in removed_inventory.items()
                    if key not in {"completed_at_utc", "seal_fingerprint"}
                }
            )
            orchestrator.save_json(marker_path, removed_inventory)
            with self.assertRaisesRegex(RuntimeError, "artifact inventory mismatch"):
                orchestrator._validate_grid_completion(grid_root)

            orchestrator.save_json(marker_path, original)
            path_tamper = json.loads(json.dumps(original))
            relative, digest = next(iter(path_tamper["artifacts"].items()))
            del path_tamper["artifacts"][relative]
            path_tamper["artifacts"]["../outside-grid.json"] = digest
            path_tamper["seal_fingerprint"] = orchestrator.canonical_sha256(
                {
                    key: value
                    for key, value in path_tamper.items()
                    if key not in {"completed_at_utc", "seal_fingerprint"}
                }
            )
            orchestrator.save_json(marker_path, path_tamper)
            with self.assertRaisesRegex(RuntimeError, "artifact inventory mismatch|unsafe"):
                orchestrator._validate_grid_completion(grid_root)

            orchestrator.save_json(marker_path, original)
            candidate_dir = grid_root / f"candidate_{provenance['candidate_id']}"
            removed_candidate_dir = grid_root / "removed_candidate_directory"
            candidate_dir.rename(removed_candidate_dir)
            with self.assertRaisesRegex(RuntimeError, "required artifacts are missing"):
                orchestrator._validate_grid_completion(grid_root)

    def test_pipeline_preserves_grid_context_identity_for_cv_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom_grid.json"
            write_project_grid(config)
            args = make_args(
                root,
                parameter_source="grid",
                grid_config=config,
                max_grid_candidates=1,
                resume=True,
            )
            execution_context = {
                "schema_version": 2,
                "percentage": 100,
                "seed": 42,
                "split_manifest_sha256": "1" * 64,
                "cache_plan_fingerprint": "2" * 64,
                "model_identity": {
                    "upstream_revision": "synthetic-upstream-revision",
                    "pretrained_checkpoint_sha256": "f" * 64,
                },
            }
            selected, search_provenance, _ = run_synthetic_grid(
                args, execution_context
            )
            execution_fingerprint = orchestrator.canonical_sha256(
                execution_context
            )
            self.assertEqual(
                search_provenance["execution_context_fingerprint"],
                execution_fingerprint,
            )
            self.assertNotEqual(
                search_provenance["grid_context_fingerprint"],
                execution_fingerprint,
            )

            args.stage = "cv"
            model_identity = {
                "upstream_revision": "synthetic-upstream-revision",
                "pretrained_checkpoint_sha256": "f" * 64,
            }
            with (
                mock.patch.object(orchestrator, "load_manifest", return_value={}),
                mock.patch.object(orchestrator, "load_cache_manifest", return_value={}),
                mock.patch.object(orchestrator, "validate_cache"),
                mock.patch.object(
                    orchestrator,
                    "_runtime_model_identity",
                    return_value=model_identity,
                ),
                mock.patch.object(
                    orchestrator,
                    "_runtime_execution_context",
                    return_value=execution_context,
                ),
                mock.patch.object(orchestrator, "run_cv", return_value={}) as cv_mock,
            ):
                pipeline_result = orchestrator.run_pipeline(args, "linear_probe")

            resolved_provenance = pipeline_result["parameter_provenance"]
            self.assertEqual(
                resolved_provenance["execution_context_fingerprint"],
                execution_fingerprint,
            )
            self.assertEqual(
                resolved_provenance["grid_context_fingerprint"],
                search_provenance["grid_context_fingerprint"],
            )
            self.assertNotEqual(
                resolved_provenance["grid_context_fingerprint"],
                resolved_provenance["execution_context_fingerprint"],
            )
            self.assertEqual(cv_mock.call_args.args[4], resolved_provenance)

            actual_validation = orchestrator._validate_grid_completion
            with (
                mock.patch.object(
                    orchestrator,
                    "_validate_grid_completion",
                    wraps=actual_validation,
                ) as validation_mock,
                mock.patch.object(
                    orchestrator,
                    "register_cv_run",
                    side_effect=RuntimeError("stop-after-grid-validation"),
                ),
                self.assertRaisesRegex(RuntimeError, "stop-after-grid-validation"),
            ):
                orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {},
                    pipeline_result["parameters"],
                    resolved_provenance,
                    model_identity,
                )
            self.assertEqual(
                validation_mock.call_args.kwargs["expected_context_fingerprint"],
                search_provenance["grid_context_fingerprint"],
            )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SelectedGridFoldPublicationTests(unittest.TestCase):
    @staticmethod
    def _source_fold(
        directory: Path,
        *,
        candidate_identity: str,
        parameters: dict,
        model_identity: dict,
        training_rows_fingerprint: str = "b" * 64,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        source_provenance = {
            "upstream_revision": model_identity["upstream_revision"],
            "checkpoint_sha256": model_identity[
                "pretrained_checkpoint_sha256"
            ],
        }
        parameters_fingerprint = orchestrator.canonical_sha256(parameters)
        runtime_plan = {
            "method": "linear_probe",
            "max_iterations": 1,
            "scheduler_total_updates": 1,
            "training_rows_fingerprint": training_rows_fingerprint,
            "candidate_id": candidate_identity,
            "candidate_parameters_fingerprint": parameters_fingerprint,
            "joint_candidate_count": 52,
            "candidate_execution": "joint_shared_encoder_multi_head",
            "head_initialization_schema": training.PROBE_HEAD_INITIALIZATION_SCHEMA,
        }
        runtime_plan_fingerprint = orchestrator._runtime_plan_fingerprint(
            parameters, runtime_plan
        )
        orchestrator.save_json(
            directory / "parameters.json",
            {
                "candidate_id": candidate_identity,
                "parameters": parameters,
            },
        )
        orchestrator.save_json(
            directory / "shared_training_reference.json",
            {
                "candidate_id": candidate_identity,
                "shared_training_directory": str(directory.parent.parent / "shared_training"),
                "shared_encoder_pass_per_candidate_bank": True,
            },
        )
        state = {
            "schema_version": training.SCHEMA_VERSION,
            "model_name": "3dino_linear_probe",
            "model_family": "3DINO",
            "completed": True,
            "method": "linear_probe",
            "encoder_frozen": True,
            "candidate_id": candidate_identity,
            "parameters": parameters,
            "parameters_fingerprint": parameters_fingerprint,
            "runtime_plan": runtime_plan,
            "runtime_plan_fingerprint": runtime_plan_fingerprint,
            "classes": list(CLASS_NAMES),
            "source_provenance": source_provenance,
            "seed": parameters["seed"],
            "best_optimizer_step": 1,
            "validation_metrics": validation_metrics(0.75),
        }
        orchestrator.save_json(directory / "training_state.json", state)
        torch.save(
            {
                "schema_version": training.SCHEMA_VERSION,
                "model_family": "3DINO",
                "method": "linear_probe",
                "checkpoint_scope": "single_linear_head",
                "encoder_frozen": True,
                "candidate_id": candidate_identity,
                "candidate": parameters,
                "parameters": parameters,
                "parameters_fingerprint": parameters_fingerprint,
                "runtime_plan": runtime_plan,
                "runtime_plan_fingerprint": runtime_plan_fingerprint,
                "classes": list(CLASS_NAMES),
                "source_provenance": source_provenance,
                "head_state_dict": {},
                "optimizer_step": 1,
                "validation_metrics": validation_metrics(0.75),
            },
            directory / "best_model.pt",
        )
        orchestrator.save_json(
            directory / "metrics_summary.json", validation_metrics(0.75)
        )
        orchestrator.save_json(
            directory / "validation_patient_predictions.json",
            [{"patient_id": "validation", "prediction": CLASS_NAMES[0]}],
        )
        (directory / "validation_probabilities.csv").write_text(
            "patient_id,Normal\nvalidation,1.0\n", encoding="utf-8"
        )

    def test_selected_publication_fold_is_published_with_hashed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
                / "candidate_abc123def456"
                / "fold_01"
            )
            destination = (
                root
                / "LinearProbe"
                / "CV"
                / "CV_1"
                / "100_percent"
                / "fold_01"
            )
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            self._source_fold(
                source,
                candidate_identity="abc123def456",
                parameters=parameters,
                model_identity=model_identity,
            )

            metadata = orchestrator._publish_selected_grid_fold(
                source,
                destination,
                expected_candidate_id="abc123def456",
                expected_parameters=parameters,
                model_identity=model_identity,
                result_origin="selected_publication_grid_run",
            )

            self.assertTrue(metadata["reused_without_training"])
            self.assertEqual(
                metadata["result_origin"], "selected_publication_grid_run"
            )
            self.assertEqual(metadata["candidate_id"], "abc123def456")
            self.assertEqual(Path(metadata["source_directory"]), source.resolve())
            self.assertEqual(
                Path(metadata["destination_directory"]), destination.resolve()
            )
            self.assertEqual(
                orchestrator.read_json(destination / "grid_reuse.json"), metadata
            )
            for name in (
                "best_model.pt",
                "training_state.json",
                "metrics_summary.json",
                "validation_patient_predictions.json",
                "validation_probabilities.csv",
            ):
                record = metadata["artifacts"][name]
                self.assertTrue((destination / name).is_file(), name)
                self.assertEqual(record["source_sha256"], record["destination_sha256"])
                self.assertEqual(
                    record["destination_sha256"],
                    orchestrator.sha256_file(destination / name),
                )
                self.assertIn(
                    record["materialization"], {"hard_link", "copied", "existing"}
                )

    def test_selected_fold_rejects_a_different_candidate_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate_abc123def456" / "fold_01"
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            self._source_fold(
                source,
                candidate_identity="abc123def456",
                parameters=parameters,
                model_identity=model_identity,
            )
            with self.assertRaisesRegex(RuntimeError, "candidate"):
                orchestrator._publish_selected_grid_fold(
                    source,
                    root / "cv" / "fold_01",
                    expected_candidate_id="wrong-candidate",
                    expected_parameters=parameters,
                    model_identity=model_identity,
                    result_origin="selected_publication_grid_run",
                )

    def test_grid_reuse_marker_rejects_mixed_origin_and_source_grid_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publication_root = (
                root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            source = (
                publication_root
                / "candidate_abc123def456"
                / "fold_01"
            )
            destination = root / "LinearProbe" / "CV" / "CV_1" / "fold_01"
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            self._source_fold(
                source,
                candidate_identity="abc123def456",
                parameters=parameters,
                model_identity=model_identity,
            )
            orchestrator._publish_selected_grid_fold(
                source,
                destination,
                expected_candidate_id="abc123def456",
                expected_parameters=parameters,
                model_identity=model_identity,
                result_origin="selected_publication_grid_run",
            )
            marker = destination / "grid_reuse.json"

            with self.assertRaisesRegex(RuntimeError, "result origin mismatch"):
                orchestrator._validate_or_refresh_grid_reuse(
                    marker,
                    expected_candidate_id="abc123def456",
                    expected_result_origin="selected_grid_run",
                    expected_source_grid_root=publication_root,
                )
            with self.assertRaisesRegex(RuntimeError, "source grid root mismatch"):
                orchestrator._validate_or_refresh_grid_reuse(
                    marker,
                    expected_candidate_id="abc123def456",
                    expected_result_origin="selected_publication_grid_run",
                    expected_source_grid_root=(
                        root / "LinearProbe" / "Grid" / "100_percent"
                    ),
                )

    def test_cv_reuses_all_selected_grid_folds_without_training_them_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="cv")
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            grid_root = (
                args.results_root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            train_rows_by_fold = {
                fold: [
                    {
                        "patient_id": f"train-{fold}",
                        "label": 0,
                        "label_str": CLASS_NAMES[0],
                    }
                ]
                for fold in range(1, 6)
            }
            for fold, train_rows in train_rows_by_fold.items():
                self._source_fold(
                    grid_root
                    / "candidate_abc123def456"
                    / f"fold_{fold:02d}",
                    candidate_identity="abc123def456",
                    parameters=parameters,
                    model_identity=model_identity,
                    training_rows_fingerprint=(
                        orchestrator._training_rows_fingerprint(train_rows)
                    ),
                )

            def split(_manifest, fold):
                return {"rows": train_rows_by_fold[int(fold)]}, {
                    "rows": [
                        {
                            "patient_id": f"validation-{fold}",
                            "label": 1,
                            "label_str": CLASS_NAMES[1],
                        }
                    ]
                }

            def split_rows(_args, mapping):
                return [dict(row) for row in mapping["rows"]]

            provenance = {
                "parameter_source": "publication",
                "search_type": "publication_grid",
                "candidate_id": "abc123def456",
                "grid_root": str(grid_root.resolve()),
            }

            def seal_cv_folds(percentage_dir, **_kwargs):
                for fold in range(1, 6):
                    state_path = Path(percentage_dir) / f"fold_{fold:02d}" / "training_state.json"
                    state = orchestrator.read_json(state_path)
                    checkpoint_sha = orchestrator.sha256_file(
                        Path(percentage_dir) / f"fold_{fold:02d}" / "best_model.pt"
                    )
                    state["best_model_sha256"] = checkpoint_sha
                    state["best_model_binding"] = {
                        "checkpoint_sha256": checkpoint_sha
                    }
                    orchestrator.save_json(state_path, state)
                for fold in range(1, 6):
                    orchestrator._validate_or_refresh_grid_reuse(
                        Path(percentage_dir) / f"fold_{fold:02d}" / "grid_reuse.json",
                        expected_candidate_id="abc123def456",
                        refresh_after_cv_binding=True,
                    )
                marker_hashes = {
                    f"fold_{fold:02d}/grid_reuse.json": orchestrator.sha256_file(
                        Path(percentage_dir) / f"fold_{fold:02d}" / "grid_reuse.json"
                    )
                    for fold in range(1, 6)
                }
                completion = {
                    "seal_fingerprint": "c" * 64,
                    "best_fold": 1,
                    "refit_budget": 1,
                    "selected_grid_folds_reused_without_training": True,
                    "grid_reuse_markers": marker_hashes,
                }
                orchestrator.save_json(
                    Path(percentage_dir) / "cv_completion.json", completion
                )
                return completion

            with (
                mock.patch.object(
                    orchestrator, "cross_validation_split", side_effect=split
                ),
                mock.patch.object(
                    orchestrator, "_split_rows", side_effect=split_rows
                ),
                mock.patch.object(orchestrator, "validate_no_overlap"),
                mock.patch.object(orchestrator, "_validate_grid_completion"),
                mock.patch.object(
                    orchestrator,
                    "development_mapping",
                    return_value={
                        "rows": [row for rows in train_rows_by_fold.values() for row in rows]
                    },
                ),
                mock.patch.object(orchestrator, "_common_training_kwargs", return_value={}),
                mock.patch.object(orchestrator, "_train_single") as train_single,
                mock.patch.object(orchestrator, "refit_model") as refit,
                mock.patch.object(
                    orchestrator,
                    "_write_cv_completion_seal",
                    side_effect=seal_cv_folds,
                ),
            ):
                result = orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {},
                    parameters,
                    provenance,
                    model_identity,
                )

                train_single.assert_not_called()
                refit.assert_called_once()
                cv_percentage = Path(result["percentage_directory"])
                first_marker = orchestrator.read_json(
                    cv_percentage / "fold_01" / "grid_reuse.json"
                )
                state_record = first_marker["artifacts"]["training_state.json"]
                self.assertNotEqual(
                    state_record["source_sha256"],
                    state_record["destination_sha256"],
                )
                self.assertEqual(
                    state_record["destination_sha256"],
                    orchestrator.sha256_file(
                        cv_percentage / "fold_01" / "training_state.json"
                    ),
                )
                self.assertEqual(
                    state_record["post_materialization_update"],
                    "cv_seal_binding",
                )
                self.assertTrue(first_marker["cv_sealed"])

            train_single.assert_not_called()
            self.assertEqual(refit.call_count, 1)
            self.assertTrue(result["selected_grid_folds_reused_without_training"])
            self.assertEqual(result["result_origin"], "selected_publication_grid_run")
            self.assertEqual(result["selected_grid_candidate_id"], "abc123def456")
            self.assertEqual(len(result["grid_reuse_records"]), 5)
            self.assertEqual(result["cv_id"], "CV_1")
            repeated_cv_id, repeated_cv_dir = orchestrator.register_cv_run(
                args.results_root,
                "linear_probe",
                parameters,
                provenance,
                model_identity,
            )
            self.assertEqual(repeated_cv_id, "CV_1")
            self.assertEqual(repeated_cv_dir, Path(result["cv_directory"]))
            cv_percentage = Path(result["percentage_directory"])
            for fold in range(1, 6):
                marker = orchestrator.read_json(
                    cv_percentage / f"fold_{fold:02d}" / "grid_reuse.json"
                )
                self.assertTrue(marker["reused_without_training"])
                self.assertEqual(
                    marker["result_origin"], "selected_publication_grid_run"
                )

    def test_cv_resume_finishes_missing_reuse_marker_after_core_materialization(self):
        """A marker-write crash must resume without retraining or overwriting bytes."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="cv", resume=True)
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            grid_root = (
                args.results_root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            provenance = {
                "parameter_source": "publication",
                "search_type": "publication_grid",
                "candidate_id": "abc123def456",
                "grid_root": str(grid_root.resolve()),
            }
            train_rows_by_fold = {
                fold: [
                    {
                        "patient_id": f"train-{fold}",
                        "label": 0,
                        "label_str": CLASS_NAMES[0],
                    }
                ]
                for fold in range(1, 6)
            }
            for fold, train_rows in train_rows_by_fold.items():
                self._source_fold(
                    grid_root
                    / "candidate_abc123def456"
                    / f"fold_{fold:02d}",
                    candidate_identity="abc123def456",
                    parameters=parameters,
                    model_identity=model_identity,
                    training_rows_fingerprint=(
                        orchestrator._training_rows_fingerprint(train_rows)
                    ),
                )

            _, cv_dir = orchestrator.register_cv_run(
                args.results_root,
                "linear_probe",
                parameters,
                provenance,
                model_identity,
            )
            interrupted_fold = cv_dir / "100_percent" / "fold_01"
            partial_fold = cv_dir / "100_percent" / "fold_02"
            for fold, destination in ((1, interrupted_fold), (2, partial_fold)):
                orchestrator._publish_selected_grid_fold(
                    grid_root
                    / "candidate_abc123def456"
                    / f"fold_{fold:02d}",
                    destination,
                    expected_candidate_id="abc123def456",
                    expected_parameters=parameters,
                    model_identity=model_identity,
                    result_origin="selected_publication_grid_run",
                )
                (destination / "grid_reuse.json").unlink()
            marker = interrupted_fold / "grid_reuse.json"
            partial_marker = partial_fold / "grid_reuse.json"
            # The first crash happened after every core file; the second
            # happened earlier, after training_state but before metrics_summary.
            partial_training_state_sha = orchestrator.sha256_file(
                partial_fold / "training_state.json"
            )
            (partial_fold / "metrics_summary.json").unlink()
            core_hashes = {
                name: orchestrator.sha256_file(interrupted_fold / name)
                for name in CANDIDATE_FOLD_ARTIFACTS
            }

            def split(_manifest, fold):
                return {"rows": train_rows_by_fold[int(fold)]}, {
                    "rows": [
                        {
                            "patient_id": f"validation-{fold}",
                            "label": 1,
                            "label_str": CLASS_NAMES[1],
                        }
                    ]
                }

            with (
                mock.patch.object(
                    orchestrator, "cross_validation_split", side_effect=split
                ),
                mock.patch.object(
                    orchestrator,
                    "_split_rows",
                    side_effect=lambda _args, mapping: [
                        dict(row) for row in mapping["rows"]
                    ],
                ),
                mock.patch.object(orchestrator, "validate_no_overlap"),
                mock.patch.object(orchestrator, "_validate_grid_completion"),
                mock.patch.object(
                    orchestrator,
                    "development_mapping",
                    return_value={
                        "rows": [
                            row
                            for rows in train_rows_by_fold.values()
                            for row in rows
                        ]
                    },
                ),
                mock.patch.object(
                    orchestrator, "_common_training_kwargs", return_value={}
                ),
                mock.patch.object(orchestrator, "_train_single") as train_single,
                mock.patch.object(orchestrator, "refit_model") as refit,
                mock.patch.object(
                    orchestrator,
                    "_write_cv_completion_seal",
                    return_value={"seal_fingerprint": "c" * 64},
                ),
                mock.patch.object(
                    orchestrator,
                    "_publish_selected_grid_fold",
                    wraps=orchestrator._publish_selected_grid_fold,
                ) as publish,
            ):
                result = orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {},
                    parameters,
                    provenance,
                    model_identity,
                )

            train_single.assert_not_called()
            refit.assert_called_once()
            self.assertEqual(publish.call_count, 5)
            self.assertTrue(marker.is_file())
            repaired = orchestrator.read_json(marker)
            self.assertTrue(repaired["reused_without_training"])
            self.assertEqual(repaired["candidate_id"], "abc123def456")
            self.assertTrue(
                all(
                    record["materialization"] == "existing"
                    for record in repaired["artifacts"].values()
                )
            )
            self.assertEqual(
                core_hashes,
                {
                    name: orchestrator.sha256_file(interrupted_fold / name)
                    for name in CANDIDATE_FOLD_ARTIFACTS
                },
            )
            self.assertTrue(partial_marker.is_file())
            partially_repaired = orchestrator.read_json(partial_marker)
            self.assertEqual(
                partially_repaired["artifacts"]["training_state.json"][
                    "materialization"
                ],
                "existing",
            )
            self.assertNotEqual(
                partially_repaired["artifacts"]["metrics_summary.json"][
                    "materialization"
                ],
                "existing",
            )
            self.assertEqual(
                partial_training_state_sha,
                orchestrator.sha256_file(partial_fold / "training_state.json"),
            )
            self.assertEqual(
                orchestrator.sha256_file(
                    grid_root
                    / "candidate_abc123def456"
                    / "fold_02"
                    / "metrics_summary.json"
                ),
                orchestrator.sha256_file(partial_fold / "metrics_summary.json"),
            )
            self.assertTrue(result["selected_grid_folds_reused_without_training"])

    def test_no_resume_partial_cv_preflight_blocks_all_training_and_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="cv", resume=False)
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            grid_root = (
                args.results_root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            provenance = {
                "parameter_source": "publication",
                "search_type": "publication_grid",
                "candidate_id": "abc123def456",
                "grid_root": str(grid_root.resolve()),
            }
            _, cv_dir = orchestrator.register_cv_run(
                args.results_root,
                "linear_probe",
                parameters,
                provenance,
                model_identity,
            )
            partial = cv_dir / "100_percent" / "fold_01" / "training_state.json"
            orchestrator.save_json(partial, {"completed": False})

            with (
                mock.patch.object(orchestrator, "_validate_grid_completion"),
                mock.patch.object(
                    orchestrator, "cross_validation_split"
                ) as split,
                mock.patch.object(orchestrator, "_train_single") as train_single,
                mock.patch.object(orchestrator, "refit_model") as refit,
                mock.patch.object(
                    orchestrator, "_publish_selected_grid_fold"
                ) as publish,
                self.assertRaisesRegex(RuntimeError, "Partial CV artifacts"),
            ):
                orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {},
                    parameters,
                    provenance,
                    model_identity,
                )

            split.assert_not_called()
            train_single.assert_not_called()
            refit.assert_not_called()
            publish.assert_not_called()

    def test_cv_fails_closed_when_selected_grid_has_no_completion_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="cv")
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            grid_root = (
                args.results_root
                / "LinearProbe"
                / "Publication_Grid"
                / "100_percent"
            )
            provenance = {
                "parameter_source": "publication",
                "search_type": "publication_grid",
                "candidate_id": "abc123def456",
                "grid_root": str(grid_root.resolve()),
            }
            with mock.patch.object(orchestrator, "_train_single") as train_single:
                with self.assertRaisesRegex(RuntimeError, "grid_completion.json"):
                    orchestrator.run_cv(
                        args,
                        "linear_probe",
                        {},
                        parameters,
                        provenance,
                        model_identity,
                    )
            train_single.assert_not_called()

    def test_selected_grid_fold_missing_an_artifact_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate_abc123def456" / "fold_01"
            destination = root / "CV_1" / "fold_01"
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            self._source_fold(
                source,
                candidate_identity="abc123def456",
                parameters=parameters,
                model_identity=model_identity,
            )
            (source / "metrics_summary.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                orchestrator._publish_selected_grid_fold(
                    source,
                    destination,
                    expected_candidate_id="abc123def456",
                    expected_parameters=parameters,
                    model_identity=model_identity,
                    result_origin="selected_publication_grid_run",
                )
            self.assertFalse(destination.exists())

    def test_custom_grid_cv_uses_selected_grid_run_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="cv", parameter_source="grid")
            parameters = {
                "candidate_id": "abc123def456",
                "nominal_lr": 0.01,
                "lr": 0.00125,
                "n_last_blocks": 4,
                "use_mean_patch_token": True,
                "seed": 42,
            }
            model_identity = {
                "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            grid_root = args.results_root / "LinearProbe" / "Grid" / "100_percent"
            train_rows = [
                {
                    "patient_id": "train-1",
                    "label": 0,
                    "label_str": CLASS_NAMES[0],
                }
            ]
            source = grid_root / "candidate_abc123def456" / "fold_01"
            source.mkdir(parents=True)
            orchestrator.save_json(
                source / "training_state.json",
                {
                    "runtime_plan": {
                        "training_rows_fingerprint": (
                            orchestrator._training_rows_fingerprint(train_rows)
                        )
                    }
                },
            )
            provenance = {
                "parameter_source": "grid",
                "search_type": "grid",
                "candidate_id": "abc123def456",
                "grid_root": str(grid_root.resolve()),
            }
            publish = mock.Mock(side_effect=RuntimeError("stop-after-origin"))
            with (
                mock.patch.object(orchestrator, "_validate_grid_completion"),
                mock.patch.object(
                    orchestrator,
                    "cross_validation_split",
                    return_value=({"rows": train_rows}, {"rows": []}),
                ),
                mock.patch.object(
                    orchestrator,
                    "_split_rows",
                    side_effect=lambda _args, mapping: mapping["rows"],
                ),
                mock.patch.object(orchestrator, "validate_no_overlap"),
                mock.patch.object(
                    orchestrator, "_publish_selected_grid_fold", publish
                ),
                self.assertRaisesRegex(RuntimeError, "stop-after-origin"),
            ):
                orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {},
                    parameters,
                    provenance,
                    model_identity,
                )
            self.assertEqual(
                publish.call_args.kwargs["result_origin"], "selected_grid_run"
            )


TorchModule = nn.Module if nn is not None else object


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SharedEncoderEfficiencyTests(unittest.TestCase):
    class FrozenEncoder(TorchModule):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.calls = 0

    @staticmethod
    def _rows():
        return [
            {
                "image": torch.full((1, 2, 2, 2), float(index + 1) / 10),
                "label": index,
                "patient_id": f"patient-{index}",
                "label_str": name,
            }
            for index, name in enumerate(CLASS_NAMES)
        ]

    def _encoder_calls(self, root: Path, candidate_count: int) -> int:
        encoder = self.FrozenEncoder()

        def extract_bank(model, images, representations):
            model.calls += 1
            base = images.flatten(1).mean(dim=1, keepdim=True)
            return {
                representation: base.repeat(1, 4) * model.scale
                for representation in representations
            }

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
        feature_recipes = ((1, False), (1, True), (4, False), (4, True))
        candidates = [
            {
                "nominal_lr": 0.01,
                "lr": 0.01,
                "learning_rate_scale_rule": "none",
                "n_last_blocks": feature_recipes[index][0],
                "use_mean_patch_token": feature_recipes[index][1],
                "feature_dim": 4,
            }
            for index in range(candidate_count)
        ]
        training.train_linear_probe_grid(
            self._rows(),
            self._rows(),
            parameters,
            candidates,
            root,
            device=torch.device("cpu"),
            encoder_factory=lambda **_: encoder,
            feature_bank_extractor=extract_bank,
            feature_dimension_fn=lambda _blocks, _mean: 4,
            dataset_factory=lambda rows, **_: [dict(row) for row in rows],
        )
        return encoder.calls

    def test_encoder_call_count_is_independent_of_candidate_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one_head_calls = self._encoder_calls(root / "one", 1)
            four_head_calls = self._encoder_calls(root / "four", 4)
        self.assertGreater(one_head_calls, 0)
        self.assertEqual(four_head_calls, one_head_calls)

    def test_head_initialization_is_independent_of_bank_order_and_size(self):
        selected = {
            "candidate_id": "candidate-selected",
            "n_last_blocks": 1,
            "use_mean_patch_token": False,
            "feature_dim": 4,
        }
        other = {
            "candidate_id": "candidate-other",
            "n_last_blocks": 4,
            "use_mean_patch_token": True,
            "feature_dim": 4,
        }
        dimension = lambda _blocks, _mean: 4
        alone = training.ProbeHeadBank(
            [selected], 5, feature_dimension_fn=dimension, seed=42
        )
        first = training.ProbeHeadBank(
            [selected, other], 5, feature_dimension_fn=dimension, seed=42
        )
        second = training.ProbeHeadBank(
            [other, selected], 5, feature_dimension_fn=dimension, seed=42
        )
        changed_seed = training.ProbeHeadBank(
            [selected], 5, feature_dimension_fn=dimension, seed=43
        )
        for key, tensor in alone.heads["candidate-selected"].state_dict().items():
            self.assertTrue(
                torch.equal(tensor, first.heads["candidate-selected"].state_dict()[key])
            )
            self.assertTrue(
                torch.equal(tensor, second.heads["candidate-selected"].state_dict()[key])
            )
        self.assertFalse(
            torch.equal(
                alone.heads["candidate-selected"].weight,
                changed_seed.heads["candidate-selected"].weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
