"""Mocked actual-stage contracts for the 3DINO orchestrator.

These tests deliberately exercise orchestration and artifact bookkeeping only.
They never load NIfTI volumes, an upstream checkout, or either model checkpoint.
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
except ImportError:  # pragma: no cover - environment-dependent skip
    torch = None

from Code.Models.ThreeDINO import CLASS_NAMES  # noqa: E402
from Code.Models.ThreeDINO import orchestrator  # noqa: E402
from Code.Models.ThreeDINO.cli import build_parser  # noqa: E402


def make_args(root: Path, **overrides):
    """Return the runtime attributes consumed by orchestration helpers."""

    values = {
        "dataset_root": root / "Dataset",
        "results_root": root / "Results" / "3DINO",
        "cache_root": root / "Results" / "3DINO" / "Cache",
        "cache_namespace": "synthetic-prefix-cache",
        "cache_dir": root / "Results" / "3DINO" / "Cache" / "Shared" / "synthetic-prefix-cache",
        "upstream_root": root / "external" / "3DINO",
        "upstream_revision": "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
        "pretrained_checkpoint": root / "weights" / "3dino_vit_weights.pth",
        "pretrained_checkpoint_sha256": "a" * 64,
        "device": "cpu",
        "seed": 17,
        "num_workers": 0,
        "resume": False,
        "dry_run": False,
        "stage": "cv",
        "percentage": 40,
        "parameter_source": "default",
        "parameter_config": None,
        "search_config": None,
        "max_search_candidates": None,
        "model_source": "refit_full_development",
        "checkpoint": None,
        "force_test": False,
        "allow_experimental_full_finetune": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def validation_metrics(f1: float, balanced_accuracy: float = 0.5) -> dict:
    return {
        "accuracy": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1,
        "average_precision_macro": f1,
        "per_class": {
            name: {"recall": f1 if name == "Perforation" else balanced_accuracy}
            for name in CLASS_NAMES
        },
    }


def write_synthetic_completed_classifier(
    output_dir: Path,
    *,
    method: str,
    parameters: dict,
    training_rows: list[dict],
    model_identity: dict,
    max_iterations: int,
    state_extra: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    """Write a tiny torch checkpoint with the real provenance contract."""

    runtime_plan = {
        "method": method,
        "max_iterations": int(max_iterations),
        "scheduler_total_updates": int(max_iterations),
        "training_patient_count": len(training_rows),
        "training_rows_fingerprint": orchestrator._training_rows_fingerprint(
            training_rows
        ),
    }
    parameters_fingerprint = orchestrator.canonical_sha256(parameters)
    runtime_plan_fingerprint = orchestrator._runtime_plan_fingerprint(
        parameters, runtime_plan
    )
    source = {
        "upstream_revision": model_identity["upstream_revision"],
        "checkpoint_sha256": model_identity["pretrained_checkpoint_sha256"],
    }
    state = {
        "completed": True,
        "method": method,
        "parameters": dict(parameters),
        "parameters_fingerprint": parameters_fingerprint,
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": runtime_plan_fingerprint,
        "classes": list(CLASS_NAMES),
        "source_provenance": source,
        "seed": int(parameters.get("seed", 17)),
        **dict(state_extra or {}),
    }
    payload = {
        "method": method,
        "parameters": dict(parameters),
        "parameters_fingerprint": parameters_fingerprint,
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": runtime_plan_fingerprint,
        "classes": list(CLASS_NAMES),
        "source_provenance": source,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "best_model.pt")
    orchestrator.save_json(output_dir / "training_state.json", state)
    orchestrator.save_json(
        output_dir / "metrics_summary.json",
        metrics or validation_metrics(0.5),
    )
    return state


def write_synthetic_cv_completion(
    root: Path,
    *,
    parameters: dict,
    model_identity: dict,
    best_fold: int = 1,
    cv_id: str = "CV_3",
    duplicate_fold_sha: bool = False,
) -> tuple[Path, dict]:
    """Create six tiny internally bound classifiers and their final CV seal."""

    percentage_dir = (
        root / "Results" / "3DINO" / "LinearProbe" / "CV" / cv_id / "40_percent"
    )
    fold_fingerprints = {}
    for fold in range(1, 6):
        patient_fold = 1 if duplicate_fold_sha and fold in {1, 2} else fold
        rows = [
            {
                "patient_id": f"fold-{patient_fold}-training",
                "label": patient_fold - 1,
                "label_str": CLASS_NAMES[patient_fold - 1],
            }
        ]
        fold_fingerprints[fold] = orchestrator._training_rows_fingerprint(rows)
        write_synthetic_completed_classifier(
            percentage_dir / f"fold_{fold:02d}",
            method="linear_probe",
            parameters=parameters,
            training_rows=rows,
            model_identity=model_identity,
            max_iterations=10,
        )
    if duplicate_fold_sha:
        first_checkpoint = percentage_dir / "fold_01" / "best_model.pt"
        (percentage_dir / "fold_02" / "best_model.pt").write_bytes(
            first_checkpoint.read_bytes()
        )
    refit_budget = 7
    refit_parameters = {
        **parameters,
        "evaluation_interval_iterations": refit_budget,
        "early_stopping_patience": 0,
    }
    refit_rows = [
        {
            "patient_id": "full-development",
            "label": 0,
            "label_str": CLASS_NAMES[0],
        }
    ]
    write_synthetic_completed_classifier(
        percentage_dir / "refit_full_development",
        method="linear_probe",
        parameters=refit_parameters,
        training_rows=refit_rows,
        model_identity=model_identity,
        max_iterations=refit_budget,
        state_extra={"refit_budget": refit_budget},
    )
    percentage_dir.mkdir(parents=True, exist_ok=True)
    (percentage_dir / "cv_fold_summary.csv").write_text(
        "fold,f1_macro\n1,0.8\n", encoding="utf-8"
    )
    selected_checkpoint = percentage_dir / f"fold_{best_fold:02d}" / "best_model.pt"
    orchestrator.save_json(
        percentage_dir / "best_fold_model.json",
        {
            "cv_id": cv_id,
            "method": "linear_probe",
            "best_fold": best_fold,
            "checkpoint": str(selected_checkpoint.resolve()),
        },
    )
    seal = orchestrator._write_cv_completion_seal(
        percentage_dir,
        cv_id=cv_id,
        method="linear_probe",
        percentage=40,
        parameters=parameters,
        model_identity=model_identity,
        best_fold=best_fold,
        refit_budget=refit_budget,
        fold_training_rows_fingerprints=fold_fingerprints,
        refit_training_rows_fingerprint=orchestrator._training_rows_fingerprint(
            refit_rows
        ),
    )
    return percentage_dir, seal


class FullFineTuneGateTests(unittest.TestCase):
    def test_actual_pipeline_fails_before_cache_or_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(Path(temporary), method="full_finetune")
            with (
                mock.patch.object(orchestrator, "load_manifest", return_value={}),
                mock.patch.object(orchestrator, "load_cache_manifest") as load_cache,
                mock.patch.object(orchestrator, "validate_upstream_checkout") as validate_upstream,
                self.assertRaisesRegex(PermissionError, "disabled by default"),
            ):
                orchestrator.run_pipeline(args, "full_finetune")

            load_cache.assert_not_called()
            validate_upstream.assert_not_called()

    def test_explicit_gate_adds_both_training_and_model_authorization_keys(self):
        args = SimpleNamespace(allow_experimental_full_finetune=True)
        original = {"encoder_lr": 1e-6}
        with self.assertRaisesRegex(PermissionError, "CLI switch alone"):
            orchestrator._authorise_parameters(
                "full_finetune", original, args
            )
        authorized = orchestrator._authorise_parameters(
            "full_finetune",
            original,
            args,
            {
                "license_approval_claimed": True,
                "memory_preflight_passed": True,
            },
        )
        self.assertEqual(original, {"encoder_lr": 1e-6})
        self.assertTrue(authorized["allow_experimental_full_finetune"])
        self.assertTrue(authorized["allow_full_finetune"])

    def test_dry_run_authorization_is_cli_and_license_and_memory(self):
        combinations = [
            (False, False, False),
            (True, False, False),
            (False, True, True),
            (True, True, False),
            (True, False, True),
            (True, True, True),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(orchestrator, "load_manifest", return_value={}):
                for index, (cli, license_review, memory_preflight) in enumerate(
                    combinations
                ):
                    with self.subTest(
                        cli=cli,
                        license_review=license_review,
                        memory_preflight=memory_preflight,
                    ):
                        profile = root / f"manual-full-finetune-{index}.json"
                        orchestrator.save_json(
                            profile,
                            {
                                "method": "full_finetune",
                                "profile_type": "project_adapted_conditional",
                                "parameters": {"encoder_lr": 1e-6},
                                "authorization": {
                                    "license_approval_claimed": license_review,
                                    "memory_preflight_passed": memory_preflight,
                                },
                            },
                        )
                        args = make_args(
                            root,
                            dry_run=True,
                            parameter_source="manual",
                            parameter_config=profile,
                            allow_experimental_full_finetune=cli,
                        )
                        result = orchestrator.run_pipeline(
                            args, "full_finetune"
                        )
                        status = result["full_finetune_authorization"]
                        self.assertEqual(status["cli_acknowledged"], cli)
                        self.assertEqual(
                            status["profile_license_review"], license_review
                        )
                        self.assertEqual(
                            status["profile_memory_preflight"],
                            memory_preflight,
                        )
                        expected = cli and license_review and memory_preflight
                        self.assertEqual(status["authorized"], expected)
                        self.assertEqual(
                            result["full_finetune_authorized"], expected
                        )


class IdentityAndRegistrationTests(unittest.TestCase):
    def test_fingerprint_is_path_independent_but_checkpoint_sensitive(self):
        parameters = {"lr": 0.01, "n_last_blocks": 4}
        first = {
            "model_repo_id": "AICONSlab/3DINO-ViT",
            "upstream_repository": "https://github.com/AICONSlab/3DINO",
            "upstream_revision": "fixed-revision",
            "upstream_root": "C:/machine-one/3DINO",
            "pretrained_checkpoint": "C:/machine-one/weights/model.pth",
            "pretrained_checkpoint_sha256": "1" * 64,
            "upstream_validation": {"local_only": True},
        }
        relocated = {
            **first,
            "upstream_root": "D:/machine-two/vendor/3DINO",
            "pretrained_checkpoint": "D:/machine-two/model.pth",
            "upstream_validation": {"local_only": False},
        }
        changed_weights = {
            **relocated,
            "pretrained_checkpoint_sha256": "2" * 64,
        }

        first_fingerprint = orchestrator.parameter_fingerprint(
            "linear_probe", parameters, first
        )
        self.assertEqual(
            first_fingerprint,
            orchestrator.parameter_fingerprint(
                "linear_probe", parameters, relocated
            ),
        )
        self.assertNotEqual(
            first_fingerprint,
            orchestrator.parameter_fingerprint(
                "linear_probe", parameters, changed_weights
            ),
        )

    def test_register_cv_run_reuses_exact_identity_and_allocates_changed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Results" / "3DINO"
            parameters = {"lr": 0.01, "n_last_blocks": 1}
            identity = {
                "model_repo_id": "AICONSlab/3DINO-ViT",
                "upstream_repository": "https://github.com/AICONSlab/3DINO",
                "upstream_revision": "fixed-revision",
                "upstream_root": "C:/first/location",
                "pretrained_checkpoint": "C:/first/model.pth",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            first_id, first_dir = orchestrator.register_cv_run(
                root,
                "linear_probe",
                parameters,
                {"parameter_source": "default"},
                identity,
            )
            relocated = {
                **identity,
                "upstream_root": "D:/relocated/source",
                "pretrained_checkpoint": "D:/relocated/model.pth",
            }
            repeated_id, repeated_dir = orchestrator.register_cv_run(
                root,
                "linear_probe",
                parameters,
                {"parameter_source": "manual"},
                relocated,
            )
            changed_id, _ = orchestrator.register_cv_run(
                root,
                "linear_probe",
                {**parameters, "lr": 0.02},
                {"parameter_source": "manual"},
                relocated,
            )

            self.assertEqual(first_id, "CV_1")
            self.assertEqual(repeated_id, first_id)
            self.assertEqual(repeated_dir, first_dir)
            self.assertEqual(changed_id, "CV_2")
            registry = orchestrator.read_json(
                root / "LinearProbe" / "CV" / "cv_runs.json"
            )
            self.assertEqual(len(registry["runs"]), 2)
            cv_parameters = orchestrator.read_json(
                first_dir / "cv_parameters.json"
            )
            self.assertEqual(len(cv_parameters["provenance_history"]), 2)

    def test_effective_seed_distinguishes_cv_but_worker_count_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_root = root / "Results" / "3DINO"
            base_parameters = {
                "lr": 0.01,
                "seed": 999,
                "n_last_blocks": 1,
            }
            identity = {
                "model_repo_id": "AICONSlab/3DINO-ViT",
                "upstream_repository": "https://github.com/AICONSlab/3DINO",
                "upstream_revision": "fixed-revision",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            first_args = make_args(root, seed=17, num_workers=0)
            first_context = {
                "schema_version": 2,
                "seed": 17,
                "split_manifest_semantic_sha256": "split-a",
                "cache_plan_fingerprint": "cache-a",
            }
            second_args = make_args(root, seed=18, num_workers=0)
            second_context = {**first_context, "seed": 18}
            worker_args = make_args(root, seed=17, num_workers=10)
            first_parameters = orchestrator._apply_execution_context(
                base_parameters, first_args, first_context
            )
            second_parameters = orchestrator._apply_execution_context(
                base_parameters, second_args, second_context
            )
            worker_parameters = orchestrator._apply_execution_context(
                base_parameters, worker_args, first_context
            )

            first_id, _ = orchestrator.register_cv_run(
                results_root,
                "linear_probe",
                first_parameters,
                {"parameter_source": "default"},
                identity,
            )
            second_id, _ = orchestrator.register_cv_run(
                results_root,
                "linear_probe",
                second_parameters,
                {"parameter_source": "default"},
                identity,
            )
            worker_changed_id, _ = orchestrator.register_cv_run(
                results_root,
                "linear_probe",
                worker_parameters,
                {
                    "parameter_source": "default",
                    "operational_loader": orchestrator._operational_loader_provenance(
                        worker_args
                    ),
                },
                identity,
            )

            self.assertEqual(first_parameters["seed"], 17)
            self.assertEqual(second_parameters["seed"], 18)
            self.assertNotEqual(
                orchestrator.parameter_fingerprint(
                    "linear_probe", first_parameters, identity
                ),
                orchestrator.parameter_fingerprint(
                    "linear_probe", second_parameters, identity
                ),
            )
            self.assertEqual(first_id, "CV_1")
            self.assertEqual(second_id, "CV_2")
            self.assertEqual(worker_changed_id, first_id)
            self.assertEqual(first_parameters, worker_parameters)
            self.assertEqual(
                orchestrator.parameter_fingerprint(
                    "linear_probe", first_parameters, identity
                ),
                orchestrator.parameter_fingerprint(
                    "linear_probe", worker_parameters, identity
                ),
            )

    def test_runtime_identity_uses_validated_checkpoint_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(Path(temporary))
            resolved = SimpleNamespace(sha256="f" * 64)
            provisioned = SimpleNamespace(
                upstream_root=args.upstream_root.resolve(strict=False),
                checkpoint_path=args.pretrained_checkpoint.resolve(strict=False),
                upstream={
                    "revision": args.upstream_revision,
                    "revision_claim_verified_from_scm": True,
                    "tracked_modifications_present": False,
                    "modification_status": "clean_tracked_tree",
                },
                downloads_performed=False,
                git_operations_performed=False,
                to_dict=lambda: {
                    "downloads_performed": False,
                    "git_operations_performed": False,
                    "upstream": {"action": "reused"},
                    "checkpoint": {"action": "reused"},
                },
            )
            with (
                mock.patch.object(
                    orchestrator, "provision_assets", return_value=provisioned
                ) as provision,
                mock.patch.object(
                    orchestrator, "resolve_checkpoint", return_value=resolved
                ) as resolve_checkpoint,
            ):
                identity = orchestrator._runtime_model_identity(
                    args, require_files=True
                )

            provision.assert_called_once()
            resolve_checkpoint.assert_called_once_with(
                args.pretrained_checkpoint.resolve(strict=False),
                expected_sha256=args.pretrained_checkpoint_sha256,
            )
            self.assertEqual(identity["pretrained_checkpoint_sha256"], "f" * 64)
            self.assertEqual(
                identity["upstream_validation"],
                provisioned.upstream,
            )

    def test_execution_context_requires_cache_artifact_integrity_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, num_workers=10)
            split_path = args.dataset_root / "Splits" / "40_percent.json"
            orchestrator.save_json(split_path, {"synthetic": True})
            manifest = {"synthetic": True}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            cache_manifest = {
                "plan_fingerprint": "plan-a",
                "split_manifest_sha256": "split-a",
                "artifact_integrity": {"seal_sha256": "c" * 64},
                "entries": [
                    {"patient_id": "one", "label": 0, "cache_key": "key-one"}
                ],
            }
            context = orchestrator._runtime_execution_context(
                args, manifest, cache_manifest, model_identity
            )
            self.assertEqual(
                context["cache_artifact_integrity_sha256"], "c" * 64
            )
            self.assertEqual(context["schema_version"], 2)
            self.assertNotIn("num_workers", context)
            same_context = orchestrator._runtime_execution_context(
                make_args(root, num_workers=0), manifest, cache_manifest, model_identity
            )
            self.assertEqual(context, same_context)
            loader = orchestrator._operational_loader_provenance(args)
            self.assertEqual(loader["train_num_workers"], 10)
            self.assertTrue(loader["persistent_train_workers"])
            self.assertEqual(loader["validation_num_workers"], 0)
            self.assertEqual(loader["test_num_workers"], 0)
            self.assertFalse(loader["semantic_identity"])

            without_artifact_seal = dict(cache_manifest)
            without_artifact_seal.pop("artifact_integrity")
            with self.assertRaisesRegex(ValueError, "artifact-integrity seal"):
                orchestrator._runtime_execution_context(
                    args, manifest, without_artifact_seal, model_identity
                )


class FoldArtifactTests(unittest.TestCase):
    def test_fold_summary_accepts_wrapped_metrics_and_all_budget_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            percentage_dir = Path(temporary) / "40_percent"
            states = [
                {"best_optimizer_step": 2},
                {"best_iteration": 4},
                {"best_step": 6},
                {"best_epoch": 8},
                {
                    "selected_candidate_id": "head-a",
                    "candidate_best_records": {
                        "head-a": {"optimizer_step": 10}
                    },
                },
            ]
            expected_budgets = [2, 4, 6, 8, 10]
            for fold, state in enumerate(states, start=1):
                directory = percentage_dir / f"fold_{fold:02d}"
                metrics = validation_metrics(f1=fold / 10)
                orchestrator.save_json(directory / "training_state.json", state)
                document = (
                    {"validation_metrics": metrics}
                    if fold % 2
                    else metrics
                )
                orchestrator.save_json(directory / "metrics_summary.json", document)

            rows = orchestrator._fold_summary(
                percentage_dir, {"max_iterations": 99}
            )

            self.assertEqual(
                [row["refit_budget"] for row in rows], expected_budgets
            )
            self.assertEqual(rows[0]["best_optimizer_step"], 2)
            self.assertEqual(rows[1]["best_optimizer_step"], 4)
            self.assertIsNone(rows[3]["best_optimizer_step"])
            self.assertAlmostEqual(rows[4]["f1_macro"], 0.5)
            self.assertAlmostEqual(
                rows[4]["selection_score"], 0.7 * 0.5 + 0.3 * 0.5
            )

    def test_completed_resume_state_compares_all_effective_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "fold_01"
            execution_context = {
                "schema_version": 2,
                "seed": 17,
                "cache_plan_fingerprint": "cache-a",
            }
            registered_parameters = {
                "lr": 0.01,
                "batch_size": 4,
                "seed": 17,
                "_execution_context": execution_context,
            }
            model_identity = {
                "upstream_revision": "fixed-revision",
                "pretrained_checkpoint_sha256": "a" * 64,
            }
            runtime_plan = {
                "method": "linear_probe",
                "max_iterations": 10,
                "training_rows_fingerprint": "b" * 64,
            }
            runtime_plan_fingerprint = orchestrator._runtime_plan_fingerprint(
                registered_parameters, runtime_plan
            )
            checkpoint_path = output_dir / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"synthetic-checkpoint")
            checkpoint_sha256 = orchestrator.sha256_file(checkpoint_path)
            orchestrator.save_json(
                output_dir / "training_state.json",
                {
                    "completed": True,
                    "parameters": registered_parameters,
                    "parameters_fingerprint": orchestrator.canonical_sha256(
                        registered_parameters
                    ),
                    "runtime_plan": runtime_plan,
                    "runtime_plan_fingerprint": runtime_plan_fingerprint,
                    "seed": 17,
                    "source_provenance": {
                        "upstream_revision": "fixed-revision",
                        "checkpoint_sha256": "a" * 64,
                    },
                    "best_model_sha256": checkpoint_sha256,
                    "best_model_binding": {
                        "checkpoint_sha256": checkpoint_sha256,
                        "parameters_fingerprint": orchestrator.canonical_sha256(
                            registered_parameters
                        ),
                        "runtime_plan_fingerprint": runtime_plan_fingerprint,
                        "training_rows_fingerprint": "b" * 64,
                    },
                },
            )
            orchestrator.save_json(
                output_dir / "metrics_summary.json", validation_metrics(0.5)
            )
            self.assertTrue(
                orchestrator._is_completed(
                    output_dir, registered_parameters, model_identity
                )
            )
            changed_non_context_parameter = {
                **registered_parameters,
                "lr": 0.02,
            }
            with self.assertRaisesRegex(
                RuntimeError, "different effective parameters"
            ):
                orchestrator._is_completed(
                    output_dir,
                    changed_non_context_parameter,
                    model_identity,
                )
            checkpoint_path.write_bytes(b"mutated-checkpoint")
            with self.assertRaisesRegex(
                RuntimeError, "changed after CV sealing"
            ):
                orchestrator._is_completed(
                    output_dir, registered_parameters, model_identity
                )


class SearchNormalizationTests(unittest.TestCase):
    def test_grid_candidate_mapping_and_list_forms_are_normalized(self):
        candidates = [
            {"candidate_id": "head-a"},
            {"candidate_id": "head-b"},
        ]
        metrics_a = validation_metrics(0.4)
        metrics_b = validation_metrics(0.6)
        mapping_result = {
            "candidates": {
                "head-a": {"metrics": metrics_a},
                "head-b": {"metrics": metrics_b},
            }
        }
        normalized = orchestrator._normalise_grid_candidate_results(
            mapping_result, candidates
        )
        self.assertEqual(
            {item["candidate_id"] for item in normalized},
            {"head-a", "head-b"},
        )
        self.assertEqual(
            orchestrator._metric_record(
                "head-a", 2, {"nominal_lr": 0.1}, normalized[0]
            )["fold"],
            2,
        )

        list_result = {
            "candidates": [
                {"candidate_id": "head-a", "metrics": metrics_a},
                {"candidate_id": "head-b", "metrics": metrics_b},
            ]
        }
        self.assertEqual(
            len(
                orchestrator._normalise_grid_candidate_results(
                    list_result, candidates
                )
            ),
            2,
        )

    def test_grid_candidate_identity_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "candidate mismatch"):
            orchestrator._normalise_grid_candidate_results(
                {
                    "candidates": {
                        "unexpected": {
                            "metrics": validation_metrics(0.5)
                        }
                    }
                },
                [{"candidate_id": "expected"}],
            )

    def test_selected_search_parameters_reject_stale_execution_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            search_config = root / "search.json"
            orchestrator.save_json(
                search_config,
                {"method": "linear_probe", "strategy": "synthetic"},
            )
            args = make_args(root, search_config=search_config)
            selection_path = (
                args.results_root
                / "LinearProbe"
                / "Grid"
                / "40_percent"
                / "selected_parameters.json"
            )
            stale_context = {
                "schema_version": 2,
                "seed": 17,
                "split_manifest_semantic_sha256": "old-split",
                "cache_plan_fingerprint": "old-cache",
            }
            orchestrator.save_json(
                selection_path,
                {
                    "parameter_source": "search",
                    "parameters": {
                        "lr": 0.01,
                        "_execution_context": stale_context,
                    },
                    "candidate_id": "candidate-a",
                    "search_config_sha256": orchestrator.sha256_file(
                        search_config
                    ),
                    "execution_context": stale_context,
                    "execution_context_fingerprint": orchestrator.canonical_sha256(
                        stale_context
                    ),
                },
            )
            current_context = {
                **stale_context,
                "split_manifest_semantic_sha256": "new-split",
            }

            with self.assertRaisesRegex(
                RuntimeError, "different split, cache plan, seed"
            ):
                orchestrator._selected_parameters_from_search(
                    args,
                    "linear_probe",
                    "search",
                    current_context,
                )

            with self.assertRaisesRegex(RuntimeError, "Rerun the"):
                orchestrator._apply_execution_context(
                    {"lr": 0.01, "_execution_context": stale_context},
                    args,
                    current_context,
                )


class ExternalEvidenceNotebookContractTests(unittest.TestCase):
    def test_results_notebook_does_not_attribute_evidence_to_external_checkpoint(self):
        notebook_path = (
            SYSTEMATIC_ROOT
            / "Notebooks"
            / "3DINO"
            / "3DINO_Test_Results_Summary.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

        self.assertIn(
            "External classifier checkpoint; selection history and publication evidence unverified",
            source,
        )
        self.assertIn("profile_type = 'external_checkpoint_unverified'", source)
        self.assertIn("parameter_source = 'external_checkpoint_unverified'", source)
        self.assertIn("attributed_parameters = {}", source)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class MockedStageFlowTests(unittest.TestCase):
    def test_completed_cv_resume_rejects_missing_or_tampered_fold_and_refit_checkpoints(self):
        cases = (
            ("missing_fold", "fold_03", "delete"),
            ("tampered_fold", "fold_03", "tamper"),
            ("missing_refit", "refit_full_development", "delete"),
            ("tampered_refit", "refit_full_development", "tamper"),
        )
        for case_name, classifier_dir, mutation in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = make_args(root, resume=True, stage="cv", percentage=40)
                parameters = {
                    "lr": 0.01,
                    "seed": args.seed,
                    "_execution_context": {
                        "seed": args.seed,
                        "cache_plan_fingerprint": "cache-a",
                    },
                }
                provenance = {"parameter_source": "default"}
                model_identity = {
                    "upstream_revision": args.upstream_revision,
                    "pretrained_checkpoint_sha256": (
                        args.pretrained_checkpoint_sha256
                    ),
                }
                cv_id, _ = orchestrator.register_cv_run(
                    args.results_root,
                    "linear_probe",
                    parameters,
                    provenance,
                    model_identity,
                )
                percentage_dir, _ = write_synthetic_cv_completion(
                    root,
                    parameters=parameters,
                    model_identity=model_identity,
                    cv_id=cv_id,
                )
                checkpoint = percentage_dir / classifier_dir / "best_model.pt"
                if mutation == "delete":
                    checkpoint.unlink()
                else:
                    checkpoint.write_bytes(b"tampered-after-cv-completion")

                with (
                    mock.patch.object(
                        orchestrator, "cross_validation_split"
                    ) as split,
                    mock.patch.object(
                        orchestrator, "_train_single"
                    ) as train_single,
                    mock.patch.object(orchestrator, "refit_model") as refit,
                    mock.patch.object(
                        orchestrator, "_publish_selected_grid_fold"
                    ) as publish,
                    self.assertRaisesRegex(
                        (FileNotFoundError, RuntimeError),
                        "[Cc]lassifier checkpoint",
                    ),
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

    def test_mocked_cv_uses_linear_state_budget_and_refits_median(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root)
            parameters = {
                "lr": 0.01,
                "nominal_lr": 0.01,
                "n_last_blocks": 1,
                "use_mean_patch_token": False,
                "max_iterations": 50,
            }
            provenance = {"parameter_source": "default"}
            identity = {
                "model_repo_id": "AICONSlab/3DINO-ViT",
                "upstream_repository": "https://github.com/AICONSlab/3DINO",
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            budgets = [1, 3, 5, 7, 9]
            f1_values = [0.2, 0.3, 0.8, 0.4, 0.5]
            training_calls: list[dict] = []
            refit_calls: list[dict] = []
            write_order: list[str] = []
            original_save_json = orchestrator.save_json

            def tracked_save_json(path, value):
                write_order.append(Path(path).name)
                return original_save_json(path, value)

            def split(_manifest, fold):
                return (
                    {"partition": "training", "fold": fold},
                    {"partition": "validation", "fold": fold},
                )

            def rows(_args, mapping):
                return [
                    {
                        "patient_id": f"{mapping['partition']}-{mapping.get('fold', 0)}",
                        "label": 0,
                        "label_str": CLASS_NAMES[0],
                    }
                ]

            def train_single(
                method,
                train_rows,
                validation_rows,
                selected_parameters,
                output_dir,
                runtime_args,
                *,
                budget=None,
            ):
                fold = int(output_dir.name.split("_")[-1])
                training_calls.append(
                    {
                        "method": method,
                        "train_rows": train_rows,
                        "validation_rows": validation_rows,
                        "parameters": dict(selected_parameters),
                        "budget": budget,
                        "base_checkpoint": runtime_args.pretrained_checkpoint,
                    }
                )
                state = write_synthetic_completed_classifier(
                    output_dir,
                    method=method,
                    parameters=dict(selected_parameters),
                    training_rows=train_rows,
                    model_identity=identity,
                    max_iterations=int(selected_parameters["max_iterations"]),
                    state_extra={
                    "selected_candidate_id": "resolved-head",
                    "candidate_best_records": {
                        "resolved-head": {
                            "optimizer_step": budgets[fold - 1]
                        }
                    },
                    },
                    metrics={
                        "validation_metrics": validation_metrics(
                            f1_values[fold - 1]
                        )
                    },
                )
                return {"state": state}

            def refit(rows, selected_parameters, output_dir, **kwargs):
                refit_calls.append(
                    {
                        "rows": rows,
                        "parameters": dict(selected_parameters),
                        "output_dir": output_dir,
                        **kwargs,
                    }
                )
                refit_parameters = {
                    **dict(selected_parameters),
                    "evaluation_interval_iterations": int(
                        kwargs["refit_budget"]
                    ),
                    "early_stopping_patience": 0,
                }
                state = write_synthetic_completed_classifier(
                    output_dir,
                    method=kwargs["method"],
                    parameters=refit_parameters,
                    training_rows=rows,
                    model_identity=identity,
                    max_iterations=int(kwargs["refit_budget"]),
                    state_extra={"refit_budget": int(kwargs["refit_budget"])},
                )
                return {"state": state}

            with (
                mock.patch.object(
                    orchestrator, "cross_validation_split", side_effect=split
                ),
                mock.patch.object(
                    orchestrator,
                    "development_mapping",
                    return_value={"partition": "development"},
                ),
                mock.patch.object(orchestrator, "_split_rows", side_effect=rows),
                mock.patch.object(orchestrator, "validate_no_overlap") as no_overlap,
                mock.patch.object(
                    orchestrator, "_train_single", side_effect=train_single
                ),
                mock.patch.object(
                    orchestrator, "refit_model", side_effect=refit
                ),
                mock.patch.object(
                    orchestrator, "save_json", side_effect=tracked_save_json
                ),
            ):
                result = orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    provenance,
                    identity,
                )
                args.resume = True
                resumed_result = orchestrator.run_cv(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    provenance,
                    identity,
                )

            self.assertEqual(result["cv_id"], "CV_1")
            self.assertEqual(result["best_fold"], 3)
            self.assertEqual(result["refit_budget"], 5)
            self.assertEqual(
                resumed_result["cv_completion_seal_fingerprint"],
                result["cv_completion_seal_fingerprint"],
            )
            self.assertEqual(len(training_calls), 5)
            self.assertEqual(no_overlap.call_count, 5)
            self.assertEqual(len(refit_calls), 1)
            self.assertEqual(refit_calls[0]["refit_budget"], 5)
            self.assertEqual(refit_calls[0]["method"], "linear_probe")
            self.assertEqual(
                refit_calls[0]["checkpoint_path"],
                args.pretrained_checkpoint,
            )
            best_document = orchestrator.read_json(
                Path(result["percentage_directory"]) / "best_fold_model.json"
            )
            self.assertEqual(best_document["best_fold"], 3)
            self.assertFalse(
                best_document["heldout_test_used_during_selection"]
            )
            completion_path = Path(result["cv_completion"])
            self.assertTrue(completion_path.is_file())
            self.assertEqual(write_order[-1], "cv_completion.json")
            completion = orchestrator.read_json(completion_path)
            self.assertTrue(completion["complete"])
            self.assertEqual(len(completion["fold_models"]), 5)
            self.assertEqual(completion["best_fold"], 3)
            self.assertEqual(completion["refit_budget"], 5)
            self.assertEqual(
                completion["refit_model"]["effective_checkpoint_parameters"][
                    "evaluation_interval_iterations"
                ],
                5,
            )
            self.assertEqual(
                completion["refit_model"]["effective_checkpoint_parameters"][
                    "early_stopping_patience"
                ],
                0,
            )
            self.assertEqual(
                len(
                    {
                        record["checkpoint_sha256"]
                        for record in completion["fold_models"]
                    }
                ),
                5,
            )
            for fold, record in enumerate(completion["fold_models"], start=1):
                self.assertEqual(
                    record["classifier_identity"],
                    {"role": "cv_fold", "fold": fold},
                )
                self.assertEqual(record["method"], "linear_probe")
                self.assertEqual(
                    record["effective_checkpoint_parameters"], parameters
                )
                self.assertEqual(len(record["checkpoint_sha256"]), 64)
                self.assertEqual(len(record["training_rows_fingerprint"]), 64)
                self.assertEqual(len(record["runtime_plan_fingerprint"]), 64)

    def test_prediction_kwargs_keep_base_checkpoint_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, resume=True, num_workers=10)
            kwargs = orchestrator._prediction_kwargs(args)

            self.assertEqual(kwargs["checkpoint_path"], args.pretrained_checkpoint)
            self.assertEqual(
                kwargs["expected_checkpoint_sha256"],
                args.pretrained_checkpoint_sha256,
            )
            self.assertEqual(kwargs["num_workers"], 0)
            self.assertNotIn("resume", kwargs)
            self.assertNotIn("seed", kwargs)
            self.assertNotIn("classifier_checkpoint_path", kwargs)

            parsed = build_parser("linear_probe").parse_args(
                [
                    "--base_checkpoint",
                    str(root / "base.pth"),
                    "--model_source",
                    "provided_checkpoint",
                    "--checkpoint",
                    str(root / "classifier.pt"),
                ]
            )
            self.assertEqual(parsed.pretrained_checkpoint, root / "base.pth")
            self.assertEqual(parsed.checkpoint, root / "classifier.pt")

    def test_internal_model_sources_require_final_cv_completion_seal(self):
        for model_source in (
            "refit_full_development",
            "best_cv_fold",
            "cv_ensemble",
        ):
            with self.subTest(model_source=model_source), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = make_args(
                    root,
                    stage="test",
                    model_source=model_source,
                )
                cv_dir = args.results_root / "LinearProbe" / "CV" / "CV_3"
                percentage_dir = cv_dir / "40_percent"
                registered_parameters = {
                    "lr": 0.01,
                    "seed": args.seed,
                    "_execution_context": {
                        "seed": args.seed,
                        "cache_plan_fingerprint": "cache-a",
                    },
                }
                model_identity = {
                    "upstream_revision": args.upstream_revision,
                    "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
                }
                mismatched_payload = {
                    "method": "linear_probe",
                    "classes": list(CLASS_NAMES),
                    "parameters": {
                        **registered_parameters,
                        "lr": 0.02,
                    },
                    "source_provenance": {
                        "upstream_revision": args.upstream_revision,
                        "checkpoint_sha256": args.pretrained_checkpoint_sha256,
                    },
                    "runtime_plan": {"max_iterations": 5},
                }

                if model_source == "refit_full_development":
                    checkpoint = (
                        percentage_dir
                        / "refit_full_development"
                        / "best_model.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(mismatched_payload, checkpoint)
                    orchestrator.save_json(
                        checkpoint.parent / "training_state.json",
                        {"refit_budget": 5},
                    )
                elif model_source == "best_cv_fold":
                    checkpoint = percentage_dir / "fold_01" / "best_model.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(mismatched_payload, checkpoint)
                    orchestrator.save_json(
                        percentage_dir / "best_fold_model.json",
                        {"checkpoint": str(checkpoint)},
                    )
                else:
                    for fold in range(1, 6):
                        checkpoint = (
                            percentage_dir
                            / f"fold_{fold:02d}"
                            / "best_model.pt"
                        )
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(mismatched_payload, checkpoint)

                with (
                    mock.patch.object(
                        orchestrator,
                        "_resolve_cv",
                        return_value=("CV_3", cv_dir),
                    ),
                    mock.patch.object(orchestrator, "_split_rows") as split_rows,
                    self.assertRaisesRegex(
                        RuntimeError, "requires the final CV completion seal"
                    ),
                ):
                    orchestrator.run_test(
                        args,
                        "linear_probe",
                        {"synthetic": True},
                        registered_parameters,
                        {"parameter_source": "default"},
                        model_identity,
                    )
                split_rows.assert_not_called()

    def test_internal_single_sources_use_sealed_identity_and_refit_overrides(self):
        for model_source in ("best_cv_fold", "refit_full_development"):
            with self.subTest(model_source=model_source), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = make_args(root, stage="test", model_source=model_source)
                parameters = {
                    "lr": 0.01,
                    "seed": args.seed,
                    "_execution_context": {
                        "seed": args.seed,
                        "cache_plan_fingerprint": "cache-a",
                    },
                }
                model_identity = {
                    "upstream_revision": args.upstream_revision,
                    "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
                }
                percentage_dir, seal = write_synthetic_cv_completion(
                    root,
                    parameters=parameters,
                    model_identity=model_identity,
                    best_fold=3,
                )
                cv_dir = percentage_dir.parent
                args.cache_dir.mkdir(parents=True, exist_ok=True)
                orchestrator.save_json(
                    args.cache_dir / "cache_manifest.json",
                    {"schema_version": 1},
                )
                heldout_rows = [
                    {
                        "patient_id": "heldout",
                        "label": 0,
                        "label_str": CLASS_NAMES[0],
                    }
                ]
                development_rows = [
                    {
                        "patient_id": "development",
                        "label": 0,
                        "label_str": CLASS_NAMES[0],
                    }
                ]

                def split_rows(_args, mapping):
                    return heldout_rows if mapping == "heldout" else development_rows

                def evaluate_artifacts(_rows, _checkpoint, output_dir, **_kwargs):
                    metrics = validation_metrics(0.7)
                    orchestrator.save_json(
                        output_dir / "metrics_summary.json", metrics
                    )
                    orchestrator.save_json(
                        output_dir / "test_patient_predictions.json", []
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "test_probabilities.csv").write_text(
                        "patient_id\nheldout\n", encoding="utf-8"
                    )
                    return metrics

                with (
                    mock.patch.object(
                        orchestrator,
                        "_resolve_cv",
                        return_value=("CV_3", cv_dir),
                    ),
                    mock.patch.object(
                        orchestrator, "heldout_mapping", return_value="heldout"
                    ),
                    mock.patch.object(
                        orchestrator,
                        "development_mapping",
                        return_value="development",
                    ),
                    mock.patch.object(
                        orchestrator, "_split_rows", side_effect=split_rows
                    ),
                    mock.patch.object(orchestrator, "validate_no_overlap"),
                    mock.patch.object(
                        orchestrator,
                        "evaluate_checkpoint",
                        side_effect=evaluate_artifacts,
                    ),
                ):
                    result = orchestrator.run_test(
                        args,
                        "linear_probe",
                        {"synthetic": True},
                        parameters,
                        {"parameter_source": "default"},
                        model_identity,
                    )

                selection = orchestrator.read_json(
                    Path(result["output_directory"]) / "model_selection.json"
                )
                classifier_record = selection["classifier_checkpoints"][0]
                self.assertTrue(classifier_record["cv_seal_binding_verified"])
                self.assertEqual(
                    classifier_record["binding_status"],
                    "verified_against_final_cv_completion_seal",
                )
                self.assertEqual(
                    selection["cv_completion_seal_fingerprint"],
                    seal["seal_fingerprint"],
                )
                self.assertEqual(
                    selection["checkpoint_parameter_binding"],
                    "internal_classifier_bound_by_final_cv_completion_seal",
                )
                self.assertFalse(
                    selection["heldout_test_used_during_selection"]
                )
                if model_source == "refit_full_development":
                    self.assertEqual(
                        selection["checkpoint_training_parameters"][
                            "evaluation_interval_iterations"
                        ],
                        7,
                    )
                    self.assertEqual(
                        selection["checkpoint_training_parameters"][
                            "early_stopping_patience"
                        ],
                        0,
                    )
                    self.assertEqual(
                        classifier_record["classifier_identity"],
                        {"role": "refit_full_development", "fold": None},
                    )
                else:
                    self.assertEqual(
                        classifier_record["classifier_identity"],
                        {"role": "cv_fold", "fold": 3},
                    )

    def test_internal_checkpoint_mutation_is_rejected_before_heldout_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="test", model_source="best_cv_fold")
            parameters = {"lr": 0.01, "seed": args.seed}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            percentage_dir, _ = write_synthetic_cv_completion(
                root,
                parameters=parameters,
                model_identity=model_identity,
                best_fold=1,
            )
            (percentage_dir / "fold_01" / "best_model.pt").write_bytes(
                b"mutated-after-sealing"
            )
            with (
                mock.patch.object(
                    orchestrator,
                    "_resolve_cv",
                    return_value=("CV_3", percentage_dir.parent),
                ),
                mock.patch.object(orchestrator, "_split_rows") as split_rows,
                self.assertRaisesRegex(RuntimeError, "changed after CV sealing"),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    {"parameter_source": "default"},
                    model_identity,
                )
            split_rows.assert_not_called()

    def test_run_test_rejects_resealed_checkpoint_parameter_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="test", model_source="best_cv_fold")
            parameters = {"lr": 0.01, "seed": args.seed}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            percentage_dir, seal = write_synthetic_cv_completion(
                root,
                parameters=parameters,
                model_identity=model_identity,
                best_fold=1,
            )
            checkpoint = percentage_dir / "fold_01" / "best_model.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["parameters"] = {**parameters, "lr": 0.02}
            payload["parameters_fingerprint"] = orchestrator.canonical_sha256(
                payload["parameters"]
            )
            payload["runtime_plan_fingerprint"] = (
                orchestrator._runtime_plan_fingerprint(
                    payload["parameters"], payload["runtime_plan"]
                )
            )
            torch.save(payload, checkpoint)
            changed_sha = orchestrator.sha256_file(checkpoint)

            state_path = checkpoint.parent / "training_state.json"
            state = orchestrator.read_json(state_path)
            state["best_model_sha256"] = changed_sha
            state["best_model_binding"]["checkpoint_sha256"] = changed_sha
            state["best_model_binding"]["checkpoint_size_bytes"] = int(
                checkpoint.stat().st_size
            )
            orchestrator.save_json(state_path, state)

            fold_record = seal["fold_models"][0]
            fold_record["checkpoint_sha256"] = changed_sha
            fold_record["checkpoint_size_bytes"] = int(checkpoint.stat().st_size)
            fold_record["training_state_sha256"] = orchestrator.sha256_file(
                state_path
            )
            seal["best_fold_binding"]["checkpoint_sha256"] = changed_sha
            seal.pop("seal_fingerprint")
            seal["seal_fingerprint"] = orchestrator.canonical_sha256(seal)
            orchestrator.save_json(
                percentage_dir / "cv_completion.json", seal
            )

            with (
                mock.patch.object(
                    orchestrator,
                    "_resolve_cv",
                    return_value=("CV_3", percentage_dir.parent),
                ),
                mock.patch.object(orchestrator, "_split_rows") as split_rows,
                self.assertRaisesRegex(RuntimeError, "parameters do not match"),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    {"parameter_source": "default"},
                    model_identity,
                )
            split_rows.assert_not_called()

    def test_best_fold_selection_cannot_switch_fold_after_sealing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="test", model_source="best_cv_fold")
            parameters = {"lr": 0.01, "seed": args.seed}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            percentage_dir, seal = write_synthetic_cv_completion(
                root,
                parameters=parameters,
                model_identity=model_identity,
                best_fold=1,
            )
            switched_checkpoint = percentage_dir / "fold_02" / "best_model.pt"
            orchestrator.save_json(
                percentage_dir / "best_fold_model.json",
                {
                    "cv_id": "CV_3",
                    "method": "linear_probe",
                    "best_fold": 2,
                    "checkpoint": str(switched_checkpoint.resolve()),
                },
            )
            seal["artifacts"]["best_fold_model.json"] = orchestrator.sha256_file(
                percentage_dir / "best_fold_model.json"
            )
            seal.pop("seal_fingerprint")
            seal["seal_fingerprint"] = orchestrator.canonical_sha256(seal)
            orchestrator.save_json(
                percentage_dir / "cv_completion.json", seal
            )

            with (
                mock.patch.object(
                    orchestrator,
                    "_resolve_cv",
                    return_value=("CV_3", percentage_dir.parent),
                ),
                mock.patch.object(orchestrator, "_split_rows") as split_rows,
                self.assertRaisesRegex(
                    RuntimeError, "Best-fold selection document differs"
                ),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    {"parameter_source": "default"},
                    model_identity,
                )
            split_rows.assert_not_called()

    def test_cv_ensemble_rejects_duplicate_classifier_sha_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_args(root, stage="test", model_source="cv_ensemble")
            parameters = {"lr": 0.01, "seed": args.seed}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }
            percentage_dir, seal = write_synthetic_cv_completion(
                root,
                parameters=parameters,
                model_identity=model_identity,
                duplicate_fold_sha=True,
            )
            self.assertEqual(
                seal["fold_models"][0]["checkpoint_sha256"],
                seal["fold_models"][1]["checkpoint_sha256"],
            )
            with (
                mock.patch.object(
                    orchestrator,
                    "_resolve_cv",
                    return_value=("CV_3", percentage_dir.parent),
                ),
                mock.patch.object(orchestrator, "_split_rows") as split_rows,
                self.assertRaisesRegex(RuntimeError, "duplicate classifier SHA-256"),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    {"parameter_source": "default"},
                    model_identity,
                )
            split_rows.assert_not_called()

    def test_mocked_provided_checkpoint_test_passes_classifier_positionally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier = root / "trained-classifier.pt"
            classifier.write_bytes(b"synthetic-classifier-checkpoint")
            args = make_args(
                root,
                stage="test",
                model_source="provided_checkpoint",
                checkpoint=classifier,
            )
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            (args.cache_dir / "cache_manifest.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            cv_dir = args.results_root / "LinearProbe" / "CV" / "CV_7"
            heldout_rows = [
                {
                    "patient_id": "test-1",
                    "label": 0,
                    "label_str": CLASS_NAMES[0],
                }
            ]
            development_rows = [
                {
                    "patient_id": "development-1",
                    "label": 0,
                    "label_str": CLASS_NAMES[0],
                }
            ]
            evaluated_metrics = validation_metrics(0.75, 0.7)
            parameters = {
                "lr": 0.01,
                "_execution_context": {
                    "seed": args.seed,
                    "split_manifest_semantic_sha256": "synthetic-split",
                },
            }
            provenance = {"parameter_source": "default"}
            model_identity = {
                "upstream_revision": args.upstream_revision,
                "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
            }

            def split_rows(_args, mapping):
                return (
                    heldout_rows
                    if mapping == {"partition": "heldout"}
                    else development_rows
                )

            def evaluate_artifacts(
                rows, classifier_checkpoint, output_dir, **_kwargs
            ):
                self.assertEqual(rows, heldout_rows)
                self.assertEqual(classifier_checkpoint, classifier)
                orchestrator.save_json(
                    output_dir / "metrics_summary.json", evaluated_metrics
                )
                orchestrator.save_json(
                    output_dir / "test_patient_predictions.json",
                    [{"patient_id": "test-1", "predicted_class": CLASS_NAMES[0]}],
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "test_probabilities.csv").write_text(
                    "patient_id,predicted_class\ntest-1,Appendicitis\n",
                    encoding="utf-8",
                )
                return evaluated_metrics

            with (
                mock.patch.object(
                    orchestrator, "_resolve_cv", return_value=("CV_7", cv_dir)
                ),
                mock.patch.object(
                    orchestrator,
                    "heldout_mapping",
                    return_value={"partition": "heldout"},
                ),
                mock.patch.object(
                    orchestrator,
                    "development_mapping",
                    return_value={"partition": "development"},
                ),
                mock.patch.object(
                    orchestrator, "_split_rows", side_effect=split_rows
                ),
                mock.patch.object(orchestrator, "validate_no_overlap") as no_overlap,
                mock.patch.object(
                    orchestrator,
                    "evaluate_checkpoint",
                    side_effect=evaluate_artifacts,
                ) as evaluate,
            ):
                result = orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    provenance,
                    model_identity,
                )

            no_overlap.assert_called_once_with(development_rows, heldout_rows)
            positional = evaluate.call_args.args
            keyword = evaluate.call_args.kwargs
            self.assertEqual(positional[0], heldout_rows)
            self.assertEqual(positional[1], classifier)
            self.assertEqual(keyword["checkpoint_path"], args.pretrained_checkpoint)
            self.assertEqual(keyword["method"], "linear_probe")
            self.assertNotIn("resume", keyword)
            self.assertNotIn("seed", keyword)
            self.assertEqual(result["metrics"], evaluated_metrics)
            output_dir = Path(result["output_directory"])
            artifact_names = set(orchestrator._TEST_ARTIFACT_NAMES)
            self.assertEqual(set(result["completion"]["artifacts"]), artifact_names)
            for artifact_name in artifact_names:
                artifact = output_dir / artifact_name
                self.assertTrue(artifact.is_file())
                self.assertEqual(
                    result["completion"]["artifacts"][artifact_name],
                    orchestrator.sha256_file(artifact),
                )
            completion_path = output_dir / "test_completion.json"
            self.assertTrue(completion_path.is_file())
            self.assertTrue(
                orchestrator.read_json(completion_path)["complete"]
            )

            selection = orchestrator.read_json(output_dir / "model_selection.json")
            self.assertEqual(
                selection["checkpoints"], [str(classifier.resolve())]
            )
            classifier_sha = orchestrator.sha256_file(classifier)
            self.assertEqual(
                selection["classifier_checkpoints"][0]["sha256"],
                classifier_sha,
            )
            self.assertEqual(
                selection["test_identity"]["classifier_checkpoint_sha256"],
                [classifier_sha],
            )
            self.assertEqual(
                selection["parameters_role"],
                "requested_cv_context_not_attributed_to_external_checkpoint",
            )
            self.assertEqual(
                selection["checkpoint_parameter_binding"],
                "external_checkpoint_not_asserted_to_match_requested_cv_parameters",
            )
            self.assertEqual(
                selection["selection_provenance_status"],
                "unknown_external_checkpoint_history",
            )
            self.assertIsNone(selection["parameters"])
            self.assertEqual(selection["cv_selection_parameters"], parameters)
            self.assertIsNone(selection["checkpoint_training_parameters"])
            self.assertFalse(
                selection["classifier_checkpoints"][0]["binding_verified"]
            )
            self.assertEqual(
                selection["classifier_checkpoints"][0]["binding_status"],
                "external_checkpoint_parameters_and_selection_history_unverified",
            )
            self.assertIsNone(
                selection["heldout_test_used_during_selection"]
            )

            args.resume = True
            with mock.patch.object(
                orchestrator, "_resolve_cv", return_value=("CV_7", cv_dir)
            ):
                reused = orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    provenance,
                    model_identity,
                )
            self.assertTrue(reused["reused_completed_result"])
            self.assertEqual(evaluate.call_count, 1)

            orchestrator.save_json(
                output_dir / "metrics_summary.json", {"mutated": True}
            )
            with (
                mock.patch.object(
                    orchestrator, "_resolve_cv", return_value=("CV_7", cv_dir)
                ),
                self.assertRaisesRegex(RuntimeError, "artifact changed"),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    parameters,
                    provenance,
                    model_identity,
                )
            self.assertEqual(evaluate.call_count, 1)

    def test_resume_rejects_partial_heldout_directory_without_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier = root / "trained-classifier.pt"
            classifier.write_bytes(b"synthetic-classifier-checkpoint")
            args = make_args(
                root,
                stage="test",
                model_source="provided_checkpoint",
                checkpoint=classifier,
                resume=True,
            )
            cv_dir = args.results_root / "LinearProbe" / "CV" / "CV_8"
            partial_dir = (
                args.results_root
                / "LinearProbe"
                / "Test"
                / "CV_8"
                / "provided_checkpoint"
                / "40_percent"
            )
            orchestrator.save_json(
                partial_dir / "metrics_summary.json", {"partial": True}
            )

            with (
                mock.patch.object(
                    orchestrator, "_resolve_cv", return_value=("CV_8", cv_dir)
                ),
                self.assertRaisesRegex(RuntimeError, "partial"),
            ):
                orchestrator.run_test(
                    args,
                    "linear_probe",
                    {"synthetic": True},
                    {"lr": 0.01},
                    {"parameter_source": "default"},
                    {
                        "upstream_revision": args.upstream_revision,
                        "pretrained_checkpoint_sha256": args.pretrained_checkpoint_sha256,
                    },
                )


if __name__ == "__main__":
    unittest.main()
