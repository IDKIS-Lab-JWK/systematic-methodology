"""Synthetic dry-run checks that never load weights, cache tensors, or patients."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import CLASS_NAMES  # noqa: E402
from Code.Models.ThreeDINO import cli  # noqa: E402
from Code.Models.ThreeDINO.cli import build_parser  # noqa: E402
from Code.Models.ThreeDINO.orchestrator import run_pipeline  # noqa: E402


def write_manifest(dataset_root: Path) -> None:
    folds = [
        {
            "fold": fold,
            "patients_by_class": {
                name: [f"{index}{fold:03d}"]
                for index, name in enumerate(CLASS_NAMES, start=1)
            },
        }
        for fold in range(1, 6)
    ]
    heldout = {
        name: [f"{index}900"]
        for index, name in enumerate(CLASS_NAMES, start=1)
    }
    document = {
        "heldout_test": {
            "patients_by_class": heldout,
            "counts_by_class": {name: 1 for name in CLASS_NAMES},
            "total": 5,
        },
        "cross_validation": {"number_of_folds": 5, "folds": folds},
        "metadata": {"percentage": 100},
    }
    split_dir = dataset_root / "Splits"
    split_dir.mkdir(parents=True)
    (split_dir / "100_percent.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def parse(method: str, dataset: Path, results: Path, source: str):
    return build_parser(method).parse_args(
        [
            "--dry_run",
            "--stage",
            "all",
            "--percentage",
            "100",
            "--parameter_source",
            source,
            "--dataset_root",
            str(dataset),
            "--results_root",
            str(results),
            "--cache_root",
            str(results / "Cache"),
            "--upstream_root",
            str(results / "missing-upstream-is-allowed-in-dry-run"),
            "--pretrained_checkpoint",
            str(results / "missing-checkpoint-is-allowed-in-dry-run.pth"),
        ]
    )


class DryRunTests(unittest.TestCase):
    def test_parser_is_independent_of_eager_orchestration_and_device_imports(self):
        source = Path(cli.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(
                module.endswith(("orchestrator", "training"))
                for module in top_level_modules
            )
        )

        original = cli._default_device
        try:
            cli._default_device = lambda: (_ for _ in ()).throw(
                AssertionError("parser construction must not probe PyTorch")
            )
            args = build_parser("linear_probe").parse_args(["--dry_run"])
        finally:
            cli._default_device = original
        self.assertIsNone(args.device)
        self.assertEqual(args.num_workers, 0)
        workers_action = next(
            action
            for action in build_parser("linear_probe")._actions
            if action.dest == "num_workers"
        )
        self.assertIn("operational throughput setting", workers_action.help)
        with self.assertRaises(SystemExit):
            build_parser("linear_probe").parse_args(["--num_workers", "-1"])

    def test_grid_names_are_canonical_and_search_names_remain_aliases(self):
        parser = build_parser("linear_probe")
        canonical = parser.parse_args(
            [
                "--stage",
                "grid",
                "--parameter_source",
                "grid",
                "--grid_config",
                "canonical.json",
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
                "legacy.json",
                "--max_search_candidates",
                "2",
            ]
        )
        self.assertEqual(canonical.stage, "grid")
        self.assertEqual(canonical.parameter_source, "grid")
        self.assertEqual(canonical.grid_config, Path("canonical.json"))
        self.assertEqual(canonical.max_grid_candidates, 3)
        self.assertEqual(legacy.stage, "grid")
        self.assertEqual(legacy.parameter_source, "grid")
        self.assertEqual(legacy.grid_config, Path("legacy.json"))
        self.assertEqual(legacy.max_grid_candidates, 2)
        self.assertFalse(hasattr(canonical, "search_config"))
        self.assertFalse(hasattr(canonical, "max_search_candidates"))

    def test_publication_grid_rejects_overrides_before_orchestration(self):
        with self.assertRaises(SystemExit):
            cli.main(
                "linear_probe",
                ["--parameter_source", "publication", "--grid_config", "x.json"],
            )
        with self.assertRaises(SystemExit):
            cli.main(
                "linear_probe",
                [
                    "--parameter_source",
                    "publication",
                    "--max_grid_candidates",
                    "1",
                ],
            )

    def test_publication_probe_plans_all_52_heads_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, results = root / "Dataset", root / "Results" / "3DINO"
            write_manifest(dataset)
            args = parse("linear_probe", dataset, results, "publication")
            args.num_workers = 10
            result = run_pipeline(args, "linear_probe")
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["parameters_or_search"]["candidate_count"], 52)
            self.assertNotIn("num_workers", result["parameters_or_search"])
            self.assertEqual(result["operational_loader"]["train_num_workers"], 10)
            self.assertTrue(
                result["operational_loader"]["persistent_train_workers"]
            )
            self.assertEqual(
                result["operational_loader"]["validation_num_workers"], 0
            )
            self.assertEqual(result["operational_loader"]["test_num_workers"], 0)
            self.assertTrue(result["heldout_test_planned_only"])
            self.assertFalse(result["writes_performed"])
            self.assertFalse(results.exists())

    def test_full_finetune_default_remains_unauthorized_in_dry_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, results = root / "Dataset", root / "Results" / "3DINO"
            write_manifest(dataset)
            result = run_pipeline(
                parse("full_finetune", dataset, results, "default"),
                "full_finetune",
            )
            self.assertEqual(result["status"], "dry_run")
            self.assertFalse(result["full_finetune_authorized"])
            self.assertIn(
                "project_adapted_conditional",
                result["parameter_provenance"]["profile_type"],
            )
            self.assertFalse(results.exists())

    def test_curated_full_finetune_search_is_three_project_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, results = root / "Dataset", root / "Results" / "3DINO"
            write_manifest(dataset)
            args = parse("full_finetune", dataset, results, "grid")
            args.stage = "grid"
            result = run_pipeline(args, "full_finetune")
            self.assertEqual(result["parameters_or_search"]["candidate_count"], 3)
            self.assertFalse(result["full_finetune_authorized"])
            self.assertFalse(results.exists())

    def test_no_zero_shot_parser_exists(self):
        with self.assertRaisesRegex(ValueError, "Unknown 3DINO method"):
            build_parser("zero_shot")


if __name__ == "__main__":
    unittest.main()
