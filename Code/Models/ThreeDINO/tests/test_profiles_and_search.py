"""Validate publication provenance and deterministic 3DINO candidate expansion."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import CLASS_NAMES  # noqa: E402
from Code.Models.ThreeDINO.search import (  # noqa: E402
    build_probe_candidates,
    candidate_id,
    effective_learning_rate,
    load_search_definition,
    rank_records,
)

MODEL_ROOT = Path(__file__).resolve().parents[1]


def load_profile(name: str) -> dict:
    with (MODEL_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class PublicationProfileTests(unittest.TestCase):
    def test_publication_profile_expands_exactly_thirteen_by_four_heads(self):
        profile = load_profile("publication_linear_probe.json")
        rates = profile["search_space"]["learning_rates_nominal"]
        recipes = profile["search_space"]["representations"]
        self.assertEqual(
            rates,
            [
                0.0001,
                0.0002,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.3,
                0.5,
            ],
        )
        self.assertEqual(
            {(item["n_last_blocks"], item["use_mean_patch_token"]) for item in recipes},
            {(1, False), (1, True), (4, False), (4, True)},
        )
        candidates = build_probe_candidates(profile)
        self.assertEqual(len(candidates), 52)
        self.assertEqual(len({item["candidate_id"] for item in candidates}), 52)
        self.assertTrue(all(item["encoder_frozen"] for item in candidates))
        self.assertTrue(
            math.isclose(min(item["lr"] for item in candidates), 0.0001 / 8)
        )
        self.assertTrue(math.isclose(max(item["lr"] for item in candidates), 0.5 / 8))

    def test_lr_reconciliation_is_explicit_and_reproducible(self):
        profile = load_profile("publication_linear_probe.json")
        self.assertEqual(
            profile["fixed_parameters"]["learning_rate_scale_rule"],
            "global_batch_over_256",
        )
        self.assertIn("paper", profile["learning_rate_reconciliation"])
        self.assertIn("official_code", profile["learning_rate_reconciliation"])
        self.assertAlmostEqual(
            effective_learning_rate(
                0.1,
                batch_size=32,
                world_size=1,
                scale_rule="global_batch_over_256",
            ),
            0.0125,
        )

    def test_all_profiles_preserve_class_order_and_method_status(self):
        names = [
            "publication_linear_probe.json",
            "default_linear_probe.json",
            "search_linear_probe.json",
            "default_full_finetune.json",
            "search_full_finetune.json",
        ]
        for name in names:
            with self.subTest(name=name):
                profile = load_profile(name)
                self.assertEqual(
                    profile["task_adaptations"]["class_order"], list(CLASS_NAMES)
                )
        for name in ("default_full_finetune.json", "search_full_finetune.json"):
            profile = load_profile(name)
            self.assertIn("project_adapted_conditional", profile["profile_type"])
            self.assertFalse(profile["authorization"]["enabled_by_default"])
            self.assertFalse(profile["authorization"]["license_approval_claimed"])
            parameters = profile.get("parameters", profile.get("fixed_parameters"))
            self.assertNotIn("allow_experimental_full_finetune", parameters)
            self.assertNotIn("allow_full_finetune", parameters)
        self.assertFalse((MODEL_ROOT / "publication_full_finetune.json").exists())
        self.assertFalse((MODEL_ROOT / "zero_shot.py").exists())


class CandidateSafetyTests(unittest.TestCase):
    def test_curated_probe_search_loads_and_has_stable_ids(self):
        definition = load_search_definition(MODEL_ROOT / "search_linear_probe.json")
        candidates = definition["resolved_candidates"]
        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidate_id(candidates[0]), candidates[0]["candidate_id"])
        reordered = dict(reversed(list(candidates[0].items())))
        self.assertEqual(candidate_id(reordered), candidates[0]["candidate_id"])

    def test_candidate_rejects_inconsistent_nominal_and_effective_lr(self):
        definition = {
            "fixed_parameters": {
                "batch_size": 32,
                "learning_rate_scale_rule": "global_batch_over_256",
            },
            "candidates": [
                {
                    "nominal_lr": 0.01,
                    "lr": 0.01,
                    "n_last_blocks": 1,
                    "use_mean_patch_token": False,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_probe_candidates(definition)

    def test_rank_is_variance_penalized_then_minority_sensitive(self):
        records = [
            {
                "candidate_id": "stable",
                "mean_f1_macro": 0.70,
                "std_f1_macro": 0.01,
                "mean_balanced_accuracy": 0.70,
                "std_balanced_accuracy": 0.01,
                "minority_perforation_recall": 0.6,
                "nominal_lr": 0.01,
            },
            {
                "candidate_id": "unstable",
                "mean_f1_macro": 0.75,
                "std_f1_macro": 0.20,
                "mean_balanced_accuracy": 0.75,
                "std_balanced_accuracy": 0.20,
                "minority_perforation_recall": 0.9,
                "nominal_lr": 0.01,
            },
        ]
        ranked = rank_records(records)
        self.assertEqual(ranked[0]["candidate_id"], "stable")
        self.assertEqual([item["rank"] for item in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
