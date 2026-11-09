from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from metrics import compute_metrics, format_predictions  # noqa: E402
from search import candidate_id, load_search_definition, rank_records  # noqa: E402


class MetricsSearchSafetyTests(unittest.TestCase):
    def test_metric_schema_matches_five_classes(self):
        classes = ["A", "B", "C", "D", "E"]
        truth = np.arange(5)
        probability = np.eye(5)
        metrics = compute_metrics(truth, truth, probability, classes)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["class_order"], classes)
        self.assertEqual(set(metrics["per_class"]), set(classes))
        rows, frame = format_predictions([str(i) for i in range(5)], classes, probability, classes)
        self.assertEqual(len(rows), 5)
        self.assertEqual(frame.columns[-1], "patient_id")

    def test_candidate_id_is_key_order_independent(self):
        self.assertEqual(candidate_id({"a": 1, "b": 2}), candidate_id({"b": 2, "a": 1}))

    def test_search_rejects_cartesian_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.json"
            path.write_text(json.dumps({"strategy": "cartesian", "candidates": [{}, {}], "rounds": [{"budget": 1, "keep": 1, "folds": [1]}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_search_definition(path)

    def test_rank_uses_robust_score_then_perforation(self):
        records = [
            {"candidate_id": "b", "mean_f1_macro": 0.7, "std_f1_macro": 0.1, "mean_balanced_accuracy": 0.6, "std_balanced_accuracy": 0.1, "minority_perforation_recall": 0.8},
            {"candidate_id": "a", "mean_f1_macro": 0.8, "std_f1_macro": 0.1, "mean_balanced_accuracy": 0.7, "std_balanced_accuracy": 0.1, "minority_perforation_recall": 0.2},
        ]
        self.assertEqual(rank_records(records)[0]["candidate_id"], "a")

if __name__ == "__main__":
    unittest.main()
