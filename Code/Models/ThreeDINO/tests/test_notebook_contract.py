"""Static validation for the 3DINO held-out summary notebook."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK_PATH = (
    SYSTEMATIC_ROOT / "Notebooks" / "3DINO" / "3DINO_Test_Results_Summary.ipynb"
)


class NotebookCompletionContractTests(unittest.TestCase):
    def test_notebook_requires_hashed_completion_marker_and_full_artifact_set(self):
        document = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in document["cells"]
            if cell.get("cell_type") == "code"
        )
        for index, cell in enumerate(document["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")

        self.assertIn("test_completion.json", code)
        self.assertIn("marker.get('complete') is not True", code)
        self.assertIn("marker.get('test_fingerprint'", code)
        self.assertIn("validate_completion(path.parent)", code)
        self.assertIn("sha256_file(artifact_path)", code)
        for artifact in (
            "metrics_summary.json",
            "model_selection.json",
            "test_patient_predictions.json",
            "test_probabilities.csv",
        ):
            self.assertIn(artifact, code)


if __name__ == "__main__":
    unittest.main()
