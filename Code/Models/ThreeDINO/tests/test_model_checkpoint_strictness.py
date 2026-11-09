"""Synthetic fail-closed checkpoint coverage tests; no official weights are used."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent skip
    torch = None
    nn = None

from Code.Models.ThreeDINO import model as threedino_model  # noqa: E402

TorchModule = nn.Module if nn is not None else object


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class StrictCheckpointCoverageTests(unittest.TestCase):
    class TinyEncoder(TorchModule):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
            self.pos_embed = nn.Parameter(torch.zeros(1, 2, 4))
            self.mask_token = nn.Parameter(torch.zeros(1, 4))
            self.patch_embed = nn.Linear(4, 4)
            self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.norm = nn.LayerNorm(4)

    def _checkpoint(self, root: Path, state: dict) -> Path:
        path = root / "synthetic_teacher.pt"
        torch.save({"teacher": state}, path)
        return path

    def test_exact_encoder_state_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            encoder = self.TinyEncoder()
            report = threedino_model._load_checkpoint_fail_closed(
                encoder,
                self._checkpoint(Path(temporary), encoder.state_dict()),
            )
            self.assertEqual(report["coverage"], 1.0)
            self.assertEqual(report["missing_keys"], [])
            self.assertFalse(report["random_fallback_used"])

    def test_missing_transformer_tensor_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            encoder = self.TinyEncoder()
            state = dict(encoder.state_dict())
            state.pop("blocks.1.weight")
            with self.assertRaisesRegex(
                threedino_model.CheckpointValidationError,
                "does not fully cover",
            ):
                threedino_model._load_checkpoint_fail_closed(
                    encoder, self._checkpoint(Path(temporary), state)
                )

    def test_only_inference_unused_mask_token_may_be_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            encoder = self.TinyEncoder()
            state = dict(encoder.state_dict())
            state.pop("mask_token")
            report = threedino_model._load_checkpoint_fail_closed(
                encoder, self._checkpoint(Path(temporary), state)
            )
            self.assertEqual(report["missing_keys"], ["mask_token"])
            self.assertEqual(report["allowed_missing_keys"], ["mask_token"])


if __name__ == "__main__":
    unittest.main()
