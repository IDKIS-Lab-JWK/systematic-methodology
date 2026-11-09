"""Regression for Merlin's frozen stem and reentrant image-stage checkpoints."""
import sys
import unittest
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conv_lora import ConvLoRA3d
from model import MerlinClassifier


class CheckpointedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(1, 4, 1)
        self.stem.requires_grad_(False)
        self.layer1 = nn.Sequential(ConvLoRA3d(nn.Conv3d(4, 4, 1)), nn.SiLU())
        self.layer2 = nn.Sequential(ConvLoRA3d(nn.Conv3d(4, 4, 1)), nn.SiLU())
        self.input_required_grad = None

    def forward(self, images):
        self.input_required_grad = images.requires_grad
        images = self.stem(images)
        # Match the installed Merlin image-stage checkpoint behavior exactly.
        images = checkpoint(self.layer1, images, use_reentrant=True)
        images = checkpoint(self.layer2, images, use_reentrant=True)
        return images.mean(dim=(2, 3, 4))


class CheckpointGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.images = torch.randn(2, 1, 3, 3, 3)
        self.labels = torch.tensor([0, 1])

    def model(self, enable):
        return MerlinClassifier(CheckpointedEncoder(), 3, 4, 0.0,
                                checkpoint_input_gradients=enable)

    def test_trainable_adapters_receive_gradients_behind_frozen_stem(self):
        model = self.model(True).train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        adapters = [m for m in model.modules() if isinstance(m, ConvLoRA3d)]
        original_b = [m.lora_b.detach().clone() for m in adapters]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(self.images), self.labels).backward()
            for adapter in adapters:
                for parameter in (adapter.lora_a, adapter.lora_b):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(adapter.lora_b.grad.abs().sum().item(), 0)
                if step:
                    self.assertGreater(adapter.lora_a.grad.abs().sum().item(), 0)
                self.assertIsNone(adapter.convolution.weight.grad)
            self.assertTrue(all(p.grad is None for p in model.encoder.stem.parameters()))
            optimizer.step()
        self.assertTrue(all(not torch.equal(before, m.lora_b) for before, m in zip(original_b, adapters)))
        self.assertFalse(self.images.requires_grad)
        self.assertIsNone(self.images.grad)

    def test_regression_reproduces_head_only_gradients_without_input_policy(self):
        model = self.model(False).train()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            nn.functional.cross_entropy(model(self.images), self.labels).backward()
        self.assertTrue(all(p.grad is None for n, p in model.encoder.named_parameters() if 'lora_' in n))
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_evaluation_and_no_grad_do_not_enable_input_gradients(self):
        model = self.model(True)
        for training in (True, False):
            model.train(training)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                output = model(self.images)
            self.assertFalse(model.encoder.input_required_grad)
            self.assertFalse(output.requires_grad)
            self.assertTrue(torch.isfinite(output).all())
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            model(self.images)
        self.assertFalse(model.encoder.input_required_grad)


if __name__ == '__main__':
    unittest.main()
