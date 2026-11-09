from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


MERLIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MERLIN_DIR))

from training import (  # noqa: E402
    build_grad_scaler,
    build_optimizer,
    build_optimizer_update_scheduler,
    optimizer_group_manifest,
    train_epoch,
)


class DummyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = nn.Parameter(torch.ones(2, 3))
        self.lora_b = nn.Parameter(torch.ones(4, 2))


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            setattr(self, stage, DummyAdapter())


class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DummyEncoder()
        self.classifier = nn.Sequential(
            nn.LayerNorm(4),
            nn.Dropout(0.0),
            nn.Linear(4, 2),
        )


def llrd_parameters() -> dict:
    return {
        "adapter_lr": 1e-4,
        "head_lr": 1e-4,
        "image_layer_decay": 0.7,
        "encoder_weight_decay": 0.01,
        "head_weight_decay": 0.01,
        "exclude_norm_bias_from_weight_decay": True,
        "scheduler_step_unit": "optimizer_update",
    }


class LLRDOptimizerTests(unittest.TestCase):
    def test_stage_rates_decay_groups_and_complete_assignment(self):
        model = DummyClassifier()
        optimizer = build_optimizer(
            model, "conv_lora_llrd", llrd_parameters()
        )
        groups = {group["name"]: group for group in optimizer.param_groups}

        self.assertAlmostEqual(groups["adapter_layer4_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(groups["adapter_layer3_decay"]["lr"], 7e-5)
        self.assertAlmostEqual(groups["adapter_layer2_decay"]["lr"], 4.9e-5)
        self.assertAlmostEqual(groups["adapter_layer1_decay"]["lr"], 3.43e-5)
        self.assertAlmostEqual(groups["classifier_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(groups["classifier_no_decay"]["lr"], 1e-4)

        self.assertEqual(groups["adapter_layer4_decay"]["weight_decay"], 0.01)
        self.assertEqual(groups["classifier_decay"]["weight_decay"], 0.01)
        self.assertEqual(groups["classifier_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(
            groups["classifier_decay"]["parameter_names"],
            ["classifier.2.weight"],
        )
        self.assertEqual(
            set(groups["classifier_no_decay"]["parameter_names"]),
            {"classifier.0.weight", "classifier.0.bias", "classifier.2.bias"},
        )

        assigned = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        expected = [
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(expected))
        manifest = optimizer_group_manifest(optimizer)
        self.assertEqual(manifest["strategy"], "image_conv_lora_llrd")
        self.assertEqual(len(manifest["signature"]), 16)

    def test_unexpected_trainable_encoder_parameter_is_rejected(self):
        model = DummyClassifier()
        model.encoder.extra = nn.Parameter(torch.ones(1))

        with self.assertRaisesRegex(RuntimeError, "unsupported trainable"):
            build_optimizer(model, "conv_lora_llrd", llrd_parameters())

    def test_missing_adapter_stage_is_rejected(self):
        model = DummyClassifier()
        del model.encoder.layer1

        with self.assertRaisesRegex(RuntimeError, "no trainable parameters in layer1"):
            build_optimizer(model, "conv_lora_llrd", llrd_parameters())

    def test_legacy_optimizer_shapes_are_unchanged(self):
        model = DummyClassifier()
        old_parameters = {
            "adapter_lr": 1e-4,
            "encoder_lr": 1e-5,
            "head_lr": 1e-4,
            "encoder_weight_decay": 1e-4,
            "head_weight_decay": 1e-4,
        }
        conv_optimizer = build_optimizer(model, "conv_lora", old_parameters)
        self.assertEqual(
            [group["name"] for group in conv_optimizer.param_groups],
            ["adapters", "classifier"],
        )
        full_optimizer = build_optimizer(model, "full_finetune", old_parameters)
        self.assertEqual(
            [group["name"] for group in full_optimizer.param_groups],
            ["encoder", "classifier"],
        )

    def test_scheduler_advances_once_per_accumulated_optimizer_update(self):
        model = nn.Linear(4, 2)
        loader = [
            {"image": torch.ones(1, 4), "label": torch.tensor([index % 2])}
            for index in range(5)
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = build_optimizer_update_scheduler(
            optimizer,
            total_updates=math.ceil(len(loader) / 2),
            warmup_updates=0,
            minimum_lr_factor=0.0,
        )
        scaler = build_grad_scaler(False)
        initial_scheduler_epoch = scheduler.last_epoch

        _, stats = train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            {"grad_accum_steps": 2, "max_grad_norm": 0.0},
            False,
            update_scheduler=scheduler,
            normalize_partial_accumulation=True,
            return_update_stats=True,
        )

        self.assertEqual(stats["successful_optimizer_updates"], 3)
        self.assertEqual(stats["skipped_optimizer_updates"], 0)
        self.assertEqual(scheduler.last_epoch - initial_scheduler_epoch, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)


if __name__ == "__main__":
    unittest.main()
