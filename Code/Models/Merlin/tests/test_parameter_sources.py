from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


MERLIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MERLIN_DIR))

from cli import _explicit_parameter_names, build_parser  # noqa: E402
from orchestrator import (  # noqa: E402
    GRID_CONFIG_FILENAMES,
    VARIANT_FOLDERS,
    grid_candidates,
    parse_bool,
    resolve_parameters,
    resolve_cv_run,
)


def parsed_args(variant: str, *arguments: str):
    parser = build_parser(variant)
    values = list(arguments)
    args = parser.parse_args(values)
    args.explicit_parameter_names = _explicit_parameter_names(parser, values)
    args.grid = parse_bool(args.grid)
    args.percentage_specific_grid = parse_bool(args.percentage_specific_grid)
    args.train_image_projection = parse_bool(args.train_image_projection)
    args.gradient_checkpointing = parse_bool(args.gradient_checkpointing)
    if hasattr(args, "exclude_norm_bias_from_weight_decay"):
        args.exclude_norm_bias_from_weight_decay = parse_bool(
            args.exclude_norm_bias_from_weight_decay
        )
    return args


class ParameterSourceTests(unittest.TestCase):
    def test_corrected_adapter_training_does_not_reuse_untagged_cv(self):
        args = parsed_args('conv_lora_llrd', '--grid', 'false')
        parameters, provenance = resolve_parameters(args, 'conv_lora_llrd')
        with tempfile.TemporaryDirectory() as temp:
            args.results_root = Path(temp)
            old = resolve_cv_run(args, 'conv_lora_llrd', parameters, provenance, create=True)
            path = old['cv_dir'] / 'cv_parameters.json'
            document = json.loads(path.read_text())
            document.pop('training_implementation')
            path.write_text(json.dumps(document))
            corrected = resolve_cv_run(args, 'conv_lora_llrd', parameters, provenance, create=True)
            self.assertEqual(corrected['cv_id'], 'CV_2')
            self.assertNotIn('training_implementation', json.loads(path.read_text()))
            matched = resolve_cv_run(args, 'conv_lora_llrd', parameters, provenance, create=False)
            self.assertEqual(matched['cv_id'], 'CV_2')
            self.assertEqual(corrected['document']['parameters'], parameters)

    def test_auto_grid_false_preserves_full_project_defaults(self):
        args = parsed_args("full_finetune", "--grid", "false")

        parameters, provenance = resolve_parameters(args, "full_finetune")

        self.assertEqual(
            parameters,
            {
                "head_lr": 1e-4,
                "dropout_prob": 0.1,
                "encoder_weight_decay": 1e-4,
                "head_weight_decay": 1e-4,
                "warmup_epochs": 2,
                "min_lr_factor": 0.1,
                "max_grad_norm": 1.0,
                "batch_size": 1,
                "grad_accum_steps": 4,
                "epochs": 50,
                "early_stopping_patience": 8,
                "min_delta": 0.0,
                "embedding_dim": 2048,
                "gradient_checkpointing": True,
                "encoder_lr": 1e-5,
            },
        )
        self.assertEqual(provenance["parameter_origin"], "default")

    def test_auto_grid_false_preserves_convlora_project_defaults(self):
        args = parsed_args("conv_lora", "--grid", "false")

        parameters, provenance = resolve_parameters(args, "conv_lora")

        self.assertEqual(parameters["batch_size"], 1)
        self.assertEqual(parameters["grad_accum_steps"], 4)
        self.assertEqual(parameters["epochs"], 100)
        self.assertEqual(parameters["early_stopping_patience"], 10)
        self.assertEqual(parameters["adapter_lr"], 1e-4)
        self.assertEqual(parameters["conv_lora_rank"], 2)
        self.assertEqual(parameters["conv_lora_alpha"], 2.0)
        self.assertEqual(provenance["parameter_origin"], "default")

    def test_publication_full_profile_uses_accepted_paper_values(self):
        args = parsed_args(
            "full_finetune", "--parameter_source", "publication"
        )

        parameters, provenance = resolve_parameters(args, "full_finetune")

        self.assertEqual(parameters["encoder_lr"], 1e-5)
        self.assertEqual(parameters["head_lr"], 1e-5)
        self.assertEqual(parameters["batch_size"], 18)
        self.assertEqual(parameters["grad_accum_steps"], 8)
        self.assertEqual(parameters["epochs"], 300)
        self.assertEqual(parameters["early_stopping_patience"], 5)
        self.assertEqual(parameters["warmup_epochs"], 0)
        self.assertEqual(parameters["min_lr_factor"], 0.0)
        self.assertEqual(parameters["max_grad_norm"], 0.0)
        self.assertEqual(provenance["parameter_origin"], "publication_backed")
        self.assertEqual(
            provenance["sources"]["audited_commit"],
            "0e5fc47829b74033a6d72a8778c0c16c8644d6ad",
        )

    def test_publication_convlora_profile_uses_accepted_paper_values(self):
        args = parsed_args("conv_lora", "--parameter_source", "publication")

        parameters, provenance = resolve_parameters(args, "conv_lora")

        self.assertEqual(parameters["adapter_lr"], 1e-4)
        self.assertEqual(parameters["head_lr"], 1e-4)
        self.assertEqual(parameters["batch_size"], 18)
        self.assertEqual(parameters["grad_accum_steps"], 8)
        self.assertEqual(parameters["epochs"], 300)
        self.assertEqual(parameters["early_stopping_patience"], 5)
        self.assertEqual(parameters["conv_lora_rank"], 2)
        self.assertEqual(parameters["conv_lora_alpha"], 2.0)
        self.assertTrue(parameters["train_image_projection"])
        self.assertEqual(provenance["parameter_origin"], "publication_backed")

    def test_llrd_defaults_are_independent_from_existing_convlora(self):
        args = parsed_args("conv_lora_llrd", "--grid", "false")

        parameters, provenance = resolve_parameters(args, "conv_lora_llrd")

        self.assertEqual(parameters["adapter_lr"], 1e-4)
        self.assertEqual(parameters["head_lr"], 1e-4)
        self.assertEqual(parameters["image_layer_decay"], 0.7)
        self.assertEqual(parameters["encoder_weight_decay"], 0.01)
        self.assertEqual(parameters["head_weight_decay"], 0.01)
        self.assertEqual(parameters["batch_size"], 2)
        self.assertEqual(parameters["grad_accum_steps"], 8)
        self.assertEqual(parameters["epochs"], 300)
        self.assertFalse(parameters["train_image_projection"])
        self.assertTrue(parameters["exclude_norm_bias_from_weight_decay"])
        self.assertEqual(parameters["scheduler_step_unit"], "optimizer_update")
        self.assertEqual(provenance["parameter_origin"], "default")

        old_args = parsed_args("conv_lora", "--grid", "false")
        old_parameters, _ = resolve_parameters(old_args, "conv_lora")
        self.assertNotIn("image_layer_decay", old_parameters)
        self.assertNotIn("scheduler_step_unit", old_parameters)
        self.assertEqual(old_parameters["batch_size"], 1)
        self.assertEqual(old_parameters["epochs"], 100)

    def test_llrd_publication_profile_is_explicitly_adapted(self):
        args = parsed_args(
            "conv_lora_llrd", "--parameter_source", "publication"
        )

        parameters, provenance = resolve_parameters(args, "conv_lora_llrd")

        self.assertEqual(parameters["adapter_lr"], 1e-4)
        self.assertEqual(parameters["head_lr"], 1e-4)
        self.assertEqual(parameters["image_layer_decay"], 0.7)
        self.assertEqual(parameters["batch_size"], 18)
        self.assertEqual(parameters["grad_accum_steps"], 8)
        self.assertFalse(parameters["train_image_projection"])
        self.assertEqual(provenance["parameter_origin"], "publication_adapted")
        self.assertEqual(
            provenance["publication_fidelity"],
            "publication_inspired_image_only_supervised_adaptation",
        )

    def test_llrd_publication_derived_batch_override_is_recorded(self):
        args = parsed_args(
            "conv_lora_llrd",
            "--parameter_source",
            "publication_derived",
            "--batch_size",
            "2",
        )

        parameters, provenance = resolve_parameters(args, "conv_lora_llrd")

        self.assertEqual(parameters["batch_size"], 2)
        self.assertEqual(provenance["parameter_origin"], "publication_derived")
        override = provenance["publication_parameter_overrides"]
        self.assertEqual(override["changed_parameter_count"], 1)
        self.assertEqual(
            override["changed_parameters"]["batch_size"]["publication_value"],
            18,
        )

    def test_llrd_has_separate_results_profile_and_grid(self):
        self.assertEqual(VARIANT_FOLDERS["conv_lora_llrd"], "ConvLoRA_LLRD")
        self.assertEqual(
            GRID_CONFIG_FILENAMES["conv_lora_llrd"],
            "grid_conv_lora_llrd.json",
        )
        candidates = grid_candidates(
            MERLIN_DIR / GRID_CONFIG_FILENAMES["conv_lora_llrd"]
        )
        self.assertEqual(len(candidates), 3)
        default_args = parsed_args("conv_lora_llrd", "--grid", "false")
        default_parameters, _ = resolve_parameters(
            default_args, "conv_lora_llrd"
        )
        self.assertEqual(set(candidates[0]), set(default_parameters))

    def test_publication_rejects_manual_parameter_overrides(self):
        args = parsed_args(
            "full_finetune",
            "--parameter_source",
            "publication",
            "--encoder_lr",
            "2e-5",
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_parameters(args, "full_finetune")

    def test_explicit_default_source_overrides_legacy_grid_default(self):
        args = parsed_args("full_finetune", "--parameter_source", "default")

        parameters, provenance = resolve_parameters(args, "full_finetune")

        self.assertEqual(parameters["batch_size"], 1)
        self.assertEqual(provenance["parameter_origin"], "default")


if __name__ == "__main__":
    unittest.main()
