"""CLI for the isolated COLIPRI pooling-optimised probe workflow."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE, METHOD_LABEL, MODEL_REVISION
    from .backend import COLIPRIBackend, DeterministicBatchStream
    from .runtime_options import resolve_micro_batch_size
except ImportError:  # pragma: no cover
    from COLIPRI import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE, METHOD_LABEL, MODEL_REVISION
    from COLIPRI.backend import COLIPRIBackend, DeterministicBatchStream
    from COLIPRI.runtime_options import resolve_micro_batch_size

from _linear_probe.common import (
    MODEL_SOURCES,
    PERCENTAGES,
    atomic_write_csv,
    atomic_write_json,
    completion_seal,
    fingerprint,
    methodology_root,
    sha256_file,
    utc_now,
)
from _pooling_optimised_linear_probe.workflow import run_pipeline


def _install_cpu_batch_stream_resume_compatibility() -> None:
    """Keep CPU generator state on CPU when a checkpoint was loaded on CUDA.

    PyTorch applies ``map_location`` recursively, including to the byte tensor
    returned by ``torch.Generator.get_state()``.  A CPU generator rejects that
    state after it has been remapped to CUDA.  Normalizing only the bookkeeping
    tensors here is an operational checkpoint-restore compatibility measure;
    it does not alter model, optimizer, scheduler, or numerical training state.
    """

    original = DeterministicBatchStream.load_state_dict
    if getattr(original, "_colipri_cpu_rng_compatibility", False):
        return

    @functools.wraps(original)
    def compatible_load_state_dict(self, state):
        normalized = dict(state)
        for key in ("order", "generator_state"):
            value = normalized.get(key)
            if hasattr(value, "detach") and hasattr(value, "to"):
                normalized[key] = value.detach().to(device="cpu")
        return original(self, normalized)

    compatible_load_state_dict._colipri_cpu_rng_compatibility = True
    DeterministicBatchStream.load_state_dict = compatible_load_state_dict


def _install_failed_candidate_materialization_compatibility() -> None:
    """Seal an excluded diverged grid trial without inventing predictions.

    A failed trial can retain finite parameters whose forward pass nevertheless
    produces non-finite logits.  It is already excluded from selection, so its
    fold record must describe the failure instead of calculating metrics from
    fabricated probabilities.  This hook affects failure bookkeeping only and
    leaves every successful candidate on the original evaluation path.
    """

    original_complete = COLIPRIBackend._complete_validation_run
    if getattr(original_complete, "_colipri_failed_materialization", False):
        return

    @functools.wraps(original_complete)
    def compatible_complete(
        self,
        *,
        model,
        parameters,
        probabilities,
        train_rows,
        validation_rows,
        output_dir,
        args,
        context,
        training,
        logger,
    ):
        failure = training.get("candidate_failure")
        if failure is None:
            return original_complete(
                self,
                model=model,
                parameters=parameters,
                probabilities=probabilities,
                train_rows=train_rows,
                validation_rows=validation_rows,
                output_dir=output_dir,
                args=args,
                context=context,
                training=training,
                logger=logger,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = self._write_model_artifact(
            model, parameters, output_dir, args, training, context
        )
        per_class = {}
        for label, class_name in enumerate(CLASS_NAMES):
            per_class[class_name] = {
                "precision": None,
                "recall": None,
                "sensitivity": None,
                "specificity": None,
                "f1": None,
                "average_precision": None,
                "support": sum(
                    int(row["label"]) == label for row in validation_rows
                ),
            }
        metrics = {
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_macro": None,
            "average_precision_macro": None,
            "per_class": per_class,
            "confusion_matrix": None,
            "support": len(validation_rows),
            "candidate_status": "failed",
            "candidate_failure": dict(failure),
            "metrics_status": "not_computed_failed_optimizer_divergence",
        }
        atomic_write_json(output_dir / "parameters.json", dict(parameters))
        atomic_write_json(output_dir / "metrics_summary.json", metrics)
        atomic_write_json(output_dir / "validation_predictions.json", [])
        probability_fields = [
            "patient_id",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            *(f"probability_{name}" for name in CLASS_NAMES),
        ]
        atomic_write_csv(
            output_dir / "validation_probabilities.csv", [], probability_fields
        )
        state = {
            "schema_version": 1,
            "status": "failed",
            "completed_at": utc_now(),
            "method_label": self._method_label(context),
            "strict_linear": bool(model.strict_linear),
            "non_strict_reason": None
            if model.strict_linear
            else "trainable attention pooling precedes the linear classifier",
            "encoder_frozen": True,
            "shared_frozen_feature_batches": True,
            "training_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in train_rows]
            ),
            "validation_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in validation_rows]
            ),
            "feature_identity": self._context_feature_identity(args, context),
            "training": dict(training),
            "context": dict(context),
        }
        atomic_write_json(output_dir / "training_state.json", state)
        seal_paths = [
            artifact,
            output_dir / "parameters.json",
            output_dir / "metrics_summary.json",
            output_dir / "validation_predictions.json",
            output_dir / "validation_probabilities.csv",
            output_dir / "training_state.json",
        ]
        atomic_write_json(
            output_dir / "completion.json",
            completion_seal(
                seal_paths,
                context={
                    "model": self.name,
                    "parameters_sha256": fingerprint(dict(parameters)),
                    "artifact_sha256": sha256_file(artifact),
                },
            ),
        )
        logger.info(
            "COLIPRI pooling=%s lr=%s status=failed; validation metrics were not "
            "computed for the diverged trial",
            parameters["pooling"],
            parameters["learning_rate"],
        )
        return {"metrics": metrics, "artifact": str(artifact)}

    compatible_complete._colipri_failed_materialization = True
    COLIPRIBackend._complete_validation_run = compatible_complete

    original_summary = COLIPRIBackend.selection_summary

    @functools.wraps(original_summary)
    def compatible_summary(self, parameters, fold_metrics):
        failures = [
            {
                "fold": fold_number,
                "failure": item.get("candidate_failure"),
            }
            for fold_number, item in enumerate(fold_metrics, start=1)
            if item.get("candidate_status") == "failed"
        ]
        if not failures:
            return original_summary(self, parameters, fold_metrics)
        return {
            "status": "failed",
            "eligible_for_selection": False,
            "fold_failures": failures,
            "selection_value": None,
            "mean_average_precision_macro": None,
            "std_average_precision_macro": None,
            "project_robust_score": None,
            "mean_f1_macro": None,
            "std_f1_macro": None,
            "mean_balanced_accuracy": None,
            "std_balanced_accuracy": None,
            "mean_perforation_recall": None,
        }

    COLIPRIBackend.selection_summary = compatible_summary


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _micro_batch_size(value: str) -> int:
    try:
        return resolve_micro_batch_size(int(value), 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    results = root / "Results" / "COLIPRI"
    parser = argparse.ArgumentParser(
        description="COLIPRI publication-backed pooling-optimised frozen probing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("grid", "cv", "test", "all"), default="all")
    parser.add_argument("--percentage", choices=PERCENTAGES, type=int, default=40)
    parser.add_argument(
        "--parameter_source",
        choices=("publication", "grid", "default", "manual"),
        default="publication",
    )
    parser.add_argument("--parameter_config", type=Path)
    parser.add_argument("--grid_config", type=Path)
    parser.add_argument("--max_grid_candidates", type=_positive)
    parser.add_argument(
        "--model_source", choices=MODEL_SOURCES, default="refit_full_development"
    )
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--results_root", type=Path, default=results)
    parser.add_argument(
        "--cache_root",
        type=Path,
        help="feature-cache root; defaults to <results_root>/Cache",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--upstream_root", type=Path, default=root / "tmp" / "upstream_colipri"
    )
    parser.add_argument("--upstream_revision", default=MODEL_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--provision_log",
        type=Path,
        help="asset audit log; defaults to <results_root>/Setup/asset_provisioning.jsonl",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature_precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--training_precision", choices=("auto", "bf16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--micro_batch_size", type=_micro_batch_size, default=16,
        help="patients per microbatch; defaults to the effective batch size (16)",
    )
    parser.add_argument("--checkpoint_interval", type=_positive, default=250)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="load one CPU microbatch ahead; no extra GPU batch")
    parser.add_argument("--source_fingerprint", choices=("stat", "sha256"), default="sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=_nonnegative, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    if args.cache_root is None:
        args.cache_root = args.results_root / "Cache"
    if args.provision_log is None:
        args.provision_log = args.results_root / "Setup" / "asset_provisioning.jsonl"
    if args.upstream_revision != MODEL_REVISION:
        parser.error(f"COLIPRI source revision is immutable and must be {MODEL_REVISION}")
    if args.parameter_source == "publication":
        if (
            args.parameter_config is not None
            or args.grid_config is not None
            or args.max_grid_candidates is not None
        ):
            parser.error(
                "publication source uses its packaged complete immutable 20-candidate grid"
            )
    elif args.parameter_source == "grid" and args.parameter_config is not None:
        parser.error("grid parameter source does not accept --parameter_config")
    elif args.parameter_source in ("default", "manual") and (
        args.grid_config is not None or args.max_grid_candidates is not None
    ):
        parser.error(
            "fixed default/manual parameter sources do not accept grid controls"
        )
    if args.parameter_source == "manual" and args.parameter_config is None:
        parser.error("manual parameter source requires --parameter_config")
    if args.stage in ("grid", "all") and args.parameter_source not in (
        "publication",
        "grid",
    ):
        parser.error("grid/all stages require publication or grid parameter source")
    if args.num_workers != 0:
        parser.error(
            "COLIPRI uses bounded thread prefetch, not multiprocessing loaders; --num_workers must be 0"
        )
    if not args.dry_run and args.source_fingerprint != "sha256":
        parser.error(
            "COLIPRI PoolingOptimisedLinearProbe execution requires "
            "--source_fingerprint sha256; "
            "stat mode is cache diagnostics only"
        )
    _install_cpu_batch_stream_resume_compatibility()
    _install_failed_candidate_materialization_compatibility()
    from COLIPRI.prefetched_backends import PrefetchedCOLIPRIBackend
    try:
        result = run_pipeline(args, PrefetchedCOLIPRIBackend())
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


__all__ = ["build_parser", "main"]

