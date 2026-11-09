"""Token-safe environment, dataset, cache, and checkpoint preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import DEFAULT_CACHE_NAMESPACE
    from .audit_split_patient_ids import run_audit as run_patient_id_split_audit
    from .cache import validate_cache
    from .data import collect_required_rows, load_manifest
    from .model import (
        MODEL_REPO_ID,
        MODEL_REVISION,
        checkpoint_access_report,
        checkpoint_logit_scale,
        load_official_model,
        project_qwen_embeddings,
        zero_shot_capability_report,
    )
    from .zero_shot import ZERO_SHOT_VARIANT
    from .orchestrator import default_dataset_root, default_results_root
    from .training import save_json
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE
    from audit_split_patient_ids import run_audit as run_patient_id_split_audit
    from cache import validate_cache
    from data import collect_required_rows, load_manifest
    from model import MODEL_REPO_ID, MODEL_REVISION, checkpoint_access_report, checkpoint_logit_scale, load_official_model, project_qwen_embeddings, zero_shot_capability_report
    from zero_shot import ZERO_SHOT_VARIANT
    from orchestrator import default_dataset_root, default_results_root
    from training import save_json


def build_parser() -> argparse.ArgumentParser:
    results = default_results_root()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=results)
    parser.add_argument("--cache_root", type=Path, default=results / "Cache")
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--online_access_check", action="store_true")
    parser.add_argument(
        "--inspect_model_capabilities",
        action="store_true",
        help="Download/load the pinned checkpoint on CPU and test its zero-shot text projection API.",
    )
    parser.add_argument("--validate_cached_volumes", action="store_true")
    parser.add_argument("--output", type=Path, default=results / "Setup" / "setup_preflight.json")
    return parser


def collect(args) -> dict:
    import torch

    dataset_root, results_root = Path(args.dataset_root).resolve(), Path(args.results_root).resolve()
    cache_manifest = Path(args.cache_root).resolve() / "Shared" / args.cache_namespace / "cache_manifest.json"
    manifest = load_manifest(dataset_root, 100)
    rows = collect_required_rows(dataset_root, manifest)
    patient_id_split_validation = run_patient_id_split_audit(
        dataset_root, (20, 40, 60, 80, 100)
    )
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "huggingface_hub", "nibabel", "SimpleITK", "sklearn", "pandas", "yaml", "rve", "vision_engine", "lz4")
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]},
        "dependencies": dependencies,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "practical_cache_exporter": "lz4",
        "publication_reference_exporter": "hevc_10bit_requires_ffmpeg",
        "dataset": {"primary_input": "NIfTI", "patient_count": len(rows), "all_source_files_present": True, "png_2p5d_selected": False, "dicom_selected": False},
        "patient_id_split_validation": patient_id_split_validation,
        "cache": {"manifest": str(cache_manifest), "exists": cache_manifest.is_file()},
        "checkpoint": {"repo_id": args.model_repo_id, "revision": args.model_revision, "access_checked": False},
        "secrets_logged": False,
    }
    if cache_manifest.is_file():
        try:
            report["cache"]["validation"] = validate_cache(cache_manifest, verify_volumes=args.validate_cached_volumes)
        except Exception as exc:
            report["cache"]["validation_error"] = f"{type(exc).__name__}: {exc}"
    if args.online_access_check:
        report["checkpoint"] = checkpoint_access_report(args.model_repo_id, args.model_revision)
        report["checkpoint"]["access_checked"] = True
    if args.inspect_model_capabilities:
        try:
            model = load_official_model(
                args.model_repo_id, args.model_revision, device="cpu"
            )
            capability = zero_shot_capability_report(model)
            try:
                projected = project_qwen_embeddings(
                    model, torch.zeros(1, 4096, dtype=torch.float32)
                )
                capability.update(
                    {
                        "runtime_projection_probe": "passed",
                        "runtime_output_shape": list(projected.shape),
                    }
                )
                scale = checkpoint_logit_scale(model)
                if scale is None or not torch.isfinite(torch.tensor(scale)) or scale <= 0:
                    raise RuntimeError(
                        "Checkpoint logit scale is missing, non-finite, or non-positive."
                    )
                capability.update(
                    {
                        "checkpoint_logit_scale_probe": "passed",
                        "resolved_checkpoint_logit_scale": float(scale),
                    }
                )
            except Exception as exc:
                capability.update(
                    {
                        "runtime_projection_probe": capability.get(
                            "runtime_projection_probe", "failed"
                        ),
                        "checkpoint_logit_scale_probe": "failed",
                        "runtime_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            report["model_capabilities"] = capability
            del model
        except Exception as exc:
            report["model_capabilities"] = {
                "load_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    report["ready_for_cache"] = all(dependencies.get(name, False) for name in ("torch", "nibabel", "SimpleITK", "yaml")) and (dependencies["rve"] or dependencies["vision_engine"]) and dependencies["lz4"]
    technically_ready = bool(
        report["cache"]["exists"] and report["checkpoint"].get("accessible")
    )
    report["ready_for_model_execution"] = technically_ready
    report["ready_for_patient_id_disjoint_results"] = bool(
        technically_ready and patient_id_split_validation["status"] == "passed"
    )
    report["ready_for_model_runs"] = technically_ready
    report["completed_results"] = {
        "linear_probe_100_percent_refit_test": (
            results_root
            / "LinearProbe"
            / "Test"
            / "CV_1"
            / "refit_full_development"
            / "100_percent"
            / "metrics_summary.json"
        ).is_file(),
        "experimental_zero_shot_five_class_margin": any(
            (
                results_root
                / "ZeroShot"
                / "Test"
                / ZERO_SHOT_VARIANT
            ).glob("*/metrics_summary.json")
        ),
        "full_finetune_classification": any(
            (results_root / "FullFineTune" / "Test").glob("**/metrics_summary.json")
        )
        if (results_root / "FullFineTune" / "Test").is_dir()
        else False,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = collect(args)
        save_json(args.output, report)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
