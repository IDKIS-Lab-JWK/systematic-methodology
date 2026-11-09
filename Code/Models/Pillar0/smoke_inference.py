"""One-volume end-to-end Pillar-0 cache/model smoke inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from .cache import load_cache_manifest, load_pillar_volume
    from .model import MODEL_REPO_ID, MODEL_REVISION, PillarClassifier, extract_vision_features, load_official_model, set_gradient_checkpointing
    from .training import full_finetune_optimizer, save_json
except ImportError:  # pragma: no cover
    from cache import load_cache_manifest, load_pillar_volume
    from model import MODEL_REPO_ID, MODEL_REVISION, PillarClassifier, extract_vision_features, load_official_model, set_gradient_checkpointing
    from training import full_finetune_optimizer, save_json


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache_manifest", type=Path, required=True)
    parser.add_argument("--patient_id", default=None)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also perform one mixed-precision optimizer step and verify encoder gradients.",
    )
    parser.add_argument(
        "--bf16_weights",
        action="store_true",
        help="Experimental: load master weights as BF16 (not the validated default).",
    )
    parser.add_argument(
        "--activation_offload",
        action="store_true",
        help="Move autograd-saved tensors to pinned host memory during the backward smoke.",
    )
    parser.add_argument("--output", type=Path, default=root / "Results" / "Pillar0" / "Setup" / "model_smoke_inference.json")
    return parser


def execute(args) -> dict:
    manifest = load_cache_manifest(args.cache_manifest)
    entries = manifest["entries"]
    if args.patient_id is None:
        row = entries[0]
    else:
        row = next((item for item in entries if str(item["patient_id"]) == str(args.patient_id)), None)
        if row is None:
            raise ValueError(f"Patient {args.patient_id} is not in {args.cache_manifest}")
    device = torch.device(args.device)
    raw = load_pillar_volume(Path(row["cache_path"])).unsqueeze(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model_dtype = (
        torch.bfloat16
        if args.bf16_weights and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    model = load_official_model(
        args.model_repo_id,
        args.model_revision,
        device=device,
        dtype=model_dtype,
    )
    model.eval()
    with torch.inference_mode():
        model_input = raw.to(
            device=device,
            dtype=model_dtype if model_dtype is not None else torch.float32,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=model_dtype is not None,
        ):
            features = extract_vision_features(model, model_input, normalize=False)
    elapsed = time.perf_counter() - started
    finite = bool(torch.isfinite(features).all().item())
    result = {
        "schema_version": 1,
        "status": "passed" if finite and tuple(features.shape) == (1, 1152) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo_id": args.model_repo_id,
        "model_revision": args.model_revision,
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_namespace": manifest["namespace"],
        "smoke_cache_only_not_experiment_result": "smoke" in manifest["namespace"].lower(),
        "patient_id": str(row["patient_id"]),
        "label_str": str(row["label_str"]),
        "raw_archive_shape": row.get("shape"),
        "model_input_shape": list(raw.shape),
        "model_input_min_hu": float(raw.min()),
        "model_input_max_hu": float(raw.max()),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype).replace("torch.", ""),
        "model_weight_dtype": str(model_dtype or torch.float32).replace("torch.", ""),
        "feature_finite": finite,
        "feature_l2_norm": float(features.float().norm(dim=1).item()),
        "elapsed_seconds_including_model_load": elapsed,
        "device": str(device),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "secrets_logged": False,
    }
    if args.backward:
        classifier = PillarClassifier(
            model, num_classes=5, frozen_encoder=False, dropout=0.1
        ).to(device)
        checkpointing = set_gradient_checkpointing(classifier.encoder, True)
        optimizer = full_finetune_optimizer(
            classifier,
            {
                "encoder_lr": 1e-5,
                "head_lr": 1e-4,
                "weight_decay": 0.01,
                "layer_decay": 0.9,
            },
        )
        optimizer.zero_grad(set_to_none=True)
        classifier.train()
        backward_started = time.perf_counter()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if args.activation_offload and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with offload_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                logits = classifier(model_input)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([0], device=device)
                )
        loss.backward()
        encoder_gradients = [
            parameter.grad
            for parameter in classifier.encoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        optimizer.step()
        result["backward_smoke"] = {
            "status": "passed" if encoder_gradients else "failed_no_encoder_gradients",
            "gradient_checkpointing_active": checkpointing,
            "activation_offload_active": bool(
                args.activation_offload and device.type == "cuda"
            ),
            "loss": float(loss.detach().cpu()),
            "encoder_tensors_with_gradients": len(encoder_gradients),
            "all_encoder_gradients_finite": bool(
                encoder_gradients
                and all(torch.isfinite(value).all().item() for value in encoder_gradients)
            ),
            "elapsed_seconds": time.perf_counter() - backward_started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None,
        }
        if result["backward_smoke"]["status"] != "passed":
            result["status"] = "failed"
    save_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
