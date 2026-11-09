"""One-volume, forward-only smoke inference for the official 3DINO encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:  # Package imports.
    from . import (
        INPUT_SIZE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REVISION,
    )
    from .data import make_dataset, require_nifti_path
    from .model import (
        DEFAULT_REPRESENTATIONS,
        extract_representation_bank,
        feature_dimension,
        load_official_model,
    )
    from .provisioning import provision_assets
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import (
        INPUT_SIZE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REVISION,
    )
    from data import make_dataset, require_nifti_path
    from model import DEFAULT_REPRESENTATIONS, extract_representation_bank, feature_dimension, load_official_model
    from provisioning import provision_assets


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description="Run one forward-only 3DINO smoke; no training is performed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--upstream_root", type=Path, default=root / "tmp" / "upstream_3dino"
    )
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Optional local checkpoint override; otherwise use the pinned HF cache.",
    )
    parser.add_argument("--checkpoint_sha256", default=MODEL_SHA256)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "3DINO" / "Setup" / "asset_provisioning.jsonl",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional NIfTI. Without it, a synthetic correctly-shaped tensor is used.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Results" / "3DINO" / "Setup" / "smoke_inference.json",
    )
    return parser


def execute(args: argparse.Namespace) -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError("smoke_inference.py requires PyTorch") from exc

    assets = provision_assets(
        upstream_root=args.upstream_root,
        upstream_revision=args.upstream_revision,
        checkpoint_path=args.checkpoint_path,
        repo_id=args.model_repo_id,
        model_revision=args.model_revision,
        expected_sha256=args.checkpoint_sha256,
        expected_size_bytes=MODEL_SIZE_BYTES,
        local_files_only=args.local_files_only,
        log_path=args.provision_log,
    )
    args.upstream_root = assets.upstream_root
    args.checkpoint_path = assets.checkpoint_path
    device = torch.device(args.device)
    if args.image is None:
        images = torch.zeros((1, 1, *INPUT_SIZE), dtype=torch.float32)
        input_source = "synthetic_zero_tensor"
    else:
        source = require_nifti_path(args.image, must_exist=True).resolve()
        row = {
            "image": str(source),
            "source_path": str(source),
            "label": 0,
            "patient_id": "smoke",
            "label_str": "Appendicitis",
            "accession": "smoke",
        }
        images = make_dataset([row], split="test", cache_dir=None)[0]["image"].unsqueeze(0)
        input_source = str(source)
    if tuple(images.shape) != (1, 1, *INPUT_SIZE):
        raise RuntimeError(
            f"Smoke input should be {(1, 1, *INPUT_SIZE)}, got {tuple(images.shape)}"
        )

    started = time.perf_counter()
    model = load_official_model(
        upstream_root=args.upstream_root,
        upstream_revision=args.upstream_revision,
        checkpoint_path=args.checkpoint_path,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        device=device,
    )
    model.eval()
    images = images.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        bank = extract_representation_bank(
            model, images, representations=DEFAULT_REPRESENTATIONS
        )
    elapsed = time.perf_counter() - started

    representations: dict[str, dict] = {}
    passed = True
    for (blocks, use_mean), features in bank.items():
        expected = (1, feature_dimension(blocks, use_mean))
        finite = bool(torch.isfinite(features).all().item())
        shape_ok = tuple(features.shape) == expected
        passed = passed and finite and shape_ok
        key = f"last_{blocks}_blocks_mean_patch_{str(use_mean).lower()}"
        representations[key] = {
            "shape": list(features.shape),
            "expected_shape": list(expected),
            "finite": finite,
            "l2_norm": float(features.float().norm(dim=1).item()),
        }
    provenance = getattr(model, "_threedino_provenance", {})
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": input_source,
        "input_shape": list(images.shape),
        "input_min": float(images.min().item()),
        "input_max": float(images.max().item()),
        "device": str(device),
        "representations": representations,
        # Preserve the provisioner's stronger Hub and SCM attestations; the
        # loader provenance is intentionally limited to local byte/import
        # validation and cannot establish acquisition identity by itself.
        "checkpoint": dict(assets.checkpoint),
        "upstream": dict(assets.upstream),
        "checkpoint_load": provenance.get("load_report"),
        "asset_provisioning": assets.to_dict(),
        "elapsed_seconds_including_model_load": elapsed,
        "training_performed": False,
        "downloads_performed": assets.downloads_performed,
        "git_operations_performed": assets.git_operations_performed,
    }
    _write_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "passed" else 2


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
