"""Run one non-persistent FlexiCT preprocessing and embedding smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import UPSTREAM_REVISION
    from .cache import all_rows
    from .data import preprocess_nifti
    from .model import build_vision_only_encoder, extract_embedding
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REVISION
    from cache import all_rows
    from data import preprocess_nifti
    from model import build_vision_only_encoder, extract_embedding
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import methodology_root


def main(argv: list[str] | None = None) -> int:
    root = methodology_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--upstream_root", type=Path, default=root / "tmp" / "upstream_flexict")
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument(
        "--allow_trusted_legacy_checkpoint",
        action="store_true",
        help=(
            "opt in to weights_only=False only if safe loading fails; requires "
            "--checkpoint_sha256 for a trusted official checkpoint"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args(argv)
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT requires pinned upstream revision {UPSTREAM_REVISION}")
    if args.allow_trusted_legacy_checkpoint and not args.checkpoint_sha256:
        parser.error("--allow_trusted_legacy_checkpoint requires --checkpoint_sha256")
    try:
        ensure_upstream(args.upstream_root, revision=args.upstream_revision, local_files_only=args.local_files_only)
        checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
        rows, _ = all_rows(args.dataset_root, require_files=True)
        volume, preprocessing = preprocess_nifti(rows[0]["image_path"])
        model, model_details = build_vision_only_encoder(
            upstream_root=args.upstream_root,
            checkpoint_path=checkpoint["path"],
            device=args.device,
            allow_trusted_legacy_checkpoint=args.allow_trusted_legacy_checkpoint,
            trusted_checkpoint_sha256=args.checkpoint_sha256,
        )
        embedding, extraction = extract_embedding(
            model, volume, device=args.device, precision=args.feature_precision
        )
        result = {
            "status": "passed",
            "patient_id_omitted": True,
            "input_shape": list(volume.shape),
            "embedding_shape": list(embedding.shape),
            "preprocessing": preprocessing,
            "extraction": extraction,
            "model": model_details,
        }
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
