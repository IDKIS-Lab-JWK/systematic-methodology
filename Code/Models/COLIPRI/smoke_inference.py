"""Run one non-persistent official preprocessing and dense-feature smoke."""

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
    from . import MODEL_REVISION
    from .cache import all_rows
    from .model import build_image_encoder, extract_dense_features, preprocess_nifti
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import MODEL_REVISION
    from cache import all_rows
    from model import build_image_encoder, extract_dense_features, preprocess_nifti
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import methodology_root


def main(argv: list[str] | None = None) -> int:
    root = methodology_root()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument(
        "--upstream_root", type=Path, default=root / "tmp" / "upstream_colipri"
    )
    parser.add_argument("--upstream_revision", default=MODEL_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "COLIPRI" / "Setup" / "asset_provisioning.jsonl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature_precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args(argv)
    if args.upstream_revision != MODEL_REVISION:
        parser.error(f"COLIPRI source revision is immutable and must be {MODEL_REVISION}")
    try:
        ensure_upstream(
            args.upstream_root,
            revision=args.upstream_revision,
            local_files_only=args.local_files_only,
            provision_log=args.provision_log,
        )
        checkpoint = resolve_checkpoint(
            args.checkpoint,
            upstream_root=args.upstream_root,
            local_files_only=args.local_files_only,
            provision_log=args.provision_log,
        )
        rows, _ = all_rows(args.dataset_root, require_files=True)
        model, processor, model_details = build_image_encoder(
            upstream_root=args.upstream_root,
            checkpoint_path=checkpoint["path"],
            device=args.device,
        )
        images, preprocessing = preprocess_nifti(processor, rows[0]["image_path"])
        dense, extraction = extract_dense_features(
            model, images, device=args.device, precision=args.feature_precision
        )
        public_preprocessing = {
            key: value for key, value in preprocessing.items() if key != "source"
        }
        result = {
            "status": "passed",
            "patient_id_omitted": True,
            "processed_shape": preprocessing["shape_bcdhw"],
            "dense_feature_shape": list(dense.shape),
            "preprocessing": public_preprocessing,
            "extraction": extraction,
            "model": model_details,
        }
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


if __name__ == "__main__":
    raise SystemExit(main())
