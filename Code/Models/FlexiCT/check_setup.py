"""FlexiCT environment, source, checkpoint, and optional model preflight."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from . import UPSTREAM_REVISION
    from .model import build_vision_only_encoder
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REVISION
    from model import build_vision_only_encoder
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import methodology_root


PINNED_RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "numpy": "2.3.2",
    "scipy": "1.16.1",
    "sklearn": "1.7.1",
    "SimpleITK": "2.5.2",
    "transformers": "4.56.1",
    "huggingface_hub": "0.34.4",
}


def _base_version(value: str) -> str:
    return str(value).split("+", 1)[0]


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("--provision_log", type=Path, default=root / "Results" / "FlexiCT" / "Setup" / "asset_provisioning.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT requires pinned upstream revision {UPSTREAM_REVISION}")
    if args.allow_trusted_legacy_checkpoint and not args.checkpoint_sha256:
        parser.error("--allow_trusted_legacy_checkpoint requires --checkpoint_sha256")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "side_effects": False,
                    "python": sys.version,
                    "upstream_root": str(args.upstream_root),
                    "upstream_revision": args.upstream_revision,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else "environment override",
                    "load_model": args.load_model,
                },
                indent=2,
            )
        )
        return 0
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(
                f"FlexiCT requires Python 3.11, observed {sys.version_info.major}.{sys.version_info.minor}"
            )
        versions = {}
        for name in (
            "torch",
            "numpy",
            "scipy",
            "sklearn",
            "SimpleITK",
            "transformers",
            "huggingface_hub",
            "joblib",
        ):
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "available")
        mismatches = {
            name: {"expected": expected, "observed": versions[name]}
            for name, expected in PINNED_RUNTIME_VERSIONS.items()
            if _base_version(versions[name]) != expected
        }
        if mismatches:
            raise RuntimeError(f"FlexiCT pinned runtime version mismatch: {mismatches}")
        import torch

        if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        source = ensure_upstream(
            args.upstream_root,
            revision=args.upstream_revision,
            local_files_only=args.local_files_only,
            provision_log=args.provision_log,
        )
        checkpoint = resolve_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
        model_details = None
        if args.load_model:
            _, model_details = build_vision_only_encoder(
                upstream_root=args.upstream_root,
                checkpoint_path=checkpoint["path"],
                device=args.device,
                allow_trusted_legacy_checkpoint=args.allow_trusted_legacy_checkpoint,
                trusted_checkpoint_sha256=args.checkpoint_sha256,
            )
        result = {
            "status": "ready",
            "python": sys.version,
            "versions": versions,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source": source,
            "checkpoint": checkpoint,
            "model_loaded": args.load_model,
            "model_details": model_details,
        }
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
