"""CLI entry point for publication-method-backed FlexiCT zero-shot inference."""

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
    from .zero_shot import run_zero_shot
    from .zero_shot_model import OFFICIAL_CHECKPOINT_SHA256, TEXT_MODEL_REVISION
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REVISION
    from zero_shot import run_zero_shot
    from zero_shot_model import OFFICIAL_CHECKPOINT_SHA256, TEXT_MODEL_REVISION

from _linear_probe.common import methodology_root


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description="FlexiCT publication-method-backed five-class zero-shot evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--results_root", type=Path, default=root / "Results" / "FlexiCT")
    parser.add_argument(
        "--prompt_config",
        type=Path,
        default=PACKAGE_DIR / "zero_shot_prompts.json",
    )
    parser.add_argument(
        "--upstream_root",
        type=Path,
        default=root / "tmp" / "upstream_flexict",
    )
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Models" / "3D_VLM_final_model.pth",
    )
    parser.add_argument("--checkpoint_sha256", default=OFFICIAL_CHECKPOINT_SHA256)
    parser.add_argument("--text_model_revision", default=TEXT_MODEL_REVISION)
    parser.add_argument(
        "--hf_cache_root",
        type=Path,
        help="optional Hugging Face cache root for the pinned Qwen config/tokenizer snapshot",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_trusted_legacy_checkpoint", action="store_true")
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Setup" / "asset_provisioning.jsonl",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT revision is immutable and must be {UPSTREAM_REVISION}")
    if args.text_model_revision != TEXT_MODEL_REVISION:
        parser.error(
            f"FlexiCT zero-shot Qwen revision is immutable and must be {TEXT_MODEL_REVISION}"
        )
    if args.checkpoint_sha256.lower() != OFFICIAL_CHECKPOINT_SHA256:
        parser.error(
            "FlexiCT zero-shot requires the audited official checkpoint SHA-256 "
            f"{OFFICIAL_CHECKPOINT_SHA256}"
        )
    try:
        result = run_zero_shot(args)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
