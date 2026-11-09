"""CLI for the frozen-attention depth-token cache."""

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
    from .attention_cache import DEFAULT_ATTENTION_CACHE_NAMESPACE, prepare_cache
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REVISION
    from attention_cache import DEFAULT_ATTENTION_CACHE_NAMESPACE, prepare_cache

from _linear_probe.common import methodology_root


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description="Prepare sealed FlexiCT depth tokens for frozen single-head attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--percentage", choices=(100,), type=int, default=100)
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--cache_namespace", default=DEFAULT_ATTENTION_CACHE_NAMESPACE)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Cache" / "AttentionTokens" / DEFAULT_ATTENTION_CACHE_NAMESPACE,
    )
    parser.add_argument("--upstream_root", type=Path, default=root / "tmp" / "upstream_flexict")
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_trusted_legacy_checkpoint", action="store_true")
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "FlexiCT" / "Setup" / "asset_provisioning.jsonl",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_shards", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.upstream_revision != UPSTREAM_REVISION:
        parser.error(f"FlexiCT revision is immutable and must be {UPSTREAM_REVISION}")
    if args.allow_trusted_legacy_checkpoint and not args.checkpoint_sha256:
        parser.error("--allow_trusted_legacy_checkpoint requires --checkpoint_sha256")
    try:
        result = prepare_cache(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
