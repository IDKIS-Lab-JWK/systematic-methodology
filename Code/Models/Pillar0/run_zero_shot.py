"""CLI for the single capability-gated Pillar-0 five-class margin experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

try:
    from . import DEFAULT_CACHE_NAMESPACE
    from .cache import load_cached_rows
    from .data import heldout_mapping, load_manifest, rows_from_mapping
    from .model import MODEL_REPO_ID, MODEL_REVISION, checkpoint_access_report
    from .orchestrator import default_dataset_root, default_results_root
    from .training import save_json
    from .zero_shot import (
        CLASS_SCORE_RULE,
        PROBABILITY_RULE,
        ZERO_SHOT_VARIANT,
        load_prompt_profile,
        run_zero_shot,
    )
except ImportError:  # pragma: no cover
    from __init__ import DEFAULT_CACHE_NAMESPACE
    from cache import load_cached_rows
    from data import heldout_mapping, load_manifest, rows_from_mapping
    from model import MODEL_REPO_ID, MODEL_REVISION, checkpoint_access_report
    from orchestrator import default_dataset_root, default_results_root
    from training import save_json
    from zero_shot import (
        CLASS_SCORE_RULE,
        PROBABILITY_RULE,
        ZERO_SHOT_VARIANT,
        load_prompt_profile,
        run_zero_shot,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heldout_rows(dataset_root: Path) -> list[dict]:
    manifests = [load_manifest(dataset_root, percentage) for percentage in (20, 40, 60, 80, 100)]
    reference = heldout_mapping(manifests[0])
    for percentage, manifest in zip((40, 60, 80, 100), manifests[1:]):
        if heldout_mapping(manifest) != reference:
            raise RuntimeError(f"Held-out membership differs in {percentage}% manifest.")
    return rows_from_mapping(dataset_root, reference)


def build_parser() -> argparse.ArgumentParser:
    results = default_results_root()
    model_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Pillar-0 experimental five-class positive-minus-negative "
            "prompt-margin setup/access/run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--action", choices=("plan", "access", "run"), default="plan")
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--results_root", type=Path, default=results)
    parser.add_argument("--cache_root", type=Path, default=results / "Cache")
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument("--feature_cache_root", type=Path, default=None)
    parser.add_argument(
        "--prompt_profile",
        type=Path,
        default=model_root / "zero_shot_prompts_five_class_margin_v2.json",
    )
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace an existing result for this exact prompt-set id.",
    )
    return parser


def execute(args) -> dict:
    dataset_root, results_root = Path(args.dataset_root).resolve(), Path(args.results_root).resolve()
    cache_dir = Path(args.cache_root).resolve() / "Shared" / args.cache_namespace
    raw_manifest = cache_dir / "cache_manifest.json"
    profile = load_prompt_profile(Path(args.prompt_profile))
    planned = {
        "model_repo_id": args.model_repo_id,
        "model_revision": args.model_revision,
        "publication_demonstrated": False,
        "methodological_label": ZERO_SHOT_VARIANT,
        "strict_zero_shot": True,
        "prompt_set_id": profile["prompt_set_id"],
        "class_score_rule": CLASS_SCORE_RULE,
        "prediction_rule": "argmax_class_score",
        "probability_rule": PROBABILITY_RULE,
        "probability_interpretation": "uncalibrated_relative_score",
        "valid_only_if_checkpoint_text_projection_runtime_probe_passes": True,
        "valid_only_if_checkpoint_logit_scale_runtime_probe_passes": True,
        "prompt_profile": str(Path(args.prompt_profile).resolve()),
        "cache_manifest": str(raw_manifest.resolve()),
        "heldout_training_examples": 0,
    }
    if args.action == "plan":
        return {"status": "plan", **planned}
    if args.action == "access":
        report = checkpoint_access_report(args.model_repo_id, args.model_revision)
        report.update({"zero_shot_plan": planned, "token_logged": False})
        save_json(results_root / "Setup" / "checkpoint_access.json", report)
        return report
    if not raw_manifest.is_file():
        raise FileNotFoundError(f"Pillar cache manifest not found: {raw_manifest}")
    source_rows = _heldout_rows(dataset_root)
    rows = load_cached_rows(cache_dir, source_rows, expected_namespace=args.cache_namespace)
    feature_root = Path(args.feature_cache_root) if args.feature_cache_root else results_root / "Cache" / "Features" / f"{args.cache_namespace}_{args.model_revision[:12]}_{_sha256_file(raw_manifest)[:12]}"
    return run_zero_shot(
        rows,
        args.prompt_profile,
        raw_manifest,
        feature_root,
        results_root,
        device=torch.device(args.device),
        repo_id=args.model_repo_id,
        revision=args.model_revision,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, 'dry_run', False):
        raise RuntimeError('Legacy zero-shot testing requires persistent test caches and is disabled in this distribution. The raw inference wrapper supports the published supervised classifiers.')
    try:
        result = execute(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if args.action != "access" or result.get("accessible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
