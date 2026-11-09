"""Launch the shared systematic workflow with the CNN-specific backend."""
from __future__ import annotations
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

from filelock import FileLock
import torch
from _linear_probe.common import append_jsonl, utc_now, load_manifest, development_mapping, heldout_mapping, rows_from_mapping
from _supervised_adaptation.cli import build_parser, validate_args, _positive
from _supervised_adaptation.workflow import run_pipeline
from .adapter import CNNAdapter
from .trainer import CNNBackend


def add_arguments(parser, root):
    parser.add_argument("--checkpoint", type=Path, default=root.parent / "resnet_18_23dataset.pth")
    parser.add_argument("--cache_root", type=Path)
    parser.add_argument("--cache_policy", choices=("temporary", "retain"), default="temporary")
    parser.add_argument("--cpu_threads", type=_positive, default=8)


def main(model, argv=None):
    parser = build_parser(lambda: CNNAdapter(model), add_arguments)
    parser.set_defaults(percentage=100, micro_batch_size=4)
    args = parser.parse_args(argv)
    if getattr(args, 'stage', None) in ('test', 'all') and not getattr(args, 'dry_run', False):
        parser.error('Legacy held-out testing is disabled because it depends on persistent test caches. Use python run.py test with source images and explicit selected checkpoint(s). For legacy training select --stage cv or --stage grid/search.')
    validate_args(parser, args)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = CNNBackend(CNNAdapter(model))
    started = time.perf_counter()
    status, error = "error", None
    try:
        with contextlib.ExitStack() as stack:
            if not args.dry_run:
                args.results_root.mkdir(parents=True, exist_ok=True)
                stack.enter_context(FileLock(str(args.results_root / ".cnn_run.lock"), timeout=0))
                cache_root = args.cache_root or args.results_root / "Cache" / "Volumes"
                cache_root.mkdir(parents=True, exist_ok=True)
                stack.enter_context(FileLock(str(cache_root / ".cnn_cache.lock"), timeout=0))
                # Conservative cache capacity for one active fold/refit, not all folds.
                from _supervised_adaptation.workflow import _load_candidates
                candidates, _, _ = _load_candidates(args, backend)
                voxels = max(__import__("math").prod(p["canvas_xyz"]) for p in candidates)
                manifest = load_manifest(args.dataset_root, args.percentage)
                mapping = heldout_mapping(manifest) if args.stage == "test" else development_mapping(manifest)
                count = len(rows_from_mapping(args.dataset_root, mapping))
                required = int(count * voxels * 4 * 1.2 + 2 * 1024**3)
                free = shutil.disk_usage(cache_root).free
                if free < required:
                    raise RuntimeError(f"Need approximately {required / 1024**3:.1f} GiB free for one active cache plus headroom; available {free / 1024**3:.1f}. Choose --cache_root on a larger disk.")
            result = run_pipeline(args, backend)
            status = "complete"
            print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"status": status, "error": error}, indent=2), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt; rerun the same command to resume"
        print(error, file=sys.stderr)
        return 130
    finally:
        if not args.dry_run:
            append_jsonl(args.results_root / "Logs" / "invocations.jsonl", {
                "time": utc_now(), "model": model, "stage": args.stage,
                "percentage": args.percentage, "parameter_source": args.parameter_source,
                "model_source": args.model_source, "status": status, "error": error,
                "invocation_seconds": time.perf_counter() - started,
                "cache_policy": args.cache_policy})

