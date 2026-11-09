"""Development-only geometry audit and bounded real-data/GPU setup check."""
from __future__ import annotations
import argparse
import collections
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from _linear_probe.common import (methodology_root, load_manifest, development_mapping,
    cross_validation_split, rows_from_mapping, atomic_write_json, sha256_file, utc_now)
from _cnn.geometry import fit_plan, image_geometry, grid_for_image
from _cnn.data import preprocess


def geometry_report(root, canvas):
    records, pools = {}, []
    for pct in (20, 40, 60, 80, 100):
        manifest = load_manifest(root / "Dataset", pct)
        rows = rows_from_mapping(root / "Dataset", development_mapping(manifest))
        for row in rows:
            records.setdefault(row["image_path"], None)
            if records[row["image_path"]] is None:
                records[row["image_path"]] = image_geometry(row["image_path"])
        plans = []
        for fold in range(1, 6):
            train, val = cross_validation_split(manifest, fold)
            train_rows = rows_from_mapping(root / "Dataset", train)
            val_rows = rows_from_mapping(root / "Dataset", val)
            plan = fit_plan(train_rows, canvas, geometries=[records[r["image_path"]] for r in train_rows])
            grids = [grid_for_image(records[r["image_path"]], plan) for r in val_rows]
            plans.append({"fold": fold, "plan": plan,
                          "validation_count": len(val_rows),
                          "validation_expanded_canvas_count": sum(g["expanded_canvas"] for g in grids),
                          "validation_largest_shape_xyz": np.max([g["shape_xyz"] for g in grids], axis=0).tolist()})
        refit = fit_plan(rows, canvas, geometries=[records[r["image_path"]] for r in rows])
        pools.append({"percentage": pct, "development_count": len(rows),
                      "class_counts": dict(collections.Counter(r["class_name"] for r in rows)),
                      "manifest_sha256": manifest["_sha256"], "folds": plans, "refit": refit})
    return {"time": utc_now(), "scope": "development headers only; no held-out test header or image inspected",
            "canvas_xyz": list(canvas), "pools": pools}


def benchmark(root, model_name, shape, batch, real_patient=False):
    from MedicalNet.model import MedicalNetClassifier
    from DenseNet.model import build_model
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    prep_seconds = None
    if real_patient:
        manifest = load_manifest(root / "Dataset", 20)
        train, _ = cross_validation_split(manifest, 1)
        rows = rows_from_mapping(root / "Dataset", train)
        plan = fit_plan(rows, shape)
        # One development example from each class for preprocessing throughput.
        selected = list({r["class_name"]: r for r in reversed(rows)}.values())
        times, x = [], None
        for row in selected:
            array, metadata = preprocess(row["image_path"], plan)
            times.append(metadata["preprocessing_seconds"])
            x = torch.from_numpy(array)
        prep_seconds = times
        x = x.unsqueeze(0).expand(batch, -1, -1, -1, -1).contiguous().cuda()
    else:
        x = torch.randn(batch, 1, *reversed(shape), device="cuda")
    if model_name == "MedicalNet":
        model = MedicalNetClassifier()
        coverage = model.load_pretrained(root.parent / "resnet_18_23dataset.pth")
    else:
        model, coverage = build_model(), None
    model = model.cuda().train()
    optimizer = (torch.optim.Adam(model.parameters(), lr=.001) if model_name == "MedicalNet"
                 else torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, nesterov=True, weight_decay=1e-4))
    y = torch.arange(batch, device="cuda") % 5
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Non-finite benchmark loss/gradients")
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return {"time": utc_now(), "status": "ok", "model": model_name, "canvas_xyz": list(shape),
            "batch": batch, "precision": "bf16", "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__), "cudnn_deterministic": True,
            "tf32": False, "checkpoint_coverage": coverage,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "step_seconds": durations, "preprocessing_seconds_five_development_examples": prep_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "scope": "3 bounded optimizer steps; no production training, validation tuning, or test evaluation"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--action", choices=("geometry", "benchmark"), required=True)
    parser.add_argument("--model", choices=("MedicalNet", "DenseNet"), default="MedicalNet")
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--real_patient", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = methodology_root()
    result = (geometry_report(root, args.shape) if args.action == "geometry"
              else benchmark(root, args.model, args.shape, args.batch, args.real_patient))
    output = args.output or root / "Logs" / "CNN_Baselines" / (
        "geometry_audit.json" if args.action == "geometry" else f"benchmark_{args.model}.json")
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "status": "ok"}))

if __name__ == "__main__":
    main()
