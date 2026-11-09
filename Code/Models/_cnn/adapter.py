"""Model adapters for the existing five-fold systematic workflow."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from _linear_probe.common import ContractError, fingerprint, sha256_file, ensure_same_or_write
from .data import VolumeCache
from .geometry import fit_plan, image_geometry, RATIONALE, SOURCES


class CNNAdapter:
    def __init__(self, model_name):
        if model_name not in ("MedicalNet", "DenseNet"):
            raise ValueError(model_name)
        self.name = model_name
        self.method_label = "FullFineTune" if model_name == "MedicalNet" else "FromScratch"
        self.method_id = "medicalnet_resnet18_full_finetune" if model_name == "MedicalNet" else "densenet121_3d_scratch"
        self.package = Path(__file__).resolve().parent.parent / model_name
        self.publication_profile_path = self.package / "publication.json"
        self.default_profile_path = self.package / "default.json"
        self.search_profile_path = self.package / "grid.json"
        self.plan = None
        self.cache = None
        self.geometries = {}
        self.inputs_observed = {}
        self._identity = None

    def implementation_paths(self):
        base = Path(__file__).resolve().parent
        return sorted(base.glob("*.py")) + sorted(self.package.glob("*.py")) + sorted((self.package / "vendor").glob("*.py")) + [
            base.parent / "_linear_probe" / "common.py", base.parent / "_linear_probe" / "metrics.py",
            base.parent / "_supervised_adaptation" / "runtime.py"]

    def planned_identity(self, args):
        return {"model": self.name, "pretrained": self.name == "MedicalNet",
                "geometry_rule": "training-median spacing coarsened per axis to fit training world extents",
                "profile_controls_canvas": True, "assets_resolved": False}

    def preflight(self, args, rows):
        self.resolve_identity(args)
        for row in rows:
            path = row["image_path"]
            if path not in self.geometries:
                self.geometries[path] = image_geometry(path)
        if self.cache is None:
            runtime = {name: version(name) for name in ("numpy", "scipy", "nibabel", "torch")}
            identity = {"runtime": runtime,
                        "code": {p.name: sha256_file(p) for p in [Path(__file__).with_name("data.py"), Path(__file__).with_name("geometry.py")]}}
            cache_root = args.cache_root or Path(args.results_root) / "Cache" / "Volumes"
            self.cache = VolumeCache(cache_root, identity)

    def resolve_identity(self, args):
        if self._identity is None:
            from _supervised_adaptation.trainer import _runtime_identity
            from _supervised_adaptation.runtime import resolve_precision
            precision, _ = resolve_precision(args.device, args.training_precision)
            self._identity = {"name": self.name, "pretrained": self.name == "MedicalNet",
                              "runtime": _runtime_identity(args.device, precision),
                              "packages": {k: version(k) for k in ("numpy", "scipy", "nibabel", "monai", "filelock")},
                              "micro_batch_size": args.micro_batch_size,
                              "cpu_threads": torch.get_num_threads(),
                              "torch_math": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                             "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                             "cudnn_tf32": torch.backends.cudnn.allow_tf32}}
            if self.name == "MedicalNet":
                from MedicalNet import CHECKPOINT_SHA256, UPSTREAM_REVISION
                checkpoint = Path(args.checkpoint)
                observed = sha256_file(checkpoint)
                if observed != CHECKPOINT_SHA256:
                    raise ContractError("Expected the audited resnet_18_23dataset.pth checkpoint; SHA-256 differs")
                self._identity.update({"checkpoint_sha256": observed, "upstream_revision": UPSTREAM_REVISION,
                                       "shortcut_autograd_fix": True})
            else:
                # Importlib avoids the package's densenet factory alias.
                import importlib
                dense_module = importlib.import_module("monai.networks.nets.densenet")
                self._identity["monai_densenet_source_sha256"] = sha256_file(dense_module.__file__)
        return self._identity

    def representation_contract(self, args):
        return {"kind": "whole_volume_3d", "layout": "N,C,S,A,R", "rationale": RATIONALE, "sources": SOURCES,
                "fit_scope": "each fold's training patients; refit uses selected development pool"}

    def configure_training(self, args, train_rows, parameters, output_dir):
        self.preflight(args, train_rows)
        self.plan = fit_plan(train_rows, parameters["canvas_xyz"],
                             geometries=[self.geometries[r["image_path"]] for r in train_rows])
        self.inputs_observed = {}
        ensure_same_or_write(Path(output_dir) / "preprocessing_plan.json", self.plan)

    def prepare_rows(self, args, rows):
        return [dict(row) for row in rows]

    def validate_parameters(self, p):
        if p.get("eval_batch_size") != 1:
            raise ContractError("eval_batch_size must be 1 for fixed-spacing variable canvas inference")
        if p.get("augmentation") != "none":
            raise ContractError("This baseline implements augmentation=none only")
        if len(p.get("canvas_xyz", [])) != 3 or any(not isinstance(x, int) or x < 32 or x % 32 for x in p["canvas_xyz"]):
            raise ContractError("canvas_xyz requires three multiples of 32 >=32")
        if not np.isfinite(p["learning_rate"]) or p["learning_rate"] <= 0:
            raise ContractError("learning_rate must be positive and finite")
        if not 0 <= p.get("dropout", 0) < 1:
            raise ContractError("dropout must be in [0,1)")
        if self.name == "MedicalNet" and p.get("dropout", 0) != 0:
            raise ContractError("This MedicalNet classifier implements dropout=0 only")
        if p.get("warmup_steps", 0) != 0 or p.get("minimum_lr_ratio", 0) != 0:
            raise ContractError("CNN schedules do not implement warmup or a minimum LR ratio")
        for key in ("weight_decay", "class_weight_beta", "gradient_clip_norm", "early_stopping_min_delta"):
            if not np.isfinite(p.get(key, 0)):
                raise ContractError(f"{key} must be finite")
        if p["scheduler"] not in ("constant", "multistep"):
            raise ContractError("CNN scheduler must be constant or multistep")
        if p["scheduler"] == "multistep":
            ms = p.get("milestone_epochs", [])
            if not ms or any(not isinstance(x, int) for x in ms) or list(ms) != sorted(set(ms)) or min(ms) <= 0 or max(ms) >= p.get("epochs", 0) or not 0 < p.get("lr_gamma", 0) < 1:
                raise ContractError("Invalid multistep schedule")

    def build_model(self, args, parameters, device):
        if self.name == "MedicalNet":
            from MedicalNet.model import MedicalNetClassifier
            model = MedicalNetClassifier()
            model.load_pretrained(args.checkpoint)
        else:
            from DenseNet.model import build_model
            model = build_model(parameters["dropout"])
        return model.to(device)

    def optimizer_groups(self, model, parameters):
        return [{"params": list(model.parameters()), "lr": parameters["learning_rate"],
                 "weight_decay": parameters["weight_decay"], "group_name": "all_parameters"}]

    def validate_trainable_scope(self, model, parameters):
        if not all(p.requires_grad for p in model.parameters()):
            raise ContractError("CNN baseline must train all model parameters")

    def architecture(self, model, parameters):
        return {"architecture": "MedicalNet dilated ResNet18 shortcut A" if self.name == "MedicalNet" else "MONAI DenseNet121 spatial_dims=3",
                "initialization": "23-dataset pretrained backbone, random head" if self.name == "MedicalNet" else "random",
                "classes": 5, "channels": 1, "spatial_dims": 3, "canvas_xyz": parameters["canvas_xyz"]}

    def set_training_mode(self, model, training):
        model.train(training)

    def load_batch(self, args, rows, indices, training, generator):
        if self.plan is None or self.cache is None:
            raise ContractError("Preprocessing plan/cache have not been configured")
        images, labels = [], []
        for index in indices:
            row = rows[index]
            image, metadata = self.cache.get(row["image_path"], self.plan)
            images.append(image)
            labels.append(row["label"])
            self.inputs_observed[row["patient_id"]] = {"patient_id": row["patient_id"], "class_name": row["class_name"],
                **metadata["grid"], "antialias_sigma_native_voxels": metadata["antialias_sigma_native_voxels"],
                "native_spacing_ras_mm": metadata["source_geometry"]["native_spacing_ras_mm"]}
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    def move_inputs(self, inputs, device):
        return inputs.to(device)
