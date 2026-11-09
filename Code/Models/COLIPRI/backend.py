"""Native COLIPRI pooling-optimised probe backend."""

from __future__ import annotations

import contextlib
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn


PUBLICATION_POOLINGS = (
    "average",
    "max",
    "learned_attention",
    "average_attention",
    "multi_learned_attention",
)
PUBLICATION_LEARNING_RATES = (0.003, 0.01, 0.03, 0.1)
LEGACY_METHOD_LABEL = "LinearProbe"
MIGRATION_MANIFEST = "METHOD_IDENTITY_MIGRATION.json"
PUBLICATION_FIXED_PARAMETERS = {
    "optimizer": "sgd",
    "momentum": 0.95,
    "weight_decay": 0.0,
    "scheduler": "cosine",
    "max_steps": 12500,
    "effective_batch_size": 16,
    "embed_dim": 864,
    "attention_heads": 12,
    "num_classes": 5,
    "loss": "weighted_multiclass_cross_entropy_effective_number",
    "class_weight_beta": 0.999,
}

try:
    from . import (
        CLASS_NAMES,
        FEATURE_SHAPE_CDHW,
        METHOD_ID,
        METHOD_LABEL,
        MODEL_REVISION,
    )
    from .cache import FeatureStore
    from .model import resolve_precision
except ImportError:  # pragma: no cover
    from __init__ import (
        CLASS_NAMES,
        FEATURE_SHAPE_CDHW,
        METHOD_ID,
        METHOD_LABEL,
        MODEL_REVISION,
    )
    from cache import FeatureStore
    from model import resolve_precision

try:
    from _linear_probe.common import (
        atomic_write_csv,
        atomic_write_json,
        candidate_id,
        completion_seal,
        fingerprint,
        read_json,
        sha256_file,
        utc_now,
        validate_completion,
    )
    from _linear_probe.metrics import (
        compute_metrics,
        perforation_recall,
        prediction_documents,
        project_grid_score,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _linear_probe.common import (
        atomic_write_csv,
        atomic_write_json,
        candidate_id,
        completion_seal,
        fingerprint,
        read_json,
        sha256_file,
        utc_now,
        validate_completion,
    )
    from _linear_probe.metrics import (
        compute_metrics,
        perforation_recall,
        prediction_documents,
        project_grid_score,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for COLIPRI probe training") from exc
    return torch


def _atomic_torch_save(value: Any, destination: Path) -> None:
    torch = _torch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path, *, map_location: Any = "cpu") -> dict:
    torch = _torch()
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected COLIPRI artifact payload: {path}")
    return payload


def effective_number_weights(labels: Sequence[int], num_classes: int, beta: float):
    """Return mean-one effective-number class weights from training labels."""

    torch = _torch()
    if not 0 <= beta < 1:
        raise ValueError("class_weight_beta must be in [0, 1)")
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes)
    if counts.numel() != num_classes or bool((counts == 0).any()):
        raise ValueError(f"Every class must occur in training; counts={counts.tolist()}")
    counts64 = counts.to(torch.float64)
    if beta == 0:
        weights = torch.ones_like(counts64)
    else:
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts64))
        weights = weights / weights.mean()
    return weights.to(torch.float32), counts


class DeterministicBatchStream:
    """Infinite epoch-shuffled index stream with a serializable exact state."""

    def __init__(self, length: int, seed: int, state: Mapping[str, Any] | None = None):
        torch = _torch()
        if length <= 0:
            raise ValueError("Training rows cannot be empty")
        self.length = int(length)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.epoch = 0
        self.position = 0
        self.order = torch.randperm(self.length, generator=self.generator)
        if state is not None:
            self.load_state_dict(state)

    def next(self, count: int) -> list[int]:
        torch = _torch()
        if count <= 0:
            raise ValueError("Batch count must be positive")
        values: list[int] = []
        while len(values) < count:
            if self.position >= self.length:
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(self.length, generator=self.generator)
            take = min(count - len(values), self.length - self.position)
            values.extend(
                int(value)
                for value in self.order[self.position : self.position + take].tolist()
            )
            self.position += take
        return values

    def state_dict(self) -> dict:
        return {
            "length": self.length,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        torch = _torch()
        if int(state["length"]) != self.length:
            raise RuntimeError("Batch-stream row count changed across resume")
        order = torch.as_tensor(state["order"], dtype=torch.long).clone()
        if tuple(order.shape) != (self.length,):
            raise RuntimeError("Invalid resumed batch-stream order")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        if not 0 <= self.position <= self.length:
            raise RuntimeError("Invalid resumed batch-stream position")
        self.order = order
        self.generator.set_state(torch.as_tensor(state["generator_state"], dtype=torch.uint8))


class COLIPRIBackend:
    name = "COLIPRI"
    artifact_filename = "model.pt"
    selection_metric = "mean five-fold macro one-vs-rest average precision (AUPRC)"

    def __init__(self) -> None:
        package = Path(__file__).resolve().parent
        self.publication_profile_path = (
            package / "publication_pooling_optimised_linear_probe.json"
        )
        self.default_profile_path = package / "default_pooling_optimised_linear_probe.json"
        self.search_profile_path = package / "search_pooling_optimised_linear_probe.json"
        self._store: FeatureStore | None = None

    @staticmethod
    def _method_label(context: Mapping[str, Any]) -> str:
        observed = context.get("method", METHOD_LABEL)
        if observed not in (METHOD_LABEL, LEGACY_METHOD_LABEL):
            raise RuntimeError(f"Unsupported COLIPRI method identity: {observed!r}")
        return str(observed)

    @staticmethod
    def _without_probe_implementation(identity: Mapping[str, Any]) -> dict:
        comparable = dict(identity)
        comparable.pop("probe_implementation_identity", None)
        return comparable

    def _legacy_migration_document(self, args: Any) -> dict:
        path = Path(args.results_root) / METHOD_LABEL / MIGRATION_MANIFEST
        if not path.is_file():
            raise RuntimeError(
                "Legacy COLIPRI artifacts require their method-identity migration "
                f"manifest: {path}"
            )
        document = read_json(path)
        required = {
            "schema_version": 1,
            "migration_type": "metadata_only_result_namespace",
            "model": self.name,
            "legacy_method_label": LEGACY_METHOD_LABEL,
            "canonical_method_id": METHOD_ID,
            "canonical_method_label": METHOD_LABEL,
        }
        if any(document.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Incompatible COLIPRI migration manifest: {path}")
        if not isinstance(
            document.get("legacy_probe_implementation_identity"), dict
        ):
            raise RuntimeError(
                f"COLIPRI migration manifest lacks the legacy implementation: {path}"
            )
        return document

    def _validate_legacy_feature_identity(
        self, args: Any, identity: Mapping[str, Any]
    ) -> dict:
        migration = self._legacy_migration_document(args)
        if (
            identity.get("probe_implementation_identity")
            != migration["legacy_probe_implementation_identity"]
        ):
            raise RuntimeError(
                "Legacy COLIPRI implementation identity is not declared by the "
                "migration manifest"
            )
        current = self.feature_identity(args)
        if self._without_probe_implementation(
            identity
        ) != self._without_probe_implementation(current):
            raise RuntimeError("Legacy COLIPRI feature/cache/runtime identity changed")
        return dict(identity)

    def _context_feature_identity(
        self, args: Any, context: Mapping[str, Any]
    ) -> dict:
        if self._method_label(context) == LEGACY_METHOD_LABEL:
            identity = context.get("model_identity")
            if not isinstance(identity, Mapping):
                raise RuntimeError(
                    "Legacy COLIPRI context has no preserved model identity"
                )
            return self._validate_legacy_feature_identity(args, identity)
        return self.feature_identity(args)

    @staticmethod
    def _validate_candidate_parameters(parameters: Mapping[str, Any]) -> None:
        allowed = {
            "candidate_id",
            "pooling",
            "learning_rate",
            *PUBLICATION_FIXED_PARAMETERS.keys(),
        }
        unknown = set(parameters).difference(allowed)
        if unknown:
            raise ValueError(f"Unsupported COLIPRI parameters: {sorted(unknown)}")
        missing = allowed.difference({"candidate_id"}).difference(parameters)
        if missing:
            raise ValueError(f"Missing COLIPRI parameters: {sorted(missing)}")
        pooling = str(parameters["pooling"])
        if pooling not in PUBLICATION_POOLINGS:
            raise ValueError(f"Unsupported COLIPRI pooling method: {pooling!r}")
        learning_rate = float(parameters["learning_rate"])
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("COLIPRI learning_rate must be finite and positive")
        max_steps = float(parameters["max_steps"])
        if not math.isfinite(max_steps) or max_steps <= 0 or not max_steps.is_integer():
            raise ValueError("COLIPRI max_steps must be a positive integer")
        exact = {
            "optimizer": "sgd",
            "momentum": 0.95,
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "effective_batch_size": 16,
            "embed_dim": 864,
            "attention_heads": 12,
            "num_classes": 5,
            "loss": "weighted_multiclass_cross_entropy_effective_number",
        }
        for key, expected in exact.items():
            observed = parameters[key]
            if isinstance(expected, str):
                matches = str(observed).lower() == expected
            else:
                matches = float(observed) == float(expected)
            if not matches:
                raise ValueError(
                    f"COLIPRI {key} is fixed at {expected!r}; observed {observed!r}"
                )
        beta = float(parameters["class_weight_beta"])
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("COLIPRI class_weight_beta must be in [0, 1)")
        if parameters.get("candidate_id") is not None:
            explicit = str(parameters["candidate_id"])
            canonical_parameters = {
                key: value for key, value in parameters.items() if key != "candidate_id"
            }
            expected_identifier = candidate_id(canonical_parameters)
            if explicit != expected_identifier:
                raise ValueError(
                    "COLIPRI candidate_id must be the canonical 12-character "
                    f"parameter fingerprint {expected_identifier}; observed {explicit!r}"
                )

    @staticmethod
    def _validate_publication_profile(profile: Mapping[str, Any]) -> None:
        search = profile.get("search_space", {})
        references = profile.get("references", {})
        observed_poolings = tuple(search.get("pooling", ()))
        observed_lrs = tuple(float(value) for value in search.get("learning_rate", ()))
        if profile.get("method") != METHOD_LABEL or profile.get("method_id") != METHOD_ID:
            raise ValueError("The COLIPRI publication profile has the wrong method identity")
        if profile.get("parameter_source") != "publication":
            raise ValueError("The COLIPRI publication profile must identify its source")
        if profile.get("publication_backed_grid") is not True:
            raise ValueError("The COLIPRI publication profile must preserve the full grid")
        if references.get("official_revision") != MODEL_REVISION:
            raise ValueError("The COLIPRI publication profile has a non-pinned revision")
        if observed_poolings != PUBLICATION_POOLINGS:
            raise ValueError("The COLIPRI publication pooling grid/order was modified")
        if observed_lrs != PUBLICATION_LEARNING_RATES:
            raise ValueError("The COLIPRI publication learning-rate grid/order was modified")
        if int(search.get("candidate_count", -1)) != 20:
            raise ValueError("The COLIPRI publication grid must declare 20 candidates")
        if dict(profile.get("fixed_parameters", {})) != PUBLICATION_FIXED_PARAMETERS:
            raise ValueError("The COLIPRI publication fixed training recipe was modified")

    def build_candidates(self, profile: Mapping[str, Any]) -> list[dict]:
        if profile.get("method") is not None and (
            profile.get("method") != METHOD_LABEL or profile.get("method_id") != METHOD_ID
        ):
            raise ValueError("The COLIPRI parameter profile has the wrong method identity")
        fixed = dict(profile.get("fixed_parameters", {}))
        search = profile.get("search_space", {})
        poolings = search.get("pooling")
        learning_rates = search.get("learning_rate")
        if poolings is None and learning_rates is None:
            parameters = profile.get("parameters") or profile.get("fixed_parameters")
            if not isinstance(parameters, dict):
                return []
            self._validate_candidate_parameters(parameters)
            return [dict(parameters)]
        if not isinstance(poolings, list) or not isinstance(learning_rates, list):
            raise ValueError("COLIPRI grids require pooling and learning_rate lists")
        if profile.get("parameter_source") == "publication":
            self._validate_publication_profile(profile)
        candidates = [
            {**fixed, "pooling": str(pooling), "learning_rate": float(learning_rate)}
            for pooling in poolings
            for learning_rate in learning_rates
        ]
        declared_count = search.get("candidate_count")
        if declared_count is not None and int(declared_count) != len(candidates):
            raise ValueError(
                f"COLIPRI grid declares {declared_count} candidates but builds {len(candidates)}"
            )
        for parameters in candidates:
            self._validate_candidate_parameters(parameters)
        return candidates

    def dry_run_model_claim(self, args: Any) -> dict:
        return {
            "upstream_repository": "https://huggingface.co/microsoft/colipri",
            "upstream_revision": args.upstream_revision,
            "checkpoint": str(args.checkpoint) if args.checkpoint else "pinned Hugging Face asset",
            "checkpoint_sha256": "97d66586507d1f48577b5149777d02e12d25787435a2a295cac5563df9291c14",
            "feature_cache_namespace": args.cache_namespace,
            "feature_precision": args.feature_precision,
            "source_fingerprint": args.source_fingerprint,
            "training_precision": args.training_precision,
            "micro_batch_size": args.micro_batch_size,
            "num_workers": args.num_workers,
            "representation": "raw 864 x 24 x 24 x 24 Primus-M backbone features",
        }

    def _feature_store(self, args: Any) -> FeatureStore:
        if self._store is None:
            self._store = FeatureStore(args)
        return self._store

    def ensure_feature_rows(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> None:
        self._feature_store(args).validate_rows(rows)

    @staticmethod
    def _probe_runtime(args: Any) -> dict:
        torch = _torch()
        requested_device = str(args.device)
        device = torch.device(requested_device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("COLIPRI probe execution supports CPU or CUDA devices")
        resolved_precision, _ = COLIPRIBackend._resolve_training_precision(
            requested_device, str(args.training_precision)
        )
        runtime = {
            "seed": int(args.seed),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "training_precision_requested": str(args.training_precision),
            "training_precision_resolved": resolved_precision,
            "micro_batch_size": int(args.micro_batch_size),
            "torch_version": str(torch.__version__),
            "device_requested": requested_device,
            "device_type": device.type,
            "device_name": None,
            "cuda_runtime": None,
            "cudnn_version": None,
            "cuda_device_capability": None,
        }
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            actual_device = torch.device("cuda", index)
            runtime.update(
                {
                    "device_resolved": str(actual_device),
                    "device_name": torch.cuda.get_device_name(actual_device),
                    "cuda_runtime": str(torch.version.cuda),
                    "cudnn_version": torch.backends.cudnn.version(),
                    "cuda_device_capability": list(
                        torch.cuda.get_device_capability(actual_device)
                    ),
                }
            )
        else:
            runtime["device_resolved"] = "cpu"
        return runtime

    def feature_identity(self, args: Any) -> dict:
        package = Path(__file__).resolve().parent
        shared = package.parent / "_linear_probe"
        pooling_workflow = package.parent / "_pooling_optimised_linear_probe"
        return {
            **self._feature_store(args).identity,
            "probe_implementation_identity": {
                "schema_version": 1,
                "files": {
                    "COLIPRI/backend.py": sha256_file(package / "backend.py"),
                    "COLIPRI/pooling.py": sha256_file(package / "pooling.py"),
                    "_linear_probe/common.py": sha256_file(shared / "common.py"),
                    "_linear_probe/metrics.py": sha256_file(shared / "metrics.py"),
                    "_pooling_optimised_linear_probe/workflow.py": sha256_file(
                        pooling_workflow / "workflow.py"
                    ),
                },
            },
            "probe_runtime": self._probe_runtime(args),
        }

    @staticmethod
    def _candidate_key(parameters: Mapping[str, Any]) -> str:
        explicit = parameters.get("candidate_id")
        if explicit:
            return str(explicit)
        canonical = {key: value for key, value in parameters.items() if key != "candidate_id"}
        return candidate_id(canonical)

    @staticmethod
    def _validate_shared_recipe(candidates: Sequence[Mapping[str, Any]]) -> dict:
        if not candidates:
            raise ValueError("No COLIPRI candidates were provided")
        for parameters in candidates:
            COLIPRIBackend._validate_candidate_parameters(parameters)
        varying = {"candidate_id", "pooling", "learning_rate"}
        first = {key: value for key, value in candidates[0].items() if key not in varying}
        for parameters in candidates[1:]:
            current = {key: value for key, value in parameters.items() if key not in varying}
            if current != first:
                raise ValueError("Shared COLIPRI training requires identical fixed recipes")
        if str(first.get("optimizer", "sgd")).lower() != "sgd":
            raise ValueError("The native COLIPRI probe supports the publication SGD optimizer")
        if str(first.get("scheduler", "cosine")).lower() != "cosine":
            raise ValueError("The native COLIPRI probe supports the publication cosine schedule")
        return first

    @staticmethod
    def _build_probe(parameters: Mapping[str, Any], *, seed: int, device: str):
        torch = _torch()
        try:
            from .pooling import COLIPRIPoolingOptimisedLinearProbe
        except ImportError:  # pragma: no cover
            from pooling import COLIPRIPoolingOptimisedLinearProbe

        # Probe heads initialise on CPU.  Fork only the CPU generator so
        # building 20 candidates neither snapshots nor mutates CUDA RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = COLIPRIPoolingOptimisedLinearProbe(
                pooling=str(parameters["pooling"]),
                embed_dim=int(parameters.get("embed_dim", 864)),
                num_heads=int(parameters.get("attention_heads", 12)),
                num_classes=int(parameters.get("num_classes", len(CLASS_NAMES))),
            )
        return model.to(device)

    @staticmethod
    def _seed_for_candidate(base_seed: int, fold: int, key: str) -> int:
        return int(base_seed) + int(fold) * 100_003 + int(key[:8], 16) % 1_000_003

    def _load_batch(
        self,
        args: Any,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
    ):
        torch = _torch()
        tensors = [self._feature_store(args).load(rows[index]) for index in indices]
        batch = tensors[0].unsqueeze(0) if len(tensors) == 1 else torch.stack(tensors, dim=0)
        expected = (len(indices), *FEATURE_SHAPE_CDHW)
        if tuple(batch.shape) != expected or batch.dtype != torch.float32:
            raise RuntimeError(f"Unexpected cached COLIPRI batch: {tuple(batch.shape)}, {batch.dtype}")
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return batch, labels

    @staticmethod
    def _dense_to_tokens(features: Any):
        try:
            from .pooling import dense_to_tokens
        except ImportError:  # pragma: no cover
            from pooling import dense_to_tokens

        return dense_to_tokens(features)

    @staticmethod
    def _autocast(device: str, dtype: Any | None):
        torch = _torch()
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _resolve_training_precision(device: str, requested: str) -> tuple[str, Any | None]:
        """Resolve the supported probe precision without FP16 loss scaling."""

        torch = _torch()
        if requested not in ("auto", "bf16", "fp32"):
            raise ValueError("COLIPRI probe training supports auto, bf16, or fp32")
        if not str(device).startswith("cuda"):
            if requested == "bf16":
                raise RuntimeError("BF16 probe training requires CUDA")
            return "fp32", None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA probe training was requested but CUDA is unavailable")
        if requested == "auto":
            requested = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        if requested == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 probe training was requested but CUDA BF16 is unsupported"
                )
            return "bf16", torch.bfloat16
        if requested == "fp32":
            return "fp32", None
        raise ValueError("COLIPRI probe training supports auto, bf16, or fp32")

    def _training_identity(
        self,
        candidates: Sequence[Mapping[str, Any]],
        train_rows: Sequence[Mapping[str, Any]],
        args: Any,
        context: Mapping[str, Any],
        precision: str,
    ) -> dict:
        probe_runtime = self._probe_runtime(args)
        if probe_runtime["training_precision_resolved"] != precision:
            raise RuntimeError("Resolved COLIPRI precision changed while building identity")
        return {
            "schema_version": 1,
            "model": self.name,
            "method": self._method_label(context),
            "candidate_parameters": [dict(item) for item in candidates],
            "training_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in train_rows]
            ),
            "feature_identity": self._feature_store(args).identity,
            "probe_runtime": probe_runtime,
            "seed": int(args.seed),
            "fold": int(context.get("fold", 0)),
            "percentage": context.get("percentage"),
            "training_precision": precision,
            "training_precision_requested": str(args.training_precision),
            "micro_batch_size": int(args.micro_batch_size),
        }

    @staticmethod
    def _checkpoint_payload(
        *,
        identity: Mapping[str, Any],
        completed_steps: int,
        skipped_steps: int,
        stream: DeterministicBatchStream,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any],
        schedulers: Mapping[str, Any],
        class_counts: Any,
        class_weights: Any,
        candidate_failures: Mapping[str, Mapping[str, Any]],
    ) -> dict:
        return {
            "schema_version": 1,
            "status": "complete" if completed_steps >= int(identity["candidate_parameters"][0].get("max_steps", 12500)) else "partial",
            "identity": dict(identity),
            "identity_sha256": fingerprint(identity),
            "completed_optimizer_steps": int(completed_steps),
            "skipped_optimizer_steps": int(skipped_steps),
            "batch_stream": stream.state_dict(),
            "models": {key: model.state_dict() for key, model in models.items()},
            "optimizers": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
            "schedulers": {key: scheduler.state_dict() for key, scheduler in schedulers.items()},
            "class_counts": class_counts.detach().cpu(),
            "class_weights": class_weights.detach().cpu(),
            "candidate_failures": {
                str(key): dict(value) for key, value in candidate_failures.items()
            },
            "updated_at": utc_now(),
        }

    def _train_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        train_rows: Sequence[Mapping[str, Any]],
        checkpoint_path: Path,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
    ) -> tuple[dict[str, Any], dict]:
        torch = _torch()
        fixed = self._validate_shared_recipe(candidates)
        max_steps = int(fixed.get("max_steps", 12500))
        effective_batch = int(fixed.get("effective_batch_size", 16))
        micro_batch = int(args.micro_batch_size)
        if micro_batch <= 0 or micro_batch > effective_batch or effective_batch % micro_batch:
            raise ValueError(
                "micro_batch_size must be positive and divide the publication effective batch size"
            )
        accumulation_steps = effective_batch // micro_batch
        device = str(args.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        precision_name, precision_dtype = self._resolve_training_precision(
            device, args.training_precision
        )
        fold = int(context.get("fold", 0))
        identity = self._training_identity(
            candidates, train_rows, args, context, precision_name
        )
        identity_sha = fingerprint(identity)
        labels = [int(row["label"]) for row in train_rows]
        class_weights, class_counts = effective_number_weights(
            labels,
            int(fixed.get("num_classes", len(CLASS_NAMES))),
            float(fixed.get("class_weight_beta", 0.999)),
        )
        class_weights_device = class_weights.to(device)
        models: dict[str, Any] = {}
        optimizers: dict[str, Any] = {}
        schedulers: dict[str, Any] = {}
        for parameters in candidates:
            key = self._candidate_key(parameters)
            model = self._build_probe(
                parameters,
                seed=self._seed_for_candidate(args.seed, fold, key),
                device=device,
            )
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(parameters["learning_rate"]),
                momentum=float(parameters.get("momentum", 0.95)),
                weight_decay=float(parameters.get("weight_decay", 0.0)),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps, eta_min=0.0
            )
            models[key] = model
            optimizers[key] = optimizer
            schedulers[key] = scheduler

        stream_seed = int(args.seed) + fold * 1_000_003
        stream = DeterministicBatchStream(len(train_rows), stream_seed)
        completed_steps = 0
        skipped_steps = 0
        candidate_failures: dict[str, dict[str, Any]] = {}
        if checkpoint_path.is_file():
            if not args.resume:
                raise RuntimeError(f"Partial training checkpoint exists and resume is disabled: {checkpoint_path}")
            payload = _torch_load(checkpoint_path, map_location=device)
            if payload.get("identity_sha256") != identity_sha or payload.get("identity") != identity:
                raise RuntimeError(f"Incompatible COLIPRI resume checkpoint: {checkpoint_path}")
            for key in models:
                models[key].load_state_dict(payload["models"][key], strict=True)
                optimizers[key].load_state_dict(payload["optimizers"][key])
                schedulers[key].load_state_dict(payload["schedulers"][key])
            stream.load_state_dict(payload["batch_stream"])
            completed_steps = int(payload["completed_optimizer_steps"])
            skipped_steps = int(payload.get("skipped_optimizer_steps", 0))
            observed_failures = payload.get("candidate_failures", {})
            if not isinstance(observed_failures, Mapping) or not set(
                observed_failures
            ).issubset(models):
                raise RuntimeError(
                    f"Malformed failed-candidate state in COLIPRI checkpoint: {checkpoint_path}"
                )
            candidate_failures = {
                str(key): dict(value) for key, value in observed_failures.items()
            }
            logger.info("Resumed shared COLIPRI probe training at optimizer step %s", completed_steps)

        checkpoint_interval = max(1, int(getattr(args, "checkpoint_interval", 250)))
        report_interval = max(1, min(250, max_steps // 20 if max_steps >= 20 else 1))

        def mark_failed(key: str, *, reason: str, micro_batch_index: int | None) -> None:
            if key in candidate_failures:
                return
            parameters = next(
                item for item in candidates if self._candidate_key(item) == key
            )
            failure = {
                "schema_version": 1,
                "status": "failed_optimizer_divergence",
                "reason": reason,
                "optimizer_step": completed_steps + 1,
                "micro_batch_index": micro_batch_index,
                "pooling": str(parameters["pooling"]),
                "learning_rate": float(parameters["learning_rate"]),
                "training_precision": precision_name,
            }
            candidate_failures[key] = failure
            optimizers[key].zero_grad(set_to_none=True)
            logger.warning(
                "COLIPRI candidate %s diverged at optimizer step %s (%s); "
                "the fixed grid trial is retained as failed and excluded from selection",
                key,
                completed_steps + 1,
                reason,
            )

        while completed_steps < max_steps:
            active_keys = [key for key in models if key not in candidate_failures]
            if not active_keys:
                logger.warning(
                    "All COLIPRI candidates diverged by optimizer step %s; stopping shared training",
                    completed_steps,
                )
                break
            for key in active_keys:
                optimizers[key].zero_grad(set_to_none=True)
            # Drawing the full effective batch first is essential for exact
            # weighted-loss accumulation: PyTorch's weighted CE mean divides
            # by the sum of target weights over the whole batch, not by the
            # number of microbatches.
            effective_indices = stream.next(effective_batch)
            effective_targets = torch.as_tensor(
                [int(train_rows[index]["label"]) for index in effective_indices],
                dtype=torch.long,
                device=device,
            )
            weight_denominator = class_weights_device[effective_targets].sum()
            for micro_start in range(0, effective_batch, micro_batch):
                indices = effective_indices[micro_start : micro_start + micro_batch]
                features, targets = self._load_batch(args, train_rows, indices)
                features = features.to(device)
                targets = targets.to(device)
                tokens = self._dense_to_tokens(features)
                for key in list(active_keys):
                    if key in candidate_failures:
                        continue
                    model = models[key]
                    with self._autocast(device, precision_dtype):
                        loss = torch.nn.functional.cross_entropy(
                            model.forward_tokens(tokens),
                            targets,
                            weight=class_weights_device,
                            reduction="sum",
                        ) / weight_denominator
                    if not bool(torch.isfinite(loss)):
                        mark_failed(
                            key,
                            reason="non_finite_loss",
                            micro_batch_index=micro_start // micro_batch,
                        )
                        continue
                    loss.backward()
                del features, targets, tokens
            for key in active_keys:
                if key in candidate_failures:
                    continue
                gradients = [
                    parameter.grad
                    for parameter in models[key].parameters()
                    if parameter.grad is not None
                ]
                if not gradients or not all(
                    bool(torch.isfinite(gradient).all()) for gradient in gradients
                ):
                    mark_failed(
                        key,
                        reason="non_finite_gradient",
                        micro_batch_index=None,
                    )
                    continue
                optimizers[key].step()
                if not all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in models[key].parameters()
                ):
                    raise RuntimeError(
                        "COLIPRI optimizer produced non-finite parameters for "
                        f"candidate {key}; the last finite state cannot be recovered"
                    )
                schedulers[key].step()
            completed_steps += 1
            if completed_steps % report_interval == 0 or completed_steps == max_steps:
                logger.info(
                    "COLIPRI shared probe optimizer step %s/%s (%s candidates)",
                    completed_steps,
                    max_steps,
                    len(models),
                )
            if completed_steps % checkpoint_interval == 0 or completed_steps == max_steps:
                _atomic_torch_save(
                    self._checkpoint_payload(
                        identity=identity,
                        completed_steps=completed_steps,
                        skipped_steps=skipped_steps,
                        stream=stream,
                        models=models,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        class_counts=class_counts,
                        class_weights=class_weights,
                        candidate_failures=candidate_failures,
                    ),
                    checkpoint_path,
                )
        metadata = {
            "method_label": identity["method"],
            "optimizer_steps": completed_steps,
            "skipped_optimizer_steps": skipped_steps,
            "effective_batch_size": effective_batch,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation_steps,
            "training_precision": precision_name,
            "training_precision_requested": str(args.training_precision),
            "probe_runtime": identity["probe_runtime"],
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "training_identity_sha256": identity_sha,
            "checkpoint": str(checkpoint_path.resolve()),
            "candidate_failures": candidate_failures,
            "successful_candidate_ids": [
                key for key in models if key not in candidate_failures
            ],
        }
        return models, metadata

    def _evaluate_models(
        self,
        models: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        args: Any,
    ) -> dict[str, np.ndarray]:
        torch = _torch()
        device = str(args.device)
        _, precision_dtype = self._resolve_training_precision(
            device, args.training_precision
        )
        micro_batch = max(1, int(args.micro_batch_size))
        outputs: dict[str, list[Any]] = {key: [] for key in models}
        for model in models.values():
            model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), micro_batch):
                indices = list(range(start, min(start + micro_batch, len(rows))))
                features, _ = self._load_batch(args, rows, indices)
                features = features.to(device)
                tokens = self._dense_to_tokens(features)
                for key, model in models.items():
                    with self._autocast(device, precision_dtype):
                        logits = model.forward_tokens(tokens)
                    probabilities = torch.softmax(logits.float(), dim=1).cpu()
                    outputs[key].append(probabilities)
                del features, tokens
        return {
            key: torch.cat(chunks, dim=0).numpy().astype(np.float64, copy=False)
            for key, chunks in outputs.items()
        }

    @staticmethod
    def _cpu_state_dict(model: Any) -> dict:
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def _write_model_artifact(
        self,
        model: Any,
        parameters: Mapping[str, Any],
        output_dir: Path,
        args: Any,
        training: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / self.artifact_filename
        payload = {
            "schema_version": 1,
            "model": self.name,
            "method_label": self._method_label(context),
            "architecture": model.architecture(),
            "parameters": dict(parameters),
            "class_names": list(CLASS_NAMES),
            "feature_identity": self._context_feature_identity(args, context),
            "training": dict(training),
            "state_dict": self._cpu_state_dict(model),
        }
        _atomic_torch_save(payload, artifact)
        return artifact

    def _complete_validation_run(
        self,
        *,
        model: Any,
        parameters: Mapping[str, Any],
        probabilities: np.ndarray,
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
        output_dir: Path,
        args: Any,
        context: Mapping[str, Any],
        training: Mapping[str, Any],
        logger: Any,
    ) -> dict:
        labels = np.asarray([int(row["label"]) for row in validation_rows], dtype=np.int64)
        predicted = probabilities.argmax(axis=1)
        metrics = compute_metrics(labels, predicted, probabilities, CLASS_NAMES)
        candidate_failure = training.get("candidate_failure")
        candidate_status = "failed" if candidate_failure is not None else "complete"
        metrics["candidate_status"] = candidate_status
        metrics["candidate_failure"] = candidate_failure
        json_rows, csv_rows = prediction_documents(validation_rows, probabilities, CLASS_NAMES)
        artifact = self._write_model_artifact(
            model, parameters, output_dir, args, training, context
        )
        atomic_write_json(output_dir / "parameters.json", dict(parameters))
        atomic_write_json(output_dir / "metrics_summary.json", metrics)
        atomic_write_json(output_dir / "validation_predictions.json", json_rows)
        atomic_write_csv(
            output_dir / "validation_probabilities.csv", csv_rows, list(csv_rows[0])
        )
        state = {
            "schema_version": 1,
            "status": candidate_status,
            "completed_at": utc_now(),
            "method_label": self._method_label(context),
            "strict_linear": bool(model.strict_linear),
            "non_strict_reason": None
            if model.strict_linear
            else "trainable attention pooling precedes the linear classifier",
            "encoder_frozen": True,
            "shared_frozen_feature_batches": True,
            "training_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in train_rows]
            ),
            "validation_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in validation_rows]
            ),
            "feature_identity": self._context_feature_identity(args, context),
            "training": dict(training),
            "context": dict(context),
        }
        atomic_write_json(output_dir / "training_state.json", state)
        seal_paths = [
            artifact,
            output_dir / "parameters.json",
            output_dir / "metrics_summary.json",
            output_dir / "validation_predictions.json",
            output_dir / "validation_probabilities.csv",
            output_dir / "training_state.json",
        ]
        atomic_write_json(
            output_dir / "completion.json",
            completion_seal(
                seal_paths,
                context={
                    "model": self.name,
                    "parameters_sha256": fingerprint(dict(parameters)),
                    "artifact_sha256": sha256_file(artifact),
                },
            ),
        )
        logger.info(
            "COLIPRI pooling=%s lr=%s status=%s validation macro-AUPRC=%.4f macro-F1=%.4f",
            parameters["pooling"],
            parameters["learning_rate"],
            candidate_status,
            metrics["average_precision_macro"],
            metrics["f1_macro"],
        )
        return {"metrics": metrics, "artifact": str(artifact)}

    def fit_grid_fold(
        self,
        candidates: Sequence[Mapping[str, Any]],
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
        grid_root: Path,
        fold_number: int,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
    ) -> dict:
        shared_root = grid_root / "shared_training" / f"fold_{fold_number:02d}"
        shared_root.mkdir(parents=True, exist_ok=True)
        models, training = self._train_candidates(
            candidates,
            train_rows,
            shared_root / "last_checkpoint.pt",
            args,
            {**context, "fold": fold_number},
            logger,
        )
        shared_completion_path = shared_root / "completion.json"
        if shared_completion_path.is_file():
            shared_completion = validate_completion(shared_root, "completion.json")
            shared_state = read_json(shared_root / "training_state.json")
            expected_ids = [self._candidate_key(item) for item in candidates]
            if (
                shared_state.get("candidate_ids") != expected_ids
                or shared_state.get("training", {}).get("training_identity_sha256")
                != training["training_identity_sha256"]
            ):
                raise RuntimeError(
                    f"Incompatible sealed shared COLIPRI training state: {shared_root}"
                )
        else:
            shared_state = {
                "schema_version": 1,
                "status": "complete",
                "model": self.name,
                "fold": fold_number,
                "candidate_ids": [self._candidate_key(item) for item in candidates],
                "candidate_count": len(candidates),
                "training": training,
                "checkpoint_sha256": sha256_file(shared_root / "last_checkpoint.pt"),
            }
            atomic_write_json(shared_root / "training_state.json", shared_state)
            atomic_write_json(
                shared_completion_path,
                completion_seal(
                    [
                        shared_root / "last_checkpoint.pt",
                        shared_root / "training_state.json",
                    ],
                    context={
                        "model": self.name,
                        "fold": fold_number,
                        "candidate_count": len(candidates),
                        "training_identity_sha256": training[
                            "training_identity_sha256"
                        ],
                    },
                ),
            )
            shared_completion = read_json(shared_completion_path)
        training = {
            **training,
            "shared_training_completion": str(shared_completion_path.resolve()),
            "shared_training_completion_sha256": sha256_file(shared_completion_path),
            "shared_training_seal_sha256": shared_completion["seal_sha256"],
            "shared_training_checkpoint_sha256": shared_state["checkpoint_sha256"],
        }
        pending_ids = set(context.get("pending_candidate_ids", []))
        requested = [
            parameters
            for parameters in candidates
            if not pending_ids or parameters["candidate_id"] in pending_ids
        ]
        requested_models = {
            self._candidate_key(parameters): models[self._candidate_key(parameters)]
            for parameters in requested
        }
        probabilities = self._evaluate_models(requested_models, validation_rows, args)
        results = {}
        for parameters in requested:
            key = self._candidate_key(parameters)
            output = grid_root / f"candidate_{key}" / f"fold_{fold_number:02d}"
            results[key] = self._complete_validation_run(
                model=models[key],
                parameters=parameters,
                probabilities=probabilities[key],
                train_rows=train_rows,
                validation_rows=validation_rows,
                output_dir=output,
                args=args,
                context={**context, "fold": fold_number, "candidate_id": key},
                training={
                    **training,
                    "shared_candidate_count": len(candidates),
                    "candidate_failure": training.get("candidate_failures", {}).get(key),
                },
                logger=logger,
            )
        return results

    def fit_one(
        self,
        parameters: Mapping[str, Any],
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
        output_dir: Path,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        key = self._candidate_key(parameters)
        models, training = self._train_candidates(
            [parameters], train_rows, output_dir / "last_checkpoint.pt", args, context, logger
        )
        if training["candidate_failures"]:
            raise RuntimeError(
                "The selected fixed COLIPRI probe diverged; choose a completed grid "
                "candidate or a different explicitly documented fixed profile"
            )
        probabilities = self._evaluate_models({key: models[key]}, validation_rows, args)[key]
        return self._complete_validation_run(
            model=models[key],
            parameters=parameters,
            probabilities=probabilities,
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=output_dir,
            args=args,
            context=context,
            training=training,
            logger=logger,
        )

    def refit(
        self,
        parameters: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        output_dir: Path,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        key = self._candidate_key(parameters)
        models, training = self._train_candidates(
            [parameters], rows, output_dir / "last_checkpoint.pt", args, context, logger
        )
        if training["candidate_failures"]:
            raise RuntimeError(
                "The selected COLIPRI refit candidate diverged; its CV/grid identity "
                "cannot be represented as a completed refit"
            )
        model = models[key]
        artifact = self._write_model_artifact(
            model, parameters, output_dir, args, training, context
        )
        atomic_write_json(output_dir / "parameters.json", dict(parameters))
        atomic_write_json(
            output_dir / "training_state.json",
            {
                "schema_version": 1,
                "status": "complete",
                "completed_at": utc_now(),
                "method_label": self._method_label(context),
                "strict_linear": bool(model.strict_linear),
                "encoder_frozen": True,
                "refit_full_development": True,
                "row_count": len(rows),
                "rows_sha256": fingerprint(
                    [(row["patient_id"], row["label"]) for row in rows]
                ),
                "feature_identity": self._context_feature_identity(args, context),
                "training": training,
                "context": dict(context),
            },
        )
        atomic_write_json(
            output_dir / "completion.json",
            completion_seal(
                [artifact, output_dir / "parameters.json", output_dir / "training_state.json"],
                context={
                    "model": self.name,
                    "parameters_sha256": fingerprint(dict(parameters)),
                    "artifact_sha256": sha256_file(artifact),
                },
            ),
        )
        logger.info("Refit COLIPRI probe on %s development cases", len(rows))
        return {"artifact": str(artifact)}

    def _load_artifact_model(self, artifact: Path, args: Any):
        payload = _torch_load(artifact, map_location="cpu")
        if (
            payload.get("schema_version") != 1
            or payload.get("model") != self.name
            or payload.get("method_label")
            not in (METHOD_LABEL, LEGACY_METHOD_LABEL)
            or payload.get("class_names") != list(CLASS_NAMES)
        ):
            raise RuntimeError(f"Incompatible COLIPRI classifier artifact: {artifact}")
        if payload.get("training", {}).get("candidate_failure") is not None:
            raise RuntimeError(
                f"COLIPRI classifier artifact is a recorded failed grid trial: {artifact}"
            )
        artifact_identity = payload.get("feature_identity")
        if payload.get("method_label") == LEGACY_METHOD_LABEL:
            if not isinstance(artifact_identity, Mapping):
                raise RuntimeError(
                    f"Legacy COLIPRI artifact has no feature identity: {artifact}"
                )
            self._validate_legacy_feature_identity(args, artifact_identity)
        elif artifact_identity != self.feature_identity(args):
            raise RuntimeError(
                f"COLIPRI feature identity mismatch for artifact: {artifact}"
            )
        architecture = payload["architecture"]
        parameters = payload["parameters"]
        self._validate_candidate_parameters(parameters)
        model = self._build_probe(parameters, seed=0, device=str(args.device))
        if model.architecture() != architecture:
            raise RuntimeError(f"COLIPRI artifact architecture mismatch: {artifact}")
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.eval()

    def predict(
        self,
        artifact: Path,
        rows: Sequence[Mapping[str, Any]],
        args: Any,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        model = self._load_artifact_model(artifact, args)
        key = artifact.resolve().as_posix()
        return self._evaluate_models({key: model}, rows, args)[key]

    def selection_summary(
        self, parameters: Mapping[str, Any], fold_metrics: Sequence[Mapping[str, Any]]
    ) -> dict:
        fold_failures = [
            {
                "fold": fold_number,
                "failure": item.get("candidate_failure"),
            }
            for fold_number, item in enumerate(fold_metrics, start=1)
            if item.get("candidate_status") == "failed"
        ]
        average_precision = [
            float(item["average_precision_macro"])
            for item in fold_metrics
            if item.get("average_precision_macro") is not None
        ]
        if len(average_precision) != len(fold_metrics):
            raise RuntimeError("Macro-AUPRC is undefined in at least one COLIPRI validation fold")
        return {
            "status": "failed" if fold_failures else "complete",
            "eligible_for_selection": not fold_failures,
            "fold_failures": fold_failures,
            "selection_value": None
            if fold_failures
            else float(np.mean(average_precision)),
            "mean_average_precision_macro": float(np.mean(average_precision)),
            "std_average_precision_macro": float(np.std(average_precision)),
            "project_robust_score": project_grid_score(fold_metrics),
            "mean_f1_macro": float(np.mean([item["f1_macro"] for item in fold_metrics])),
            "std_f1_macro": float(np.std([item["f1_macro"] for item in fold_metrics])),
            "mean_balanced_accuracy": float(
                np.mean([item["balanced_accuracy"] for item in fold_metrics])
            ),
            "std_balanced_accuracy": float(
                np.std([item["balanced_accuracy"] for item in fold_metrics])
            ),
            "mean_perforation_recall": perforation_recall(fold_metrics),
        }

    @staticmethod
    def ranking_key(summary: Mapping[str, Any]) -> tuple:
        pooling_order = {
            "average": 0,
            "max": 1,
            "learned_attention": 2,
            "average_attention": 3,
            "multi_learned_attention": 4,
        }
        if summary.get("status", "complete") != "complete":
            return (1, str(summary["candidate_id"]))
        return (
            0,
            -float(summary["selection_value"]),
            float(summary["std_average_precision_macro"]),
            -float(summary["mean_perforation_recall"]),
            float(summary["parameters"]["learning_rate"]),
            pooling_order[str(summary["parameters"]["pooling"])],
            str(summary["candidate_id"]),
        )

    @staticmethod
    def summary_csv_row(summary: Mapping[str, Any]) -> dict:
        return {
            "rank": summary.get("rank"),
            "status": summary.get("status", "complete"),
            "eligible_for_selection": summary.get("eligible_for_selection", True),
            "failed_fold_count": len(summary.get("fold_failures", [])),
            "candidate_id": summary["candidate_id"],
            "pooling": summary["parameters"]["pooling"],
            "learning_rate": summary["parameters"]["learning_rate"],
            "mean_macro_auprc": summary["mean_average_precision_macro"],
            "std_macro_auprc": summary["std_average_precision_macro"],
            "project_robust_score": summary["project_robust_score"],
            "mean_f1_macro": summary["mean_f1_macro"],
            "std_f1_macro": summary["std_f1_macro"],
            "mean_balanced_accuracy": summary["mean_balanced_accuracy"],
            "std_balanced_accuracy": summary["std_balanced_accuracy"],
            "mean_perforation_recall": summary["mean_perforation_recall"],
        }


__all__ = [
    "COLIPRIBackend",
    "DeterministicBatchStream",
    "effective_number_weights",
]
