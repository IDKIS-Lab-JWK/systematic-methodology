"""Generic resumable trainer used by the four new adaptation methods."""

from __future__ import annotations

import math
import platform
import statistics
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _linear_probe.common import (
    CLASS_NAMES,
    ContractError,
    atomic_write_csv,
    atomic_write_json,
    candidate_id,
    completion_seal,
    ensure_same_or_write,
    fingerprint,
    rows_fingerprint,
    sha256_file,
    utc_now,
)
from _linear_probe.metrics import (
    compute_metrics,
    fold_selection_score,
    perforation_recall,
    prediction_documents,
)

from . import CHECKPOINT_SCHEMA_VERSION, METHOD_SCHEMA_VERSION
from .profiles import build_candidates as expand_candidates
from .runtime import (
    DeterministicBatchStream,
    atomic_torch_save,
    autocast_context,
    effective_number_weights,
    load_trainable_state_dict,
    optimizer_from_groups,
    require_torch,
    resolve_precision,
    restore_torch_rng_state,
    scheduler_from_optimizer,
    torch_load,
    torch_rng_state,
    trainable_state_dict,
)
from .safe_io import same_or_write_csv, same_or_write_json


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_identity(device: str, precision: str) -> dict:
    torch = require_torch()
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "scikit_learn": _package_version("scikit-learn"),
        "device_requested": str(device),
        "training_precision_resolved": precision,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn": int(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        identity["cuda_device_name"] = torch.cuda.get_device_name(index)
        identity["cuda_device_capability"] = list(torch.cuda.get_device_capability(index))
    return identity


def _implementation_identity(paths: Sequence[str | Path]) -> dict:
    records = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {"files": records, "sha256": fingerprint(records)}


def _resolved_max_steps(parameters: Mapping[str, Any], row_count: int) -> int:
    if parameters.get("max_steps") is not None:
        value = int(parameters["max_steps"])
    elif parameters.get("epochs") is not None:
        effective_batch = int(parameters["effective_batch_size"])
        value = int(math.ceil(row_count / effective_batch) * int(parameters["epochs"]))
    else:
        raise ContractError("Parameters require either max_steps or epochs")
    if value <= 0:
        raise ContractError("Resolved optimizer-step budget must be positive")
    return value


def _seed(base: int, fold: int, identifier: str) -> int:
    return int(base) + int(fold) * 1_000_003 + int(identifier[:8], 16)


def input_rows_identity(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Bind the concrete input files without repeatedly hashing large volumes."""

    records = []
    for row in rows:
        record = {
            "patient_id": str(row["patient_id"]),
            "class_name": str(row["class_name"]),
            "label": int(row["label"]),
        }
        raw_path = row.get("token_path") or row.get("cache_path") or row.get(
            "image_path"
        )
        if raw_path is None:
            record["input_kind"] = "adapter_materialized_non_file"
            records.append(record)
            continue
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        record.update({
            "path": str(path),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
        declared_sha256 = row.get("token_sha256") or row.get("cache_sha256")
        if declared_sha256 is not None:
            record["declared_sha256"] = str(declared_sha256)
        records.append(record)
    return {
        "schema_version": 1,
        "row_count": len(records),
        "identity_sha256": fingerprint(records),
        "policy": (
            "ordered patient/label plus resolved input path, byte size, nanosecond "
            "mtime, and any cache-declared SHA-256 when the adapter uses file inputs"
        ),
    }


class AdaptationBackend:
    """Workflow-facing backend around a model-specific adapter object.

    The adapter owns model construction, input preparation, and trainable-scope
    validation.  This class owns deterministic optimization, exact checkpoints,
    metrics, artifacts, and completion seals.
    """

    artifact_filename = "model.pt"
    selection_metric = "robust five-fold macro-F1 and balanced accuracy"

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.name = str(adapter.name)
        self.method_label = str(adapter.method_label)
        self.method_id = str(adapter.method_id)
        self.publication_profile_path = Path(adapter.publication_profile_path)
        self.default_profile_path = Path(adapter.default_profile_path)
        self.search_profile_path = Path(adapter.search_profile_path)
        self._model_identity: dict | None = None

    def build_candidates(self, profile: Mapping[str, Any]) -> list[dict]:
        candidates = expand_candidates(profile)
        for parameters in candidates:
            self._validate_parameters(parameters)
        return candidates

    def _validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        canonical = dict(parameters)
        explicit = canonical.pop("candidate_id", None)
        if explicit is not None and str(explicit) != candidate_id(canonical):
            raise ContractError("candidate_id does not match the executable parameter fingerprint")
        effective_batch = int(parameters.get("effective_batch_size", 0))
        if effective_batch <= 0:
            raise ContractError("effective_batch_size must be positive")
        if int(parameters.get("eval_batch_size", 0)) <= 0:
            raise ContractError("eval_batch_size must be positive")
        budgets = [
            key
            for key in ("max_steps", "epochs")
            if parameters.get(key) is not None
        ]
        if len(budgets) != 1 or int(parameters[budgets[0]]) <= 0:
            raise ContractError("Parameters require exactly one positive max_steps or epochs budget")
        optimizer = str(parameters.get("optimizer", "")).lower()
        if optimizer not in ("adamw", "sgd"):
            raise ContractError(f"Unsupported optimizer: {optimizer!r}")
        scheduler = str(parameters.get("scheduler", "")).lower()
        if scheduler not in ("cosine", "polynomial", "constant"):
            raise ContractError(f"Unsupported scheduler: {scheduler!r}")
        warmup = int(parameters.get("warmup_steps", 0))
        if warmup < 0 or (
            parameters.get("max_steps") is not None
            and warmup >= int(parameters["max_steps"])
        ):
            raise ContractError("warmup_steps must be non-negative and smaller than max_steps")
        minimum_ratio = float(parameters.get("minimum_lr_ratio", 0.0))
        if not 0.0 <= minimum_ratio <= 1.0:
            raise ContractError("minimum_lr_ratio must lie in [0,1]")
        if scheduler == "polynomial" and float(parameters.get("polynomial_power", 0.0)) <= 0.0:
            raise ContractError("polynomial_power must be positive")
        for key in ("weight_decay", "head_weight_decay"):
            if key in parameters and float(parameters[key]) < 0.0:
                raise ContractError(f"{key} must be non-negative")
        if optimizer == "adamw":
            betas = [float(value) for value in parameters.get("betas", ())]
            if len(betas) != 2 or any(not 0.0 <= value < 1.0 for value in betas):
                raise ContractError("AdamW betas must contain two values in [0,1)")
            if float(parameters.get("epsilon", 0.0)) <= 0.0:
                raise ContractError("AdamW epsilon must be positive")
        if optimizer == "sgd":
            momentum = float(parameters.get("momentum", 0.0))
            if not 0.0 <= momentum < 1.0:
                raise ContractError("SGD momentum must lie in [0,1)")
            if bool(parameters.get("nesterov", False)) and momentum <= 0.0:
                raise ContractError("Nesterov SGD requires positive momentum")
        if not 0.0 <= float(parameters.get("class_weight_beta", -1.0)) < 1.0:
            raise ContractError("class_weight_beta must lie in [0,1)")
        if float(parameters.get("gradient_clip_norm", 0.0)) < 0.0:
            raise ContractError("gradient_clip_norm must be non-negative")
        if int(parameters.get("validation_interval_steps", 0)) <= 0:
            raise ContractError("validation_interval_steps must be positive")
        if int(parameters.get("early_stopping_patience_steps", 0)) < 0:
            raise ContractError("early_stopping_patience_steps must be non-negative")
        self.adapter.validate_parameters(parameters)

    def dry_run_model_claim(self, args: Any) -> dict:
        return self.adapter.planned_identity(args)

    def ensure_feature_rows(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> None:
        self.adapter.preflight(args, rows)

    @staticmethod
    def input_rows_identity(rows: Sequence[Mapping[str, Any]]) -> dict:
        return input_rows_identity(rows)

    def feature_identity(self, args: Any) -> dict:
        if self._model_identity is None:
            implementation_paths = [
                Path(__file__),
                Path(__file__).with_name("runtime.py"),
                Path(__file__).with_name("profiles.py"),
                Path(__file__).with_name("safe_io.py"),
                Path(__file__).with_name("workflow.py"),
                Path(__file__).with_name("cli.py"),
                *[Path(item) for item in self.adapter.implementation_paths()],
            ]
            self._model_identity = {
                "schema_version": METHOD_SCHEMA_VERSION,
                "model": self.name,
                "method": self.method_label,
                "method_id": self.method_id,
                "pretrained_model": self.adapter.resolve_identity(args),
                "representation": self.adapter.representation_contract(args),
                "implementation": _implementation_identity(implementation_paths),
            }
        return dict(self._model_identity)

    def _prepare_rows(
        self, args: Any, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict]:
        prepared = [dict(item) for item in self.adapter.prepare_rows(args, rows)]
        if len(prepared) != len(rows):
            raise ContractError("Prepared row count differs from the split row count")
        expected = [
            (str(row["patient_id"]), int(row["label"])) for row in rows
        ]
        observed = [
            (str(row["patient_id"]), int(row["label"])) for row in prepared
        ]
        if observed != expected:
            raise ContractError("Prepared rows changed patient order or labels")
        for row in prepared:
            row.setdefault("class_name", CLASS_NAMES[int(row["label"])])
        return prepared

    def _training_identity(
        self,
        *,
        parameters: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        args: Any,
        context: Mapping[str, Any],
        precision: str,
        max_steps: int,
    ) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": self.name,
            "method": self.method_label,
            "parameters": dict(parameters),
            "training_rows_sha256": rows_fingerprint(rows),
            "training_input_identity": input_rows_identity(rows),
            "model_identity": self.feature_identity(args),
            "context": dict(context),
            "seed": int(args.seed),
            "training_precision_requested": str(args.training_precision),
            "training_precision_resolved": precision,
            "micro_batch_size": int(args.micro_batch_size),
            "effective_batch_size": int(parameters["effective_batch_size"]),
            "max_steps_resolved": int(max_steps),
            "runtime": _runtime_identity(str(args.device), precision),
        }

    @staticmethod
    def _validate_optimizer_groups(model: Any, groups: Sequence[Mapping[str, Any]]) -> None:
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        observed: list[int] = []
        for group in groups:
            if "lr" not in group or "weight_decay" not in group:
                raise ContractError("Every optimizer group must declare lr and weight_decay")
            observed.extend(id(parameter) for parameter in group.get("params", []))
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ContractError("Optimizer groups do not exactly cover trainable parameters")

    @staticmethod
    def _move_optimizer_state(optimizer: Any, device: str) -> None:
        torch = require_torch()
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _probabilities(
        self,
        model: Any,
        rows: Sequence[Mapping[str, Any]],
        args: Any,
        parameters: Mapping[str, Any],
        precision_dtype: Any | None,
    ) -> np.ndarray:
        torch = require_torch()
        model.eval()
        batch_size = max(1, int(parameters.get("eval_batch_size", args.micro_batch_size)))
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                indices = list(range(start, min(start + batch_size, len(rows))))
                inputs, _ = self.adapter.load_batch(
                    args, rows, indices, training=False, generator=None
                )
                inputs = self.adapter.move_inputs(inputs, str(args.device))
                with autocast_context(str(args.device), precision_dtype):
                    logits = model(inputs)
                if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                    raise RuntimeError(f"Expected five logits, got {tuple(logits.shape)}")
                chunks.append(torch.softmax(logits.float(), dim=1).cpu())
        probabilities = torch.cat(chunks, dim=0).numpy().astype(np.float64, copy=False)
        if probabilities.shape != (len(rows), len(CLASS_NAMES)):
            raise RuntimeError("Prediction shape changed during evaluation")
        return probabilities

    def _checkpoint_payload(
        self,
        *,
        identity: Mapping[str, Any],
        model: Any,
        optimizer: Any,
        scheduler: Any,
        stream: DeterministicBatchStream,
        augmentation_generator: Any,
        completed_steps: int,
        best_step: int,
        best_score: float,
        best_state: Mapping[str, Any] | None,
        history: Sequence[Mapping[str, Any]],
        class_counts: Any,
        class_weights: Any,
        status: str,
        early_stopped: bool,
    ) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": status,
            "identity": dict(identity),
            "identity_sha256": fingerprint(identity),
            "completed_optimizer_steps": int(completed_steps),
            "best_optimizer_step": int(best_step),
            "best_selection_score": float(best_score),
            "model_trainable_state": trainable_state_dict(model),
            "best_trainable_state": dict(best_state) if best_state is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batch_stream": stream.state_dict(),
            "augmentation_generator_state": augmentation_generator.get_state().clone(),
            "torch_rng": torch_rng_state(),
            "history": [dict(item) for item in history],
            "class_counts": class_counts.detach().cpu(),
            "class_weights": class_weights.detach().cpu(),
            "early_stopped": bool(early_stopped),
            "updated_at": utc_now(),
        }

    def _train(
        self,
        parameters: Mapping[str, Any],
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]] | None,
        output_dir: Path,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
        *,
        budget_steps: int | None = None,
    ) -> tuple[Any, dict, np.ndarray | None, dict | None]:
        torch = require_torch()
        self._validate_parameters(parameters)
        train = self._prepare_rows(args, train_rows)
        validation = (
            self._prepare_rows(args, validation_rows)
            if validation_rows is not None
            else None
        )
        effective_batch = int(parameters["effective_batch_size"])
        micro_batch = int(args.micro_batch_size)
        if micro_batch <= 0 or micro_batch > effective_batch or effective_batch % micro_batch:
            raise ContractError(
                "micro_batch_size must be positive, no larger than, and divide effective_batch_size"
            )
        max_steps = int(budget_steps or _resolved_max_steps(parameters, len(train)))
        precision_name, precision_dtype = resolve_precision(
            str(args.device), str(args.training_precision)
        )
        identity = self._training_identity(
            parameters=parameters,
            rows=train,
            args=args,
            context=context,
            precision=precision_name,
            max_steps=max_steps,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_same_or_write(output_dir / "training_context.json", identity)

        fold = int(context.get("fold", 0))
        identifier = str(parameters.get("candidate_id") or candidate_id(parameters))
        initialization_seed = _seed(int(args.seed), fold, identifier)
        torch.manual_seed(initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initialization_seed)
        model = self.adapter.build_model(args, parameters, str(args.device))
        model.to(str(args.device))
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable_names:
            raise ContractError("Adaptation model has no trainable parameters")
        self.adapter.validate_trainable_scope(model, parameters)
        groups = self.adapter.optimizer_groups(model, parameters)
        self._validate_optimizer_groups(model, groups)
        optimizer = optimizer_from_groups(groups, parameters)
        scheduler = scheduler_from_optimizer(optimizer, parameters, max_steps)

        labels = [int(row["label"]) for row in train]
        class_weights, class_counts = effective_number_weights(
            labels, len(CLASS_NAMES), float(parameters["class_weight_beta"])
        )
        class_weights_device = class_weights.to(str(args.device))
        stream = DeterministicBatchStream(len(train), initialization_seed + 17)
        augmentation_generator = torch.Generator(device="cpu")
        augmentation_generator.manual_seed(initialization_seed + 29)
        completed_steps = 0
        best_step = 0
        best_score = float("-inf")
        best_state: dict | None = None
        history: list[dict] = []
        checkpoint_path = output_dir / "last_checkpoint.pt"
        checkpoint_complete = False
        early_stopped = False

        if checkpoint_path.is_file():
            if not args.resume:
                raise ContractError(
                    f"Partial checkpoint exists and resume is disabled: {checkpoint_path}"
                )
            payload = torch_load(checkpoint_path, map_location=str(args.device))
            if (
                payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or payload.get("identity_sha256") != fingerprint(identity)
                or payload.get("identity") != identity
            ):
                raise ContractError(f"Incompatible adaptation resume checkpoint: {checkpoint_path}")
            load_trainable_state_dict(model, payload["model_trainable_state"])
            optimizer.load_state_dict(payload["optimizer"])
            self._move_optimizer_state(optimizer, str(args.device))
            scheduler.load_state_dict(payload["scheduler"])
            stream.load_state_dict(payload["batch_stream"])
            augmentation_generator.set_state(
                torch.as_tensor(
                    payload["augmentation_generator_state"], dtype=torch.uint8
                ).cpu()
            )
            restore_torch_rng_state(payload["torch_rng"])
            completed_steps = int(payload["completed_optimizer_steps"])
            best_step = int(payload.get("best_optimizer_step", 0))
            best_score = float(payload.get("best_selection_score", float("-inf")))
            loaded_best = payload.get("best_trainable_state")
            best_state = dict(loaded_best) if isinstance(loaded_best, Mapping) else None
            history = [dict(item) for item in payload.get("history", [])]
            checkpoint_complete = payload.get("status") == "complete"
            early_stopped = bool(payload.get("early_stopped", False))
            logger.info("Resumed %s at optimizer step %s/%s", self.method_label, completed_steps, max_steps)

        validation_interval = int(parameters["validation_interval_steps"])
        patience = int(parameters.get("early_stopping_patience_steps", 0))
        min_delta = float(parameters.get("early_stopping_min_delta", 0.0))
        checkpoint_interval = int(args.checkpoint_interval)
        report_interval = max(1, min(100, max_steps // 20 if max_steps >= 20 else 1))
        last_evaluated_step = history[-1]["optimizer_step"] if history else 0

        def evaluate() -> tuple[np.ndarray, dict, float]:
            assert validation is not None
            probabilities = self._probabilities(
                model, validation, args, parameters, precision_dtype
            )
            truth = np.asarray([int(row["label"]) for row in validation], dtype=np.int64)
            metrics = compute_metrics(
                truth, probabilities.argmax(axis=1), probabilities, CLASS_NAMES
            )
            score = float(fold_selection_score(metrics))
            return probabilities, metrics, score

        def save_checkpoint(status: str) -> None:
            atomic_torch_save(
                self._checkpoint_payload(
                    identity=identity,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stream=stream,
                    augmentation_generator=augmentation_generator,
                    completed_steps=completed_steps,
                    best_step=best_step,
                    best_score=best_score,
                    best_state=best_state,
                    history=history,
                    class_counts=class_counts,
                    class_weights=class_weights,
                    status=status,
                    early_stopped=early_stopped,
                ),
                checkpoint_path,
            )

        while completed_steps < max_steps and not checkpoint_complete and not early_stopped:
            self.adapter.set_training_mode(model, True)
            optimizer.zero_grad(set_to_none=True)
            effective_indices = stream.next(effective_batch)
            effective_targets = torch.as_tensor(
                [int(train[index]["label"]) for index in effective_indices],
                dtype=torch.long,
                device=str(args.device),
            )
            denominator = class_weights_device[effective_targets].sum()
            for start in range(0, effective_batch, micro_batch):
                indices = effective_indices[start : start + micro_batch]
                inputs, targets = self.adapter.load_batch(
                    args,
                    train,
                    indices,
                    training=True,
                    generator=augmentation_generator,
                )
                inputs = self.adapter.move_inputs(inputs, str(args.device))
                targets = targets.to(str(args.device))
                with autocast_context(str(args.device), precision_dtype):
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        targets,
                        weight=class_weights_device,
                        reduction="sum",
                    ) / denominator
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer step {completed_steps + 1}"
                    )
                loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(item).all()) for item in gradients):
                raise FloatingPointError(
                    f"Non-finite or missing gradient at optimizer step {completed_steps + 1}"
                )
            clip = float(parameters.get("gradient_clip_norm", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=clip,
                )
            optimizer.step()
            scheduler.step()
            completed_steps += 1

            due_validation = validation is not None and (
                completed_steps % validation_interval == 0 or completed_steps == max_steps
            )
            if due_validation:
                probabilities, metrics, score = evaluate()
                last_evaluated_step = completed_steps
                history.append(
                    {
                        "optimizer_step": completed_steps,
                        "selection_score": score,
                        "f1_macro": metrics["f1_macro"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "learning_rates": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                    }
                )
                if score > best_score + min_delta:
                    best_score = score
                    best_step = completed_steps
                    best_state = trainable_state_dict(model)
                if patience and best_step and completed_steps - best_step >= patience:
                    early_stopped = True
            if completed_steps % report_interval == 0 or completed_steps == max_steps:
                logger.info(
                    "%s optimizer step %s/%s", self.method_label, completed_steps, max_steps
                )
            if (
                completed_steps % checkpoint_interval == 0
                or due_validation
                or completed_steps == max_steps
                or early_stopped
            ):
                save_checkpoint("partial")
            if early_stopped:
                logger.info(
                    "Early stopping at step %s; best validation step was %s",
                    completed_steps,
                    best_step,
                )
                break

        if validation is not None and last_evaluated_step != completed_steps:
            _, metrics, score = evaluate()
            history.append(
                {
                    "optimizer_step": completed_steps,
                    "selection_score": score,
                    "f1_macro": metrics["f1_macro"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                }
            )
            if score > best_score + min_delta:
                best_score = score
                best_step = completed_steps
                best_state = trainable_state_dict(model)
        if validation is None:
            best_step = completed_steps
            best_state = trainable_state_dict(model)
            best_score = float("nan")
        if best_state is None or best_step <= 0:
            raise RuntimeError("Training completed without a selectable model state")
        if not checkpoint_complete:
            save_checkpoint("complete")
        load_trainable_state_dict(model, best_state)

        final_probabilities = None
        final_metrics = None
        if validation is not None:
            final_probabilities = self._probabilities(
                model, validation, args, parameters, precision_dtype
            )
            truth = np.asarray([int(row["label"]) for row in validation], dtype=np.int64)
            final_metrics = compute_metrics(
                truth,
                final_probabilities.argmax(axis=1),
                final_probabilities,
                CLASS_NAMES,
            )
        trainable_count = sum(
            int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
        )
        frozen_count = sum(
            int(parameter.numel()) for parameter in model.parameters() if not parameter.requires_grad
        )
        training = {
            "schema_version": 1,
            "status": "complete",
            "context": dict(context),
            "training_rows_sha256": rows_fingerprint(train),
            "validation_rows_sha256": rows_fingerprint(validation) if validation else None,
            "feature_identity": self.feature_identity(args),
            "parameters_sha256": fingerprint(dict(parameters)),
            "training_identity_sha256": fingerprint(identity),
            "completed_optimizer_steps": completed_steps,
            "max_steps_resolved": max_steps,
            "selected_optimizer_step": best_step,
            "best_validation_selection_score": best_score if np.isfinite(best_score) else None,
            "early_stopped": early_stopped,
            "effective_batch_size": effective_batch,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": effective_batch // micro_batch,
            "training_precision_requested": str(args.training_precision),
            "training_precision_resolved": precision_name,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "trainable_parameter_count": trainable_count,
            "frozen_parameter_count": frozen_count,
            "trainable_parameter_names": trainable_names,
            "optimizer_groups": [
                {
                    "group_name": str(group.get("group_name", f"group_{index}")),
                    "learning_rate": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "parameter_count": sum(int(item.numel()) for item in group["params"]),
                }
                for index, group in enumerate(groups)
            ],
            "history": history,
            "last_checkpoint": str(checkpoint_path.resolve()),
        }
        return model, training, final_probabilities, final_metrics

    @staticmethod
    def _cpu_state(model: Any) -> dict:
        return trainable_state_dict(model)

    def _write_artifact(
        self,
        model: Any,
        parameters: Mapping[str, Any],
        output_dir: Path,
        args: Any,
        training: Mapping[str, Any],
    ) -> Path:
        artifact = output_dir / self.artifact_filename
        payload = {
            "schema_version": METHOD_SCHEMA_VERSION,
            "model": self.name,
            "method": self.method_label,
            "method_id": self.method_id,
            "class_names": list(CLASS_NAMES),
            "parameters": dict(parameters),
            "model_identity": self.feature_identity(args),
            "architecture": self.adapter.architecture(model, parameters),
            "training": dict(training),
            "trainable_state_dict": self._cpu_state(model),
        }
        if artifact.is_file():
            existing = torch_load(artifact, map_location="cpu")
            existing_state = existing.get("trainable_state_dict")
            expected_state = payload["trainable_state_dict"]
            torch = require_torch()
            same_header = {
                key: value for key, value in existing.items() if key != "trainable_state_dict"
            } == {key: value for key, value in payload.items() if key != "trainable_state_dict"}
            same_state = (
                isinstance(existing_state, Mapping)
                and set(existing_state) == set(expected_state)
                and all(
                    isinstance(existing_state[key], torch.Tensor)
                    and torch.equal(existing_state[key].cpu(), expected_state[key].cpu())
                    for key in expected_state
                )
            )
            if not same_header or not same_state:
                raise ContractError(f"Existing adaptation artifact differs; refusing overwrite: {artifact}")
            return artifact
        atomic_torch_save(payload, artifact)
        return artifact

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
        model, training, probabilities, metrics = self._train(
            parameters,
            train_rows,
            validation_rows,
            Path(output_dir),
            args,
            context,
            logger,
        )
        assert probabilities is not None and metrics is not None
        prepared_validation = self._prepare_rows(args, validation_rows)
        json_rows, csv_rows = prediction_documents(
            prepared_validation, probabilities, CLASS_NAMES
        )
        artifact = self._write_artifact(
            model, parameters, Path(output_dir), args, training
        )
        same_or_write_json(Path(output_dir) / "parameters.json", dict(parameters))
        same_or_write_json(Path(output_dir) / "metrics_summary.json", metrics)
        same_or_write_json(Path(output_dir) / "validation_predictions.json", json_rows)
        same_or_write_csv(
            Path(output_dir) / "validation_probabilities.csv",
            csv_rows,
            list(csv_rows[0]),
        )
        same_or_write_json(Path(output_dir) / "training_state.json", training)
        artifacts = [
            artifact,
            Path(output_dir) / "last_checkpoint.pt",
            Path(output_dir) / "training_context.json",
            Path(output_dir) / "parameters.json",
            Path(output_dir) / "metrics_summary.json",
            Path(output_dir) / "validation_predictions.json",
            Path(output_dir) / "validation_probabilities.csv",
            Path(output_dir) / "training_state.json",
        ]
        atomic_write_json(
            Path(output_dir) / "completion.json",
            completion_seal(
                artifacts,
                context={
                    "model": self.name,
                    "method": self.method_label,
                    "candidate_id": parameters.get("candidate_id"),
                    "training_identity_sha256": training["training_identity_sha256"],
                    "artifact_sha256": sha256_file(artifact),
                },
            ),
        )
        return {"metrics": metrics, "artifact": str(artifact)}

    def refit(
        self,
        parameters: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        output_dir: Path,
        args: Any,
        context: Mapping[str, Any],
        logger: Any,
    ) -> dict:
        budget = int(context["refit_optimizer_steps"])
        model, training, _, _ = self._train(
            parameters,
            rows,
            None,
            Path(output_dir),
            args,
            context,
            logger,
            budget_steps=budget,
        )
        artifact = self._write_artifact(
            model, parameters, Path(output_dir), args, training
        )
        same_or_write_json(Path(output_dir) / "parameters.json", dict(parameters))
        same_or_write_json(Path(output_dir) / "training_state.json", training)
        artifacts = [
            artifact,
            Path(output_dir) / "last_checkpoint.pt",
            Path(output_dir) / "training_context.json",
            Path(output_dir) / "parameters.json",
            Path(output_dir) / "training_state.json",
        ]
        atomic_write_json(
            Path(output_dir) / "completion.json",
            completion_seal(
                artifacts,
                context={
                    "model": self.name,
                    "method": self.method_label,
                    "refit_optimizer_steps": budget,
                    "training_identity_sha256": training["training_identity_sha256"],
                    "artifact_sha256": sha256_file(artifact),
                },
            ),
        )
        return {"artifact": str(artifact), "optimizer_steps": budget}

    def _load_artifact(self, artifact: Path, args: Any) -> tuple[Any, dict]:
        payload = torch_load(artifact, map_location="cpu")
        if (
            payload.get("schema_version") != METHOD_SCHEMA_VERSION
            or payload.get("model") != self.name
            or payload.get("method") != self.method_label
            or payload.get("method_id") != self.method_id
            or payload.get("class_names") != list(CLASS_NAMES)
        ):
            raise ContractError(f"Incompatible adaptation artifact: {artifact}")
        if payload.get("model_identity") != self.feature_identity(args):
            raise ContractError(f"Pretrained checkpoint/code identity changed: {artifact}")
        parameters = dict(payload["parameters"])
        self._validate_parameters(parameters)
        model = self.adapter.build_model(args, parameters, str(args.device))
        model.to(str(args.device))
        load_trainable_state_dict(model, payload["trainable_state_dict"])
        model.eval()
        return model, parameters

    def predict(
        self,
        artifact: Path,
        rows: Sequence[Mapping[str, Any]],
        args: Any,
        _context: Mapping[str, Any],
    ) -> np.ndarray:
        model, parameters = self._load_artifact(Path(artifact), args)
        prepared = self._prepare_rows(args, rows)
        _, dtype = resolve_precision(str(args.device), str(args.training_precision))
        return self._probabilities(model, prepared, args, parameters, dtype)

    @staticmethod
    def refit_budget(fold_states: Sequence[Mapping[str, Any]]) -> int:
        steps = [int(item["selected_optimizer_step"]) for item in fold_states]
        if len(steps) != 5 or any(value <= 0 for value in steps):
            raise ContractError("Refit budget requires five positive selected fold budgets")
        return max(1, int(round(statistics.median(steps))))

    def selection_summary(
        self,
        parameters: Mapping[str, Any],
        fold_metrics: Sequence[Mapping[str, Any]],
    ) -> dict:
        if len(fold_metrics) != 5:
            raise ContractError("Candidate selection requires exactly five fold metrics")
        f1 = np.asarray([float(item["f1_macro"]) for item in fold_metrics])
        balanced = np.asarray(
            [float(item["balanced_accuracy"]) for item in fold_metrics]
        )
        robust_f1 = float(f1.mean() - f1.std(ddof=0))
        robust_balanced = float(balanced.mean() - balanced.std(ddof=0))
        return {
            "status": "complete",
            "eligible_for_selection": True,
            "mean_f1_macro": float(f1.mean()),
            "std_f1_macro": float(f1.std(ddof=0)),
            "mean_balanced_accuracy": float(balanced.mean()),
            "std_balanced_accuracy": float(balanced.std(ddof=0)),
            "robust_selection_score": 0.7 * robust_f1 + 0.3 * robust_balanced,
            "mean_perforation_recall": float(perforation_recall(fold_metrics)),
        }

    @staticmethod
    def ranking_key(summary: Mapping[str, Any]) -> tuple:
        return (
            -float(summary["robust_selection_score"]),
            -float(summary["mean_perforation_recall"]),
            -float(summary["mean_f1_macro"]),
            str(summary["candidate_id"]),
        )

    @staticmethod
    def summary_csv_row(summary: Mapping[str, Any]) -> dict:
        return {
            "rank": summary.get("rank"),
            "candidate_id": summary["candidate_id"],
            "robust_selection_score": summary["robust_selection_score"],
            "mean_f1_macro": summary["mean_f1_macro"],
            "std_f1_macro": summary["std_f1_macro"],
            "mean_balanced_accuracy": summary["mean_balanced_accuracy"],
            "std_balanced_accuracy": summary["std_balanced_accuracy"],
            "mean_perforation_recall": summary["mean_perforation_recall"],
        }


__all__ = ["AdaptationBackend", "input_rows_identity"]
