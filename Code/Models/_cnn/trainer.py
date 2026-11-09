"""CNN-owned optimizer loop, based on the systematic adaptation trainer.

The generic CV/grid/test workflow is reused unchanged. Adam, epoch schedules,
fold-fitted preprocessing and timing are local to this package.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from _linear_probe.common import (CLASS_NAMES, ContractError, candidate_id, fingerprint,
    rows_fingerprint, ensure_same_or_write, atomic_write_json, atomic_write_csv,
    completion_seal, read_json, append_jsonl, utc_now)
from _linear_probe.metrics import compute_metrics, fold_selection_score, prediction_documents
from _supervised_adaptation.safe_io import same_or_write_json, same_or_write_csv
from _supervised_adaptation import CHECKPOINT_SCHEMA_VERSION
from _supervised_adaptation.trainer import AdaptationBackend, _resolved_max_steps, _seed
from _supervised_adaptation.runtime import (require_torch, resolve_precision, effective_number_weights,
    DeterministicBatchStream, torch_load, load_trainable_state_dict, restore_torch_rng_state,
    atomic_torch_save, autocast_context, trainable_state_dict)
from .runtime import optimizer_from_groups, scheduler_from_optimizer


class CNNBackend(AdaptationBackend):
    def _validate_parameters(self, parameters):
        # Validate common fields through the unchanged parent, translating only
        # this local extension's enum/epoch fields for validation (never execution).
        translated = dict(parameters)
        explicit = translated.pop("candidate_id", None)
        if explicit is not None and explicit != candidate_id(translated):
            raise ContractError("CNN candidate fingerprint mismatch")
        self.adapter.validate_parameters(parameters)
        if parameters.get("optimizer") not in ("adam", "adamw", "sgd"):
            raise ContractError("Unsupported CNN optimizer")
        if parameters.get("optimizer") == "adam":
            translated["optimizer"] = "adamw"
        translated["scheduler"] = "constant"
        translated["validation_interval_steps"] = 1
        translated["early_stopping_patience_steps"] = 0
        super()._validate_parameters(translated)
        if int(parameters["validation_interval_epochs"]) <= 0 or int(parameters["early_stopping_patience_epochs"]) < 0:
            raise ContractError("Invalid CNN validation/stopping epoch interval")

    def _training_identity(self, **kwargs):
        return {**super()._training_identity(**kwargs), "preprocessing_plan": self.adapter.plan}

    def _checkpoint_payload(self, **kwargs):
        payload = super()._checkpoint_payload(**kwargs)
        self._saved_active_seconds = self._active_seconds + time.perf_counter() - self._session_start
        payload["cumulative_active_seconds"] = self._saved_active_seconds
        payload["preprocessing_observed"] = self.adapter.inputs_observed
        return payload

    def _load_artifact(self, artifact, args):
        payload = torch_load(artifact, map_location="cpu")
        plan = payload["training"]["preprocessing_plan"]
        if fingerprint(plan["numeric"]) != plan["numeric_sha256"]:
            raise ContractError("Artifact preprocessing fingerprint mismatch")
        self.adapter.plan = plan
        self.adapter.inputs_observed = {}
        return super()._load_artifact(artifact, args)

    def _extra_outputs(self, output_dir, artifacts, context):
        root = Path(output_dir)
        state = read_json(root / "training_state.json")
        # Derived history and coverage audit are immutable members of completion.
        history = state["history"]
        columns = ["optimizer_step", "epoch_equivalent", "selection_score", "f1_macro", "balanced_accuracy", "learning_rates", "cumulative_active_seconds"]
        atomic_write_csv(root / "training_history.csv", history, columns)
        atomic_write_json(root / "preprocessing_observed.json", {
            "scope": "patients loaded during training/validation, including restored checkpoint observations; plan fitted on all training patients",
            "patients": list(self.adapter.inputs_observed.values())})
        artifacts.extend(root / name for name in ("preprocessing_plan.json", "preprocessing_observed.json", "training_history.csv"))
        # Publish completion once, only after the CNN-specific audit is durable.
        atomic_write_json(root / "completion.json", completion_seal(artifacts, context=context))

    def fit_one(self, parameters, train_rows, validation_rows, output_dir, args, context, logger):
        root = Path(output_dir)
        model, training, probabilities, metrics = self._train(
            parameters, train_rows, validation_rows, root, args, context, logger)
        json_rows, csv_rows = prediction_documents(self._prepare_rows(args, validation_rows), probabilities, CLASS_NAMES)
        artifact = self._write_artifact(model, parameters, root, args, training)
        for name, value in (("parameters.json", dict(parameters)), ("training_state.json", training),
                            ("metrics_summary.json", metrics), ("validation_predictions.json", json_rows)):
            same_or_write_json(root / name, value)
        same_or_write_csv(root / "validation_probabilities.csv", csv_rows, list(csv_rows[0]))
        artifacts = [artifact, *[root / name for name in ("last_checkpoint.pt", "training_context.json",
            "parameters.json", "training_state.json", "metrics_summary.json", "validation_predictions.json",
            "validation_probabilities.csv")]]
        from _linear_probe.common import sha256_file
        self._extra_outputs(root, artifacts, {"model": self.name, "method": self.method_label,
            "candidate_id": parameters.get("candidate_id"), "training_identity_sha256": training["training_identity_sha256"],
            "artifact_sha256": sha256_file(artifact)})
        if args.cache_policy == "temporary":
            self.adapter.cache.release()
        return {"metrics": metrics, "artifact": str(artifact)}

    def refit(self, parameters, rows, output_dir, args, context, logger):
        root, budget = Path(output_dir), int(context["refit_optimizer_steps"])
        model, training, _, _ = self._train(parameters, rows, None, root, args, context, logger, budget_steps=budget)
        artifact = self._write_artifact(model, parameters, root, args, training)
        same_or_write_json(root / "parameters.json", dict(parameters))
        same_or_write_json(root / "training_state.json", training)
        artifacts = [artifact, *[root / name for name in ("last_checkpoint.pt", "training_context.json",
                                                       "parameters.json", "training_state.json")]]
        from _linear_probe.common import sha256_file
        self._extra_outputs(root, artifacts, {"model": self.name, "method": self.method_label,
            "refit_optimizer_steps": budget, "training_identity_sha256": training["training_identity_sha256"],
            "artifact_sha256": sha256_file(artifact)})
        if args.cache_policy == "temporary":
            self.adapter.cache.release()
        return {"artifact": str(artifact), "optimizer_steps": budget}

    def predict(self, artifact, rows, args, context):
        started = time.perf_counter()
        output = super().predict(artifact, rows, args, context)
        append_jsonl(Path(args.results_root) / "Logs" / "inference_timing.jsonl", {
            "time": utc_now(), "artifact": str(artifact), "patients": len(rows),
            "active_seconds": time.perf_counter() - started,
            "preprocessing_numeric_sha256": self.adapter.plan["numeric_sha256"],
            "expanded_canvas_patients": sum(r["expanded_canvas"] for r in self.adapter.inputs_observed.values()),
            "preprocessing_observed": list(self.adapter.inputs_observed.values())})
        if args.cache_policy == "temporary":
            self.adapter.cache.release()
        return output

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
        self._session_start = time.perf_counter()
        self._active_seconds = 0.0
        self._saved_active_seconds = 0.0
        self.adapter.configure_training(args, train_rows, parameters, output_dir)
        logger.info("Preprocessing: reference spacing %s; working spacing %s; canvas XYZ %s; no crop",
                    self.adapter.plan["reference_spacing_xyz_mm"],
                    self.adapter.plan["numeric"]["working_spacing_xyz_mm"], parameters["canvas_xyz"])
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
        steps_per_epoch = math.ceil(len(train) / effective_batch)
        scheduler = scheduler_from_optimizer(optimizer, parameters, steps_per_epoch)

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
            self._active_seconds = float(payload.get("cumulative_active_seconds", 0.0))
            self.adapter.inputs_observed.update(payload.get("preprocessing_observed", {}))
            self._saved_active_seconds = self._active_seconds
            completed_steps = int(payload["completed_optimizer_steps"])
            best_step = int(payload.get("best_optimizer_step", 0))
            best_score = float(payload.get("best_selection_score", float("-inf")))
            loaded_best = payload.get("best_trainable_state")
            best_state = dict(loaded_best) if isinstance(loaded_best, Mapping) else None
            history = [dict(item) for item in payload.get("history", [])]
            checkpoint_complete = payload.get("status") == "complete"
            early_stopped = bool(payload.get("early_stopped", False))
            logger.info("Resumed %s at optimizer step %s/%s", self.method_label, completed_steps, max_steps)

        validation_interval = int(parameters["validation_interval_epochs"]) * steps_per_epoch
        patience = int(parameters["early_stopping_patience_epochs"]) * steps_per_epoch
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
                        "epoch_equivalent": completed_steps / steps_per_epoch,
                        "cumulative_active_seconds": self._active_seconds + time.perf_counter() - self._session_start,
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
                    "epoch_equivalent": completed_steps / steps_per_epoch,
                    "cumulative_active_seconds": self._active_seconds + time.perf_counter() - self._session_start,
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
            "preprocessing_plan": self.adapter.plan,
            "steps_per_epoch": steps_per_epoch,
            "cumulative_active_seconds": self._saved_active_seconds,
            "timing_scope": "Active invocation time through last saved optimizer checkpoint; includes model/data/cache/validation work; excludes gaps and final artifact export. Lost work after last checkpoint is not counted.",
        }
        return model, training, final_probabilities, final_metrics

