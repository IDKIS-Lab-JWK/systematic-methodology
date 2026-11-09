"""FlexiCT sklearn probe backend for the shared experiment orchestrator."""

from __future__ import annotations

import os
import platform
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import CLASS_NAMES, FEATURE_DIM, UPSTREAM_REVISION
    from .cache import FeatureStore
except ImportError:  # pragma: no cover
    from __init__ import CLASS_NAMES, FEATURE_DIM, UPSTREAM_REVISION
    from cache import FeatureStore

try:
    from _linear_probe.common import (
        ContractError,
        atomic_write_csv,
        atomic_write_json,
        candidate_id,
        completion_seal,
        fingerprint,
        sha256_file,
        utc_now,
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
        ContractError,
        atomic_write_csv,
        atomic_write_json,
        candidate_id,
        completion_seal,
        fingerprint,
        sha256_file,
        utc_now,
    )
    from _linear_probe.metrics import (
        compute_metrics,
        perforation_recall,
        prediction_documents,
        project_grid_score,
    )


class FlexiCTBackend:
    name = "FlexiCT"
    artifact_filename = "model.joblib"
    selection_metric = (
        "0.7*(mean macro-F1 - population std) + "
        "0.3*(mean balanced accuracy - population std)"
    )
    publication_c_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    _standardization_claim = "StandardScaler fit on training partition only"

    def __init__(self) -> None:
        package = Path(__file__).resolve().parent
        self.publication_profile_path = package / "publication_linear_probe.json"
        self.default_profile_path = package / "default_linear_probe.json"
        self.search_profile_path = package / "search_linear_probe.json"
        self._store: FeatureStore | None = None

    def build_candidates(self, profile: Mapping[str, Any]) -> list[dict]:
        fixed = dict(profile.get("fixed_parameters", {}))
        values = profile.get("search_space", {}).get("C")
        if values is None:
            parameters = profile.get("parameters")
            candidates = [dict(parameters)] if isinstance(parameters, dict) else []
        else:
            try:
                candidates = [{**fixed, "C": float(value)} for value in values]
            except (TypeError, ValueError) as exc:
                raise ContractError("FlexiCT C search space must be a numeric sequence") from exc
        for candidate in candidates:
            self._validate_parameters(candidate)
        if str(profile.get("profile_type", "")).startswith("publication_"):
            observed = tuple(float(item["C"]) for item in candidates)
            if observed != self.publication_c_grid:
                raise ContractError(
                    "FlexiCT publication profile must retain the complete ordered seven-C grid"
                )
            required_fields = {
                "classifier",
                "penalty",
                "solver",
                "class_weight",
                "max_iter",
                "random_state",
                "feature_standardization",
                "encoder_frozen",
                "feature_dimension",
            }
            if any(not required_fields.issubset(candidate) for candidate in candidates):
                raise ContractError(
                    "FlexiCT publication profile is missing native fixed-recipe fields"
                )
            source = profile.get("source", {})
            if not isinstance(source, Mapping) or source.get("repository_revision") != UPSTREAM_REVISION:
                raise ContractError("FlexiCT publication profile revision is not the pinned source")
        return candidates

    @classmethod
    def _validate_parameters(cls, parameters: Mapping[str, Any]) -> None:
        allowed = {
            "candidate_id",
            "C",
            "classifier",
            "penalty",
            "solver",
            "class_weight",
            "max_iter",
            "random_state",
            "feature_standardization",
            "encoder_frozen",
            "feature_dimension",
        }
        unknown = set(parameters).difference(allowed)
        if unknown:
            raise ContractError(f"Unsupported FlexiCT probe parameters: {sorted(unknown)}")
        expected_fixed = {
            "classifier": "multinomial_logistic_regression",
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 20000,
            "random_state": 0,
            "feature_standardization": cls._standardization_claim,
            "encoder_frozen": True,
            "feature_dimension": FEATURE_DIM,
        }
        missing = (set(expected_fixed) | {"C"}).difference(parameters)
        if missing:
            raise ContractError(
                f"FlexiCT probe parameters are missing fixed recipe fields: {sorted(missing)}"
            )
        try:
            c_value = float(parameters["C"])
        except (TypeError, ValueError) as exc:
            raise ContractError("FlexiCT probe parameters require a numeric C") from exc
        if not np.isfinite(c_value) or c_value <= 0:
            raise ContractError("FlexiCT logistic-regression C must be finite and positive")
        for name, expected in expected_fixed.items():
            if parameters[name] != expected:
                raise ContractError(
                    f"FlexiCT native probe requires {name}={expected!r}, "
                    f"got {parameters[name]!r}"
                )
        if parameters.get("candidate_id") is not None:
            canonical_parameters = {
                key: value for key, value in parameters.items() if key != "candidate_id"
            }
            expected_identifier = candidate_id(canonical_parameters)
            if str(parameters["candidate_id"]) != expected_identifier:
                raise ContractError(
                    "FlexiCT candidate_id must be the canonical 12-character "
                    f"parameter fingerprint {expected_identifier}; observed "
                    f"{parameters['candidate_id']!r}"
                )

    def dry_run_model_claim(self, args: Any) -> dict:
        return {
            "upstream_repository": "https://github.com/ricklisz/FlexiCT",
            "upstream_revision": args.upstream_revision,
            "checkpoint": str(args.checkpoint)
            if args.checkpoint
            else "FLEXICT_CHECKPOINT or FLEXICT_VLM_CHECKPOINT",
            "checkpoint_sha256": args.checkpoint_sha256 or "resolved and recorded at execution",
            "feature_cache_namespace": args.cache_namespace,
            "feature_precision": args.feature_precision,
            "source_fingerprint": args.source_fingerprint,
            "representation": "vlm_projected_cls_mean_l2_1024",
            "text_tower_loaded": False,
            "checkpoint_deserialization_policy": (
                "weights_only_then_explicit_sha256_trusted_pickle"
                if args.allow_trusted_legacy_checkpoint
                else "weights_only_only"
            ),
        }

    def _feature_store(self, args: Any) -> FeatureStore:
        if getattr(args, "source_fingerprint", "sha256") != "sha256":
            raise ContractError(
                "FlexiCT LinearProbe execution requires SHA-256 source identity; "
                "stat mode is cache diagnostics only"
            )
        if self._store is None:
            self._store = FeatureStore(args)
        return self._store

    def ensure_feature_rows(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._store is None:
            self._store = FeatureStore(args, required_rows=rows)
        else:
            self._store.validate_rows(rows)

    def feature_identity(self, args: Any) -> dict:
        package = Path(__file__).resolve().parent
        shared = package.parent / "_linear_probe"
        return {
            **self._feature_store(args).identity,
            "probe_implementation_identity": {
                "schema_version": 1,
                "files": {
                    "FlexiCT/backend.py": sha256_file(package / "backend.py"),
                    "_linear_probe/common.py": sha256_file(shared / "common.py"),
                    "_linear_probe/metrics.py": sha256_file(shared / "metrics.py"),
                    "_linear_probe/workflow.py": sha256_file(shared / "workflow.py"),
                },
            },
            "probe_runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "scikit_learn_version": sklearn.__version__,
                "joblib_version": joblib.__version__,
                "classifier_random_state": 0,
            },
        }

    def _arrays(self, args: Any, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            raise ContractError("FlexiCT probe received an empty patient partition")
        store = self._feature_store(args)
        features = np.stack([store.load(row).numpy() for row in rows], axis=0)
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if features.shape != (len(rows), FEATURE_DIM):
            raise RuntimeError(f"Unexpected stacked FlexiCT feature shape: {features.shape}")
        if not np.isfinite(features).all():
            raise ContractError("FlexiCT cached feature matrix contains non-finite values")
        if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
            raise ContractError("FlexiCT labels fall outside the canonical five-class range")
        for row, label in zip(rows, labels, strict=True):
            if "class_name" in row and str(row["class_name"]) != CLASS_NAMES[int(label)]:
                raise ContractError(
                    f"FlexiCT class/label mismatch for patient {row['patient_id']}"
                )
        return features, labels

    @classmethod
    def _pipeline(cls, parameters: Mapping[str, Any]) -> Pipeline:
        cls._validate_parameters(parameters)
        return Pipeline(
            [
                ("standard_scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        penalty="l2",
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=20000,
                        random_state=0,
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

    @staticmethod
    def _atomic_joblib(model: Pipeline, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            joblib.dump(model, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _aligned_probabilities(model: Pipeline, features: np.ndarray) -> np.ndarray:
        classifier = model.named_steps["logistic_regression"]
        observed = np.asarray(classifier.classes_, dtype=np.int64)
        expected = np.arange(len(CLASS_NAMES), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                f"Classifier class order {observed.tolist()} does not match {expected.tolist()}"
            )
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
            raise RuntimeError(f"Unexpected FlexiCT probability shape: {probabilities.shape}")
        totals = probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all() or np.any(totals <= 0):
            raise RuntimeError("FlexiCT classifier produced invalid probabilities")
        return probabilities / totals

    @staticmethod
    def _validate_training_labels(labels: np.ndarray) -> None:
        observed = np.unique(np.asarray(labels, dtype=np.int64))
        expected = np.arange(len(CLASS_NAMES), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ContractError(
                "Every FlexiCT training partition must contain all canonical classes; "
                f"observed {observed.tolist()}, expected {expected.tolist()}"
            )

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
        features_train, labels_train = self._arrays(args, train_rows)
        features_validation, labels_validation = self._arrays(args, validation_rows)
        self._validate_training_labels(labels_train)
        model = self._pipeline(parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features_train, labels_train)
        convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        for message in convergence:
            logger.warning("FlexiCT logistic regression convergence warning: %s", message)
        probabilities = self._aligned_probabilities(model, features_validation)
        predicted = probabilities.argmax(axis=1)
        metrics = compute_metrics(labels_validation, predicted, probabilities, CLASS_NAMES)
        json_rows, csv_rows = prediction_documents(validation_rows, probabilities, CLASS_NAMES)
        artifact = output_dir / self.artifact_filename
        self._atomic_joblib(model, artifact)
        atomic_write_json(output_dir / "parameters.json", dict(parameters))
        atomic_write_json(output_dir / "metrics_summary.json", metrics)
        atomic_write_json(output_dir / "validation_predictions.json", json_rows)
        atomic_write_csv(
            output_dir / "validation_probabilities.csv", csv_rows, list(csv_rows[0])
        )
        state = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "method_label": "LinearProbe",
            "strict_linear": False,
            "linear_classifier": True,
            "strict_raw_backbone_probe": False,
            "non_strict_reason": "uses the released frozen 1728-to-1024 VLM projection",
            "encoder_frozen": True,
            "training_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in train_rows]
            ),
            "validation_rows_sha256": fingerprint(
                [(row["patient_id"], row["label"]) for row in validation_rows]
            ),
            "feature_identity": self.feature_identity(args),
            "convergence_warnings": convergence,
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
            "FlexiCT C=%s validation macro-F1=%.4f balanced-accuracy=%.4f",
            parameters["C"],
            metrics["f1_macro"],
            metrics["balanced_accuracy"],
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
        results = {}
        pending_ids = set(context.get("pending_candidate_ids", []))
        for parameters in candidates:
            if pending_ids and parameters["candidate_id"] not in pending_ids:
                continue
            output = (
                grid_root
                / f"candidate_{parameters['candidate_id']}"
                / f"fold_{fold_number:02d}"
            )
            results[parameters["candidate_id"]] = self.fit_one(
                parameters,
                train_rows,
                validation_rows,
                output,
                args,
                {**context, "fold": fold_number, "candidate_id": parameters["candidate_id"]},
                logger,
            )
        return results

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
        features, labels = self._arrays(args, rows)
        self._validate_training_labels(labels)
        model = self._pipeline(parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, labels)
        convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        for message in convergence:
            logger.warning("FlexiCT refit convergence warning: %s", message)
        artifact = output_dir / self.artifact_filename
        self._atomic_joblib(model, artifact)
        atomic_write_json(output_dir / "parameters.json", dict(parameters))
        atomic_write_json(
            output_dir / "training_state.json",
            {
                "schema_version": 1,
                "status": "complete",
                "completed_at": utc_now(),
                "refit_full_development": True,
                "method_label": "LinearProbe",
                "strict_linear": False,
                "linear_classifier": True,
                "strict_raw_backbone_probe": False,
                "non_strict_reason": "uses the released frozen 1728-to-1024 VLM projection",
                "encoder_frozen": True,
                "row_count": len(rows),
                "rows_sha256": fingerprint([(row["patient_id"], row["label"]) for row in rows]),
                "feature_identity": self.feature_identity(args),
                "convergence_warnings": convergence,
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
        logger.info("Refit FlexiCT logistic regression on %s development cases", len(rows))
        return {"artifact": str(artifact)}

    def predict(
        self,
        artifact: Path,
        rows: Sequence[Mapping[str, Any]],
        args: Any,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        model = joblib.load(artifact)
        if not isinstance(model, Pipeline):
            raise RuntimeError(f"Unexpected FlexiCT classifier artifact: {artifact}")
        features, _ = self._arrays(args, rows)
        return self._aligned_probabilities(model, features)

    def selection_summary(
        self, parameters: Mapping[str, Any], fold_metrics: Sequence[Mapping[str, Any]]
    ) -> dict:
        return {
            "selection_value": project_grid_score(fold_metrics),
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
        return (
            -float(summary["selection_value"]),
            -float(summary["mean_perforation_recall"]),
            float(summary["parameters"]["C"]),
            str(summary["candidate_id"]),
        )

    @staticmethod
    def summary_csv_row(summary: Mapping[str, Any]) -> dict:
        return {
            "rank": summary.get("rank"),
            "candidate_id": summary["candidate_id"],
            "C": summary["parameters"]["C"],
            "selection_score": summary["selection_value"],
            "mean_f1_macro": summary["mean_f1_macro"],
            "std_f1_macro": summary["std_f1_macro"],
            "mean_balanced_accuracy": summary["mean_balanced_accuracy"],
            "std_balanced_accuracy": summary["std_balanced_accuracy"],
            "mean_perforation_recall": summary["mean_perforation_recall"],
        }


__all__ = ["FlexiCTBackend"]
