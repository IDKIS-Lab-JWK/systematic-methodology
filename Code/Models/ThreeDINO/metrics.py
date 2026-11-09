"""Five-class patient-level metrics and prediction artifact formatting.

The schema intentionally matches the Pillar-0 and Merlin result readers.  All
metrics are computed from a single-label softmax probability matrix; undefined
one-vs-rest quantities are represented by ``None`` so JSON output remains
standards-compliant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CANONICAL_CLASS_COUNT = 5


def _safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _validated_inputs(y_true, y_pred, y_prob, classes: Sequence[str]):
    classes = [str(value) for value in classes]
    if len(classes) != CANONICAL_CLASS_COUNT:
        raise ValueError(
            f"3DINO evaluation requires exactly {CANONICAL_CLASS_COUNT} classes; "
            f"received {len(classes)}."
        )
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique and in canonical index order.")

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    expected_shape = (truth.size, len(classes))
    if truth.size == 0:
        raise ValueError("Cannot compute metrics for an empty partition.")
    if predicted.shape != truth.shape:
        raise ValueError(
            f"Expected prediction shape {truth.shape}; got {predicted.shape}."
        )
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probability shape {expected_shape}; got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("Softmax probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    if np.any((truth < 0) | (truth >= len(classes))):
        raise ValueError("Ground-truth labels fall outside the canonical class range.")
    if np.any((predicted < 0) | (predicted >= len(classes))):
        raise ValueError("Predicted labels fall outside the canonical class range.")
    return truth, predicted, probabilities, classes


def compute_metrics(y_true, y_pred, y_prob, classes: list[str]) -> dict:
    """Return the canonical five-class softmax metric dictionary."""

    truth, predicted, probabilities, classes = _validated_inputs(
        y_true, y_pred, y_prob, classes
    )
    labels = list(range(len(classes)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    precision = precision_score(
        truth, predicted, labels=labels, average=None, zero_division=0
    )
    sensitivity = recall_score(
        truth, predicted, labels=labels, average=None, zero_division=0
    )
    class_f1 = f1_score(
        truth, predicted, labels=labels, average=None, zero_division=0
    )

    specificity: list[float] = []
    average_precision: list[float] = []
    for class_index in labels:
        true_positive = int(matrix[class_index, class_index])
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        true_negative = int(
            matrix.sum() - true_positive - false_negative - false_positive
        )
        specificity_denominator = true_negative + false_positive
        specificity.append(
            true_negative / specificity_denominator
            if specificity_denominator
            else float("nan")
        )

        binary_truth = (truth == class_index).astype(np.int32)
        average_precision.append(
            float("nan")
            if np.unique(binary_truth).size < 2
            else float(
                average_precision_score(
                    binary_truth, probabilities[:, class_index]
                )
            )
        )

    finite_average_precision = [
        value for value in average_precision if math.isfinite(value)
    ]
    macro_average_precision = (
        float(np.mean(finite_average_precision))
        if finite_average_precision
        else float("nan")
    )
    per_class = {
        class_name: {
            "precision": _safe_float(precision[index]),
            "recall": _safe_float(sensitivity[index]),
            "sensitivity": _safe_float(sensitivity[index]),
            "specificity": _safe_float(specificity[index]),
            "f1": _safe_float(class_f1[index]),
            "average_precision": _safe_float(average_precision[index]),
            "support": int(matrix[index, :].sum()),
        }
        for index, class_name in enumerate(classes)
    }

    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "f1_macro": float(
            f1_score(truth, predicted, labels=labels, average="macro")
        ),
        "average_precision_macro": _safe_float(macro_average_precision),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in sensitivity],
        "per_class_sensitivity": [_safe_float(value) for value in sensitivity],
        "per_class_specificity": [_safe_float(value) for value in specificity],
        "per_class_average_precision": [
            _safe_float(value) for value in average_precision
        ],
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_order": list(classes),
    }


def format_predictions(
    patient_ids: list[str],
    label_strings: list[str],
    probabilities,
    classes: list[str],
) -> tuple[list[dict], pd.DataFrame]:
    """Build JSON rows and the probability CSV used by all model families."""

    classes = [str(value) for value in classes]
    if len(classes) != CANONICAL_CLASS_COUNT or len(set(classes)) != len(classes):
        raise ValueError("Prediction formatting requires five unique class names.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = (len(patient_ids), len(classes))
    if probabilities.shape != expected:
        raise ValueError(
            f"Expected probability shape {expected}; got {probabilities.shape}."
        )
    if len(label_strings) != len(patient_ids):
        raise ValueError("patient_ids and label_strings must have the same length.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    predicted = probabilities.argmax(axis=1)
    rows: list[dict] = []
    for patient_id, true_label, predicted_index, probability in zip(
        patient_ids, label_strings, predicted, probabilities
    ):
        rows.append(
            {
                "patient_id": str(patient_id),
                "true_label": str(true_label),
                "pred_label": classes[int(predicted_index)],
                "probs": {
                    class_name: float(probability[index])
                    for index, class_name in enumerate(classes)
                },
            }
        )

    frame = pd.DataFrame(probabilities, columns=classes)
    frame["true_label"] = [row["true_label"] for row in rows]
    frame["pred_label"] = [row["pred_label"] for row in rows]
    frame["patient_id"] = [row["patient_id"] for row in rows]
    return rows, frame


def fold_selection_score(metrics: dict) -> float:
    """Development-fold score shared with Pillar-0 and Merlin."""

    return 0.7 * float(metrics["f1_macro"]) + 0.3 * float(
        metrics["balanced_accuracy"]
    )


def grid_selection_score(
    mean_f1: float, std_f1: float, mean_ba: float, std_ba: float
) -> float:
    """Variance-penalized score for ranking complete CV candidates."""

    values = [mean_f1, std_f1, mean_ba, std_ba]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Grid-selection inputs must all be finite.")
    if float(std_f1) < 0 or float(std_ba) < 0:
        raise ValueError("Metric standard deviations cannot be negative.")
    return 0.7 * (float(mean_f1) - float(std_f1)) + 0.3 * (
        float(mean_ba) - float(std_ba)
    )
