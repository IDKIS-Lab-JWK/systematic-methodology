"""Patient-level metrics with schema parity to the current Merlin workflow."""

from __future__ import annotations

import math

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


def _safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_metrics(y_true, y_pred, y_prob, classes: list[str]) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    labels = list(range(len(classes)))
    if y_prob.shape != (len(y_true), len(classes)):
        raise ValueError(
            f"Expected probability shape {(len(y_true), len(classes))}; got {y_prob.shape}."
        )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    specificity = []
    average_precision = []
    for class_index in labels:
        tp = int(matrix[class_index, class_index])
        fn = int(matrix[class_index, :].sum() - tp)
        fp = int(matrix[:, class_index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        specificity.append(tn / (tn + fp) if tn + fp else float("nan"))
        binary_true = (y_true == class_index).astype(np.int32)
        average_precision.append(
            float("nan")
            if np.unique(binary_true).size < 2
            else float(average_precision_score(binary_true, y_prob[:, class_index]))
        )
    valid_ap = [value for value in average_precision if math.isfinite(value)]
    ap_macro = float(np.mean(valid_ap)) if valid_ap else float("nan")
    per_class = {
        class_name: {
            "precision": _safe_float(precision[index]),
            "recall": _safe_float(sensitivity[index]),
            "sensitivity": _safe_float(sensitivity[index]),
            "specificity": _safe_float(specificity[index]),
            "f1": _safe_float(class_f1[index]),
            "average_precision": _safe_float(average_precision[index]),
        }
        for index, class_name in enumerate(classes)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "average_precision_macro": _safe_float(ap_macro),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in sensitivity],
        "per_class_sensitivity": [_safe_float(value) for value in sensitivity],
        "per_class_specificity": [_safe_float(value) for value in specificity],
        "per_class_average_precision": [_safe_float(value) for value in average_precision],
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities.argmax(axis=1)
    rows = []
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
    return 0.7 * float(metrics["f1_macro"]) + 0.3 * float(metrics["balanced_accuracy"])


def grid_selection_score(mean_f1: float, std_f1: float, mean_ba: float, std_ba: float) -> float:
    return 0.7 * (float(mean_f1) - float(std_f1)) + 0.3 * (
        float(mean_ba) - float(std_ba)
    )
