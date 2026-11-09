"""Five-class metrics shared by the new linear-probe implementations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


def _number(value: Any) -> float | None:
    converted = float(value)
    return converted if np.isfinite(converted) else None


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    classes: Sequence[str],
) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    class_names = list(classes)
    if truth.ndim != 1 or predicted.shape != truth.shape:
        raise ValueError("y_true and y_pred must be matching one-dimensional arrays")
    if probabilities.shape != (len(truth), len(class_names)):
        raise ValueError(
            f"Expected probability shape {(len(truth), len(class_names))}, "
            f"got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    if np.any(probabilities < -1e-7):
        raise ValueError("Probabilities must be non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("Probability rows must sum to one")

    labels = np.arange(len(class_names), dtype=np.int64)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    binary = label_binarize(truth, classes=labels)
    per_class: dict[str, dict] = {}
    ap_values: list[float] = []
    total = int(matrix.sum())
    for index, class_name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        specificity = float(tn / (tn + fp)) if tn + fp else 0.0
        ap = None
        if binary[:, index].min() != binary[:, index].max():
            ap = _number(average_precision_score(binary[:, index], probabilities[:, index]))
            if ap is not None:
                ap_values.append(ap)
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "sensitivity": float(recall[index]),
            "specificity": specificity,
            "f1": float(f1[index]),
            "average_precision": ap,
            "support": int(support[index]),
        }
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "average_precision_macro": float(np.mean(ap_values)) if ap_values else None,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "support": int(len(truth)),
    }


def fold_selection_score(metrics: Mapping[str, Any]) -> float:
    return 0.7 * float(metrics["f1_macro"]) + 0.3 * float(metrics["balanced_accuracy"])


def project_grid_score(fold_metrics: Sequence[Mapping[str, Any]]) -> float:
    f1 = np.asarray([float(item["f1_macro"]) for item in fold_metrics], dtype=np.float64)
    balanced = np.asarray(
        [float(item["balanced_accuracy"]) for item in fold_metrics], dtype=np.float64
    )
    return float(0.7 * (f1.mean() - f1.std()) + 0.3 * (balanced.mean() - balanced.std()))


def mean_macro_ap(fold_metrics: Sequence[Mapping[str, Any]]) -> float:
    values = [
        float(item["average_precision_macro"])
        for item in fold_metrics
        if item.get("average_precision_macro") is not None
    ]
    return float(np.mean(values)) if values else float("-inf")


def perforation_recall(fold_metrics: Sequence[Mapping[str, Any]]) -> float:
    return float(
        np.mean([float(item["per_class"]["Perforation"]["recall"]) for item in fold_metrics])
    )


def prediction_documents(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    predicted = probs.argmax(axis=1)
    json_rows: list[dict] = []
    csv_rows: list[dict] = []
    for row, predicted_index, probability_row in zip(rows, predicted, probs, strict=True):
        item = {
            "patient_id": str(row["patient_id"]),
            "true_index": int(row["label"]),
            "true_class": str(row["class_name"]),
            "predicted_index": int(predicted_index),
            "predicted_class": str(classes[int(predicted_index)]),
            "probabilities": {
                class_name: float(probability_row[index])
                for index, class_name in enumerate(classes)
            },
        }
        json_rows.append(item)
        flat = {key: value for key, value in item.items() if key != "probabilities"}
        flat.update(
            {f"probability_{name}": float(probability_row[index]) for index, name in enumerate(classes)}
        )
        csv_rows.append(flat)
    return json_rows, csv_rows


__all__ = [
    "compute_metrics",
    "fold_selection_score",
    "mean_macro_ap",
    "perforation_recall",
    "prediction_documents",
    "project_grid_score",
]
