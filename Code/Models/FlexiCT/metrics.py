"""FlexiCT-facing exports of the systematic five-class metric contract."""

from _linear_probe.metrics import (
    compute_metrics,
    fold_selection_score,
    perforation_recall,
    prediction_documents,
    project_grid_score,
)

__all__ = [
    "compute_metrics",
    "fold_selection_score",
    "perforation_recall",
    "prediction_documents",
    "project_grid_score",
]
