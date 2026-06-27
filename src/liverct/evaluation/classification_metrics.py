from __future__ import annotations

from typing import Any

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
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: pd.Series | np.ndarray | list[int],
    y_score: pd.Series | np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Compute binary classification metrics for label 1 as positive class.

    Positive class:
    - 1 = Hepatic_Steatosis
    - 0 = Healthy
    """
    y_true_array = np.asarray(y_true).astype(int)
    y_score_array = np.asarray(y_score).astype(float)
    y_pred_array = (y_score_array >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics: dict[str, float | int] = {
        "n": int(len(y_true_array)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true_array)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
        metrics["average_precision"] = float(
            average_precision_score(y_true_array, y_score_array)
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def evaluate_predictions_by_split(
    pred_df: pd.DataFrame,
    level: str,
    label_col: str = "label",
    score_col: str = "prob_positive",
    split_col: str = "split",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute metrics for each split in a prediction dataframe.
    """
    required_columns = {label_col, score_col, split_col}
    missing = required_columns - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns in pred_df: {sorted(missing)}")

    rows: list[dict[str, Any]] = []

    for split_name, split_df in pred_df.groupby(split_col, sort=True):
        metrics = compute_binary_classification_metrics(
            y_true=split_df[label_col],
            y_score=split_df[score_col],
            threshold=threshold,
        )
        metrics["split"] = split_name
        metrics["level"] = level
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    ordered_columns = [
        "level",
        "split",
        "n",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    return metrics_df[ordered_columns]
