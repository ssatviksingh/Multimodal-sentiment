"""
eval_metrics_extended.py

Wrapper utilities around scikit-learn metrics for convenience.
"""

from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def compute_basic_metrics(
    y_true,
    y_pred,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1 for a set of predictions.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
    }

