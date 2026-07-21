# ==========================================================
# utils.py
# Shared helper functions (used mainly across Task 5's models)
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from src.config import LABELS


def build_results(model_name, y_true, y_pred):
    """
    Compute every metric the assignment asks for and package it into
    one dict, so it can be dropped straight into the model comparison
    table without re-deriving anything later.
    """

    per_class_p = precision_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "Precision_negative": per_class_p[0],
        "Recall_negative": per_class_r[0],
        "Precision_neutral": per_class_p[1],
        "Recall_neutral": per_class_r[1],
        "Precision_positive": per_class_p[2],
        "Recall_positive": per_class_r[2],
    }
