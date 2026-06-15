from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


def _specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = matrix.sum()
    scores = []
    for c in range(num_classes):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        scores.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def _safe_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(
            roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))
            )
        )
    except ValueError:
        return float("nan")


def classification_report(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    y_true = targets.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "auc": _safe_auc(y_true, probs, num_classes),
        "sensitivity": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "specificity": _specificity(y_true, preds, num_classes),
        "f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }
