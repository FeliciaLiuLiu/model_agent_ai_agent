"""Metrics calculation utilities."""
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def confusion_counts(y_true, y_pred) -> Dict[str, int]:
    """Calculate confusion matrix counts (TN, FP, FN, TP)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (1, 1):
        return {'TN': int(cm[0,0]), 'FP': 0, 'FN': 0, 'TP': 0} if y_true[0] == 0 else {'TN': 0, 'FP': 0, 'FN': 0, 'TP': int(cm[0,0])}
    tn, fp, fn, tp = cm.ravel()
    return {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def precision_recall_f1(cm: Dict[str, int]) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1 from confusion matrix."""
    tp, fp, fn = cm['TP'], cm['FP'], cm['FN']
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def auc_roc(y_true, y_score) -> float:
    """Calculate AUC-ROC."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)
    return 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))


def auc_pr(y_true, y_score) -> float:
    """Calculate AUC-PR."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)
    return float(np.mean(y_true)) if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, y_score))


def fpr_score(y_true, y_pred) -> Tuple[int, int, float]:
    """Calculate False Positive Rate. Returns (TN, FP, FPR)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tn, fp, float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
