"""Spark-friendly metric helpers."""
from typing import Dict, Tuple


def confusion_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, int]:
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def precision_recall_f1(cm: Dict[str, int]) -> Tuple[float, float, float]:
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def fpr_score(cm: Dict[str, int]) -> float:
    tn, fp = cm["TN"], cm["FP"]
    return float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
