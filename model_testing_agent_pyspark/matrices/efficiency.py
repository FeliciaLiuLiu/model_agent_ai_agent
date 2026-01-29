"""Model Efficiency Evaluation (PySpark)."""
import os
from typing import Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..core.metrics import precision_recall_f1, fpr_score
from ..core.utils import add_predictions


class ModelEfficiencySpark:
    """Evaluate model efficiency (FPR analysis) with Spark."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, df: DataFrame, label_col: str, feature_cols: List[str], threshold=0.5, **kwargs) -> Tuple[Dict, Dict, Dict]:
        df_pred = add_predictions(df, model, feature_cols, label_col=label_col, threshold=threshold)
        cm = self._confusion_counts(df_pred, label_col)
        fpr_val = fpr_score(cm)

        thresholds = np.linspace(0.05, 0.95, 19)
        fpr_list, tpr_list, precision_list = [], [], []
        for t in thresholds:
            cm_t = self._confusion_counts_at_threshold(df_pred, label_col, float(t))
            fpr_list.append(fpr_score(cm_t))
            prec, rec, _ = precision_recall_f1(cm_t)
            precision_list.append(prec)
            tpr_list.append(rec)

        plots = {
            "fpr_vs_threshold": self._plot_fpr_vs_threshold(thresholds, fpr_list),
            "efficiency_frontier": self._plot_efficiency_frontier(fpr_list, tpr_list, thresholds),
            "fpr_tpr_tradeoff": self._plot_fpr_tpr_tradeoff(thresholds, fpr_list, tpr_list),
        }

        metrics = {
            "fpr": fpr_val,
            "tn": cm["TN"],
            "fp": cm["FP"],
            "threshold": threshold,
            "fpr_at_thresholds": dict(zip([f"t_{t:.2f}" for t in thresholds], fpr_list)),
        }
        explanations = self._build_explanations(cm, fpr_val, threshold, thresholds, fpr_list, tpr_list)
        return metrics, plots, explanations

    def _confusion_counts(self, df_pred: DataFrame, label_col: str) -> Dict[str, int]:
        agg = df_pred.select(
            F.sum((F.col(label_col) == 1) & (F.col("y_pred") == 1)).alias("tp"),
            F.sum((F.col(label_col) == 0) & (F.col("y_pred") == 1)).alias("fp"),
            F.sum((F.col(label_col) == 0) & (F.col("y_pred") == 0)).alias("tn"),
            F.sum((F.col(label_col) == 1) & (F.col("y_pred") == 0)).alias("fn"),
        ).collect()[0]
        return {"TP": int(agg["tp"] or 0), "FP": int(agg["fp"] or 0), "TN": int(agg["tn"] or 0), "FN": int(agg["fn"] or 0)}

    def _confusion_counts_at_threshold(self, df_pred: DataFrame, label_col: str, threshold: float) -> Dict[str, int]:
        df_t = df_pred.withColumn("pred_t", (F.col("y_score") >= F.lit(threshold)).cast("int"))
        agg = df_t.select(
            F.sum((F.col(label_col) == 1) & (F.col("pred_t") == 1)).alias("tp"),
            F.sum((F.col(label_col) == 0) & (F.col("pred_t") == 1)).alias("fp"),
            F.sum((F.col(label_col) == 0) & (F.col("pred_t") == 0)).alias("tn"),
            F.sum((F.col(label_col) == 1) & (F.col("pred_t") == 0)).alias("fn"),
        ).collect()[0]
        return {"TP": int(agg["tp"] or 0), "FP": int(agg["fp"] or 0), "TN": int(agg["tn"] or 0), "FN": int(agg["fn"] or 0)}

    def _plot_fpr_vs_threshold(self, thresholds, fpr_list):
        path = os.path.join(self.data_dir, "fpr_vs_threshold.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, fpr_list, "b-", lw=2, marker="o")
        ax.fill_between(thresholds, fpr_list, alpha=0.2)
        ax.set_xlabel("Threshold"); ax.set_ylabel("False Positive Rate")
        ax.set_title("FPR vs Classification Threshold"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_efficiency_frontier(self, fpr_list, tpr_list, thresholds):
        path = os.path.join(self.data_dir, "efficiency_frontier.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(fpr_list, tpr_list, c=thresholds, cmap="viridis", s=100)
        ax.plot(fpr_list, tpr_list, "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("Efficiency Frontier"); plt.colorbar(scatter, label="Threshold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_fpr_tpr_tradeoff(self, thresholds, fpr_list, tpr_list):
        path = os.path.join(self.data_dir, "fpr_tpr_tradeoff.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, fpr_list, "r-", lw=2, marker="o", label="FPR")
        ax.plot(thresholds, tpr_list, "g-", lw=2, marker="s", label="TPR (Recall)")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
        ax.set_title("FPR vs TPR Tradeoff"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _build_explanations(self, cm, fpr, threshold, thresholds, fpr_list, tpr_list):
        neg = cm["TN"] + cm["FP"]
        pos = cm["TP"] + cm["FN"]
        tp = cm["TP"]; fp = cm["FP"]; fn = cm["FN"]; tn = cm["TN"]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_desc = "low" if fpr < 0.05 else ("moderate" if fpr < 0.1 else "high")
        idx = int(np.argmin([abs(v - 0.05) for v in fpr_list])) if fpr_list else 0
        best_t = float(thresholds[idx]) if len(thresholds) else threshold
        best_fpr = float(fpr_list[idx]) if fpr_list else fpr
        best_tpr = float(tpr_list[idx]) if tpr_list else tpr

        summary = []
        summary.append(
            f"At threshold {threshold:.2f}, FPR={fpr:.4f} ({fpr_desc}); FP={fp} out of {neg} negatives."
        )
        summary.append(
            f"At the same threshold, TPR={tpr:.4f} with TP={tp} and FN={fn}."
        )
        summary.append(
            f"A threshold near {best_t:.2f} yields FPR≈{best_fpr:.4f} with TPR≈{best_tpr:.4f}."
        )
        plots = {
            "fpr_vs_threshold": "FPR decreases as the threshold increases; use it to pick an operating point.",
            "efficiency_frontier": "Each point shows the FPR/TPR tradeoff; move toward the top-left for better efficiency.",
            "fpr_tpr_tradeoff": "FPR and TPR curves highlight how recall drops as you reduce false positives.",
        }
        return {"summary": summary, "plots": plots}
