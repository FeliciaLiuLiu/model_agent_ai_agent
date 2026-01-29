"""Model Effectiveness Evaluation (PySpark)."""
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import Bucketizer

from ..core.metrics import precision_recall_f1
from ..core.utils import add_predictions


class ModelEffectivenessSpark:
    """Evaluate model effectiveness with Spark DataFrame operations."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, df: DataFrame, label_col: str, feature_cols: List[str], threshold=0.5, k_list=None, **kwargs) -> Tuple[Dict, Dict, Dict]:
        k_list = k_list or [10, 50, 100, 200, 500]

        df_pred = add_predictions(df, model, feature_cols, label_col=label_col, threshold=threshold)
        cm = self._confusion_counts(df_pred, label_col)
        prec, rec, f1 = precision_recall_f1(cm)

        rdd = df_pred.select("y_score", label_col).rdd.map(lambda r: (float(r[0]), float(r[1])))
        bcm = BinaryClassificationMetrics(rdd)
        auc_roc_val = float(bcm.areaUnderROC)
        auc_pr_val = float(bcm.areaUnderPR)
        roc_points = bcm.roc().collect()
        pr_points = bcm.pr().collect()

        ks_val, ks_threshold = self._ks_statistic(df_pred, label_col)
        pk, rk = self._precision_recall_at_k(df_pred, label_col, k_list)

        plots = {
            "roc_curve": self._plot_roc_curve(roc_points, auc_roc_val),
            "pr_curve": self._plot_pr_curve(pr_points, auc_pr_val),
            "confusion_matrix": self._plot_confusion_matrix(cm, normalize=False),
            "confusion_matrix_norm": self._plot_confusion_matrix(cm, normalize=True),
            "ks_curve": self._plot_ks_curve(df_pred, label_col, ks_val),
            "precision_recall_at_k": self._plot_precision_recall_at_k(k_list, pk, rk),
            "score_distribution": self._plot_score_distribution(df_pred, label_col),
            "threshold_analysis": self._plot_threshold_analysis(df_pred, label_col),
        }

        metrics = {
            "auc_roc": auc_roc_val,
            "auc_pr": auc_pr_val,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "ks_statistic": ks_val,
            "ks_threshold": ks_threshold,
            "confusion_matrix": cm,
            "precision_at_k": dict(zip(k_list, pk)),
            "recall_at_k": dict(zip(k_list, rk)),
        }

        explanations = self._build_explanations(metrics, threshold, k_list)
        return metrics, plots, explanations

    def _confusion_counts(self, df_pred: DataFrame, label_col: str) -> Dict[str, int]:
        agg = df_pred.select(
            F.sum((F.col(label_col) == 1) & (F.col("y_pred") == 1)).alias("tp"),
            F.sum((F.col(label_col) == 0) & (F.col("y_pred") == 1)).alias("fp"),
            F.sum((F.col(label_col) == 0) & (F.col("y_pred") == 0)).alias("tn"),
            F.sum((F.col(label_col) == 1) & (F.col("y_pred") == 0)).alias("fn"),
        ).collect()[0]
        return {"TP": int(agg["tp"] or 0), "FP": int(agg["fp"] or 0), "TN": int(agg["tn"] or 0), "FN": int(agg["fn"] or 0)}

    def _ks_statistic(self, df_pred: DataFrame, label_col: str, n_buckets: int = 100):
        min_max = df_pred.agg(F.min("y_score").alias("min"), F.max("y_score").alias("max")).collect()[0]
        min_val, max_val = float(min_max["min"]), float(min_max["max"])
        if min_val == max_val:
            return 0.0, float(min_val)

        bins = [min_val + (max_val - min_val) * i / n_buckets for i in range(n_buckets + 1)]
        bucketizer = Bucketizer(splits=bins, inputCol="y_score", outputCol="bucket", handleInvalid="keep")
        df_b = bucketizer.transform(df_pred)

        pos = df_b.filter(F.col(label_col) == 1).groupBy("bucket").count().collect()
        neg = df_b.filter(F.col(label_col) == 0).groupBy("bucket").count().collect()
        pos_counts = {int(r["bucket"]): r["count"] for r in pos}
        neg_counts = {int(r["bucket"]): r["count"] for r in neg}

        total_pos = sum(pos_counts.values())
        total_neg = sum(neg_counts.values())
        if total_pos == 0 or total_neg == 0:
            return 0.0, float(min_val)

        cdf_pos, cdf_neg = [], []
        cum_pos, cum_neg = 0.0, 0.0
        for i in range(n_buckets):
            cum_pos += pos_counts.get(i, 0) / total_pos
            cum_neg += neg_counts.get(i, 0) / total_neg
            cdf_pos.append(cum_pos)
            cdf_neg.append(cum_neg)

        diffs = [abs(p - n) for p, n in zip(cdf_pos, cdf_neg)]
        ks_idx = int(np.argmax(diffs)) if diffs else 0
        ks_val = float(max(diffs)) if diffs else 0.0
        return ks_val, float(bins[ks_idx])

    def _precision_recall_at_k(self, df_pred: DataFrame, label_col: str, k_list: List[int]):
        total_pos = df_pred.filter(F.col(label_col) == 1).count()
        w = Window.orderBy(F.col("y_score").desc())
        ranked = df_pred.withColumn("rank", F.row_number().over(w))
        ranked = ranked.withColumn("cum_tp", F.sum(F.col(label_col)).over(w))
        pk, rk = [], []
        for k in k_list:
            tp_at_k = ranked.filter(F.col("rank") <= F.lit(int(k))).agg(F.max("cum_tp").alias("tp")).collect()[0]["tp"]
            tp_at_k = float(tp_at_k or 0)
            pk.append(tp_at_k / k if k > 0 else 0.0)
            rk.append(tp_at_k / total_pos if total_pos > 0 else 0.0)
        return pk, rk

    def _plot_roc_curve(self, roc_points, auc_val):
        path = os.path.join(self.data_dir, "roc_curve.png")
        fpr = [p[0] for p in roc_points]
        tpr = [p[1] for p in roc_points]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.2)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_pr_curve(self, pr_points, auc_val):
        path = os.path.join(self.data_dir, "pr_curve.png")
        recall = [p[0] for p in pr_points]
        precision = [p[1] for p in pr_points]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, "b-", lw=2, label=f"PR (AUC={auc_val:.4f})")
        ax.fill_between(recall, precision, alpha=0.2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_confusion_matrix(self, cm, normalize=False):
        suffix = "_norm" if normalize else ""
        path = os.path.join(self.data_dir, f"confusion_matrix{suffix}.png")
        matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                text = f"{matrix[i, j]:.2%}" if normalize else f"{int(matrix[i, j])}"
                ax.text(j, i, text, ha="center", va="center", fontsize=14)
        ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
        plt.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_ks_curve(self, df_pred: DataFrame, label_col: str, ks):
        path = os.path.join(self.data_dir, "ks_curve.png")
        min_max = df_pred.agg(F.min("y_score").alias("min"), F.max("y_score").alias("max")).collect()[0]
        min_val, max_val = float(min_max["min"]), float(min_max["max"])
        bins = np.linspace(min_val, max_val, 101)
        bucketizer = Bucketizer(splits=bins.tolist(), inputCol="y_score", outputCol="bucket", handleInvalid="keep")
        df_b = bucketizer.transform(df_pred)
        pos = df_b.filter(F.col(label_col) == 1).groupBy("bucket").count().collect()
        neg = df_b.filter(F.col(label_col) == 0).groupBy("bucket").count().collect()
        pos_counts = {int(r["bucket"]): r["count"] for r in pos}
        neg_counts = {int(r["bucket"]): r["count"] for r in neg}
        total_pos = sum(pos_counts.values())
        total_neg = sum(neg_counts.values())

        cdf_pos, cdf_neg = [], []
        cum_pos, cum_neg = 0.0, 0.0
        for i in range(100):
            cum_pos += pos_counts.get(i, 0) / total_pos if total_pos else 0.0
            cum_neg += neg_counts.get(i, 0) / total_neg if total_neg else 0.0
            cdf_pos.append(cum_pos)
            cdf_neg.append(cum_neg)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(bins[:-1], cdf_pos, "b-", lw=2, label="Positive Class")
        ax.plot(bins[:-1], cdf_neg, "r-", lw=2, label="Negative Class")
        ax.fill_between(bins[:-1], cdf_pos, cdf_neg, alpha=0.2, color="green")
        ax.set_xlabel("Score Threshold")
        ax.set_ylabel("Cumulative Distribution")
        ax.set_title(f"KS Curve (KS={ks:.4f})")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_precision_recall_at_k(self, k_list, pk, rk):
        path = os.path.join(self.data_dir, "precision_recall_at_k.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_list))
        width = 0.35
        ax.bar(x - width / 2, pk, width, label="Precision@K", color="steelblue")
        ax.bar(x + width / 2, rk, width, label="Recall@K", color="coral")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_title("Precision@K and Recall@K")
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in k_list])
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_score_distribution(self, df_pred: DataFrame, label_col: str, sample_frac: float = 0.3):
        path = os.path.join(self.data_dir, "score_distribution.png")
        sample = df_pred.sample(False, sample_frac, seed=42).select("y_score", label_col).collect()
        scores_neg = [r[0] for r in sample if r[1] == 0]
        scores_pos = [r[0] for r in sample if r[1] == 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores_neg, bins=50, alpha=0.6, label="Negative", color="blue")
        ax.hist(scores_pos, bins=50, alpha=0.6, label="Positive", color="red")
        ax.set_xlabel("Predicted Score"); ax.set_ylabel("Frequency")
        ax.set_title("Score Distribution by Class"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_threshold_analysis(self, df_pred: DataFrame, label_col: str):
        path = os.path.join(self.data_dir, "threshold_analysis.png")
        thresholds = np.linspace(0.1, 0.9, 17)
        precs, recs, f1s = [], [], []
        for t in thresholds:
            cm = self._confusion_counts_at_threshold(df_pred, label_col, float(t))
            p, r, f = precision_recall_f1(cm)
            precs.append(p); recs.append(r); f1s.append(f)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, precs, "b-", lw=2, marker="o", label="Precision")
        ax.plot(thresholds, recs, "r-", lw=2, marker="s", label="Recall")
        ax.plot(thresholds, f1s, "g-", lw=2, marker="^", label="F1")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title("Precision/Recall/F1 vs Threshold")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _confusion_counts_at_threshold(self, df_pred: DataFrame, label_col: str, threshold: float) -> Dict[str, int]:
        df_t = df_pred.withColumn("pred_t", (F.col("y_score") >= F.lit(threshold)).cast("int"))
        agg = df_t.select(
            F.sum((F.col(label_col) == 1) & (F.col("pred_t") == 1)).alias("tp"),
            F.sum((F.col(label_col) == 0) & (F.col("pred_t") == 1)).alias("fp"),
            F.sum((F.col(label_col) == 0) & (F.col("pred_t") == 0)).alias("tn"),
            F.sum((F.col(label_col) == 1) & (F.col("pred_t") == 0)).alias("fn"),
        ).collect()[0]
        return {"TP": int(agg["tp"] or 0), "FP": int(agg["fp"] or 0), "TN": int(agg["tn"] or 0), "FN": int(agg["fn"] or 0)}

    def _build_explanations(self, metrics, threshold, k_list):
        auc_roc = metrics["auc_roc"]
        auc_pr = metrics["auc_pr"]
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1"]
        ks = metrics["ks_statistic"]
        summary = []
        summary.append(f"AUC-ROC={auc_roc:.4f} and AUC-PR={auc_pr:.4f} indicate ranking performance.")
        summary.append(f"At threshold {threshold:.2f}, precision={prec:.4f}, recall={rec:.4f}, F1={f1:.4f}.")
        summary.append(f"KS statistic={ks:.4f} summarizes class separation in score distributions.")
        if k_list:
            summary.append(f"Precision@K and Recall@K computed for K={k_list}.")
        plots = {
            "roc_curve": "ROC shows TPR vs FPR across thresholds; higher AUC is better.",
            "pr_curve": "PR curve highlights precision-recall tradeoff under class imbalance.",
            "confusion_matrix": "Confusion matrix shows TP/FP/FN/TN counts.",
            "confusion_matrix_norm": "Normalized confusion shows class-wise error rates.",
            "ks_curve": "KS curve shows separation between positive and negative score CDFs.",
            "precision_recall_at_k": "Precision@K/Recall@K show performance at top-K alerts.",
            "score_distribution": "Score distributions show overlap between classes.",
            "threshold_analysis": "Precision/Recall/F1 vs threshold helps choose operating point.",
        }
        return {"summary": summary, "plots": plots}
