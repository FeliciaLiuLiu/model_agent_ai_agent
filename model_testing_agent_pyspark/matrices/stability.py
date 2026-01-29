"""Model Stability Evaluation (PySpark)."""
import os
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import Bucketizer

from ..core.utils import add_predictions, split_reference_current, get_numeric_columns


class ModelStabilitySpark:
    """Evaluate model stability and robustness using Spark."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(
        self,
        model,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
        df_reference: Optional[DataFrame] = None,
        n_folds: int = 5,
        n_bootstrap: int = 100,
        random_state: int = 42,
        **kwargs,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        df_pred = add_predictions(df, model, feature_cols, label_col=label_col)

        if df_reference is not None:
            df_ref = add_predictions(df_reference, model, feature_cols, label_col=label_col)
            df_curr = df_pred
        else:
            df_ref, df_curr = split_reference_current(df_pred, seed=random_state)

        psi = self._calculate_psi(df_ref, df_curr)
        drift_results = self._detect_data_drift(df_ref, df_curr, feature_cols)
        concept_drift = self._detect_concept_drift(df_curr, label_col, n_splits=5)
        cv_results = self._cross_validate(df_pred, label_col, n_folds, random_state)
        bootstrap_results = self._bootstrap_evaluate(df_curr, label_col, n_bootstrap, random_state)

        plots = {
            "psi_distribution": self._plot_psi_distribution(df_ref, df_curr),
            "data_drift_heatmap": self._plot_drift_heatmap(drift_results),
            "concept_drift": self._plot_concept_drift(concept_drift),
            "cv_results": self._plot_cv_results(cv_results),
            "bootstrap_distribution": self._plot_bootstrap_distribution(bootstrap_results),
            "stability_summary": self._plot_stability_summary(psi, cv_results, bootstrap_results),
        }

        metrics = {
            "psi": psi,
            "cv_auc_roc_mean": cv_results["auc_roc_mean"],
            "cv_auc_roc_std": cv_results["auc_roc_std"],
            "cv_auc_pr_mean": cv_results["auc_pr_mean"],
            "cv_auc_pr_std": cv_results["auc_pr_std"],
            "bootstrap_auc_roc_mean": bootstrap_results["auc_roc_mean"],
            "bootstrap_auc_roc_ci_lower": bootstrap_results["ci_lower"],
            "bootstrap_auc_roc_ci_upper": bootstrap_results["ci_upper"],
            "concept_drift_detected": concept_drift["drift_detected"],
            "concept_drift_score": concept_drift["drift_score"],
        }
        artifacts = {"data_drift": drift_results, "cv_fold_scores": cv_results["fold_scores"]}
        explanations = self._build_explanations(metrics, artifacts)
        return metrics, plots, artifacts, explanations

    def _calculate_psi(self, df_ref: DataFrame, df_curr: DataFrame, n_bins: int = 10) -> float:
        min_max = df_ref.unionByName(df_curr).agg(F.min("y_score").alias("min"), F.max("y_score").alias("max")).collect()[0]
        min_val, max_val = float(min_max["min"]), float(min_max["max"])
        if min_val == max_val:
            return 0.0
        bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
        bucketizer = Bucketizer(splits=bins, inputCol="y_score", outputCol="bucket", handleInvalid="keep")
        ref_b = bucketizer.transform(df_ref)
        cur_b = bucketizer.transform(df_curr)
        ref_counts = {int(r["bucket"]): r["count"] for r in ref_b.groupBy("bucket").count().collect()}
        cur_counts = {int(r["bucket"]): r["count"] for r in cur_b.groupBy("bucket").count().collect()}
        ref_total = sum(ref_counts.values())
        cur_total = sum(cur_counts.values())
        psi = 0.0
        for i in range(n_bins):
            exp_pct = max(ref_counts.get(i, 0) / ref_total, 1e-10) if ref_total else 1e-10
            act_pct = max(cur_counts.get(i, 0) / cur_total, 1e-10) if cur_total else 1e-10
            psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        return float(psi)

    def _detect_data_drift(self, df_ref: DataFrame, df_curr: DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        results = {}
        numeric_cols = [c for c in feature_cols if c in get_numeric_columns(df_ref)][:50]
        for col in numeric_cols:
            stats_ref = df_ref.select(F.mean(col).alias("mean"), F.stddev(col).alias("std"), F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
            stats_cur = df_curr.select(F.mean(col).alias("mean"), F.stddev(col).alias("std"), F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
            mean_diff = abs((stats_cur["mean"] or 0.0) - (stats_ref["mean"] or 0.0))
            std_ratio = (stats_cur["std"] or 0.0) / ((stats_ref["std"] or 0.0) + 1e-10)

            ks_stat = self._ks_feature(df_ref, df_curr, col)
            results[col] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": 0.0,
                "mean_diff": float(mean_diff),
                "std_ratio": float(std_ratio),
                "drift_detected": ks_stat > 0.1,
            }
        return results

    def _ks_feature(self, df_ref: DataFrame, df_curr: DataFrame, col: str, n_bins: int = 20) -> float:
        min_max = df_ref.select(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
        min_val, max_val = float(min_max["min"] or 0.0), float(min_max["max"] or 0.0)
        min_max2 = df_curr.select(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
        min_val = min(min_val, float(min_max2["min"] or 0.0))
        max_val = max(max_val, float(min_max2["max"] or 0.0))
        if min_val == max_val:
            return 0.0
        bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
        bucketizer = Bucketizer(splits=bins, inputCol=col, outputCol="bucket", handleInvalid="keep")
        ref_b = bucketizer.transform(df_ref)
        cur_b = bucketizer.transform(df_curr)
        ref_counts = {int(r["bucket"]): r["count"] for r in ref_b.groupBy("bucket").count().collect()}
        cur_counts = {int(r["bucket"]): r["count"] for r in cur_b.groupBy("bucket").count().collect()}
        ref_total = sum(ref_counts.values())
        cur_total = sum(cur_counts.values())
        cdf_ref, cdf_cur = [], []
        cum_ref, cum_cur = 0.0, 0.0
        for i in range(n_bins):
            cum_ref += ref_counts.get(i, 0) / ref_total if ref_total else 0.0
            cum_cur += cur_counts.get(i, 0) / cur_total if cur_total else 0.0
            cdf_ref.append(cum_ref)
            cdf_cur.append(cum_cur)
        diffs = [abs(a - b) for a, b in zip(cdf_ref, cdf_cur)]
        return float(max(diffs)) if diffs else 0.0

    def _detect_concept_drift(self, df_curr: DataFrame, label_col: str, n_splits: int = 5):
        n = df_curr.count()
        if n == 0:
            return {"drift_detected": False, "drift_score": 0.0, "chunk_aucs": []}
        w = Window.orderBy(F.monotonically_increasing_id())
        df_idx = df_curr.withColumn("idx", F.row_number().over(w))
        chunk_size = max(1, n // n_splits)
        aucs = []
        for i in range(n_splits):
            start = i * chunk_size + 1
            end = (i + 1) * chunk_size if i < n_splits - 1 else n
            df_chunk = df_idx.filter((F.col("idx") >= start) & (F.col("idx") <= end))
            if df_chunk.select(label_col).distinct().count() < 2:
                continue
            auc = self._auc_roc(df_chunk, label_col)
            aucs.append(auc)
        if len(aucs) < 2:
            return {"drift_detected": False, "drift_score": 0.0, "chunk_aucs": aucs}
        drift_score = float(np.std(aucs) / (np.mean(aucs) + 1e-10))
        return {"drift_detected": drift_score > 0.1, "drift_score": drift_score, "chunk_aucs": aucs}

    def _cross_validate(self, df_pred: DataFrame, label_col: str, n_folds: int, random_state: int):
        df_folds = df_pred.withColumn("fold", (F.rand(seed=random_state) * n_folds).cast("int"))
        roc_scores, pr_scores = [], []
        for f in range(n_folds):
            df_val = df_folds.filter(F.col("fold") == f)
            if df_val.select(label_col).distinct().count() < 2:
                continue
            roc = self._auc_roc(df_val, label_col)
            pr = self._auc_pr(df_val, label_col)
            roc_scores.append(roc)
            pr_scores.append(pr)
        return {
            "auc_roc_mean": float(np.mean(roc_scores)) if roc_scores else 0.0,
            "auc_roc_std": float(np.std(roc_scores)) if roc_scores else 0.0,
            "auc_pr_mean": float(np.mean(pr_scores)) if pr_scores else 0.0,
            "auc_pr_std": float(np.std(pr_scores)) if pr_scores else 0.0,
            "fold_scores": list(zip(roc_scores, pr_scores)),
        }

    def _bootstrap_evaluate(self, df_curr: DataFrame, label_col: str, n_bootstrap: int, random_state: int):
        auc_scores = []
        for i in range(n_bootstrap):
            df_boot = df_curr.sample(withReplacement=True, fraction=1.0, seed=random_state + i)
            if df_boot.select(label_col).distinct().count() < 2:
                continue
            auc_scores.append(self._auc_roc(df_boot, label_col))
        if not auc_scores:
            return {"auc_roc_mean": 0.0, "auc_roc_std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "samples": []}
        return {
            "auc_roc_mean": float(np.mean(auc_scores)),
            "auc_roc_std": float(np.std(auc_scores)),
            "ci_lower": float(np.percentile(auc_scores, 2.5)),
            "ci_upper": float(np.percentile(auc_scores, 97.5)),
            "samples": auc_scores,
        }

    def _auc_roc(self, df_pred: DataFrame, label_col: str) -> float:
        rdd = df_pred.select("y_score", label_col).rdd.map(lambda r: (float(r[0]), float(r[1])))
        return float(BinaryClassificationMetrics(rdd).areaUnderROC)

    def _auc_pr(self, df_pred: DataFrame, label_col: str) -> float:
        rdd = df_pred.select("y_score", label_col).rdd.map(lambda r: (float(r[0]), float(r[1])))
        return float(BinaryClassificationMetrics(rdd).areaUnderPR)

    def _plot_psi_distribution(self, df_ref: DataFrame, df_curr: DataFrame):
        path = os.path.join(self.data_dir, "psi_distribution.png")
        ref_scores = [r[0] for r in df_ref.select("y_score").sample(False, 0.3, seed=42).collect()]
        cur_scores = [r[0] for r in df_curr.select("y_score").sample(False, 0.3, seed=42).collect()]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ref_scores, bins=30, alpha=0.6, label="Reference", color="blue")
        ax.hist(cur_scores, bins=30, alpha=0.6, label="Current", color="orange")
        ax.set_xlabel("Score"); ax.set_ylabel("Frequency")
        ax.set_title("Score Distribution (PSI)"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_drift_heatmap(self, drift_results):
        path = os.path.join(self.data_dir, "data_drift_heatmap.png")
        features = list(drift_results.keys())[:20]
        ks_stats = [drift_results[f]["ks_statistic"] for f in features]
        pvals = [drift_results[f].get("ks_pvalue", 1.0) for f in features]

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        colors = ["red" if drift_results[f]["drift_detected"] else "green" for f in features]
        axes[0].barh(range(len(features)), ks_stats, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(features))); axes[0].set_yticklabels(features)
        axes[0].set_xlabel("KS Statistic"); axes[0].set_title("Data Drift (Approx KS)")
        axes[0].axvline(x=0.1, color="r", linestyle="--", label="Threshold")

        axes[1].barh(range(len(features)), [-np.log10(p + 1e-10) for p in pvals], color="steelblue", alpha=0.7)
        axes[1].set_yticks(range(len(features))); axes[1].set_yticklabels(features)
        axes[1].set_xlabel("-log10(p-value)"); axes[1].set_title("Statistical Significance")
        axes[1].axvline(x=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")

        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_concept_drift(self, concept_drift):
        path = os.path.join(self.data_dir, "concept_drift.png")
        aucs = concept_drift["chunk_aucs"]
        if not aucs:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            plt.savefig(path, dpi=150); plt.close()
            return path
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(aucs) + 1), aucs, "b-", lw=2, marker="o", markersize=10)
        ax.fill_between(range(1, len(aucs) + 1), aucs, alpha=0.2)
        ax.axhline(y=np.mean(aucs), color="r", linestyle="--", label=f"Mean: {np.mean(aucs):.4f}")
        ax.set_xlabel("Time Chunk"); ax.set_ylabel("AUC-ROC")
        ax.set_title("Concept Drift over Time Chunks"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_cv_results(self, cv_results):
        path = os.path.join(self.data_dir, "cv_results.png")
        scores = cv_results["fold_scores"]
        if not scores:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            plt.savefig(path, dpi=150); plt.close(); return path
        roc = [s[0] for s in scores]; pr = [s[1] for s in scores]
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(roc))
        ax.bar(x - 0.2, roc, width=0.4, label="AUC-ROC")
        ax.bar(x + 0.2, pr, width=0.4, label="AUC-PR")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in x])
        ax.set_title("Cross-Validation Results"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_bootstrap_distribution(self, bootstrap_results):
        path = os.path.join(self.data_dir, "bootstrap_distribution.png")
        samples = bootstrap_results.get("samples", [])
        fig, ax = plt.subplots(figsize=(8, 6))
        if samples:
            ax.hist(samples, bins=30, alpha=0.7, color="steelblue")
            ax.axvline(bootstrap_results["ci_lower"], color="red", linestyle="--", label="CI Lower")
            ax.axvline(bootstrap_results["ci_upper"], color="red", linestyle="--", label="CI Upper")
            ax.set_title("Bootstrap AUC-ROC Distribution"); ax.legend()
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_stability_summary(self, psi, cv_results, bootstrap_results):
        path = os.path.join(self.data_dir, "stability_summary.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        cv_std = cv_results.get("auc_roc_std", 0.0)
        ci_width = bootstrap_results.get("ci_upper", 0.0) - bootstrap_results.get("ci_lower", 0.0)
        metrics = {"PSI": psi, "CV AUC Std": cv_std, "Bootstrap CI Width": ci_width}
        ax.bar(metrics.keys(), metrics.values(), color=["steelblue", "orange", "green"], alpha=0.7)
        ax.set_title("Stability Summary"); ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _build_explanations(self, metrics, artifacts):
        summary = []
        summary.append(f"PSI={metrics['psi']:.4f} indicates score distribution shift between reference and current.")
        summary.append(
            f"CV AUC-ROC mean={metrics['cv_auc_roc_mean']:.4f}, std={metrics['cv_auc_roc_std']:.4f} "
            "indicates fold stability."
        )
        summary.append(
            f"Bootstrap AUC-ROC mean={metrics['bootstrap_auc_roc_mean']:.4f} with CI "
            f"[{metrics['bootstrap_auc_roc_ci_lower']:.4f}, {metrics['bootstrap_auc_roc_ci_upper']:.4f}]."
        )
        summary.append(
            f"Concept drift detected={metrics['concept_drift_detected']} (score={metrics['concept_drift_score']:.4f})."
        )
        plots = {
            "psi_distribution": "Reference vs current score distributions used to compute PSI.",
            "data_drift_heatmap": "Approximate KS statistics and significance per feature.",
            "concept_drift": "AUC across time chunks to detect drift.",
            "cv_results": "Fold-wise AUC-ROC and AUC-PR for stability.",
            "bootstrap_distribution": "Bootstrap AUC-ROC distribution and confidence interval.",
            "stability_summary": "Summary of PSI, CV variance, and bootstrap CI width.",
        }
        return {"summary": summary, "plots": plots}
