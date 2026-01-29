"""Model Interpretability Evaluation (PySpark)."""
import os
import warnings
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.mllib.evaluation import BinaryClassificationMetrics

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False
    lime = None


class ModelInterpretabilitySpark:
    """Evaluate model interpretability with multiple methods (no SHAP)."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(
        self,
        model,
        df,
        label_col: str,
        feature_cols: List[str],
        methods: Optional[List[str]] = None,
        sample_frac: float = 0.3,
        n_repeats: int = 10,
        top_k: int = 20,
        random_state: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        feature_names = list(feature_cols)
        model_type = self._check_model_type(model)
        X_sample, y_sample = self._sample_rows(df, feature_cols, label_col, sample_frac, random_state)

        if methods is None:
            methods = ["permutation", "pdp", "ice"]
            if LIME_AVAILABLE:
                methods.append("lime")

        results = {"metrics": {"model_type": model_type, "methods_used": methods}, "plots": {}, "artifacts": {}}

        if "permutation" in methods:
            perm = self._permutation_importance(df, model, label_col, feature_names, n_repeats, top_k, random_state)
            results["metrics"]["perm_top_features"] = perm["top_features"]
            results["plots"]["permutation_importance"] = perm["plot"]
            results["artifacts"]["perm_importances"] = perm["importances"]

        if "lime" in methods and LIME_AVAILABLE:
            try:
                lime_res = self._lime_analysis(model, X_sample, y_sample, feature_names, random_state)
                results["plots"]["lime_explanation"] = lime_res["plot"]
                results["artifacts"]["lime_weights"] = lime_res["weights"]
                results["metrics"]["lime_instances"] = len(lime_res.get("instances", []))
            except Exception as e:
                results["metrics"]["lime_error"] = str(e)

        if "pdp" in methods:
            try:
                pdp_res = self._pdp_analysis(model, X_sample, feature_names, top_k)
                results["plots"]["pdp"] = pdp_res["plot"]
                results["metrics"]["pdp_features"] = pdp_res.get("features", [])
            except Exception as e:
                results["metrics"]["pdp_error"] = str(e)

        if "ice" in methods:
            try:
                ice_res = self._ice_analysis(model, X_sample, feature_names, top_k)
                results["plots"]["ice"] = ice_res["plot"]
                results["metrics"]["ice_features"] = ice_res.get("features", [])
            except Exception as e:
                results["metrics"]["ice_error"] = str(e)

        results["explanations"] = self._build_explanations(results)
        return results

    def _sample_rows(self, df, feature_cols, label_col, sample_frac, random_state):
        df_sample = df.sample(False, sample_frac, seed=random_state).limit(5000)
        rows = df_sample.select(*(feature_cols + [label_col])).collect()
        X = [[float(r[c]) if r[c] is not None else 0.0 for c in feature_cols] for r in rows]
        y = [int(r[label_col]) if r[label_col] is not None else 0 for r in rows]
        return X, y

    def _check_model_type(self, model) -> str:
        name = type(model).__name__.lower()
        if "forest" in name or "tree" in name or "gb" in name:
            return "tree"
        if "logistic" in name or "linear" in name:
            return "linear"
        return "other"

    def _permutation_importance(self, df, model, label_col, feature_names, n_repeats, top_k, random_state):
        """Spark-based permutation importance without sklearn."""
        baseline_auc = self._auc_roc(self._score_df(df, model, label_col, feature_names), label_col)
        importances = {}

        for col in feature_names:
            imp_vals = []
            for r in range(max(1, n_repeats)):
                df_perm = self._permute_column(df, col, seed=random_state + r)
                auc_perm = self._auc_roc(self._score_df(df_perm, model, label_col, feature_names), label_col)
                imp_vals.append(baseline_auc - auc_perm)
            importances[col] = sum(imp_vals) / len(imp_vals) if imp_vals else 0.0

        ranked = sorted(importances.keys(), key=lambda k: importances[k], reverse=True)
        top_names = ranked[:top_k]
        top_vals = [importances[k] for k in top_names]
        y_pos = list(range(len(top_names)))

        path = os.path.join(self.data_dir, "permutation_importance.png")
        fig, ax = plt.subplots(figsize=(10, max(6, min(top_k, len(feature_names)) * 0.4)))
        ax.barh(y_pos, top_vals, align="center", alpha=0.7, color="steelblue")
        ax.set_yticks(y_pos); ax.set_yticklabels(top_names); ax.invert_yaxis()
        ax.set_xlabel("Importance (decrease in AUC-ROC)"); ax.set_title("Permutation Importance")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

        return {"importances": importances, "top_features": top_names, "plot": path}

    def _score_df(self, df, model, label_col, feature_cols):
        """Score Spark DataFrame using Spark ML model (PipelineModel) or sklearn fallback."""
        if hasattr(model, "transform"):
            df_pred = model.transform(df)
            if "probability" in df_pred.columns:
                df_pred = df_pred.withColumn("y_score", F.col("probability").getItem(1))
            elif "rawPrediction" in df_pred.columns:
                df_pred = df_pred.withColumn("y_score", F.col("rawPrediction").getItem(1))
            else:
                raise ValueError("Spark model output missing probability/rawPrediction columns.")
            return df_pred

        # Fallback to sklearn UDF scoring if a non-Spark model is passed
        from ..core.utils import add_predictions
        return add_predictions(df, model, feature_cols, label_col=label_col)

    def _permute_column(self, df, col, seed: int):
        """Return DataFrame with a single column permuted across rows."""
        w = Window.orderBy(F.monotonically_increasing_id())
        df_idx = df.withColumn("__idx", F.row_number().over(w))
        perm_values = df.select(col).orderBy(F.rand(seed)).withColumn("__idx", F.row_number().over(w))
        df_perm = df_idx.drop(col).join(perm_values, "__idx").drop("__idx")
        return df_perm

    def _auc_roc(self, df_pred, label_col):
        rdd = df_pred.select("y_score", label_col).rdd.map(lambda r: (float(r[0]), float(r[1])))
        return float(BinaryClassificationMetrics(rdd).areaUnderROC)

    def _lime_analysis(self, model, X, y, feature_names, random_state):
        if not LIME_AVAILABLE:
            return {"plot": None, "weights": {}}
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=["0", "1"],
            discretize_continuous=True,
            random_state=random_state,
        )
        idx = 0
        exp = explainer.explain_instance(X[idx], model.predict_proba, num_features=min(10, len(feature_names)))
        fig = exp.as_pyplot_figure()
        path = os.path.join(self.data_dir, "lime_explanation.png")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return {"plot": path, "weights": dict(exp.as_list()), "instances": [int(idx)]}

    def _pdp_analysis(self, model, X, feature_names, top_k):
        try:
            from sklearn.inspection import PartialDependenceDisplay
        except Exception as e:
            raise ImportError("scikit-learn is required for PDP. Install sklearn in the environment.") from e
        features = list(range(min(top_k, len(feature_names))))
        path = os.path.join(self.data_dir, "pdp.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PartialDependenceDisplay.from_estimator(model, X, features, feature_names=feature_names, ax=ax)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return {"plot": path, "features": [feature_names[i] for i in features]}

    def _ice_analysis(self, model, X, feature_names, top_k):
        try:
            from sklearn.inspection import PartialDependenceDisplay
        except Exception as e:
            raise ImportError("scikit-learn is required for ICE. Install sklearn in the environment.") from e
        features = list(range(min(top_k, len(feature_names))))
        path = os.path.join(self.data_dir, "ice.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PartialDependenceDisplay.from_estimator(model, X, features, kind="individual", feature_names=feature_names, ax=ax)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return {"plot": path, "features": [feature_names[i] for i in features]}

    def _build_explanations(self, results):
        metrics = results.get("metrics", {})
        summary = []
        summary.append(f"Model type: {metrics.get('model_type')}; methods used: {metrics.get('methods_used')}")
        if "perm_top_features" in metrics:
            summary.append(f"Top permutation features: {metrics['perm_top_features'][:5]}")
        if "lime_instances" in metrics:
            summary.append(f"Generated LIME explanations for {metrics['lime_instances']} instances.")
        plots = {
            "permutation_importance": "Shows global feature importance by AUC drop.",
            "lime_explanation": "Local explanation for a sample instance.",
            "pdp": "Partial dependence for top features.",
            "ice": "ICE curves show individual-level feature effects.",
        }
        return {"summary": summary, "plots": plots}
