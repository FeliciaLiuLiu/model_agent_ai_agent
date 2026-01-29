"""Model Interpretability Evaluation (PySpark)."""
import os
import warnings
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score

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
            perm = self._permutation_importance(model, X_sample, y_sample, feature_names, n_repeats, top_k, random_state)
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

    def _permutation_importance(self, model, X, y, feature_names, n_repeats, top_k, random_state):
        def scorer(est, X_in, y_in):
            try:
                y_score = est.predict_proba(X_in)[:, 1]
            except Exception:
                y_score = est.decision_function(X_in)
            return roc_auc_score(y_in, y_score) if len(set(y_in)) >= 2 else 0.5

        result = permutation_importance(
            model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=1
        )
        importances = list(result.importances_mean)
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        ranked = [feature_names[i] for i in sorted_idx]

        path = os.path.join(self.data_dir, "permutation_importance.png")
        fig, ax = plt.subplots(figsize=(10, max(6, min(top_k, len(feature_names)) * 0.4)))
        top_idx = sorted_idx[:top_k]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = [importances[i] for i in top_idx]
        stds = list(result.importances_std)
        top_stds = [stds[i] for i in top_idx]
        y_pos = list(range(len(top_names)))
        ax.barh(y_pos, top_vals, xerr=top_stds, align="center", alpha=0.7, color="steelblue")
        ax.set_yticks(y_pos); ax.set_yticklabels(top_names); ax.invert_yaxis()
        ax.set_xlabel("Importance (decrease in AUC-ROC)"); ax.set_title("Permutation Importance")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

        return {"importances": dict(zip(feature_names, list(importances))),
                "top_features": ranked[:top_k], "plot": path}

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
        features = list(range(min(top_k, len(feature_names))))
        path = os.path.join(self.data_dir, "pdp.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PartialDependenceDisplay.from_estimator(model, X, features, feature_names=feature_names, ax=ax)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return {"plot": path, "features": [feature_names[i] for i in features]}

    def _ice_analysis(self, model, X, feature_names, top_k):
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
