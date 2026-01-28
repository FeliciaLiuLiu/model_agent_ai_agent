"""
Model Interpretability Evaluation

Methods & Plots:
- Permutation Importance
- SHAP Values & Summary Plot
- LIME Explanations
- Partial Dependence Plot (PDP)
- Individual Conditional Expectation (ICE)
"""
import os
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
from ..core.utils import with_score_p1, check_model_type, get_feature_names, sample_data, unwrap_estimator

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as e:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(e)
    shap = None

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None


class ModelInterpretability:
    """Evaluate model interpretability with multiple methods."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, X, y, feature_names=None, methods=None, sample_frac=0.3,
                 n_repeats=10, top_k=20, random_state=42, **kwargs) -> Dict[str, Any]:
        """Run interpretability evaluation."""
        feature_names = get_feature_names(X, feature_names)
        model_type = check_model_type(model)
        X_sample, y_sample = sample_data(X, y, frac=sample_frac, random_state=random_state)

        if methods is None:
            methods = ['permutation', 'pdp', 'ice']
            if SHAP_AVAILABLE: methods.append('shap')
            if LIME_AVAILABLE: methods.append('lime')

        results = {'metrics': {'model_type': model_type, 'methods_used': methods}, 'plots': {}, 'artifacts': {}}
        if not SHAP_AVAILABLE:
            results['metrics']['shap_status'] = 'unavailable'
            results['metrics']['shap_reason'] = SHAP_IMPORT_ERROR or 'SHAP import failed'

        # Permutation Importance
        if 'permutation' in methods:
            perm = self._permutation_importance(model, X_sample, y_sample, feature_names, n_repeats, top_k, random_state)
            results['metrics']['perm_top_features'] = perm['top_features']
            results['plots']['permutation_importance'] = perm['plot']
            results['artifacts']['perm_importances'] = perm['importances']

        # SHAP
        if 'shap' in methods and SHAP_AVAILABLE:
            try:
                shap_res = self._shap_analysis(model, X_sample, feature_names, model_type, top_k, random_state)
                results['metrics']['shap_top_features'] = shap_res['top_features']
                results['plots'].update(shap_res['plots'])
            except Exception as e:
                results['metrics']['shap_error'] = str(e)

        # LIME
        if 'lime' in methods and LIME_AVAILABLE:
            try:
                lime_res = self._lime_analysis(model, X_sample, y_sample, feature_names, random_state)
                results['plots']['lime_explanation'] = lime_res['plot']
                results['artifacts']['lime_weights'] = lime_res['weights']
                results['metrics']['lime_instances'] = len(lime_res.get('instances', []))
            except Exception as e:
                results['metrics']['lime_error'] = str(e)

        # PDP
        if 'pdp' in methods:
            try:
                pdp_res = self._pdp_analysis(model, X_sample, feature_names, top_k)
                results['plots']['pdp'] = pdp_res['plot']
                results['metrics']['pdp_features'] = pdp_res.get('features', [])
            except Exception as e:
                results['metrics']['pdp_error'] = str(e)

        # ICE
        if 'ice' in methods:
            try:
                ice_res = self._ice_analysis(model, X_sample, feature_names, top_k)
                results['plots']['ice'] = ice_res['plot']
                results['metrics']['ice_features'] = ice_res.get('features', [])
            except Exception as e:
                results['metrics']['ice_error'] = str(e)

        results['explanations'] = self._build_explanations(results)
        return results

    def _permutation_importance(self, model, X, y, feature_names, n_repeats, top_k, random_state):
        """Calculate permutation importance."""
        X_arr = X if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y)

        def scorer(est, X, y):
            y_score, _, _ = with_score_p1(est, X)
            return roc_auc_score(y, y_score) if len(np.unique(y)) >= 2 else 0.5

        # Use single process to avoid OS semaphore limits in restricted environments.
        result = permutation_importance(
            model, X_arr, y_arr, scoring=scorer, n_repeats=n_repeats,
            random_state=random_state, n_jobs=1
        )
        importances = result.importances_mean
        sorted_idx = np.argsort(importances)[::-1]
        ranked = [feature_names[i] for i in sorted_idx]

        # Plot
        path = os.path.join(self.data_dir, 'permutation_importance.png')
        fig, ax = plt.subplots(figsize=(10, max(6, min(top_k, len(feature_names)) * 0.4)))
        top_idx = sorted_idx[:top_k]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = importances[top_idx]
        top_stds = result.importances_std[top_idx]

        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals, xerr=top_stds, align='center', alpha=0.7, color='steelblue')
        ax.set_yticks(y_pos); ax.set_yticklabels(top_names); ax.invert_yaxis()
        ax.set_xlabel('Importance (decrease in AUC-ROC)'); ax.set_title('Permutation Importance')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

        return {'importances': dict(zip(feature_names, importances.tolist())),
                'top_features': ranked[:top_k], 'plot': path}

    def _shap_analysis(self, model, X, feature_names, model_type, top_k, random_state):
        """SHAP analysis with multiple plot types."""
        X_arr = np.asarray(X)
        if len(X_arr) > 500:
            rng = np.random.RandomState(random_state)
            X_arr = X_arr[rng.choice(len(X_arr), 500, replace=False)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = unwrap_estimator(model)
            if model_type == 'tree':
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_arr)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(base_model, X_arr)
                shap_values = explainer.shap_values(X_arr)
            else:
                background = shap.sample(X_arr, min(100, len(X_arr)))
                explainer = shap.KernelExplainer(lambda x: with_score_p1(model, x)[0], background)
                shap_values = explainer.shap_values(X_arr[:min(100, len(X_arr))])

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:top_k]]

        plots = {}

        # Summary bar plot
        path1 = os.path.join(self.data_dir, 'shap_summary_bar.png')
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
        top_idx = sorted_idx[:top_k]
        ax.barh(range(len(top_idx)), mean_abs[top_idx], color='coral', alpha=0.7)
        ax.set_yticks(range(len(top_idx))); ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.invert_yaxis(); ax.set_xlabel('Mean |SHAP Value|'); ax.set_title('SHAP Feature Importance')
        plt.tight_layout(); plt.savefig(path1, dpi=150, bbox_inches='tight'); plt.close()
        plots['shap_bar'] = path1

        # Summary beeswarm plot
        path2 = os.path.join(self.data_dir, 'shap_summary_beeswarm.png')
        fig = plt.figure(figsize=(10, max(6, top_k * 0.4)))
        shap.summary_plot(shap_values, X_arr, feature_names=feature_names, max_display=top_k, show=False)
        plt.tight_layout(); plt.savefig(path2, dpi=150, bbox_inches='tight'); plt.close()
        plots['shap_beeswarm'] = path2

        return {'top_features': top_features, 'plots': plots}

    def _lime_analysis(self, model, X, y, feature_names, random_state):
        """LIME analysis for sample instances."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_arr, feature_names=feature_names, class_names=['Negative', 'Positive'],
            mode='classification', random_state=random_state
        )

        # Explain a few positive instances
        pos_idx = np.where(y_arr == 1)[0]
        if len(pos_idx) == 0:
            pos_idx = np.arange(min(3, len(X_arr)))
        sample_idx = pos_idx[:3]

        all_weights = {}
        fig, axes = plt.subplots(1, len(sample_idx), figsize=(6*len(sample_idx), 6))
        if len(sample_idx) == 1:
            axes = [axes]

        for i, idx in enumerate(sample_idx):
            exp = explainer.explain_instance(X_arr[idx], model.predict_proba, num_features=10)
            weights = dict(exp.as_list())
            all_weights[f'instance_{idx}'] = weights

            # Plot
            features = list(weights.keys())[:10]
            values = list(weights.values())[:10]
            colors = ['green' if v > 0 else 'red' for v in values]
            axes[i].barh(range(len(features)), values, color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(features))); axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Contribution'); axes[i].set_title(f'LIME: Instance {idx}')
            axes[i].axvline(x=0, color='black', linestyle='-', lw=0.5)

        path = os.path.join(self.data_dir, 'lime_explanation.png')
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

        return {'weights': all_weights, 'plot': path, 'instances': list(sample_idx)}

    def _pdp_analysis(self, model, X, feature_names, top_k):
        """Partial Dependence Plot analysis."""
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=feature_names)

        # Select top features based on variance
        X_num = X_df.select_dtypes(include=[np.number]).astype(float)
        if X_num.empty:
            raise ValueError("PDP requires numeric features.")
        # Ensure numeric dtype for PDP/ICE compatibility.
        X_df[X_num.columns] = X_num
        variances = X_num.var()
        top_features = variances.nlargest(min(top_k, 6)).index.tolist()

        path = os.path.join(self.data_dir, 'pdp.png')
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(top_features):
            row, col = i // n_cols, i % n_cols
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_df, [feature], ax=axes[row, col], line_kw={'color': 'steelblue'}
                )
                axes[row, col].set_title(f'PDP: {feature}')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Partial Dependence Plots', fontsize=14)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

        return {'plot': path, 'features': top_features}

    def _ice_analysis(self, model, X, feature_names, top_k):
        """Individual Conditional Expectation plot."""
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=feature_names)

        # Select top features
        X_num = X_df.select_dtypes(include=[np.number]).astype(float)
        if X_num.empty:
            raise ValueError("ICE requires numeric features.")
        X_df[X_num.columns] = X_num
        variances = X_num.var()
        top_features = variances.nlargest(min(top_k, 4)).index.tolist()

        path = os.path.join(self.data_dir, 'ice.png')
        n_features = len(top_features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(top_features):
            row, col = i // n_cols, i % n_cols
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_df, [feature], ax=axes[row, col], kind='both',
                    subsample=50, random_state=42,
                    ice_lines_kw={'color': 'steelblue', 'alpha': 0.1},
                    pd_line_kw={'color': 'red', 'linewidth': 2}
                )
                axes[row, col].set_title(f'ICE: {feature}')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Individual Conditional Expectation (ICE) Plots', fontsize=14)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

        return {'plot': path, 'features': top_features}

    def _build_explanations(self, results):
        """Build result-specific explanations for interpretability outputs."""
        metrics = results.get('metrics', {})
        artifacts = results.get('artifacts', {})

        summary = []
        perm_feats = metrics.get('perm_top_features', [])
        if perm_feats:
            summary.append(f"Permutation importance highlights: {perm_feats[:5]}.")

        if metrics.get('shap_top_features'):
            summary.append(f"SHAP top features: {metrics['shap_top_features'][:5]}.")
        elif metrics.get('shap_error'):
            summary.append(f"SHAP plots not generated due to error: {metrics['shap_error']}.")
        elif metrics.get('shap_status') == 'unavailable':
            reason = metrics.get('shap_reason', 'SHAP not installed')
            summary.append(f"SHAP plots not generated because SHAP is unavailable ({reason}).")

        if metrics.get('lime_instances'):
            lime_weights = artifacts.get('lime_weights', {})
            if lime_weights:
                first_key = list(lime_weights.keys())[0]
                weights = lime_weights[first_key]
                ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                top_feats = [k for k, _ in ranked]
                summary.append(
                    f"LIME generated local explanations for {metrics['lime_instances']} instances; "
                    f"top contributors for {first_key}: {top_feats}."
                )
        elif metrics.get('lime_error'):
            summary.append(f"LIME plots not generated due to error: {metrics['lime_error']}.")

        if metrics.get('pdp_features'):
            summary.append(f"PDP plots show average effects for: {metrics['pdp_features']}.")
        elif metrics.get('pdp_error'):
            summary.append(f"PDP plots not generated due to error: {metrics['pdp_error']}.")

        if metrics.get('ice_features'):
            summary.append(f"ICE plots show individual-level effects for: {metrics['ice_features']}.")
        elif metrics.get('ice_error'):
            summary.append(f"ICE plots not generated due to error: {metrics['ice_error']}.")

        plots = {
            'permutation_importance': "Bars show how much AUC drops when each feature is permuted.",
            'shap_bar': "SHAP bar plot ranks global feature impact by mean absolute SHAP values.",
            'shap_beeswarm': "SHAP beeswarm shows both impact size and direction per feature.",
            'lime_explanation': "LIME bars show local feature contributions for selected instances.",
            'pdp': "PDP curves show average model response as a feature varies.",
            'ice': "ICE curves show per-instance responses as a feature varies.",
        }
        return {'summary': summary, 'plots': plots}
