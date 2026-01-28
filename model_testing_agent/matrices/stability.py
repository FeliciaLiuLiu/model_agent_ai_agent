"""
Model Stability Evaluation

Metrics & Plots:
- Population Stability Index (PSI)
- Data Drift Detection (per feature)
- Concept Drift Detection
- Cross-Validation Stability
- Bootstrap Stability & Confidence Intervals
"""
import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from ..core.utils import with_score_p1, get_feature_names


class ModelStability:
    """Evaluate model stability and robustness."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, X, y, X_reference=None, y_reference=None, feature_names=None,
                 n_folds=5, n_bootstrap=100, random_state=42, **kwargs) -> Tuple[Dict, Dict, Dict]:
        """Run full stability evaluation."""
        X_arr, y_arr = np.asarray(X), np.asarray(y)
        feature_names = get_feature_names(X, feature_names)
        y_score, _, _ = with_score_p1(model, X)

        # PSI
        if X_reference is not None:
            y_score_ref, _, _ = with_score_p1(model, X_reference)
        else:
            mid = len(y_score) // 2
            y_score_ref = y_score[:mid]
            y_score = y_score[mid:]
            X_arr = X_arr[mid:]
            y_arr = y_arr[mid:]

        psi = self._calculate_psi(y_score_ref, y_score)

        # Data Drift
        X_ref_arr = np.asarray(X_reference) if X_reference is not None else np.asarray(X)[:len(X)//2]
        drift_results = self._detect_data_drift(X_ref_arr, X_arr, feature_names)

        # Concept Drift
        concept_drift = self._detect_concept_drift(model, X_arr, y_arr, n_splits=5, random_state=random_state)

        # Cross-Validation
        cv_results = self._cross_validate(model, np.asarray(X), np.asarray(y), n_folds, random_state)

        # Bootstrap
        bootstrap_results = self._bootstrap_evaluate(model, X_arr, y_arr, n_bootstrap, random_state)

        # Plots
        plots = {}
        plots['psi_distribution'] = self._plot_psi_distribution(y_score_ref, y_score, psi)
        plots['data_drift_heatmap'] = self._plot_drift_heatmap(drift_results, feature_names)
        plots['concept_drift'] = self._plot_concept_drift(concept_drift)
        plots['cv_results'] = self._plot_cv_results(cv_results)
        plots['bootstrap_distribution'] = self._plot_bootstrap_distribution(bootstrap_results)
        plots['stability_summary'] = self._plot_stability_summary(psi, cv_results, bootstrap_results)

        metrics = {
            'psi': psi,
            'cv_auc_roc_mean': cv_results['auc_roc_mean'], 'cv_auc_roc_std': cv_results['auc_roc_std'],
            'cv_auc_pr_mean': cv_results['auc_pr_mean'], 'cv_auc_pr_std': cv_results['auc_pr_std'],
            'bootstrap_auc_roc_mean': bootstrap_results['auc_roc_mean'],
            'bootstrap_auc_roc_ci_lower': bootstrap_results['ci_lower'],
            'bootstrap_auc_roc_ci_upper': bootstrap_results['ci_upper'],
            'concept_drift_detected': concept_drift['drift_detected'],
            'concept_drift_score': concept_drift['drift_score'],
        }
        artifacts = {'data_drift': drift_results, 'cv_fold_scores': cv_results['fold_scores']}
        return metrics, plots, artifacts

    def _calculate_psi(self, expected, actual, n_bins=10):
        """Calculate Population Stability Index."""
        min_val, max_val = min(expected.min(), actual.min()), max(expected.max(), actual.max())
        bins = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        exp_pct = np.clip(exp_counts / len(expected), 1e-10, 1)
        act_pct = np.clip(act_counts / len(actual), 1e-10, 1)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def _detect_data_drift(self, X_ref, X_curr, feature_names):
        """Detect data drift per feature using KS test."""
        results = {}
        for i, name in enumerate(feature_names[:50]):  # Limit to 50 features
            ref_col = X_ref[:, i] if X_ref.ndim > 1 else X_ref
            curr_col = X_curr[:, i] if X_curr.ndim > 1 else X_curr
            ks_stat, ks_pval = stats.ks_2samp(ref_col, curr_col)
            mean_diff = np.abs(np.mean(curr_col) - np.mean(ref_col))
            std_ratio = np.std(curr_col) / (np.std(ref_col) + 1e-10)
            results[name] = {
                'ks_statistic': float(ks_stat), 'ks_pvalue': float(ks_pval),
                'mean_diff': float(mean_diff), 'std_ratio': float(std_ratio),
                'drift_detected': ks_pval < 0.05
            }
        return results

    def _detect_concept_drift(self, model, X, y, n_splits=5, random_state=42):
        """Detect concept drift by comparing performance across time splits."""
        n = len(y)
        chunk_size = n // n_splits
        auc_scores = []
        for i in range(n_splits):
            start, end = i * chunk_size, (i + 1) * chunk_size if i < n_splits - 1 else n
            X_chunk, y_chunk = X[start:end], y[start:end]
            if len(np.unique(y_chunk)) < 2: continue
            y_score, _, _ = with_score_p1(model, X_chunk)
            auc_scores.append(roc_auc_score(y_chunk, y_score))

        if len(auc_scores) < 2:
            return {'drift_detected': False, 'drift_score': 0.0, 'chunk_aucs': []}

        drift_score = np.std(auc_scores) / (np.mean(auc_scores) + 1e-10)
        return {
            'drift_detected': drift_score > 0.1,
            'drift_score': float(drift_score),
            'chunk_aucs': auc_scores
        }

    def _cross_validate(self, model, X, y, n_folds, random_state):
        """Perform cross-validation."""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        roc_scores, pr_scores = [], []
        for _, val_idx in cv.split(X, y):
            X_val, y_val = X[val_idx], y[val_idx]
            if len(np.unique(y_val)) < 2: continue
            y_score, _, _ = with_score_p1(model, X_val)
            roc_scores.append(roc_auc_score(y_val, y_score))
            pr_scores.append(average_precision_score(y_val, y_score))
        return {
            'auc_roc_mean': float(np.mean(roc_scores)), 'auc_roc_std': float(np.std(roc_scores)),
            'auc_pr_mean': float(np.mean(pr_scores)), 'auc_pr_std': float(np.std(pr_scores)),
            'fold_scores': list(zip(roc_scores, pr_scores))
        }

    def _bootstrap_evaluate(self, model, X, y, n_bootstrap, random_state):
        """Perform bootstrap evaluation."""
        rng = np.random.RandomState(random_state)
        n = len(y)
        auc_scores = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            if len(np.unique(y_boot)) < 2: continue
            y_score, _, _ = with_score_p1(model, X_boot)
            auc_scores.append(roc_auc_score(y_boot, y_score))
        return {
            'auc_roc_mean': float(np.mean(auc_scores)), 'auc_roc_std': float(np.std(auc_scores)),
            'ci_lower': float(np.percentile(auc_scores, 2.5)),
            'ci_upper': float(np.percentile(auc_scores, 97.5)),
            'samples': auc_scores
        }

    def _plot_psi_distribution(self, y_ref, y_curr, psi):
        """Plot PSI distribution comparison."""
        path = os.path.join(self.data_dir, 'psi_distribution.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_ref, bins=30, alpha=0.6, label='Reference', color='blue')
        ax.hist(y_curr, bins=30, alpha=0.6, label='Current', color='orange')
        ax.set_xlabel('Score'); ax.set_ylabel('Frequency')
        ax.set_title(f'Score Distribution (PSI={psi:.4f})'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_drift_heatmap(self, drift_results, feature_names):
        """Plot data drift heatmap."""
        path = os.path.join(self.data_dir, 'data_drift_heatmap.png')
        features = list(drift_results.keys())[:20]  # Top 20
        ks_stats = [drift_results[f]['ks_statistic'] for f in features]
        pvals = [drift_results[f]['ks_pvalue'] for f in features]

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        # KS Statistics
        colors = ['red' if drift_results[f]['drift_detected'] else 'green' for f in features]
        axes[0].barh(range(len(features)), ks_stats, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(features))); axes[0].set_yticklabels(features)
        axes[0].set_xlabel('KS Statistic'); axes[0].set_title('Data Drift (KS Test)')
        axes[0].axvline(x=0.1, color='r', linestyle='--', label='Threshold')

        # P-values
        axes[1].barh(range(len(features)), [-np.log10(p + 1e-10) for p in pvals], color='steelblue', alpha=0.7)
        axes[1].set_yticks(range(len(features))); axes[1].set_yticklabels(features)
        axes[1].set_xlabel('-log10(p-value)'); axes[1].set_title('Statistical Significance')
        axes[1].axvline(x=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')

        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_concept_drift(self, concept_drift):
        """Plot concept drift over time chunks."""
        path = os.path.join(self.data_dir, 'concept_drift.png')
        aucs = concept_drift['chunk_aucs']
        if not aucs:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            plt.savefig(path, dpi=150); plt.close()
            return path

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(aucs) + 1), aucs, 'b-', lw=2, marker='o', markersize=10)
        ax.fill_between(range(1, len(aucs) + 1), aucs, alpha=0.2)
        ax.axhline(y=np.mean(aucs), color='r', linestyle='--', label=f'Mean: {np.mean(aucs):.4f}')
        ax.set_xlabel('Time Chunk'); ax.set_ylabel('AUC-ROC')
        status = 'DETECTED' if concept_drift['drift_detected'] else 'Not Detected'
        ax.set_title(f'Concept Drift Analysis (Drift: {status})')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_cv_results(self, cv_results):
        """Plot cross-validation results."""
        path = os.path.join(self.data_dir, 'cv_results.png')
        fold_scores = cv_results['fold_scores']
        if not fold_scores:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'No CV results')
            plt.savefig(path, dpi=150); plt.close()
            return path

        roc_scores, pr_scores = zip(*fold_scores)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(range(1, len(roc_scores)+1), roc_scores, color='steelblue', alpha=0.7)
        axes[0].axhline(y=cv_results['auc_roc_mean'], color='r', linestyle='--', 
                       label=f"Mean: {cv_results['auc_roc_mean']:.4f} ± {cv_results['auc_roc_std']:.4f}")
        axes[0].set_xlabel('Fold'); axes[0].set_ylabel('AUC-ROC')
        axes[0].set_title('Cross-Validation AUC-ROC'); axes[0].legend()

        axes[1].bar(range(1, len(pr_scores)+1), pr_scores, color='coral', alpha=0.7)
        axes[1].axhline(y=cv_results['auc_pr_mean'], color='r', linestyle='--',
                       label=f"Mean: {cv_results['auc_pr_mean']:.4f} ± {cv_results['auc_pr_std']:.4f}")
        axes[1].set_xlabel('Fold'); axes[1].set_ylabel('AUC-PR')
        axes[1].set_title('Cross-Validation AUC-PR'); axes[1].legend()

        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_bootstrap_distribution(self, bootstrap_results):
        """Plot bootstrap distribution."""
        path = os.path.join(self.data_dir, 'bootstrap_distribution.png')
        samples = bootstrap_results['samples']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(samples, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=bootstrap_results['auc_roc_mean'], color='r', linestyle='-', lw=2, label='Mean')
        ax.axvline(x=bootstrap_results['ci_lower'], color='g', linestyle='--', lw=2, label='95% CI')
        ax.axvline(x=bootstrap_results['ci_upper'], color='g', linestyle='--', lw=2)
        ax.set_xlabel('AUC-ROC'); ax.set_ylabel('Frequency')
        ax.set_title(f"Bootstrap Distribution (95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}])")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_stability_summary(self, psi, cv_results, bootstrap_results):
        """Plot stability summary dashboard."""
        path = os.path.join(self.data_dir, 'stability_summary.png')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # PSI Gauge
        ax = axes[0, 0]
        psi_color = 'green' if psi < 0.1 else ('orange' if psi < 0.25 else 'red')
        ax.barh(['PSI'], [psi], color=psi_color, height=0.5)
        ax.axvline(x=0.1, color='orange', linestyle='--', label='Warning (0.1)')
        ax.axvline(x=0.25, color='red', linestyle='--', label='Critical (0.25)')
        ax.set_xlim(0, max(0.5, psi * 1.2)); ax.set_title(f'PSI: {psi:.4f}'); ax.legend()

        # CV Variance
        ax = axes[0, 1]
        ax.bar(['AUC-ROC', 'AUC-PR'], [cv_results['auc_roc_std'], cv_results['auc_pr_std']], color=['steelblue', 'coral'])
        ax.set_ylabel('Standard Deviation'); ax.set_title('Cross-Validation Variance')

        # Bootstrap CI Width
        ax = axes[1, 0]
        ci_width = bootstrap_results['ci_upper'] - bootstrap_results['ci_lower']
        ax.bar(['95% CI Width'], [ci_width], color='purple', alpha=0.7)
        ax.set_ylabel('CI Width'); ax.set_title(f'Bootstrap Confidence Interval Width: {ci_width:.4f}')

        # Summary Text
        ax = axes[1, 1]
        ax.axis('off')
        summary = f"""
        STABILITY SUMMARY
        ─────────────────
        PSI: {psi:.4f} ({'OK' if psi < 0.1 else 'WARNING' if psi < 0.25 else 'CRITICAL'})

        CV AUC-ROC: {cv_results['auc_roc_mean']:.4f} ± {cv_results['auc_roc_std']:.4f}
        CV AUC-PR:  {cv_results['auc_pr_mean']:.4f} ± {cv_results['auc_pr_std']:.4f}

        Bootstrap 95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]
        """
        ax.text(0.1, 0.5, summary, fontsize=12, family='monospace', va='center')

        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path
