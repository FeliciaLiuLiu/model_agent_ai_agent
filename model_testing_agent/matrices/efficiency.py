"""
Model Efficiency Evaluation

Metrics & Plots:
- False Positive Rate (FPR)
- FPR vs Threshold
- True Negatives & False Positives
- Efficiency Frontier
"""
import os
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..core.metrics import fpr_score
from ..core.utils import ensure_predictions


class ModelEfficiency:
    """Evaluate model efficiency (FPR analysis)."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, X, y, threshold=0.5, **kwargs) -> Tuple[Dict, Dict]:
        """Run efficiency evaluation."""
        y_true, y_pred, y_score = ensure_predictions(model, X, y, threshold)
        tn, fp, fpr_val = fpr_score(y_true, y_pred)

        # FPR at different thresholds
        thresholds = np.linspace(0.05, 0.95, 19)
        fpr_list, tpr_list, precision_list = [], [], []
        for t in thresholds:
            y_p = (y_score >= t).astype(int)
            tn_t, fp_t, fpr_t = fpr_score(y_true, y_p)
            tp_t = np.sum((y_true == 1) & (y_p == 1))
            fn_t = np.sum((y_true == 1) & (y_p == 0))
            tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            fpr_list.append(fpr_t); tpr_list.append(tpr_t); precision_list.append(prec_t)

        plots = {}
        plots['fpr_vs_threshold'] = self._plot_fpr_vs_threshold(thresholds, fpr_list)
        plots['efficiency_frontier'] = self._plot_efficiency_frontier(fpr_list, tpr_list, thresholds)
        plots['fpr_tpr_tradeoff'] = self._plot_fpr_tpr_tradeoff(thresholds, fpr_list, tpr_list)

        metrics = {
            'fpr': fpr_val, 'tn': tn, 'fp': fp, 'threshold': threshold,
            'fpr_at_thresholds': dict(zip([f't_{t:.2f}' for t in thresholds], fpr_list)),
        }
        explanations = {
            'metrics': {
                'fpr': 'False Positive Rate at the chosen threshold.',
                'tn': 'Count of true negatives at the chosen threshold.',
                'fp': 'Count of false positives at the chosen threshold.',
                'threshold': 'Decision threshold used for current metrics.',
                'fpr_at_thresholds': 'FPR values computed across a range of thresholds.',
            },
            'plots': {
                'fpr_vs_threshold': 'How FPR changes as the decision threshold moves.',
                'efficiency_frontier': 'Tradeoff between FPR and TPR across thresholds.',
                'fpr_tpr_tradeoff': 'FPR and TPR curves across thresholds for operating-point selection.',
            },
        }
        return metrics, plots, explanations

    def _plot_fpr_vs_threshold(self, thresholds, fpr_list):
        """Plot FPR vs threshold."""
        path = os.path.join(self.data_dir, 'fpr_vs_threshold.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, fpr_list, 'b-', lw=2, marker='o')
        ax.fill_between(thresholds, fpr_list, alpha=0.2)
        ax.set_xlabel('Threshold'); ax.set_ylabel('False Positive Rate')
        ax.set_title('FPR vs Classification Threshold'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_efficiency_frontier(self, fpr_list, tpr_list, thresholds):
        """Plot efficiency frontier (FPR vs TPR)."""
        path = os.path.join(self.data_dir, 'efficiency_frontier.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(fpr_list, tpr_list, c=thresholds, cmap='viridis', s=100)
        ax.plot(fpr_list, tpr_list, 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Efficiency Frontier'); plt.colorbar(scatter, label='Threshold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_fpr_tpr_tradeoff(self, thresholds, fpr_list, tpr_list):
        """Plot FPR and TPR vs threshold."""
        path = os.path.join(self.data_dir, 'fpr_tpr_tradeoff.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, fpr_list, 'r-', lw=2, marker='o', label='FPR')
        ax.plot(thresholds, tpr_list, 'g-', lw=2, marker='s', label='TPR (Recall)')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Rate')
        ax.set_title('FPR vs TPR Tradeoff'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path
