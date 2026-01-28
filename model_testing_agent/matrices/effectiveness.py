"""
Model Effectiveness Evaluation

Metrics & Plots:
- ROC Curve & AUC-ROC
- PR Curve & AUC-PR
- Confusion Matrix (raw & normalized)
- Precision, Recall, F1 Score
- KS Statistic & KS Curve
- Precision@K & Recall@K
"""
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from ..core.metrics import confusion_counts, precision_recall_f1, auc_roc, auc_pr
from ..core.utils import ensure_predictions


class ModelEffectiveness:
    """Evaluate model effectiveness with classification metrics."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "./output"
        os.makedirs(self.data_dir, exist_ok=True)

    def evaluate(self, model, X, y, threshold=0.5, k_list=None, **kwargs) -> Tuple[Dict, Dict]:
        """Run full effectiveness evaluation with all plots."""
        k_list = k_list or [10, 50, 100, 200, 500]
        y_true, y_pred, y_score = ensure_predictions(model, X, y, threshold)

        # Metrics
        cm = confusion_counts(y_true, y_pred)
        prec, rec, f1 = precision_recall_f1(cm)
        auc_roc_val = auc_roc(y_true, y_score)
        auc_pr_val = auc_pr(y_true, y_score)
        ks, ks_threshold = self._ks_statistic(y_true, y_score)
        pk, rk = self._precision_recall_at_k(y_true, y_score, k_list)

        # Plots
        plots = {}
        plots['roc_curve'] = self._plot_roc_curve(y_true, y_score, auc_roc_val)
        plots['pr_curve'] = self._plot_pr_curve(y_true, y_score, auc_pr_val)
        plots['confusion_matrix'] = self._plot_confusion_matrix(cm, normalize=False)
        plots['confusion_matrix_norm'] = self._plot_confusion_matrix(cm, normalize=True)
        plots['ks_curve'] = self._plot_ks_curve(y_true, y_score, ks)
        plots['precision_recall_at_k'] = self._plot_precision_recall_at_k(k_list, pk, rk)
        plots['score_distribution'] = self._plot_score_distribution(y_true, y_score)
        plots['threshold_analysis'] = self._plot_threshold_analysis(y_true, y_score)

        metrics = {
            'auc_roc': auc_roc_val, 'auc_pr': auc_pr_val,
            'precision': prec, 'recall': rec, 'f1': f1,
            'ks_statistic': ks, 'ks_threshold': ks_threshold,
            'confusion_matrix': cm,
            'precision_at_k': dict(zip(k_list, pk)),
            'recall_at_k': dict(zip(k_list, rk)),
        }
        explanations = {
            'metrics': {
                'auc_roc': 'Area under the ROC curve; higher means better ranking of positives above negatives.',
                'auc_pr': 'Area under the Precision-Recall curve; higher is better for imbalanced classes.',
                'precision': 'Fraction of predicted positives that are true positives.',
                'recall': 'Fraction of true positives correctly detected.',
                'f1': 'Harmonic mean of precision and recall.',
                'ks_statistic': 'Maximum separation between positive and negative score distributions.',
                'ks_threshold': 'Score threshold at which the KS statistic is maximal.',
                'confusion_matrix': 'Counts of TN/FP/FN/TP at the chosen decision threshold.',
                'precision_at_k': 'Precision when selecting the top-K scored records.',
                'recall_at_k': 'Recall captured within the top-K scored records.',
            },
            'plots': {
                'roc_curve': 'ROC curve shows TPR vs FPR across thresholds; larger area is better.',
                'pr_curve': 'Precision-Recall curve highlights performance on the positive class.',
                'confusion_matrix': 'Raw confusion matrix counts at the chosen threshold.',
                'confusion_matrix_norm': 'Row-normalized confusion matrix showing per-class rates.',
                'ks_curve': 'CDF separation between positive and negative scores; peak is KS.',
                'precision_recall_at_k': 'Precision and recall measured at several K cutoffs.',
                'score_distribution': 'Score histogram by class to visualize separation.',
                'threshold_analysis': 'Precision, recall, and F1 across decision thresholds.',
            },
        }
        return metrics, plots, explanations

    def _ks_statistic(self, y_true, y_score, n_buckets=100):
        """Calculate KS statistic and optimal threshold."""
        buckets = np.linspace(0, 1, n_buckets + 1)
        pos_scores, neg_scores = y_score[y_true == 1], y_score[y_true == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0, 0.5
        cdf_pos = np.array([np.mean(pos_scores <= b) for b in buckets])
        cdf_neg = np.array([np.mean(neg_scores <= b) for b in buckets])
        ks_idx = np.argmax(np.abs(cdf_pos - cdf_neg))
        return float(np.max(np.abs(cdf_pos - cdf_neg))), float(buckets[ks_idx])

    def _precision_recall_at_k(self, y_true, y_score, k_list):
        """Calculate Precision@K and Recall@K."""
        n, total_pos = len(y_true), np.sum(y_true)
        sorted_labels = y_true[np.argsort(y_score)[::-1]]
        pk, rk = [], []
        for k in k_list:
            k = min(k, n)
            tp_at_k = np.sum(sorted_labels[:k])
            pk.append(float(tp_at_k / k) if k > 0 else 0.0)
            rk.append(float(tp_at_k / total_pos) if total_pos > 0 else 0.0)
        return pk, rk

    def _plot_roc_curve(self, y_true, y_score, auc_val):
        """Plot ROC curve."""
        path = os.path.join(self.data_dir, 'roc_curve.png')
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={auc_val:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.2)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve'); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_pr_curve(self, y_true, y_score, auc_val):
        """Plot Precision-Recall curve."""
        path = os.path.join(self.data_dir, 'pr_curve.png')
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(rec, prec, 'b-', lw=2, label=f'PR (AUC={auc_val:.4f})')
        ax.fill_between(rec, prec, alpha=0.2)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_confusion_matrix(self, cm, normalize=False):
        """Plot confusion matrix."""
        suffix = '_norm' if normalize else ''
        path = os.path.join(self.data_dir, f'confusion_matrix{suffix}.png')
        matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1']); ax.set_yticklabels(['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                text = f'{matrix[i,j]:.2%}' if normalize else f'{int(matrix[i,j])}'
                ax.text(j, i, text, ha='center', va='center', fontsize=14)
        ax.set_title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_ks_curve(self, y_true, y_score, ks):
        """Plot KS curve."""
        path = os.path.join(self.data_dir, 'ks_curve.png')
        buckets = np.linspace(0, 1, 101)
        pos_scores, neg_scores = y_score[y_true == 1], y_score[y_true == 0]
        cdf_pos = [np.mean(pos_scores <= b) for b in buckets] if len(pos_scores) > 0 else [0]*101
        cdf_neg = [np.mean(neg_scores <= b) for b in buckets] if len(neg_scores) > 0 else [0]*101
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(buckets, cdf_pos, 'b-', lw=2, label='Positive Class')
        ax.plot(buckets, cdf_neg, 'r-', lw=2, label='Negative Class')
        ax.fill_between(buckets, cdf_pos, cdf_neg, alpha=0.2, color='green')
        ax.set_xlabel('Score Threshold'); ax.set_ylabel('Cumulative Distribution')
        ax.set_title(f'KS Curve (KS={ks:.4f})'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_precision_recall_at_k(self, k_list, pk, rk):
        """Plot Precision@K and Recall@K."""
        path = os.path.join(self.data_dir, 'precision_recall_at_k.png')
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_list))
        width = 0.35
        ax.bar(x - width/2, pk, width, label='Precision@K', color='steelblue')
        ax.bar(x + width/2, rk, width, label='Recall@K', color='coral')
        ax.set_xlabel('K'); ax.set_ylabel('Score'); ax.set_title('Precision@K and Recall@K')
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in k_list])
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_score_distribution(self, y_true, y_score):
        """Plot score distribution by class."""
        path = os.path.join(self.data_dir, 'score_distribution.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, label='Negative', color='blue')
        ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, label='Positive', color='red')
        ax.set_xlabel('Predicted Score'); ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution by Class'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def _plot_threshold_analysis(self, y_true, y_score):
        """Plot precision, recall, F1 vs threshold."""
        path = os.path.join(self.data_dir, 'threshold_analysis.png')
        thresholds = np.linspace(0.1, 0.9, 17)
        precs, recs, f1s = [], [], []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            cm = confusion_counts(y_true, y_pred)
            p, r, f = precision_recall_f1(cm)
            precs.append(p); recs.append(r); f1s.append(f)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, precs, 'b-', lw=2, marker='o', label='Precision')
        ax.plot(thresholds, recs, 'r-', lw=2, marker='s', label='Recall')
        ax.plot(thresholds, f1s, 'g-', lw=2, marker='^', label='F1')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
        ax.set_title('Precision/Recall/F1 vs Threshold'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path
