"""Utility functions."""
from typing import Tuple, Union, Any, Optional, List
import numpy as np
import pandas as pd


def with_score_p1(model, X, return_predictions=True):
    """Extract probability scores P(y=1) from model."""
    X_arr = np.asarray(X) if isinstance(X, pd.DataFrame) else X
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_arr)
            p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
            return p1, model.predict(X_arr) if return_predictions else None, True
        except: pass
    if hasattr(model, 'decision_function'):
        try:
            decision = model.decision_function(X_arr)
            p1 = 1 / (1 + np.exp(-np.clip(decision, -500, 500)))
            return p1, model.predict(X_arr) if return_predictions else None, True
        except: pass
    y_pred = model.predict(X_arr)
    return y_pred.astype(float), y_pred, False


def ensure_predictions(model, X, y, threshold=0.5):
    """Get y_true, y_pred, y_score from model."""
    y_true = np.asarray(y).astype(int)
    p1, y_pred, has_proba = with_score_p1(model, X)
    y_pred = (p1 >= threshold).astype(int) if has_proba else y_pred.astype(int)
    return y_true, y_pred, p1


def get_feature_names(X, feature_names=None):
    """Get feature names from data."""
    if feature_names: return list(feature_names)
    if isinstance(X, pd.DataFrame): return list(X.columns)
    return [f'feature_{i}' for i in range(X.shape[1] if X.ndim > 1 else 1)]


def check_model_type(model) -> str:
    """Detect model type."""
    name = type(model).__name__.lower()
    if any(k in name for k in ['tree', 'forest', 'gradient', 'xgb', 'lgb', 'catboost']): return 'tree'
    if any(k in name for k in ['linear', 'logistic', 'ridge', 'lasso']): return 'linear'
    if 'svm' in name or 'svc' in name: return 'svm'
    if any(k in name for k in ['mlp', 'neural', 'keras']): return 'neural'
    if hasattr(model, 'feature_importances_'): return 'tree'
    if hasattr(model, 'coef_'): return 'linear'
    return 'other'


def sample_data(X, y=None, frac=1.0, n=None, random_state=42):
    """Sample data for expensive operations."""
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    sample_size = min(n, n_samples) if n else int(n_samples * frac)
    if sample_size >= n_samples: return X, y
    idx = rng.choice(n_samples, size=sample_size, replace=False)
    X_s = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    y_s = (y.iloc[idx] if isinstance(y, pd.Series) else y[idx]) if y is not None else None
    return X_s, y_s
