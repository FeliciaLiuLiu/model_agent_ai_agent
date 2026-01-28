"""Train a scikit-learn GradientBoostingClassifier on the bank-like AML dataset and save it.

Defaults:
- Reads from:  ./data/synthetic_bank_aml_200k.csv   (override with DATA_PATH)
- Writes to:   ./models/bank_aml_gbt.joblib         (override with MODEL_PATH)
- Also writes: ./models/bank_aml_gbt_metrics.json
"""

import os
import json
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


if __name__ == '__main__':
    data_path = os.environ.get('DATA_PATH', './data/synthetic_bank_aml_200k.csv')
    model_path = os.environ.get('MODEL_PATH', './models/bank_aml_gbt.joblib')

    _ensure_dir(os.path.dirname(os.path.abspath(model_path)))

    print('Reading dataset from:', os.path.abspath(data_path))
    df = pd.read_csv(data_path)

    label_col = 'is_suspicious'
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    numeric_cols = [
        'txn_amount', 'account_age_days', 'kyc_risk_score', 'num_txn_24h', 'avg_amount_7d',
        'is_pep', 'sanctions_match', 'is_cross_border', 'is_fintech_rail',
    ]
    cat_cols = [
        'currency', 'origin_country', 'destination_country', 'txn_channel', 'payment_rail',
        'txn_type', 'merchant_category', 'device_type', 'ip_country',
    ]

    missing = [c for c in (numeric_cols + cat_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[numeric_cols + cat_cols]
    y = df[label_col].astype(int)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', numeric_cols),
        ],
        remainder='drop',
    )

    model = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model),
    ])

    pipeline.fit(X_train, y_train)

    if y_test.nunique() >= 2:
        y_score = pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_score))
    else:
        auc = 0.5

    joblib.dump(pipeline, model_path)
    print('Saved model to:', model_path)
    print('AUC(ROC)=', auc)

    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    metrics = {
        'model': 'GradientBoostingClassifier (scikit-learn)',
        'model_path': os.path.abspath(model_path),
        'data_path': os.path.abspath(data_path),
        'metric': {'name': 'AUC_ROC', 'value': auc},
        'class_balance': y.value_counts(normalize=True).to_dict(),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote metrics to:', metrics_path)
