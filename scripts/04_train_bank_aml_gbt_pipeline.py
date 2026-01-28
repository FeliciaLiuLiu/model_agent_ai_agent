"""Train a scikit-learn GradientBoostingClassifier on the bank-like AML dataset and save it.

Defaults:
- Reads from:  ./data/synthetic_bank_aml_200k.csv   (override with DATA_PATH)
- Writes to:   ./models/bank_aml_gbt.joblib         (override with MODEL_PATH)
- Also writes: ./models/bank_aml_gbt_metrics.json
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Train GBT model for bank AML dataset.")
    parser.add_argument('--data', default=os.environ.get('DATA_PATH', './data/synthetic_bank_aml_200k.csv'))
    parser.add_argument('--model', default=os.environ.get('MODEL_PATH', './models/bank_aml_gbt.joblib'))
    parser.add_argument('--label-col', default=os.environ.get('LABEL_COL', 'is_suspicious'))
    parser.add_argument('--test-size', type=float, default=float(os.environ.get('TEST_SIZE', 0.3)))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    parser.add_argument('--test-path', default=os.environ.get('TEST_PATH', None))
    parser.add_argument('--no-save-test', action='store_true')
    args = parser.parse_args()

    data_path = args.data
    model_path = args.model

    _ensure_dir(os.path.dirname(os.path.abspath(model_path)))

    print('Reading dataset from:', os.path.abspath(data_path))
    df = pd.read_csv(data_path)

    label_col = args.label_col
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
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
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

    test_path = None
    if not args.no_save_test:
        if args.test_path:
            test_path = args.test_path
        else:
            data_dir = os.path.dirname(os.path.abspath(data_path))
            stem = os.path.splitext(os.path.basename(data_path))[0]
            test_path = os.path.join(data_dir, f"{stem}_test.csv")
        test_df = X_test.copy()
        test_df[label_col] = y_test.values
        test_df.to_csv(test_path, index=False)
        print('Saved test set to:', test_path)

    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    metrics = {
        'model': 'GradientBoostingClassifier (scikit-learn)',
        'model_path': os.path.abspath(model_path),
        'data_path': os.path.abspath(data_path),
        'test_path': os.path.abspath(test_path) if test_path else None,
        'metric': {'name': 'AUC_ROC', 'value': auc},
        'class_balance': y.value_counts(normalize=True).to_dict(),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote metrics to:', metrics_path)
