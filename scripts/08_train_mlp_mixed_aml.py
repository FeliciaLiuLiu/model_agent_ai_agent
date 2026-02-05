"""Train a scikit-learn MLP model on the mixed bank + fintech AML dataset.

Defaults:
- Reads from:  ./data/synthetic_aml_mixed_50k_*.csv or .parquet (auto-detected)
              or override with --data / DATA_PATH
- Writes to:   ./models/<model_name>_<timestamp>.joblib
- Also writes: ./models/<model_name>_<timestamp>_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NULL_LIKE = {"", "N/A", "NA", "NULL", "UNKNOWN", "None", "null", "nan"}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_timestamp_from_name(filename: str) -> Optional[datetime]:
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        candidate = f"{parts[i]}_{parts[i + 1]}"
        try:
            return datetime.strptime(candidate, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def _detect_latest_dataset(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("synthetic_aml_mixed_50k_*.parquet")) + list(
        data_dir.glob("synthetic_aml_mixed_50k_*.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No mixed AML dataset found in {data_dir}. Run scripts/07_generate_synthetic_aml_mixed_bank_fintech.py first."
        )

    timestamped: List[Tuple[datetime, Path]] = []
    fallback: List[Tuple[float, Path]] = []
    for path in candidates:
        ts = _parse_timestamp_from_name(path.name)
        if ts:
            timestamped.append((ts, path))
        else:
            fallback.append((path.stat().st_mtime, path))

    if timestamped:
        max_ts = max(ts for ts, _ in timestamped)
        latest = [p for ts, p in timestamped if ts == max_ts]
        parquet = [p for p in latest if p.suffix == ".parquet"]
        return parquet[0] if parquet else latest[0]

    fallback.sort(key=lambda x: x[0], reverse=True)
    top_mtime = fallback[0][0]
    latest = [p for mtime, p in fallback if mtime == top_mtime]
    parquet = [p for p in latest if p.suffix == ".parquet"]
    return parquet[0] if parquet else latest[0]


def _normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(object_cols) == 0:
        return df
    df = df.copy()
    for col in object_cols:
        df[col] = df[col].replace(list(NULL_LIKE), np.nan)
    return df


def _coerce_boolean(df: pd.DataFrame, bool_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    for col in bool_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
                .where(df[col].notna(), None)
            )
        df[col] = df[col].astype("boolean")
    return df


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on mixed bank + fintech AML dataset.")
    parser.add_argument("--data", default=os.environ.get("DATA_PATH", None))
    parser.add_argument("--label-col", default=os.environ.get("LABEL_COL", "sar_actual"))
    parser.add_argument("--time-col", default=os.environ.get("TIME_COL", "txn_ts"))
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "./models"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "aml_mixed_mlp"))
    parser.add_argument("--test-size", type=float, default=float(os.environ.get("TEST_SIZE", "0.3")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-categories", type=int, default=200)
    parser.add_argument("--save-test", action="store_true", help="Save test split to CSV (disabled by default).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    model_dir = _ensure_dir(Path(args.model_dir))

    data_path = Path(args.data) if args.data else _detect_latest_dataset(data_dir)
    print("Reading dataset from:", data_path)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if args.max_rows:
        df = df.head(args.max_rows)

    df = _normalize_nulls(df)

    label_col = args.label_col
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    time_col = args.time_col
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        df["txn_hour"] = ts.dt.hour
        df["txn_dow"] = ts.dt.dayofweek
        df["txn_month"] = ts.dt.month
        df = df.drop(columns=[time_col])

    bool_cols = ["is_international", "is_new_device", "is_high_risk_country", "is_crypto_related"]
    df = _coerce_boolean(df, bool_cols)

    drop_cols = [
        "txn_id",
        "account_id",
        "customer_id",
        "counterparty_id",
        "device_id",
        "ip_address",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    if args.max_categories is not None:
        object_cols = X.select_dtypes(include=["object", "category", "string"]).columns
        high_card = [c for c in object_cols if X[c].nunique(dropna=True) > args.max_categories]
        if high_card:
            print("Dropping high-cardinality columns:", ", ".join(high_card))
            X = X.drop(columns=high_card)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    boolean_cols = [c for c in bool_cols if c in X.columns]

    numeric_cols = [c for c in numeric_cols if c not in boolean_cols]
    categorical_cols = [c for c in categorical_cols if c not in boolean_cols]

    if not numeric_cols and not categorical_cols and not boolean_cols:
        raise ValueError("No features available after preprocessing.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if boolean_cols:
        transformers.append(("bool", boolean_pipeline, boolean_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        random_state=args.seed,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if y_test.nunique() >= 2:
        y_score = pipeline.predict_proba(X_test)[:, 1]
        auc_roc = float(roc_auc_score(y_test, y_score))
    else:
        auc_roc = 0.5

    metrics = {
        "model": "MLPClassifier (scikit-learn)",
        "data_path": str(data_path),
        "label_col": label_col,
        "test_size": args.test_size,
        "metrics": {
            "auc_roc": auc_roc,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        },
        "class_balance": y.value_counts(normalize=True).to_dict(),
        "features": {
            "numeric": numeric_cols,
            "boolean": boolean_cols,
            "categorical": categorical_cols,
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    run_id = metrics["timestamp"]
    model_path = model_dir / f"{args.model_name}_{run_id}.joblib"
    metrics_path = model_dir / f"{args.model_name}_{run_id}_metrics.json"

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", model_path)
    print("AUC(ROC)=", auc_roc)
    print("Wrote metrics to:", metrics_path)

    if args.save_test:
        data_dir = data_path.parent
        stem = data_path.stem
        test_path = data_dir / f"{stem}_test.csv"
        test_df = X_test.copy()
        test_df[label_col] = y_test.values
        test_df.to_csv(test_path, index=False)
        print("Saved test set to:", test_path)
