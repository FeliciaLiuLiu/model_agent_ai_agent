"""Train a Spark ML pipeline on the latest synthetic AML dataset and save artifacts.

Defaults:
- Reads from:  ./data/synthetic_aml_200k_*.parquet or .csv (auto-detected)
- Writes to:   ./models/<model_name>_<run_id>/model
- Also writes: ./models/<model_name>_<run_id>/metrics.json and train.log
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_timestamp_from_name(filename: str) -> Optional[datetime]:
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        candidate = f"{parts[i]}_{parts[i+1]}"
        try:
            return datetime.strptime(candidate, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def _detect_latest_dataset(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("synthetic_aml_200k_*.parquet")) + list(
        data_dir.glob("synthetic_aml_200k_*.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"No synthetic dataset found in {data_dir}")

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


def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _class_balance(df, label_col: str) -> Dict[str, float]:
    total = df.count()
    rows = df.groupBy(label_col).count().collect()
    return {str(r[label_col]): float(r["count"]) / float(total) for r in rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spark ML pipeline on synthetic AML dataset.")
    parser.add_argument("--data", default=_get_env("DATA_PATH", None))
    parser.add_argument("--label-col", default=_get_env("LABEL_COL", "is_suspicious"))
    parser.add_argument("--time-col", default=_get_env("TIME_COL", "txn_ts"))
    parser.add_argument("--model-dir", default=_get_env("MODEL_DIR", "./models"))
    parser.add_argument("--model-name", default=_get_env("MODEL_NAME", "synthetic_aml_logreg"))
    parser.add_argument("--test-size", type=float, default=float(_get_env("TEST_SIZE", "0.3")))
    parser.add_argument("--seed", type=int, default=int(_get_env("SEED", "42")))
    parser.add_argument("--time-split", action="store_true", default=True)
    parser.add_argument("--no-time-split", action="store_true", help="Disable time-based split.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--model-type", default="logreg", choices=["logreg", "gbt", "rf"])
    parser.add_argument("--save-test", action="store_true", help="Save test split (disabled by default).")
    parser.add_argument("--format", default="parquet", choices=["csv", "parquet"])
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession, functions as F
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
        from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
    except Exception as exc:
        raise RuntimeError("PySpark is required to run this training script.") from exc

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_path = Path(args.data) if args.data else _detect_latest_dataset(data_dir)
    model_dir = Path(args.model_dir)
    _ensure_dir(model_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / f"{args.model_name}_{run_id}"
    _ensure_dir(run_dir)
    model_path = run_dir / "model"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "train.log"
    meta_path = run_dir / "run_meta.pkl"

    spark = SparkSession.builder.getOrCreate()

    print("Reading dataset from:", data_path)
    if data_path.suffix == ".parquet":
        df = spark.read.parquet(str(data_path))
    else:
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(data_path))

    if args.max_rows:
        df = df.limit(args.max_rows)

    label_col = args.label_col
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    time_col = args.time_col
    warnings: List[str] = []

    id_cols = ["txn_id", "account_id", "customer_id", "merchant_id", "counterparty_id"]
    numeric_cols = [
        "txn_amount",
        "account_balance",
        "num_txn_1h",
        "num_txn_24h",
        "sum_amount_24h",
        "days_since_last_txn",
        "velocity_score",
    ]
    boolean_cols = ["is_international", "is_new_beneficiary"]
    categorical_cols = [
        "channel",
        "txn_type",
        "origin_country",
        "dest_country",
        "currency",
        "device_type",
        "risk_segment",
        "merchant_name",
        "payment_memo",
    ]

    dropped_cols = [c for c in id_cols if c in df.columns]

    if time_col in df.columns:
        df = df.withColumn(time_col, F.to_timestamp(F.col(time_col)))
        time_ratio = (
            df.select(F.mean(F.when(F.col(time_col).isNotNull(), 1).otherwise(0))).collect()[0][0]
            or 0.0
        )
        if time_ratio >= 0.5:
            df = df.withColumn("txn_hour", F.hour(F.col(time_col)))
            df = df.withColumn("txn_dow", F.dayofweek(F.col(time_col)))
            df = df.withColumn("txn_month", F.month(F.col(time_col)))
            numeric_cols.extend(["txn_hour", "txn_dow", "txn_month"])
        else:
            warnings.append(f"Time column parse ratio {time_ratio:.2%} too low; skipping time features.")
    else:
        warnings.append("Time column not found; skipping time features.")

    feature_numeric = [c for c in numeric_cols if c in df.columns]
    feature_boolean = [c for c in boolean_cols if c in df.columns]
    feature_categorical = [c for c in categorical_cols if c in df.columns]

    for col in feature_boolean:
        df = df.withColumn(col, F.col(col).cast("int"))

    split_method = "random"
    split_boundary = None
    time_split_enabled = args.time_split and not args.no_time_split
    if time_split_enabled and time_col in df.columns:
        boundary = df.approxQuantile(time_col, [1 - args.test_size], 0.01)
        if boundary and boundary[0] is not None:
            split_boundary = boundary[0]
            train_df = df.filter(F.col(time_col) <= F.lit(split_boundary))
            test_df = df.filter(F.col(time_col) > F.lit(split_boundary))
            split_method = "time"
        else:
            train_df, test_df = df.randomSplit([1 - args.test_size, args.test_size], seed=args.seed)
            warnings.append("Time split boundary not available; used random split.")
    else:
        train_df, test_df = df.randomSplit([1 - args.test_size, args.test_size], seed=args.seed)
        if time_split_enabled:
            warnings.append("Time split requested but time column missing; used random split.")

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in feature_categorical
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in feature_categorical],
        outputCols=[f"{col}_ohe" for col in feature_categorical],
        handleInvalid="keep",
    )

    assembler_inputs = feature_numeric + feature_boolean + [f"{col}_ohe" for col in feature_categorical]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    if args.model_type == "logreg":
        classifier = LogisticRegression(featuresCol="features", labelCol=label_col, probabilityCol="probability")
    elif args.model_type == "gbt":
        classifier = GBTClassifier(featuresCol="features", labelCol=label_col)
    else:
        classifier = RandomForestClassifier(featuresCol="features", labelCol=label_col)

    stages = indexers + [encoder, assembler, classifier]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_roc = float(evaluator.evaluate(predictions))

    metrics = {
        "model_type": args.model_type,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "label_col": label_col,
        "time_col": time_col,
        "feature_columns_used": {
            "numeric": feature_numeric,
            "boolean": feature_boolean,
            "categorical": feature_categorical,
        },
        "dropped_columns": dropped_cols,
        "split_method": split_method,
        "split_boundary": str(split_boundary) if split_boundary else None,
        "test_size": args.test_size,
        "auc_roc": auc_roc,
        "class_balance_full": _class_balance(df, label_col),
        "class_balance_train": _class_balance(train_df, label_col),
        "timestamp": run_id,
        "spark_version": spark.version,
        "warnings": warnings,
    }

    model.write().overwrite().save(str(model_path))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Label column: {label_col}\n")
        f.write(f"Time column: {time_col}\n")
        f.write(f"Split method: {split_method}\n")
        if split_boundary:
            f.write(f"Split boundary: {split_boundary}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Features numeric: {feature_numeric}\n")
        f.write(f"Features boolean: {feature_boolean}\n")
        f.write(f"Features categorical: {feature_categorical}\n")
        if warnings:
            f.write("Warnings:\n")
            for w in warnings:
                f.write(f"- {w}\n")

    with open(meta_path, "wb") as f:
        pickle.dump(metrics, f)

    if args.save_test:
        test_path = run_dir / f"test.{args.format}"
        if args.format == "parquet":
            test_df.write.mode("overwrite").parquet(str(test_path))
        else:
            test_df.write.mode("overwrite").option("header", True).csv(str(test_path))
        print("Saved test set to:", test_path)

    print("Saved model to:", model_path)
    print("AUC(ROC)=", auc_roc)
    print("Metrics written to:", metrics_path)
    print("Log written to:", log_path)
