"""PySpark utilities for model testing."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def get_spark(app_name: str = "ModelTestingAgentSpark") -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_model(path: str):
    """Load a scikit-learn compatible model/pipeline from joblib."""
    return joblib.load(path)


def load_data(path: str, label_col: Optional[str] = None, spark: Optional[SparkSession] = None) -> Tuple[DataFrame, Optional[str], List[str]]:
    """Load dataset into Spark DataFrame."""
    spark = spark or get_spark()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    elif ext == ".parquet":
        df = spark.read.parquet(path)
    else:
        raise ValueError(f"Unsupported format for Spark: {ext}")

    if label_col is None:
        for name in ["label", "target", "y", "class", "fraud", "is_fraud", "is_suspicious"]:
            if name in df.columns:
                label_col = name
                break

    feature_cols = [c for c in df.columns if c != label_col] if label_col else list(df.columns)
    return df, label_col, feature_cols


def cast_features_to_double(df: DataFrame, feature_cols: List[str]) -> DataFrame:
    """Cast feature columns to double for model scoring."""
    for c in feature_cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df


def add_predictions(
    df: DataFrame,
    model,
    feature_cols: List[str],
    label_col: str = "label",
    threshold: float = 0.5,
    score_col: str = "y_score",
    pred_col: str = "y_pred",
) -> DataFrame:
    """Add prediction score and label columns using a scikit-learn model."""
    spark = df.sparkSession
    bc_model = spark.sparkContext.broadcast(model)

    def _score(*cols):
        arr = np.array(cols, dtype=float).reshape(1, -1)
        m = bc_model.value
        if hasattr(m, "predict_proba"):
            try:
                proba = m.predict_proba(arr)
                return float(proba[0][1])
            except Exception:
                pass
        if hasattr(m, "decision_function"):
            try:
                return float(m.decision_function(arr)[0])
            except Exception:
                pass
        return float(m.predict(arr)[0])

    score_udf = F.udf(_score, T.DoubleType())
    df_cast = cast_features_to_double(df, feature_cols)

    df_scored = df_cast.withColumn(score_col, score_udf(*[F.col(c) for c in feature_cols]))
    df_scored = df_scored.withColumn(pred_col, (F.col(score_col) >= F.lit(threshold)).cast("int"))
    df_scored = df_scored.withColumn(label_col, F.col(label_col).cast("int"))
    return df_scored


def split_reference_current(df: DataFrame, seed: int = 42) -> Tuple[DataFrame, DataFrame]:
    """Split dataset into reference and current halves."""
    ref, curr = df.randomSplit([0.5, 0.5], seed=seed)
    return ref, curr


def get_numeric_columns(df: DataFrame) -> List[str]:
    """Get numeric columns from Spark DataFrame."""
    numeric_types = ("int", "bigint", "double", "float", "smallint", "tinyint", "decimal", "long", "short")
    cols = []
    for field in df.schema.fields:
        if any(t in field.dataType.simpleString() for t in numeric_types):
            cols.append(field.name)
    return cols
