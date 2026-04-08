"""PySpark utilities for model testing."""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

LABEL_CANDIDATES = ["label", "target", "y", "class", "fraud", "is_fraud", "is_suspicious"]


def get_spark(app_name: str = "ModelTestingAgentSpark") -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_model(path: str):
    """Load a scikit-learn compatible model/pipeline from joblib."""
    if JOBLIB_AVAILABLE:
        return joblib.load(path)

    # Fallback to pickle for .pkl/.pickle files if joblib is unavailable
    if path.lower().endswith((".pkl", ".pickle")):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    raise ImportError(
        "joblib is required to load .joblib models. "
        "Install joblib in the environment or convert the model to a .pkl file."
    )


def _guess_label_col(df: DataFrame, label_col: Optional[str] = None) -> Optional[str]:
    """Infer a label column from a Spark DataFrame when one is not provided."""
    if label_col and label_col in df.columns:
        return label_col

    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _feature_cols(df: DataFrame, label_col: Optional[str]) -> List[str]:
    """Return feature columns after excluding the label column when present."""
    return [c for c in df.columns if c != label_col] if label_col else list(df.columns)


def _read_sql_text(sql: str) -> str:
    """Read SQL from a .sql file path or return the query text directly."""
    sql_path = Path(sql)
    if sql_path.exists() and sql_path.is_file() and sql_path.suffix.lower() == ".sql":
        return sql_path.read_text(encoding="utf-8")
    return sql


def _load_sql_dataframe(
    spark: SparkSession,
    sql: str,
    conn: Optional[str] = None,
    jdbc_options: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """Load a Spark DataFrame from Spark SQL or JDBC-backed SQL."""
    sql_text = _read_sql_text(sql)
    if conn:
        reader = spark.read.format("jdbc").option("url", conn).option("query", sql_text)
        for key, value in (jdbc_options or {}).items():
            reader = reader.option(key, value)
        return reader.load()
    return spark.sql(sql_text)


def _load_python_callable(loader_py: str, loader_fn: str):
    """Load a callable from a Python loader file."""
    loader_path = Path(loader_py)
    if not loader_path.exists():
        raise FileNotFoundError(f"Python loader file not found: {loader_py}")

    spec = importlib.util.spec_from_file_location(f"model_testing_spark_loader_{loader_path.stem}", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Python loader file: {loader_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, loader_fn):
        raise AttributeError(f"Loader function '{loader_fn}' not found in {loader_py}")
    return getattr(module, loader_fn)


def _invoke_loader(loader_py: str, loader_fn: str, **kwargs):
    """Invoke a Python loader function, passing supported kwargs only."""
    fn = _load_python_callable(loader_py, loader_fn)
    signature = inspect.signature(fn)
    params = signature.parameters.values()
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    filtered_kwargs = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**filtered_kwargs)


def _ensure_spark_df(df_like: Any, spark: SparkSession) -> DataFrame:
    """Return a Spark DataFrame, converting from pandas when necessary."""
    if isinstance(df_like, DataFrame):
        return df_like
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(df_like, pd.DataFrame):
        return spark.createDataFrame(df_like)

    raise ValueError("Python loader must return a Spark DataFrame or a pandas DataFrame.")


def _normalize_python_payload(
    payload: Any,
    spark: SparkSession,
    label_col: Optional[str] = None,
) -> Tuple[DataFrame, Optional[str], List[str]]:
    """Normalize Python loader outputs into df / label_col / feature_cols."""
    if isinstance(payload, dict):
        if "df" in payload:
            df = _ensure_spark_df(payload["df"], spark)
            inferred = _guess_label_col(df, payload.get("label_col", label_col))
            feature_cols = payload.get("feature_cols") or _feature_cols(df, inferred)
            return df, inferred, list(feature_cols)
    elif isinstance(payload, tuple):
        if len(payload) == 3:
            df_like, inferred_label, feature_cols = payload
            df = _ensure_spark_df(df_like, spark)
            inferred = inferred_label or _guess_label_col(df, label_col)
            return df, inferred, list(feature_cols or _feature_cols(df, inferred))
        if len(payload) == 2:
            df_like, inferred_label = payload
            df = _ensure_spark_df(df_like, spark)
            inferred = inferred_label or _guess_label_col(df, label_col)
            return df, inferred, _feature_cols(df, inferred)
    else:
        df = _ensure_spark_df(payload, spark)
        inferred = _guess_label_col(df, label_col)
        return df, inferred, _feature_cols(df, inferred)

    raise ValueError(
        "Python loader must return a Spark DataFrame, a pandas DataFrame, "
        "(df, label_col), (df, label_col, feature_cols), or a dict with df/label_col/feature_cols."
    )


def load_data(
    path: Optional[str] = None,
    label_col: Optional[str] = None,
    spark: Optional[SparkSession] = None,
    sql: Optional[str] = None,
    conn: Optional[str] = None,
    loader_py: Optional[str] = None,
    loader_fn: str = "load_data",
    jdbc_options: Optional[Dict[str, str]] = None,
) -> Tuple[DataFrame, Optional[str], List[str]]:
    """Load dataset into a Spark DataFrame from a file, SQL query, or Python loader."""
    spark = spark or get_spark()
    provided = [value is not None for value in (path, sql, loader_py)]
    if sum(provided) != 1:
        raise ValueError("Specify exactly one data source: path, sql, or loader_py.")

    if path is not None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = spark.read.option("header", True).option("inferSchema", True).csv(path)
        elif ext == ".parquet":
            df = spark.read.parquet(path)
        else:
            raise ValueError(f"Unsupported format for Spark: {ext}")

        inferred = _guess_label_col(df, label_col)
        return df, inferred, _feature_cols(df, inferred)

    if sql is not None:
        df = _load_sql_dataframe(spark, sql, conn=conn, jdbc_options=jdbc_options)
        inferred = _guess_label_col(df, label_col)
        return df, inferred, _feature_cols(df, inferred)

    payload = _invoke_loader(loader_py, loader_fn, spark=spark, label_col=label_col)
    return _normalize_python_payload(payload, spark, label_col)


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
        arr = [[float(c) if c is not None else 0.0 for c in cols]]
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
