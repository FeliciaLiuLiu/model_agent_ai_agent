"""Spark-first utility helpers for EDA."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SUPPORTED_EXTS = [".csv"]
TIMESTAMP_RE = re.compile(r"(?:^|_)(\d{8}_\d{6})(?:_|\\.|$)")

DEFAULT_NULL_LIKE_VALUES = [
    "",
    "NA",
    "N/A",
    "NULL",
    "UNKNOWN",
]

TARGET_NAME_HINTS = [
    "sar_actual",
    "is_suspicious",
    "suspicious",
    "sar",
    "fraud",
    "aml",
    "is_fraud",
    "fraud_flag",
    "alert",
    "flag",
    "target",
    "label",
    "class",
    "y",
]

DEFAULT_DATASET_PREFIX = "synthetic_aml_mixed_50k_"


def _resolve_data_dir(data_dir: str) -> Path:
    """Resolve data directory relative to cwd or package root."""
    p = Path(data_dir)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    pkg_root = Path(__file__).resolve().parents[1]
    candidate = pkg_root / data_dir
    return candidate if candidate.exists() else p


def parse_timestamp_from_filename(name: str) -> Optional[datetime]:
    """Parse YYYYMMDD_HHMMSS timestamp from a filename."""
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def detect_latest_dataset(
    data_dir: str = "./data",
    allowed_ext: Optional[List[str]] = None,
    env_var: str = "EDA_SPARK_DATA_PATH",
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> str:
    """Detect the latest dataset from the mixed AML generator (07_XXX.py)."""
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path

    allowed_ext = allowed_ext or SUPPORTED_EXTS
    base = _resolve_data_dir(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates: List[Path] = []
    for ext in allowed_ext:
        candidates.extend(base.rglob(f"{prefix}*{ext}"))

    if not candidates:
        raise FileNotFoundError(
            f"No mixed AML dataset found in {base}. "
            f"Run scripts/07_generate_synthetic_aml_mixed_bank_fintech.py first or set {env_var}."
        )

    timestamped: List[Tuple[datetime, Path]] = []
    for path in candidates:
        ts = parse_timestamp_from_filename(path.name)
        if ts:
            timestamped.append((ts, path))

    if timestamped:
        timestamped.sort(key=lambda x: x[0], reverse=True)
        return str(timestamped[0][1])

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_data_spark(spark, path: str):
    """Load dataset using Spark."""
    local_uri = to_local_file_uri(path)
    ext = _extract_suffix(path)
    if ext == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(local_uri)
    if ext == ".parquet":
        return spark.read.parquet(local_uri)
    raise ValueError(f"Unsupported file extension: {ext}")


def to_local_file_uri(path: str) -> str:
    """Force local filesystem access for Spark by returning file:// URI."""
    lowered = path.lower()
    if lowered.startswith(("file://", "hdfs://", "s3://", "s3a://", "gs://", "abfs://", "abfss://")):
        return path
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    return f"file://{resolved}"


def _extract_suffix(path: str) -> str:
    if "://" in path:
        parsed = urlparse(path)
        return Path(parsed.path).suffix.lower()
    return Path(path).suffix.lower()


def _score_target_name(name: str) -> int:
    lowered = name.lower()
    score = 0
    for hint in TARGET_NAME_HINTS:
        if lowered == hint:
            score = max(score, 3)
        elif hint in lowered:
            score = max(score, 2)
    if lowered.startswith("is_") or lowered.endswith("_flag"):
        score = max(score, 1)
    return score


def pick_target_column_from_names(names: List[str]) -> Optional[str]:
    best_name = None
    best_score = 0
    for name in names:
        score = _score_target_name(name)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score > 0 else None


def pick_target_column(
    df,
    col_types: Dict[str, List[str]],
    id_cols: Optional[List[str]] = None,
    max_candidates: int = 20,
) -> Optional[str]:
    """Pick a likely target column from a Spark DataFrame."""
    from pyspark.sql import functions as F

    id_cols = set(id_cols or [])
    candidates = [c for c in df.columns if c not in id_cols and c not in col_types.get("datetime", [])]
    by_name = pick_target_column_from_names(candidates)
    if by_name:
        return by_name

    fallback_cols = [c for c in (col_types.get("boolean", []) + col_types.get("numeric", [])) if c in candidates]
    if not fallback_cols:
        return None
    fallback_cols = fallback_cols[:max_candidates]

    exprs = [F.approx_count_distinct(F.col(c)).alias(c) for c in fallback_cols]
    counts = df.agg(*exprs).collect()[0].asDict()
    for col in fallback_cols:
        if (counts.get(col) or 0) <= 2:
            return col
    return None


def time_parse_ratio(df, time_col: str, min_valid_ratio: float = 0.9) -> Tuple[bool, float]:
    """Check whether a time column is clean enough for time-series analysis."""
    from pyspark.sql import functions as F

    if time_col not in df.columns:
        return False, 0.0
    parsed = F.to_timestamp(F.col(time_col))
    ratio = (
        df.select(F.mean(F.when(parsed.isNotNull(), F.lit(1)).otherwise(F.lit(0)))).collect()[0][0]
        or 0.0
    )
    ratio = float(ratio)
    return ratio >= min_valid_ratio, ratio


def pick_time_column(
    df,
    col_types: Dict[str, List[str]],
    min_valid_ratio: float = 0.9,
) -> Optional[str]:
    """Pick a likely time column from Spark columns."""
    if col_types.get("datetime"):
        return col_types["datetime"][0]
    for col in df.columns:
        name_hint = str(col).lower()
        if any(tok in name_hint for tok in ["date", "time", "ts", "timestamp"]):
            clean, _ = time_parse_ratio(df, col, min_valid_ratio=min_valid_ratio)
            if clean:
                return col
    return None


def infer_column_types(
    df,
    id_cols: Optional[List[str]] = None,
    text_len_threshold: int = 30,
    text_unique_ratio: float = 0.5,
    datetime_min_ratio: float = 0.8,
    sample_size: int = 5000,
) -> Dict[str, List[str]]:
    """Classify columns into numeric, categorical, datetime, text, boolean, and other (Spark)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, DateType, NumericType, StringType, TimestampType

    id_cols = set(id_cols or [])
    cols = [c for c in df.columns if c not in id_cols]

    numeric_cols: List[str] = []
    bool_cols: List[str] = []
    datetime_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []
    other_cols: List[str] = []

    string_cols: List[str] = []
    for field in df.schema.fields:
        name = field.name
        if name not in cols:
            continue
        if isinstance(field.dataType, BooleanType):
            bool_cols.append(name)
        elif isinstance(field.dataType, (TimestampType, DateType)):
            datetime_cols.append(name)
        elif isinstance(field.dataType, NumericType):
            numeric_cols.append(name)
        elif isinstance(field.dataType, StringType):
            string_cols.append(name)
        else:
            other_cols.append(name)

    if string_cols:
        sample_df = df.select(string_cols)
        if sample_size:
            sample_df = sample_df.limit(sample_size)

        exprs = []
        for col in string_cols:
            exprs.append(F.avg(F.length(F.col(col))).alias(f"{col}__avg_len"))
            exprs.append(F.approx_count_distinct(F.col(col)).alias(f"{col}__uniq"))
            exprs.append(F.count(F.col(col)).alias(f"{col}__count"))
        stats = sample_df.agg(*exprs).collect()[0].asDict()

        for col in string_cols:
            avg_len = float(stats.get(f"{col}__avg_len") or 0.0)
            uniq = float(stats.get(f"{col}__uniq") or 0.0)
            count = float(stats.get(f"{col}__count") or 0.0)
            unique_ratio = uniq / max(1.0, count)
            name_hint = str(col).lower()
            if any(tok in name_hint for tok in ["date", "time", "ts", "timestamp"]):
                clean, ratio = time_parse_ratio(sample_df, col, min_valid_ratio=datetime_min_ratio)
                if clean:
                    datetime_cols.append(col)
                    continue
            if avg_len >= text_len_threshold or unique_ratio >= text_unique_ratio:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols,
        "boolean": bool_cols,
        "other": other_cols,
    }


def detect_null_like_values(
    df,
    null_like_values: Optional[List[str]] = None,
    max_examples: int = 3,
) -> List[Dict[str, Any]]:
    """Detect null-like placeholder values in string columns (Spark)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    null_like_values = null_like_values or DEFAULT_NULL_LIKE_VALUES
    null_set = {str(v).strip().lower() for v in null_like_values}

    results: List[Dict[str, Any]] = []
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    total_rows = df.count()
    if total_rows == 0 or not string_cols:
        return results

    for col in string_cols:
        norm = F.lower(F.trim(F.col(col)))
        cond = norm.isin(list(null_set)) | (norm == "")
        count = df.select(F.sum(F.when(cond, 1).otherwise(0)).alias("cnt")).collect()[0]["cnt"]
        if count and count > 0:
            rate = float(count) / max(1, total_rows)
            examples = (
                df.select(norm.alias("val"))
                .where(cond)
                .distinct()
                .limit(max_examples)
                .toPandas()["val"]
                .tolist()
            )
            results.append(
                {
                    "column": col,
                    "null_like_count": int(count),
                    "null_like_rate": rate,
                    "examples": examples,
                }
            )

    return results


def safe_select_columns(df, cols: Optional[List[str]]):
    """Select columns if provided; otherwise return original Spark DataFrame."""
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df.select(cols)
