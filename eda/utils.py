"""EDA utility helpers."""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_EXTS = [".csv", ".parquet"]
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
    env_var: str = "EDA_DATA_PATH",
) -> str:
    """Detect the latest dataset by file modification time (mtime)."""
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path

    allowed_ext = allowed_ext or SUPPORTED_EXTS
    base = _resolve_data_dir(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates: List[Path] = []
    for ext in allowed_ext:
        candidates.extend(base.rglob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"No dataset found in {base} (extensions: {allowed_ext})")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def auto_detect_data_path(data_dir: str = "./data", env_var: str = "EDA_DATA_PATH") -> str:
    """Backward-compatible alias for detect_latest_dataset."""
    return detect_latest_dataset(data_dir=data_dir, env_var=env_var)


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from file."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def infer_column_types(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    text_len_threshold: int = 30,
    text_unique_ratio: float = 0.5,
    datetime_sample_size: int = 500,
    datetime_min_ratio: float = 0.8,
) -> Dict[str, List[str]]:
    """Classify columns into numeric, categorical, datetime, text, boolean, and other."""
    id_cols = set(id_cols or [])
    cols = [c for c in df.columns if c not in id_cols]

    numeric_cols: List[str] = []
    bool_cols: List[str] = []
    datetime_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []
    other_cols: List[str] = []

    for col in cols:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            bool_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna().astype(str)
            if non_null.empty:
                categorical_cols.append(col)
                continue
            avg_len = float(non_null.str.len().mean())
            unique_ratio = non_null.nunique() / max(1, len(non_null))
            name_hint = str(col).lower()
            if any(tok in name_hint for tok in ["date", "time", "ts", "timestamp"]):
                sample = non_null.head(datetime_sample_size)
                parsed = pd.to_datetime(sample, errors="coerce")
                ratio = float(parsed.notna().mean())
                if ratio >= datetime_min_ratio:
                    datetime_cols.append(col)
                    continue
            if avg_len >= text_len_threshold or unique_ratio >= text_unique_ratio:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
            continue

        other_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols,
        "boolean": bool_cols,
        "other": other_cols,
    }


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
    df: pd.DataFrame,
    col_types: Dict[str, List[str]],
    id_cols: Optional[List[str]] = None,
) -> Optional[str]:
    """Pick a likely target column from a dataframe."""
    id_cols = set(id_cols or [])
    candidates = [c for c in df.columns if c not in id_cols and c not in col_types.get("datetime", [])]

    by_name = pick_target_column_from_names(candidates)
    if by_name:
        return by_name

    # Fallback: choose a binary numeric/boolean column.
    fallback_cols = col_types.get("boolean", []) + col_types.get("numeric", [])
    for col in fallback_cols:
        if col in id_cols:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        unique = pd.unique(series)
        if len(unique) <= 2:
            return col

    return None


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Convert a column to datetime, preserving original order."""
    return pd.to_datetime(df[time_col], errors="coerce")


def is_time_col_clean(
    df: pd.DataFrame,
    time_col: str,
    min_valid_ratio: float = 0.9,
) -> Tuple[bool, float]:
    """Check whether a time column is clean enough for time-series analysis."""
    if time_col not in df.columns:
        return False, 0.0
    parsed = ensure_datetime(df, time_col)
    valid_ratio = float(parsed.notna().mean()) if len(parsed) > 0 else 0.0
    return valid_ratio >= min_valid_ratio, valid_ratio


def safe_select_columns(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    """Select columns if provided; otherwise return original."""
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df.loc[:, cols]


def detect_null_like_values(
    df: pd.DataFrame,
    null_like_values: Optional[List[str]] = None,
    max_examples: int = 3,
) -> List[Dict[str, Any]]:
    """Detect null-like placeholder values in string-like columns."""
    null_like_values = null_like_values or DEFAULT_NULL_LIKE_VALUES
    null_set = {str(v).strip().lower() for v in null_like_values}

    results: List[Dict[str, Any]] = []
    str_cols = [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
        or pd.api.types.is_categorical_dtype(df[col])
    ]

    total_rows = len(df)
    if total_rows == 0:
        return results

    for col in str_cols:
        series = df[col]
        non_null = series[series.notna()]
        if non_null.empty:
            continue
        normalized = non_null.astype(str).str.strip().str.lower()
        mask = normalized.isin(null_set) | (normalized == "")
        count = int(mask.sum())
        if count <= 0:
            continue
        rate = float(count) / float(total_rows)
        examples = list(normalized[mask].dropna().unique())[:max_examples]
        results.append(
            {
                "column": col,
                "null_like_count": count,
                "null_like_rate": rate,
                "examples": examples,
            }
        )

    return results


def pick_time_column(
    df: pd.DataFrame,
    col_types: Dict[str, List[str]],
    min_valid_ratio: float = 0.9,
) -> Optional[str]:
    """Pick a likely time column if not provided."""
    candidates = col_types.get("datetime", [])
    if candidates:
        return candidates[0]
    for col in df.columns:
        name_hint = str(col).lower()
        if any(tok in name_hint for tok in ["date", "time", "ts", "timestamp"]):
            clean, ratio = is_time_col_clean(df, col, min_valid_ratio=min_valid_ratio)
            if clean:
                return col
    return None
