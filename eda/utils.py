"""EDA utility helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_EXTS = [".csv", ".parquet", ".xlsx", ".xls"]


def auto_detect_data_path(
    data_dir: str = "./data",
    env_var: str = "EDA_DATA_PATH",
) -> str:
    """Find the most recently modified dataset in the data directory."""
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path

    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates = []
    for ext in SUPPORTED_EXTS:
        candidates.extend(base.rglob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"No dataset found in {data_dir} (extensions: {SUPPORTED_EXTS})")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from file."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def detect_column_types(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    text_len_threshold: int = 30,
    text_unique_ratio: float = 0.5,
) -> Dict[str, List[str]]:
    """Classify columns into numeric, categorical, datetime, text, boolean, and other."""
    id_cols = set(id_cols or [])
    cols = [c for c in df.columns if c not in id_cols]

    numeric_cols = []
    bool_cols = []
    datetime_cols = []
    categorical_cols = []
    text_cols = []
    other_cols = []

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

        # Object/string/category types
        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna().astype(str)
            if non_null.empty:
                categorical_cols.append(col)
                continue
            avg_len = float(non_null.str.len().mean())
            unique_ratio = non_null.nunique() / max(1, len(non_null))
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


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Convert a column to datetime, preserving original order."""
    return pd.to_datetime(df[time_col], errors="coerce")


def safe_select_columns(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    """Select columns if provided; otherwise return original."""
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df.loc[:, cols]
