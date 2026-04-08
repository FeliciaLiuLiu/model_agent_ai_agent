"""Data Analysis Agent."""
import importlib.util
import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


class DataAnalysisAgent:
    """Agent for data loading and profiling."""

    def load_dataset(
        self,
        path: str = None,
        sql: str = None,
        conn: str = None,
        loader_py: str = None,
        loader_fn: str = "load_data",
    ) -> pd.DataFrame:
        provided = [value is not None for value in (path, sql, loader_py)]
        if sum(provided) != 1:
            raise ValueError("Specify exactly one data source: path, sql, or loader_py.")

        if path is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                return pd.read_csv(path)
            if ext == '.parquet':
                return pd.read_parquet(path)
            if ext in ['.xlsx', '.xls']:
                return pd.read_excel(path)
            raise ValueError(f"Unsupported format: {ext}")

        if sql is not None:
            if not conn:
                raise ValueError("A SQL connection string is required when sql is provided.")
            sql_text = self._read_sql(sql)
            if conn.startswith("sqlite:///"):
                with sqlite3.connect(conn.replace("sqlite:///", "", 1)) as db_conn:
                    return pd.read_sql_query(sql_text, db_conn)
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:
                raise ImportError(
                    "SQLAlchemy is required for non-sqlite SQL connections."
                ) from exc
            engine = create_engine(conn)
            with engine.connect() as db_conn:
                return pd.read_sql_query(sql_text, db_conn)

        payload = self._load_python(loader_py, loader_fn)
        if isinstance(payload, pd.DataFrame):
            return payload
        if isinstance(payload, dict) and "df" in payload and isinstance(payload["df"], pd.DataFrame):
            return payload["df"]
        if isinstance(payload, tuple) and payload and isinstance(payload[0], pd.DataFrame):
            return payload[0]
        raise ValueError("Python loader must return a pandas DataFrame or a payload containing a DataFrame.")

    def guess_label_col(self, df):
        for name in ['label', 'target', 'y', 'class', 'fraud', 'is_fraud']:
            if name in df.columns: return name
        for col in df.columns:
            if set(df[col].dropna().unique()).issubset({0, 1}): return col
        return None

    def _read_sql(self, sql: str) -> str:
        sql_path = Path(sql)
        if sql_path.exists() and sql_path.is_file() and sql_path.suffix.lower() == ".sql":
            return sql_path.read_text(encoding="utf-8")
        return sql

    def _load_python(self, loader_py: str, loader_fn: str):
        loader_path = Path(loader_py)
        if not loader_path.exists():
            raise FileNotFoundError(f"Python loader file not found: {loader_py}")

        spec = importlib.util.spec_from_file_location(f"model_testing_loader_{loader_path.stem}", loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import Python loader file: {loader_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, loader_fn):
            raise AttributeError(f"Loader function '{loader_fn}' not found in {loader_py}")
        return getattr(module, loader_fn)()
