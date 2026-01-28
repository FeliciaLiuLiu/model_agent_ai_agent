"""Data Analysis Agent."""
import os
import numpy as np
import pandas as pd


class DataAnalysisAgent:
    """Agent for data loading and profiling."""

    def load_dataset(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv': return pd.read_csv(path)
        elif ext == '.parquet': return pd.read_parquet(path)
        elif ext in ['.xlsx', '.xls']: return pd.read_excel(path)
        raise ValueError(f"Unsupported format: {ext}")

    def guess_label_col(self, df):
        for name in ['label', 'target', 'y', 'class', 'fraud', 'is_fraud']:
            if name in df.columns: return name
        for col in df.columns:
            if set(df[col].dropna().unique()).issubset({0, 1}): return col
        return None
