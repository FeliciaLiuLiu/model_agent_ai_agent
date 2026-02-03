import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest


@pytest.fixture()
def df_no_missing():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["x", "y", "z", "w"],
    })


@pytest.fixture()
def df_missing():
    return pd.DataFrame({
        "a": [1, None, 3, 4],
        "b": ["x", "y", "z", "w"],
    })


@pytest.fixture()
def df_null_like():
    return pd.DataFrame({
        "c1": ["NA", "n/a", "null", "foo", " ", "?", None],
        "c2": ["ok", "OK", "unknown", "bar", "-", "baz", ""],
        "num": [1, 2, 3, 4, 5, 6, 7],
    })


@pytest.fixture()
def df_no_string():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4.0, 5.0, 6.0],
    })


@pytest.fixture()
def df_mixed_types():
    return pd.DataFrame({
        "num": [1.5, 2.5, 3.0, 2.2, 1.9, 2.7],
        "cat": ["alpha", "alpha", "beta", "beta", "beta", "alpha"],
        "flag": [True, False, True, False, True, False],
        "ts": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
        ),
        "text": [
            "Customer reported unexpected transfer pattern after device change.",
            "Recurring payments observed with merchant mismatch to profile.",
            "Device location shift not consistent with historical behavior.",
            "Multiple small transfers aggregated into a single payout request.",
            "Counterparty linked to prior alerts; escalating for review.",
            "New account activity with elevated risk score and overseas destination.",
        ],
    })


@pytest.fixture()
def local_tmp_path():
    base_dir = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    tmp_dir = base_dir / uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
