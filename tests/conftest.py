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
        "num": [1.5, 2.5, 3.0],
        "cat": ["alpha", "beta", "gamma"],
        "flag": [True, False, True],
        "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "text": [
            "Customer reported unexpected transfer pattern after device change.",
            "Recurring payments observed with merchant mismatch to profile.",
            "Device location shift not consistent with historical behavior.",
        ],
    })
