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
