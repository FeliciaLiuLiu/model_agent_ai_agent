import pandas as pd

from eda.utils import coerce_target_series, infer_column_types, pick_target_column


def test_pick_target_column_by_name():
    df = pd.DataFrame({
        "txn_amount": [10, 20, 30],
        "sar_actual": [0, 1, 0],
        "is_high_risk_country": [False, True, False],
    })
    col_types = infer_column_types(df)
    assert pick_target_column(df, col_types) == "sar_actual"


def test_pick_target_column_binary_fallback():
    df = pd.DataFrame({
        "feature": [1.2, 3.4, 5.6],
        "outcome": [1, 0, 1],
    })
    col_types = infer_column_types(df)
    assert pick_target_column(df, col_types) == "outcome"


def test_coerce_target_series_boolean_like_strings():
    series = pd.Series(["yes", "no", "YES", None])
    coerced, meta = coerce_target_series(series)
    assert coerced is not None
    assert meta["kind"] == "boolean_like_string"
    assert coerced.iloc[0] == 1.0
    assert coerced.iloc[1] == 0.0
    assert coerced.iloc[2] == 1.0
    assert pd.isna(coerced.iloc[3])


def test_coerce_target_series_numeric_strings():
    series = pd.Series(["1", "2", "3", None])
    coerced, meta = coerce_target_series(series)
    assert coerced is not None
    assert meta["kind"] == "numeric_like_string"
    assert coerced.iloc[0] == 1.0
    assert coerced.iloc[2] == 3.0
