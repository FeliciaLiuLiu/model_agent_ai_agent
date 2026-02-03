import pandas as pd

from adm_central_utility.eda.utils import infer_column_types, pick_target_column


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
