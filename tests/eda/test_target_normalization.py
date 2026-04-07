import pandas as pd

from eda.runner import EDA
from eda.utils import infer_column_types, is_time_col_clean


def _context(df):
    col_types = infer_column_types(df)
    time_clean, time_ratio = is_time_col_clean(df, "event_time", min_valid_ratio=0.8)
    return {
        "df": df,
        "col_types": col_types,
        "rows_original": len(df),
        "target_col": "target",
        "time_col": "event_time",
        "time_clean": time_clean,
        "time_ratio": time_ratio,
    }


def test_target_section_maps_boolean_like_strings(local_tmp_path):
    df = pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no", "yes", "no"],
            "segment": ["a", "a", "a", "b", "b", "b"],
            "event_time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-02-01",
                "2024-02-02",
                "2024-02-03",
            ],
        }
    )
    eda = EDA(output_dir=str(local_tmp_path))
    result = eda._section_target(_context(df))

    assert result["metrics"]["target_mapping"]["kind"] == "boolean_like_string"
    assert any("Target Rate by segment" == table["title"] for table in result["tables"])
    assert "target_rate_over_time" in result["plots"]


def test_bivariate_target_maps_boolean_like_strings(local_tmp_path):
    df = pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no", "yes", "no"],
            "amount": [10, 12, 30, 35, 55, 60],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )
    eda = EDA(output_dir=str(local_tmp_path))
    context = {
        "df": df,
        "col_types": infer_column_types(df),
        "rows_original": len(df),
        "target_col": "target",
    }
    result = eda._section_bivariate_target(context, selected_cols=None)

    assert result["metrics"]["target_mapping"]["kind"] == "boolean_like_string"
    assert any(table["title"] == "Numeric vs Target (Binned)" for table in result["tables"])
    assert any(table["title"] == "Categorical vs Target" for table in result["tables"])
