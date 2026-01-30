from adm_central_utility.eda.runner import EDA
from adm_central_utility.eda.utils import infer_column_types


def _context(df):
    return {
        "df": df,
        "col_types": infer_column_types(df),
        "rows_original": len(df),
    }


def test_missingness_includes_only_missing(df_missing):
    eda = EDA()
    result = eda._section_data_quality(_context(df_missing))
    payload = result["metrics"]["missingness_payload"]

    missing_cols = payload["missing_columns"]
    assert len(missing_cols) == 1
    assert missing_cols[0]["column"] == "a"
    assert payload["non_missing_columns"] == ["b"]


def test_missingness_rate(df_missing):
    eda = EDA()
    result = eda._section_data_quality(_context(df_missing))
    payload = result["metrics"]["missingness_payload"]

    missing_cols = payload["missing_columns"]
    rate = missing_cols[0]["missing_rate"]
    assert abs(rate - 0.25) < 1e-6


def test_no_missing_columns(df_no_missing):
    eda = EDA()
    result = eda._section_data_quality(_context(df_no_missing))
    payload = result["metrics"]["missingness_payload"]

    assert payload["missing_columns"] == []
    assert result["metrics"].get("missingness_skipped_reason") is not None
