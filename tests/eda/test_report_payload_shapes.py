from adm_central_utility.eda.runner import EDA
from adm_central_utility.eda.utils import infer_column_types


def _context(df):
    return {
        "df": df,
        "col_types": infer_column_types(df),
        "rows_original": len(df),
    }


def test_missingness_payload_shape(df_missing):
    eda = EDA()
    result = eda._section_data_quality(_context(df_missing))
    payload = result["metrics"]["missingness_payload"]

    assert "missing_columns" in payload
    assert "non_missing_columns" in payload
    for row in payload["missing_columns"]:
        assert set(row.keys()) == {"column", "missing_count", "missing_rate"}
        assert 0 <= row["missing_rate"] <= 1


def test_null_like_payload_shape(df_null_like):
    eda = EDA()
    result = eda._section_data_quality(_context(df_null_like))
    payload = result["metrics"]["null_like_payload"]

    for row in payload:
        assert set(row.keys()) == {"column", "null_like_count", "null_like_rate", "examples"}
        assert 0 <= row["null_like_rate"] <= 1
        assert len(row["examples"]) <= 3
