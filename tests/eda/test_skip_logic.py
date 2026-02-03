from adm_central_utility.eda.runner import EDA
from adm_central_utility.eda.utils import infer_column_types


def _context(df):
    return {
        "df": df,
        "col_types": infer_column_types(df),
        "rows_original": len(df),
    }


def test_missingness_skipped_reason(df_no_missing, local_tmp_path):
    eda = EDA(output_dir=str(local_tmp_path))
    result = eda._section_data_quality(_context(df_no_missing))
    assert result["metrics"].get("missingness_skipped_reason") is not None


def test_null_like_skipped_reason(df_no_string, local_tmp_path):
    eda = EDA(output_dir=str(local_tmp_path))
    result = eda._section_data_quality(_context(df_no_string))
    assert result["metrics"].get("null_like_skipped_reason") is not None
