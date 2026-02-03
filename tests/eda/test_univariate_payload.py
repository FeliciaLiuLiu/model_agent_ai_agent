from adm_central_utility.eda.runner import EDA
from adm_central_utility.eda.utils import infer_column_types


def _context(df):
    return {
        "df": df,
        "col_types": infer_column_types(df),
        "rows_original": len(df),
    }


def test_univariate_payload_structure(df_null_like, local_tmp_path):
    eda = EDA(output_dir=str(local_tmp_path))
    result = eda._section_univariate(_context(df_null_like), selected_cols=None)
    payload = result.get("univariate_payload")

    assert payload is not None
    assert "numeric_columns" in payload
    assert "categorical_columns" in payload
    assert "numeric_summary_rows" in payload
    assert "categorical_topk_by_column" in payload
    assert "chart_paths" in payload
