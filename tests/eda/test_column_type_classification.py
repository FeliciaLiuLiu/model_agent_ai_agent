from adm_central_utility.eda.utils import infer_column_types


def test_infer_column_types_mixed(df_mixed_types):
    col_types = infer_column_types(df_mixed_types)

    assert "num" in col_types["numeric"]
    assert "cat" in col_types["categorical"]
    assert "flag" in col_types["boolean"]
    assert "ts" in col_types["datetime"]
    assert "text" in col_types["text"]
