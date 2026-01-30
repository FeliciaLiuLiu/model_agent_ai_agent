from adm_central_utility.eda.utils import detect_null_like_values


def test_detect_null_like_values(df_null_like):
    results = detect_null_like_values(df_null_like)
    columns = {r["column"] for r in results}
    assert "c1" in columns
    assert "c2" in columns
    assert "num" not in columns


def test_null_like_case_insensitive(df_null_like):
    results = detect_null_like_values(df_null_like)
    row = next(r for r in results if r["column"] == "c1")
    assert row["null_like_count"] >= 4


def test_null_like_whitespace(df_null_like):
    results = detect_null_like_values(df_null_like)
    row = next(r for r in results if r["column"] == "c1")
    assert row["null_like_count"] >= 1
