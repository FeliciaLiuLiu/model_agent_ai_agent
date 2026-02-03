import pandas as pd

from adm_central_utility.eda.runner import EDA


def test_no_auto_sampling_for_synthetic_pattern(tmp_path):
    df = pd.DataFrame({"a": list(range(20)), "b": ["x"] * 20})
    eda = EDA(output_dir=str(tmp_path))
    payload = eda.run(
        df=df,
        file_path="synthetic_aml_200k_20240101_000000.csv",
        sections=["summary"],
        return_payload=True,
        generate_report=False,
        save_json=False,
    )

    assert payload["config"]["rows_original"] == 20
    assert payload["config"]["rows_used"] == 20
    assert "sample_frac" not in payload["config"]
