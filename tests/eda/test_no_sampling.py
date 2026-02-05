import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from adm_central_utility.eda.runner import EDA


def test_no_auto_sampling_for_synthetic_pattern():
    df = pd.DataFrame({"a": list(range(20)), "b": ["x"] * 20})
    output_dir = Path(__file__).resolve().parents[2] / ".pytest_tmp" / f"no_sampling_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        eda = EDA(output_dir=str(output_dir))
        payload = eda.run(
            df=df,
            file_path="synthetic_aml_200k_20240101_000000.csv",
            sections=["data_quality"],
            return_payload=True,
            generate_report=False,
            save_json=False,
        )

        assert payload["config"]["rows_original"] == 20
        assert payload["config"]["rows_used"] == 20
        assert "sample_frac" not in payload["config"]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
