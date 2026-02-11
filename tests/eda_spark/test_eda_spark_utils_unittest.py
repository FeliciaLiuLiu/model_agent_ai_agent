import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import (  # noqa: E402
    SparkTestCase,
    write_csv,
    write_json_array,
    write_parquet_with_spark,
    write_tsv,
)
from eda_spark import utils  # noqa: E402


class TestSparkUtils(SparkTestCase):
    def test_resolve_data_dir(self):
        resolved = utils._resolve_data_dir(str(self.tmp_path))
        self.assertEqual(resolved, self.tmp_path)

    def test_parse_timestamp_from_filename(self):
        valid = utils.parse_timestamp_from_filename("synthetic_aml_mixed_50k_20260205_094055.csv")
        invalid = utils.parse_timestamp_from_filename("no_timestamp.csv")
        self.assertIsNotNone(valid)
        self.assertIsNone(invalid)

    def test_detect_latest_dataset_with_env_override(self):
        fake_path = str(self.tmp_path / "forced.csv")
        with mock.patch.dict(os.environ, {"EDA_SPARK_DATA_PATH": fake_path}, clear=False):
            result = utils.detect_latest_dataset(data_dir=str(self.tmp_path))
        self.assertEqual(result, fake_path)

    def test_detect_latest_dataset_by_mtime_and_prefix(self):
        old_file = write_csv(self.tmp_path, name="synthetic_aml_mixed_50k_old.csv", rows=1)
        new_file = write_csv(self.tmp_path, name="synthetic_aml_mixed_50k_new.csv", rows=2)
        os.utime(old_file, (1, 1))
        os.utime(new_file, (2, 2))

        latest = utils.detect_latest_dataset(
            data_dir=str(self.tmp_path),
            allowed_ext=[".csv"],
            env_var="NON_EXISTENT_ENV",
            prefix="synthetic_aml_mixed_50k_",
        )
        self.assertTrue(latest.endswith("synthetic_aml_mixed_50k_new.csv"))

    def test_detect_latest_dataset_errors(self):
        with self.assertRaises(FileNotFoundError):
            utils.detect_latest_dataset(data_dir=str(self.tmp_path / "missing"), env_var="NON_EXISTENT_ENV")

        with self.assertRaises(FileNotFoundError):
            utils.detect_latest_dataset(data_dir=str(self.tmp_path), allowed_ext=[".csv"], env_var="NON_EXISTENT_ENV")

    def test_load_data_spark_for_supported_formats(self):
        csv_path = write_csv(self.tmp_path, rows=3)
        tsv_path = write_tsv(self.tmp_path, rows=2)
        json_path = write_json_array(self.tmp_path, rows=4)
        parquet_path = write_parquet_with_spark(self.tmp_path, self.spark, rows=5)

        self.assertEqual(utils.load_data_spark(self.spark, str(csv_path)).count(), 3)
        self.assertEqual(utils.load_data_spark(self.spark, str(tsv_path)).count(), 2)
        self.assertEqual(utils.load_data_spark(self.spark, str(json_path)).count(), 4)
        self.assertEqual(utils.load_data_spark(self.spark, str(parquet_path)).count(), 5)

    def test_load_data_spark_unsupported_extension(self):
        bad_path = self.tmp_path / "bad.xyz"
        bad_path.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            utils.load_data_spark(self.spark, str(bad_path))

    def test_to_local_file_uri(self):
        local_path = str(self.tmp_path / "x.csv")
        uri = utils.to_local_file_uri(local_path)
        self.assertTrue(uri.startswith("file://"))
        self.assertEqual(utils.to_local_file_uri("s3://bucket/path/file.csv"), "s3://bucket/path/file.csv")

    def test_extract_suffix(self):
        self.assertEqual(utils._extract_suffix("file.csv"), ".csv")
        self.assertEqual(utils._extract_suffix("s3://bucket/a/b/file.parquet"), ".parquet")

    def test_target_name_scoring_helpers(self):
        self.assertGreaterEqual(utils._score_target_name("sar_actual"), 2)
        self.assertEqual(utils._score_target_name("random_name"), 0)
        picked = utils.pick_target_column_from_names(["amount", "is_suspicious", "channel"])
        self.assertEqual(picked, "is_suspicious")

    def test_pick_target_column(self):
        pdf = pd.DataFrame(
            {
                "txn_id": [1, 2, 3, 4],
                "amount": [100, 200, 300, 400],
                "flag": [0, 1, 0, 1],
                "event_time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            }
        )
        df = self.spark.createDataFrame(pdf)
        col_types = {
            "numeric": ["amount", "flag"],
            "categorical": [],
            "datetime": ["event_time"],
            "text": [],
            "boolean": [],
            "other": [],
        }
        picked = utils.pick_target_column(df, col_types, id_cols=["txn_id"])
        self.assertEqual(picked, "flag")

    def test_time_parse_ratio_and_pick_time_column(self):
        pdf = pd.DataFrame(
            {
                "event_time": ["2024-01-01", "2024-01-02", "bad", None],
                "created_at": ["2024-02-01 10:00:00", "2024-02-02 10:00:00", "2024-02-03 10:00:00", "2024-02-04 10:00:00"],
            }
        )
        df = self.spark.createDataFrame(pdf)
        clean, ratio = utils.time_parse_ratio(df, "created_at", min_valid_ratio=0.9)
        self.assertTrue(clean)
        self.assertGreaterEqual(ratio, 0.9)

        col_types = {"datetime": [], "numeric": [], "categorical": [], "text": [], "boolean": [], "other": []}
        picked = utils.pick_time_column(df, col_types, min_valid_ratio=0.75)
        self.assertEqual(picked, "created_at")

        col_types2 = {"datetime": ["event_time"], "numeric": [], "categorical": [], "text": [], "boolean": [], "other": []}
        picked2 = utils.pick_time_column(df, col_types2, min_valid_ratio=0.9)
        self.assertEqual(picked2, "event_time")

    def test_infer_column_types(self):
        pdf = pd.DataFrame(
            {
                "id_col": [1, 2, 3, 4],
                "amount": [1.2, 2.5, 3.1, 4.0],
                "is_flag": [True, False, True, False],
                "event_time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "segment": ["retail", "retail", "vip", "vip"],
                "notes": [
                    "This is a much longer free-form text string for testing.",
                    "Another long free-form text string for testing.",
                    "Third long free-form text string for testing.",
                    "Fourth long free-form text string for testing.",
                ],
            }
        )
        df = self.spark.createDataFrame(pdf)
        out = utils.infer_column_types(df, id_cols=["id_col"], sample_size=100)
        self.assertIn("amount", out["numeric"])
        self.assertIn("is_flag", out["boolean"])
        self.assertIn("event_time", out["datetime"])
        self.assertIn("segment", out["categorical"])
        self.assertIn("notes", out["text"])

    def test_detect_null_like_values(self):
        pdf = pd.DataFrame(
            {
                "c1": ["NA", "ok", "null", "x"],
                "c2": ["UNKNOWN", "ok", "ok", "N/A"],
                "num": [1, 2, 3, 4],
            }
        )
        df = self.spark.createDataFrame(pdf)
        payload = utils.detect_null_like_values(df, max_examples=2)
        cols = {row["column"] for row in payload}
        self.assertIn("c1", cols)
        self.assertIn("c2", cols)
        self.assertTrue(all("null_like_rate" in row for row in payload))
        self.assertTrue(all(len(row["examples"]) <= 2 for row in payload))

    def test_safe_select_columns(self):
        df = self.spark.createDataFrame(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual(utils.safe_select_columns(df, None).count(), 2)
        self.assertEqual(utils.safe_select_columns(df, ["a"]).columns, ["a"])
        with self.assertRaises(ValueError):
            utils.safe_select_columns(df, ["missing"])


if __name__ == "__main__":
    unittest.main()
