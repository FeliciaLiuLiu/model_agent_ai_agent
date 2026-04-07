import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import SparkTestCase, make_runner_dataframe, write_csv, write_nb_loader, write_py_loader, write_sql_file, write_sqlite_db  # noqa: E402
from eda_spark.runner import EDASpark  # noqa: E402
from eda_spark.utils import infer_column_types, time_parse_ratio  # noqa: E402


class TestEDASparkRunner(SparkTestCase):
    def _make_eda(self) -> EDASpark:
        return EDASpark(
            output_dir=str(self.tmp_path / "out"),
            spark=self.spark,
            max_numeric_cols=6,
            max_categorical_cols=6,
            max_plots=2,
            sample_size=400,
            top_k_categories=5,
        )

    def _make_context(self):
        pdf = make_runner_dataframe(rows=240)
        sdf = self.spark.createDataFrame(pdf)
        col_types = infer_column_types(sdf, id_cols=["txn_id"], sample_size=400)
        time_clean, time_ratio = time_parse_ratio(sdf, "txn_datetime", min_valid_ratio=0.8)
        return {
            "df": sdf,
            "data_path": "in_memory",
            "target_col": "sar_actual",
            "time_col": "txn_datetime",
            "time_clean": time_clean,
            "time_ratio": time_ratio,
            "col_types": col_types,
        }

    def test_list_and_parse_helpers(self):
        functions = EDASpark.list_functions()
        self.assertTrue(functions)
        keys = [item["key"] for item in functions]
        self.assertIn("data_quality", keys)

        self.assertIsNone(EDASpark.parse_function_selection(None))
        self.assertIsNone(EDASpark.parse_function_selection("all"))
        self.assertEqual(EDASpark.parse_function_selection("1,2"), ["data_quality", "target"])
        self.assertEqual(EDASpark.parse_function_selection("univariate,target"), ["univariate", "target"])

        options = ["a", "b", "c"]
        self.assertIsNone(EDASpark.parse_column_selection("", options))
        self.assertEqual(EDASpark.parse_column_selection("all", options), options)
        self.assertEqual(EDASpark.parse_column_selection("1,3", options), ["a", "c"])
        self.assertEqual(EDASpark.parse_column_selection("b,c", options), ["b", "c"])

    def test_print_helpers_do_not_crash(self):
        eda = self._make_eda()
        with mock.patch("sys.stdout", new=io.StringIO()):
            eda.print_functions()
            eda._print_numbered_list(["a", "b", "c"], max_items=2)

    def test_prerequisites_and_applicable_columns(self):
        eda = self._make_eda()
        context = self._make_context()

        ok, reason = eda._check_prerequisites("data_quality", context)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

        bad_target_ctx = dict(context)
        bad_target_ctx["target_col"] = "missing_target"
        ok_t, reason_t = eda._check_prerequisites("target", bad_target_ctx)
        self.assertFalse(ok_t)
        self.assertIn("Target column", reason_t)

        bad_time_ctx = dict(context)
        bad_time_ctx["time_col"] = None
        ok_tm, reason_tm = eda._check_prerequisites("time_drift", bad_time_ctx)
        self.assertFalse(ok_tm)
        self.assertIn("Time column", reason_tm)

        all_cols = eda._applicable_columns_for_section("data_quality", context)
        self.assertIn("amount", all_cols)
        uv_cols = eda._applicable_columns_for_section("univariate", context)
        self.assertNotIn("sar_actual", uv_cols)
        fvf_cols = eda._applicable_columns_for_section("feature_vs_feature", context)
        self.assertTrue(fvf_cols)

    def test_filter_and_column_selection_helpers(self):
        eda = self._make_eda()
        context = self._make_context()
        df = context["df"]
        col_types = context["col_types"]

        filtered = eda._filter_selected_columns(["amount", "missing"], ["amount", "segment"])
        self.assertEqual(filtered, ["amount"])
        self.assertIsNone(eda._filter_selected_columns(None, ["amount"]))

        numeric = eda._select_numeric(df, col_types, None)
        self.assertTrue(numeric)
        selected_numeric = eda._select_numeric(df, col_types, ["amount", "segment"])
        self.assertEqual(selected_numeric, ["amount"])

        categorical = eda._select_categorical(df, col_types, None)
        self.assertTrue(categorical)
        selected_cat = eda._select_categorical(df, col_types, ["channel", "amount"])
        self.assertEqual(selected_cat, ["channel"])

    def test_section_methods(self):
        eda = self._make_eda()
        context = self._make_context()

        dq = eda._section_data_quality(context)
        self.assertIn("metrics", dq)
        self.assertIn("missingness_payload", dq["metrics"])

        target = eda._section_target(context)
        self.assertIn("metrics", target)
        self.assertIn("target_column", target["metrics"])

        uni = eda._section_univariate(context, selected_cols=None)
        self.assertIn("univariate_payload", uni)
        self.assertIn("numeric_columns", uni["univariate_payload"])

        bvt = eda._section_bivariate_target(context, selected_cols=None)
        self.assertIn("tables", bvt)

        fvf = eda._section_feature_vs_feature(context, selected_cols=None)
        self.assertIn("metrics", fvf)
        self.assertIn("correlation", fvf["metrics"])

        tdr = eda._section_time_drift(context, selected_cols=None)
        self.assertIn("tables", tdr)

        summary = eda._section_summary(context)
        self.assertIn("summary", summary)

    def test_section_methods_with_boolean_like_string_target(self):
        pdf = pd.DataFrame(
            {
                "txn_id": [1, 2, 3, 4, 5, 6],
                "amount": [10.0, 15.0, 25.0, 35.0, 45.0, 55.0],
                "segment": ["a", "a", "b", "b", "c", "c"],
                "target": ["yes", "no", "yes", "no", "yes", "no"],
                "txn_datetime": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-03-01",
                    "2024-03-02",
                ],
            }
        )
        df = self.spark.createDataFrame(pdf)
        col_types = infer_column_types(df, id_cols=["txn_id"], sample_size=200)
        time_clean, time_ratio = time_parse_ratio(df, "txn_datetime", min_valid_ratio=0.8)
        context = {
            "df": df,
            "data_path": "in_memory",
            "target_col": "target",
            "time_col": "txn_datetime",
            "time_clean": time_clean,
            "time_ratio": time_ratio,
            "col_types": col_types,
        }

        target = eda._section_target(context)
        self.assertEqual(target["metrics"]["target_mapping"]["kind"], "boolean_like_string")
        self.assertTrue(any(table["title"] == "Target Rate by segment" for table in target["tables"]))

        bvt = eda._section_bivariate_target(context, selected_cols=None)
        self.assertEqual(bvt["metrics"]["target_mapping"]["kind"], "boolean_like_string")
        self.assertTrue(any(table["title"] == "Numeric vs Target (Binned)" for table in bvt["tables"]))

    def test_run_full_pipeline_from_dataframe(self):
        eda = self._make_eda()
        df = self.spark.createDataFrame(make_runner_dataframe(rows=220))
        payload = eda.run(
            df=df,
            sections=["data_quality", "target", "univariate", "bivariate_target", "feature_vs_feature", "time_drift"],
            target_col="sar_actual",
            time_col="txn_datetime",
            save_json=False,
            generate_report=False,
            return_payload=True,
        )
        self.assertIn("results", payload)
        self.assertIn("config", payload)
        self.assertIn("data_quality", payload["results"])
        self.assertEqual(payload["config"]["rows_used"], 220)

    def test_run_auto_exec_pipeline(self):
        write_csv(self.tmp_path, name="data.csv", rows=3)
        write_sqlite_db(self.tmp_path)
        write_sql_file(self.tmp_path, sql_text="SELECT * FROM t")
        write_py_loader(self.tmp_path)
        write_nb_loader(self.tmp_path)

        eda = EDASpark(
            output_dir=str(self.tmp_path / "out_auto"),
            spark=self.spark,
            max_plots=1,
            sample_size=200,
        )
        payload = eda.run(
            df=None,
            data_dir=str(self.tmp_path),
            auto_exec=True,
            sections=["data_quality"],
            save_json=False,
            generate_report=False,
            return_payload=True,
        )
        self.assertIn("results", payload)
        self.assertGreaterEqual(payload["config"]["rows_used"], 1)

    def test_run_named_table_composition(self):
        txn_path = self.tmp_path / "transaction.csv"
        cust_path = self.tmp_path / "customer.csv"
        acct_path = self.tmp_path / "account.csv"
        pd.DataFrame(
            {
                "transaction_id": [1, 2],
                "customer_id": ["C1", "C1"],
                "account_id": ["A1", "A2"],
                "amount": [100.0, 55.0],
                "sar_actual": [0, 1],
                "txn_datetime": ["2024-01-01", "2024-01-02"],
            }
        ).to_csv(txn_path, index=False)
        pd.DataFrame({"customer_id": ["C1", "C2"], "segment": ["gold", "silver"]}).to_csv(cust_path, index=False)
        pd.DataFrame({"account_id": ["A1", "A2"], "customer_id": ["C1", "C2"], "balance": [1000.0, 2000.0]}).to_csv(
            acct_path, index=False
        )

        eda = EDASpark(
            output_dir=str(self.tmp_path / "out_compose"),
            spark=self.spark,
            max_plots=1,
            sample_size=200,
        )
        payload = eda.run(
            data=[
                f"transaction={txn_path}",
                f"customer={cust_path}",
                f"account={acct_path}",
            ],
            sections=["data_quality"],
            save_json=False,
            generate_report=False,
            return_payload=True,
        )
        self.assertEqual(payload["config"]["composition"]["mode"], "row_level")

    def test_run_no_key_aggregate_only(self):
        tx_path = self.tmp_path / "transaction.csv"
        cust_path = self.tmp_path / "customer.csv"
        pd.DataFrame({"transaction_id": [1, 2], "amount": [5.0, 8.0]}).to_csv(tx_path, index=False)
        pd.DataFrame({"cust_ref": ["x", "y"], "segment": ["a", "b"]}).to_csv(cust_path, index=False)

        eda = EDASpark(
            output_dir=str(self.tmp_path / "out_no_key"),
            spark=self.spark,
            max_plots=1,
            sample_size=200,
        )
        payload = eda.run(
            data=[f"transaction={tx_path}", f"customer={cust_path}"],
            sections=["data_quality"],
            save_json=False,
            generate_report=False,
            return_payload=True,
        )
        self.assertEqual(payload["config"]["composition"]["mode"], "aggregate_only")
        self.assertEqual(payload["config"]["rows_used"], 2)

    def test_run_interactive(self):
        eda = self._make_eda()
        df = self.spark.createDataFrame(make_runner_dataframe(rows=120))
        with mock.patch("builtins.input", side_effect=["1", ""]):
            payload = eda.run_interactive(
                df=df,
                target_col="sar_actual",
                time_col="txn_datetime",
                save_json=False,
                generate_report=False,
                return_payload=True,
            )
        self.assertIn("results", payload)
        self.assertIn("data_quality", payload["results"])

    def test_drift_and_plot_helpers(self):
        eda = self._make_eda()

        base_df = pd.DataFrame({"bucket": [0.0, 1.0], "count": [80, 20]})
        curr_df = pd.DataFrame({"bucket": [0.0, 1.0], "count": [50, 50]})
        psi = eda._psi_from_counts(base_df, curr_df)
        self.assertGreaterEqual(psi, 0.0)

        base_cat = pd.DataFrame({"channel": ["card", "wire"], "count": [70, 30]})
        curr_cat = pd.DataFrame({"channel": ["card", "wire"], "count": [40, 60]})
        drift = eda._categorical_drift(base_cat, curr_cat, top_k=2)
        self.assertGreaterEqual(drift, 0.0)

        bar_path = self.tmp_path / "bar.png"
        hist_path = self.tmp_path / "hist.png"
        heat_path = self.tmp_path / "heat.png"
        line_path = self.tmp_path / "line.png"

        eda._plot_bar(pd.Series([1, 2], index=["a", "b"]), str(bar_path), "bar", "v")
        eda._plot_hist(pd.Series([1, 2, 3, 4]), str(hist_path), "hist")
        eda._plot_heatmap(np.array([[1.0, 0.5], [0.5, 1.0]]), ["x", "y"], str(heat_path), "heat")
        eda._plot_line(pd.Series([1, 3], index=["2024-01", "2024-02"]), str(line_path), "line", "v")

        self.assertTrue(bar_path.exists())
        self.assertTrue(hist_path.exists())
        self.assertTrue(heat_path.exists())
        self.assertTrue(line_path.exists())


if __name__ == "__main__":
    unittest.main()
