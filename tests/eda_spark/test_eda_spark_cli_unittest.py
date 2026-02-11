import sys
import unittest
from pathlib import Path
from unittest import mock

from eda_spark import cli


class TestEDASparkCLI(unittest.TestCase):
    def test_parse_list(self):
        self.assertIsNone(cli._parse_list(None))
        self.assertEqual(cli._parse_list("a,b, c"), ["a", "b", "c"])

    def test_parse_multi_data(self):
        self.assertIsNone(cli._parse_multi_data(None))
        self.assertEqual(cli._parse_multi_data("a.csv,b.csv"), ["a.csv", "b.csv"])
        self.assertEqual(cli._parse_multi_data(["a.csv,b.csv", "c.csv"]), ["a.csv", "b.csv", "c.csv"])

    def test_parse_spark_conf(self):
        self.assertIsNone(cli._parse_spark_conf(None))
        conf = cli._parse_spark_conf(["spark.sql.shuffle.partitions=8", "spark.executor.memory=2g"])
        self.assertEqual(conf["spark.sql.shuffle.partitions"], "8")
        self.assertEqual(conf["spark.executor.memory"], "2g")
        with self.assertRaises(ValueError):
            cli._parse_spark_conf(["invalid-item"])

    def test_main_list_functions(self):
        with mock.patch.object(sys, "argv", ["prog", "--list-functions"]):
            with mock.patch("eda_spark.cli.EDASpark") as cls_mock:
                cli.main()
                cls_mock.assert_called_once_with()
                cls_mock.return_value.print_functions.assert_called_once_with()

    def test_main_non_interactive(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "prog",
                "--data",
                "a.csv,b.csv",
                "--output",
                "out_dir",
                "--sections",
                "data_quality,univariate",
                "--columns",
                "amount,channel",
                "--columns-data-quality",
                "amount",
                "--target-col",
                "sar_actual",
                "--time-col",
                "txn_datetime",
                "--spark-master",
                "local[*]",
                "--spark-conf",
                "spark.sql.shuffle.partitions=4",
                "--no-report",
                "--no-json",
            ],
        ):
            with mock.patch("eda_spark.cli.EDASpark") as cls_mock:
                with mock.patch("builtins.print") as print_mock:
                    cli.main()

                cls_mock.assert_called_once()
                kwargs = cls_mock.call_args.kwargs
                self.assertEqual(kwargs["output_dir"], "out_dir")
                self.assertEqual(kwargs["target_col"], "sar_actual")
                self.assertEqual(kwargs["time_col"], "txn_datetime")
                self.assertEqual(kwargs["spark_master"], "local[*]")
                self.assertEqual(kwargs["spark_conf"], {"spark.sql.shuffle.partitions": "4"})

                cls_mock.return_value.run.assert_called_once()
                run_kwargs = cls_mock.return_value.run.call_args.kwargs
                self.assertEqual(run_kwargs["data"], ["a.csv", "b.csv"])
                self.assertEqual(run_kwargs["sections"], ["data_quality", "univariate"])
                self.assertEqual(run_kwargs["columns"], ["amount", "channel"])
                self.assertEqual(run_kwargs["section_columns"]["data_quality"], ["amount"])
                self.assertFalse(run_kwargs["save_json"])
                self.assertFalse(run_kwargs["generate_report"])
                print_mock.assert_any_call("Done!")

    def test_main_interactive(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "prog",
                "--data",
                "input.csv",
                "--interactive",
                "--output",
                "out_dir",
            ],
        ):
            with mock.patch("eda_spark.cli.EDASpark") as cls_mock:
                with mock.patch("builtins.print"):
                    cli.main()
                cls_mock.return_value.run_interactive.assert_called_once()
                kwargs = cls_mock.return_value.run_interactive.call_args.kwargs
                self.assertEqual(kwargs["data"], ["input.csv"])
                self.assertTrue(kwargs["save_json"])
                self.assertTrue(kwargs["generate_report"])


if __name__ == "__main__":
    unittest.main()
