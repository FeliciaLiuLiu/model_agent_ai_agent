import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import write_png  # noqa: E402

try:
    from eda_spark.report import EDAReportBuilder
except Exception as exc:  # pragma: no cover
    EDAReportBuilder = None
    _REPORT_IMPORT_ERROR = exc
else:
    _REPORT_IMPORT_ERROR = None


class TestEDAReportBuilder(unittest.TestCase):
    def setUp(self):
        if EDAReportBuilder is None:
            raise unittest.SkipTest(f"report dependency not available: {_REPORT_IMPORT_ERROR}")
        self._tmp = tempfile.TemporaryDirectory(prefix="eda_spark_report_unittest_")
        self.tmp_path = Path(self._tmp.name)
        self.builder = EDAReportBuilder(output_dir=str(self.tmp_path), tag="eda_spark_test", max_table_rows=10)

    def tearDown(self):
        self._tmp.cleanup()

    def test_format_helpers(self):
        self.assertEqual(self.builder._format_number(None), "")
        self.assertEqual(self.builder._format_number(10), "10")
        self.assertEqual(self.builder._format_number(10.4567), "10.457")
        self.assertEqual(self.builder._format_percent(0.25), "25.00%")
        self.assertEqual(self.builder._format_percent(25), "25.00%")
        self.assertEqual(self.builder._format_cell(0.1, "missing_rate"), "10.00%")
        self.assertEqual(self.builder._format_cell(1000, "count"), "1,000")

    def test_summary_and_truncate(self):
        self.assertIsNone(self.builder._summary_list([]))
        self.assertIsNotNone(self.builder._summary_list(["a", "b"]))
        self.assertEqual(self.builder._truncate_columns(["a", "b"], max_cols=5), "a, b")
        self.assertIn("+2 more", self.builder._truncate_columns(["a", "b", "c", "d"], max_cols=2))

    def test_table_helpers(self):
        self.assertIsNone(self.builder._table_from_def({"headers": [], "rows": []}))
        self.assertIsNone(self.builder._table_from_def({"headers": ["a"], "rows": []}))
        table = self.builder._table_from_def(
            {
                "headers": ["Column", "Rate"],
                "rows": [["x", 0.1], ["y", 0.2]],
            }
        )
        self.assertIsNotNone(table)

        widths = self.builder._coerce_col_widths([1, 2, 3], available_width=200)
        self.assertIsNotNone(widths)
        self.assertEqual(len(widths), 3)
        self.assertIsNone(self.builder._coerce_col_widths("bad", available_width=200))

        calc_widths = self.builder._column_widths(["A", "B"], [["v1", 1], ["v2", 2]], available_width=120)
        self.assertEqual(len(calc_widths), 2)

        align = self.builder._infer_alignments(["count", "name"], [[1, "x"], [2, "y"]], 2)
        self.assertEqual(align[0], "RIGHT")
        self.assertEqual(align[1], "LEFT")
        self.assertTrue(self.builder._is_numeric_column([1, "2", "3.5"]))
        self.assertFalse(self.builder._is_numeric_column([1, "x"]))

    def test_chart_grid_and_scaled_image(self):
        image_path = self.tmp_path / "chart.png"
        write_png(image_path)

        table = self.builder._chart_grid([{"title": "demo", "path": str(image_path)}], available_width=400)
        self.assertIsNotNone(table)

        empty = self.builder._chart_grid([], available_width=400)
        self.assertIsNotNone(empty)

        img = self.builder._scaled_image(str(image_path), target_width=120)
        self.assertIsNotNone(img)

    def test_cover_and_render_helpers(self):
        image_path = self.tmp_path / "plot.png"
        write_png(image_path)
        elements = []
        config = {
            "data_path": "demo.csv",
            "rows_used": 10,
            "target_col": "sar_actual",
            "time_col": "txn_datetime",
        }
        cover = self.builder._cover_section(config)
        self.assertTrue(cover)

        payload_dq = {
            "summary": ["Dataset has 10 rows."],
            "metrics": {
                "missingness_payload": {
                    "missing_columns": [{"column": "c1", "missing_count": 2, "missing_rate": 0.2}],
                    "non_missing_columns": ["c2", "c3"],
                },
                "null_like_payload": [
                    {"column": "c1", "null_like_count": 2, "null_like_rate": 0.2, "examples": ["na", "null"]}
                ],
            },
            "tables": [{"title": "Column Type Classification", "headers": ["Type", "Columns"], "rows": [["numeric", "x"]]}],
            "plots": {"missingness": str(image_path)},
        }
        self.builder._render_data_quality_section(elements, payload_dq, doc_width=500)
        self.assertTrue(elements)

        elements_uni = []
        payload_uni = {
            "summary": ["Univariate summary."],
            "tables": [
                {
                    "title": "Numeric Summary Statistics",
                    "headers": ["Column", "Mean"],
                    "rows": [["amount", 10.0]],
                    "style": "wide_numeric_stats",
                },
                {
                    "title": "Top K: channel",
                    "headers": ["Category", "Count", "Rate"],
                    "rows": [["wire", 5, 0.5]],
                    "style": "categorical_topk",
                },
            ],
            "plots": {"hist_amount": str(image_path)},
            "univariate_payload": {"chart_paths": [{"title": "amount distribution", "path": str(image_path)}]},
        }
        self.builder._render_univariate_summary(elements_uni, payload_uni, doc_width=500)
        self.assertTrue(elements_uni)

    def test_build_and_build_pdf(self):
        image_path = self.tmp_path / "img.png"
        write_png(image_path)

        results = {
            "data_quality": {
                "summary": ["dq ok"],
                "metrics": {
                    "missingness_payload": {"missing_columns": [], "non_missing_columns": ["a"]},
                    "null_like_payload": [],
                },
                "tables": [{"title": "Column Type Classification", "headers": ["Type", "Columns"], "rows": [["numeric", "a"]]}],
                "plots": {"missingness": str(image_path)},
            },
            "univariate": {
                "summary": ["uni ok"],
                "tables": [
                    {
                        "title": "Numeric Summary Statistics",
                        "headers": ["Column", "Mean"],
                        "rows": [["amount", 123.0]],
                        "style": "wide_numeric_stats",
                    }
                ],
                "plots": {"hist_amount": str(image_path)},
                "univariate_payload": {"chart_paths": [{"title": "amount", "path": str(image_path)}]},
            },
        }
        config = {"data_path": "demo.csv", "rows_used": 10, "target_col": "sar_actual", "time_col": "txn_datetime"}
        out1 = self.builder.build(results, skipped_sections=[], config=config, filename="report1.pdf")
        self.assertTrue(Path(out1).exists())

        out2 = str(self.tmp_path / "report2.pdf")
        out2_path = self.builder.build_pdf(
            {"results": results, "skipped_sections": [{"section": "target", "reason": "missing"}], "config": config},
            output_path=out2,
        )
        self.assertTrue(Path(out2_path).exists())


if __name__ == "__main__":
    unittest.main()
