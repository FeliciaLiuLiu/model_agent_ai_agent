"""Spark EDA command-line interface."""
import argparse
from typing import Dict, List, Optional

from .runner import EDASpark


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_spark_conf(values: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not values:
        return None
    conf: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --spark-conf '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        conf[key.strip()] = value.strip()
    return conf


def main():
    parser = argparse.ArgumentParser(description="EDA Agent (Spark)")
    parser.add_argument(
        "--data",
        default=None,
        help="Data file (.csv, .parquet). If omitted, auto-detects latest mixed AML dataset from ./data.",
    )
    parser.add_argument("--output", default="./output_eda_spark", help="Output directory")
    parser.add_argument("--target-col", default=None, help="Target column name")
    parser.add_argument("--time-col", default=None, help="Time column name")
    parser.add_argument("--id-cols", default=None, help="Comma-separated ID columns to exclude")
    parser.add_argument("--sections", default=None, help="Comma-separated EDA sections to run")
    parser.add_argument("--columns", default=None, help="Comma-separated columns for all sections")
    parser.add_argument("--columns-data-quality", default=None, help="Columns for data quality section")
    parser.add_argument("--columns-target", default=None, help="Columns for target section")
    parser.add_argument("--columns-univariate", default=None, help="Columns for univariate section")
    parser.add_argument("--columns-bivariate-target", default=None, help="Columns for bivariate target section")
    parser.add_argument("--columns-feature-vs-feature", default=None, help="Columns for feature vs feature section")
    parser.add_argument("--columns-time-drift", default=None, help="Columns for time series and drift section")
    parser.add_argument("--columns-summary", default=None, help="Columns for summary section")
    parser.add_argument("--no-report", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON output")
    parser.add_argument("--report-name", default="EDA_Report.pdf", help="PDF report filename")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N rows for analysis")
    parser.add_argument("--interactive", action="store_true", help="Interactive selection mode")
    parser.add_argument("--list-functions", action="store_true", help="List available EDA functions and exit")
    parser.add_argument("--spark-master", default=None, help="Spark master URL (e.g., local[*])")
    parser.add_argument("--spark-app-name", default="EDASpark", help="Spark application name")
    parser.add_argument(
        "--spark-conf",
        action="append",
        default=None,
        help="Spark config key=value (repeatable)",
    )
    args = parser.parse_args()

    if args.list_functions:
        EDASpark().print_functions()
        return

    id_cols = _parse_list(args.id_cols)
    sections = _parse_list(args.sections)
    columns = _parse_list(args.columns)

    section_columns: Dict[str, List[str]] = {
        "data_quality": _parse_list(args.columns_data_quality),
        "target": _parse_list(args.columns_target),
        "univariate": _parse_list(args.columns_univariate),
        "bivariate_target": _parse_list(args.columns_bivariate_target),
        "feature_vs_feature": _parse_list(args.columns_feature_vs_feature),
        "time_drift": _parse_list(args.columns_time_drift),
        "summary": _parse_list(args.columns_summary),
    }

    eda = EDASpark(
        output_dir=args.output,
        target_col=args.target_col,
        time_col=args.time_col,
        id_cols=id_cols,
        spark_master=args.spark_master,
        spark_app_name=args.spark_app_name,
        spark_conf=_parse_spark_conf(args.spark_conf),
    )

    if args.interactive:
        eda.run_interactive(
            df=None,
            file_path=args.data,
            target_col=args.target_col,
            time_col=args.time_col,
            max_rows=args.max_rows,
            save_json=not args.no_json,
            generate_report=not args.no_report,
            report_name=args.report_name,
        )
    else:
        eda.run(
            df=None,
            file_path=args.data,
            sections=sections,
            columns=columns,
            section_columns=section_columns,
            target_col=args.target_col,
            time_col=args.time_col,
            max_rows=args.max_rows,
            save_json=not args.no_json,
            generate_report=not args.no_report,
            report_name=args.report_name,
        )

    if not args.no_report:
        print(f"PDF: {args.output}/{args.report_name}")
    if not args.no_json:
        print(f"JSON: {args.output}/eda_results.json")
    print("Done!")


if __name__ == "__main__":
    main()
