"""EDA command-line interface."""
import argparse
from typing import Dict, List, Optional

from .runner import EDA


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_multi_data(value) -> Optional[List[str]]:
    if not value:
        return None
    items: List[str] = []
    if isinstance(value, list):
        for v in value:
            items.extend([item.strip() for item in str(v).split(",") if item.strip()])
    else:
        items.extend([item.strip() for item in str(value).split(",") if item.strip()])
    return items or None


def main():
    parser = argparse.ArgumentParser(description="EDA Agent")
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="Data file/dir/glob. Repeatable. If omitted, auto-loads from ./data.",
    )
    parser.add_argument("--sql", default=None, help="SQL query to load data")
    parser.add_argument("--db", default=None, help="Database connection string for --sql")
    parser.add_argument("--py", default=None, help="Python file with load() or df variable")
    parser.add_argument("--py-code", default=None, help="Inline Python code that defines load() or df")
    parser.add_argument("--nb", default=None, help="Notebook (.ipynb) file with load() or df")
    parser.add_argument("--data-recursive", action="store_true", help="Recursively search directories/globs for data files")
    parser.add_argument(
        "--auto-exec",
        action="store_true",
        help="Auto-execute .sql/.py/.ipynb in ./data when no input is provided",
    )
    parser.add_argument("--output", default="./output_eda", help="Output directory")
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
    parser.add_argument("--spark", action="store_true", help="Use Spark implementation (requires pyspark)")
    args = parser.parse_args()

    if args.list_functions:
        EDA().print_functions()
        return

    id_cols = _parse_list(args.id_cols)
    sections = _parse_list(args.sections)
    columns = _parse_list(args.columns)
    data_inputs = _parse_multi_data(args.data)
    auto_exec = True if args.auto_exec else None

    section_columns: Dict[str, List[str]] = {
        "data_quality": _parse_list(args.columns_data_quality),
        "target": _parse_list(args.columns_target),
        "univariate": _parse_list(args.columns_univariate),
        "bivariate_target": _parse_list(args.columns_bivariate_target),
        "feature_vs_feature": _parse_list(args.columns_feature_vs_feature),
        "time_drift": _parse_list(args.columns_time_drift),
        "summary": _parse_list(args.columns_summary),
    }

    spark_mode = args.spark
    if spark_mode:
        try:
            from .spark_runner import EDASpark  # type: ignore
        except Exception as exc:
            raise RuntimeError("PySpark is required for --spark. Install pyspark and retry.") from exc
        eda = EDASpark(
            output_dir=args.output,
            target_col=args.target_col,
            time_col=args.time_col,
            id_cols=id_cols,
        )
    else:
        eda = EDA(
            output_dir=args.output,
            target_col=args.target_col,
            time_col=args.time_col,
            id_cols=id_cols,
        )

    if spark_mode:
        if args.auto_exec:
            raise RuntimeError("--auto-exec is only supported in pandas mode (no --spark).")
        if args.sql or args.py or args.py_code or args.nb or args.db or args.data_recursive:
            raise RuntimeError("SQL/Python/Notebook inputs are only supported in pandas mode (no --spark).")
        if data_inputs and len(data_inputs) > 1:
            raise RuntimeError("Spark mode supports a single --data path (use a glob or directory).")
        spark_path = data_inputs[0] if data_inputs else None
        if args.interactive:
            eda.run_interactive(
                df=None,
                file_path=spark_path,
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
                file_path=spark_path,
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
    else:
        if args.interactive:
            eda.run_interactive(
                df=None,
                file_path=None,
                data=data_inputs,
                sql=args.sql,
                db=args.db,
                py=args.py,
                py_code=args.py_code,
                nb=args.nb,
                recursive=args.data_recursive,
                auto_exec=auto_exec,
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
                file_path=None,
                data=data_inputs,
                sql=args.sql,
                db=args.db,
                py=args.py,
                py_code=args.py_code,
                nb=args.nb,
                recursive=args.data_recursive,
                auto_exec=auto_exec,
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
