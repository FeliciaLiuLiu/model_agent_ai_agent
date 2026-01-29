"""EDA command-line interface."""
import argparse
from typing import Dict, List, Optional

from .runner import EDA


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="EDA Agent")
    parser.add_argument("--data", default=None, help="Data file (.csv, .parquet, .xlsx). If omitted, auto-detects from ./data.")
    parser.add_argument("--output", default="./output_eda", help="Output directory")
    parser.add_argument("--target-col", default=None, help="Target column name")
    parser.add_argument("--time-col", default=None, help="Time column name")
    parser.add_argument("--id-cols", default=None, help="Comma-separated ID columns to exclude")
    parser.add_argument("--sections", default=None, help="Comma-separated EDA sections to run")
    parser.add_argument("--columns", default=None, help="Comma-separated columns for all sections")
    parser.add_argument("--columns-overview", default=None, help="Columns for overview section")
    parser.add_argument("--columns-missingness", default=None, help="Columns for missingness section")
    parser.add_argument("--columns-numeric", default=None, help="Columns for numeric section")
    parser.add_argument("--columns-categorical", default=None, help="Columns for categorical section")
    parser.add_argument("--columns-correlation", default=None, help="Columns for correlation section")
    parser.add_argument("--columns-outliers", default=None, help="Columns for outliers section")
    parser.add_argument("--columns-time", default=None, help="Columns for time section")
    parser.add_argument("--no-report", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON output")
    parser.add_argument("--report-name", default="EDA_Report.pdf", help="PDF report filename")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N rows for analysis")
    parser.add_argument("--interactive", action="store_true", help="Interactive selection mode")
    parser.add_argument("--list-functions", action="store_true", help="List available EDA functions and exit")
    args = parser.parse_args()

    if args.list_functions:
        EDA().print_functions()
        return

    id_cols = _parse_list(args.id_cols)
    sections = _parse_list(args.sections)
    columns = _parse_list(args.columns)

    section_columns: Dict[str, List[str]] = {
        "overview": _parse_list(args.columns_overview),
        "missingness": _parse_list(args.columns_missingness),
        "numeric": _parse_list(args.columns_numeric),
        "categorical": _parse_list(args.columns_categorical),
        "correlation": _parse_list(args.columns_correlation),
        "outliers": _parse_list(args.columns_outliers),
        "time": _parse_list(args.columns_time),
    }

    eda = EDA(
        output_dir=args.output,
        target_col=args.target_col,
        time_col=args.time_col,
        id_cols=id_cols,
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
