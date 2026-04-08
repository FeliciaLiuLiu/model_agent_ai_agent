"""Command-line interface (PySpark)."""
import argparse
import json
import sys

from .main import ModelTestingAgentSpark
from .interactive import InteractiveAgentSpark


def _parse_list(value):
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_options(values):
    if not values:
        return None
    options = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid JDBC option '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        options[key.strip()] = value.strip()
    return options


def _load_json_arg(value):
    if not value:
        return None
    try:
        if value.endswith(".json"):
            with open(value, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser(description="Model Testing Agent (PySpark)")
    parser.add_argument("--model", required=True, help="Model file (.joblib)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--data", help="Data file (.csv, .parquet)")
    source_group.add_argument("--sql", help="SQL query text or a .sql file path")
    source_group.add_argument("--loader-py", dest="loader_py", help="Python loader file path")
    parser.add_argument("--conn", default=None, help="Spark SQL JDBC URL when using external SQL data")
    parser.add_argument("--loader-fn", default="load_data", help="Function name inside the Python loader file")
    parser.add_argument(
        "--jdbc-option",
        action="append",
        default=None,
        help="Additional JDBC option in key=value form; can be provided multiple times",
    )
    parser.add_argument("--label_col", default=None, help="Label column")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--sections", default=None, help="Sections to run")
    parser.add_argument("--columns", default=None, help="Columns for all sections (comma-separated)")
    parser.add_argument("--columns-effectiveness", default=None, help="Columns for effectiveness section")
    parser.add_argument("--columns-efficiency", default=None, help="Columns for efficiency section")
    parser.add_argument("--columns-stability", default=None, help="Columns for stability section")
    parser.add_argument("--columns-interpretability", default=None, help="Columns for interpretability section")
    parser.add_argument("--segmentation", default=None, help="Segmentation config as JSON string or .json file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = ModelTestingAgentSpark.load_model(args.model)
    source_desc = args.data or args.sql or args.loader_py
    print(f"Loading data: {source_desc}")
    try:
        jdbc_options = _parse_options(args.jdbc_option)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    df, label_col, feature_cols = ModelTestingAgentSpark.load_data(
        path=args.data,
        label_col=args.label_col,
        sql=args.sql,
        conn=args.conn,
        loader_py=args.loader_py,
        loader_fn=args.loader_fn,
        jdbc_options=jdbc_options,
    )
    if label_col is None:
        print("Error: No label column detected.")
        sys.exit(1)

    if args.interactive:
        agent = InteractiveAgentSpark(output_dir=args.output)
        agent.run_interactive(model=model, df=df, label_col=label_col, feature_cols=feature_cols)
    else:
        sections = _parse_list(args.sections)
        columns = _parse_list(args.columns)
        section_columns = {
            "effectiveness": _parse_list(args.columns_effectiveness),
            "efficiency": _parse_list(args.columns_efficiency),
            "stability": _parse_list(args.columns_stability),
            "interpretability": _parse_list(args.columns_interpretability),
        }
        segmentation = _load_json_arg(args.segmentation) if args.segmentation else None
        agent = ModelTestingAgentSpark(output_dir=args.output)
        results = agent.run(
            model=model,
            df=df,
            label_col=label_col,
            feature_cols=feature_cols,
            sections=sections,
            threshold=args.threshold,
            columns=columns,
            section_columns=section_columns,
            segmentation=segmentation,
        )
        print(f"\nPDF: {agent.generate_report(results)}")
        print(f"JSON: {agent.save_results(results)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
