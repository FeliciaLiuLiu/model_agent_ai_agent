"""Command-line interface (PySpark)."""
import argparse
import sys

from .main import ModelTestingAgentSpark
from .interactive import InteractiveAgentSpark


def _parse_list(value):
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Model Testing Agent (PySpark)")
    parser.add_argument("--model", required=True, help="Model file (.joblib)")
    parser.add_argument("--data", required=True, help="Data file (.csv, .parquet)")
    parser.add_argument("--label_col", default=None, help="Label column")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--sections", default=None, help="Sections to run")
    parser.add_argument("--columns", default=None, help="Columns for all sections (comma-separated)")
    parser.add_argument("--columns-effectiveness", default=None, help="Columns for effectiveness section")
    parser.add_argument("--columns-efficiency", default=None, help="Columns for efficiency section")
    parser.add_argument("--columns-stability", default=None, help="Columns for stability section")
    parser.add_argument("--columns-interpretability", default=None, help="Columns for interpretability section")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = ModelTestingAgentSpark.load_model(args.model)
    print(f"Loading data: {args.data}")
    df, label_col, feature_cols = ModelTestingAgentSpark.load_data(args.data, args.label_col)
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
        )
        print(f"\nPDF: {agent.generate_report(results)}")
        print(f"JSON: {agent.save_results(results)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
