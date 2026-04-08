"""Command-line interface."""
import json
import argparse, sys
from .main import ModelTestingAgent
from .interactive import InteractiveAgent


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
    parser = argparse.ArgumentParser(description="Model Testing Agent")
    parser.add_argument('--model', required=True, help='Model file (.joblib)')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--data', help='Data file (.csv, .parquet, .xlsx, .xls)')
    source_group.add_argument('--sql', help='SQL query text or a .sql file path')
    source_group.add_argument('--loader-py', dest='loader_py', help='Python loader file path')
    parser.add_argument('--conn', default=None, help='Connection string used with --sql')
    parser.add_argument('--loader-fn', default='load_data', help='Function name inside the Python loader file')
    parser.add_argument('--label_col', default=None, help='Label column')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--sections', default=None, help='Sections to run')
    parser.add_argument('--columns', default=None, help='Columns for all sections (comma-separated)')
    parser.add_argument('--columns-effectiveness', default=None, help='Columns for effectiveness section')
    parser.add_argument('--columns-efficiency', default=None, help='Columns for efficiency section')
    parser.add_argument('--columns-stability', default=None, help='Columns for stability section')
    parser.add_argument('--columns-interpretability', default=None, help='Columns for interpretability section')
    parser.add_argument('--segmentation', default=None, help='Segmentation config as JSON string or .json file path')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = ModelTestingAgent.load_model(args.model)
    source_desc = args.data or args.sql or args.loader_py
    print(f"Loading data: {source_desc}")
    X, y, features = ModelTestingAgent.load_data(
        path=args.data,
        label_col=args.label_col,
        sql=args.sql,
        conn=args.conn,
        loader_py=args.loader_py,
        loader_fn=args.loader_fn,
    )
    if y is None: print("Error: No label column"); sys.exit(1)
    segmentation = _load_json_arg(args.segmentation) if args.segmentation else None

    if args.interactive:
        agent = InteractiveAgent(output_dir=args.output)
        agent.run_interactive(model=model, X=X, y=y, feature_names=features, segmentation=segmentation)
    else:
        sections = [s.strip() for s in args.sections.split(',')] if args.sections else None
        def parse_cols(value):
            if not value:
                return None
            return [c.strip() for c in value.split(',') if c.strip()]

        section_columns = {
            'effectiveness': parse_cols(args.columns_effectiveness),
            'efficiency': parse_cols(args.columns_efficiency),
            'stability': parse_cols(args.columns_stability),
            'interpretability': parse_cols(args.columns_interpretability),
        }
        columns = parse_cols(args.columns)
        agent = ModelTestingAgent(output_dir=args.output)
        results = agent.run(
            model=model,
            X=X,
            y=y,
            feature_names=features,
            sections=sections,
            threshold=args.threshold,
            columns=columns,
            section_columns=section_columns,
            segmentation=segmentation,
        )
        print(f"\nPDF: {agent.generate_report(results)}")
        print(f"JSON: {agent.save_results(results)}")
    print("\nDone!")


if __name__ == '__main__': main()
