"""Command-line interface."""
import argparse, sys
from .main import ModelTestingAgent
from .interactive import InteractiveAgent


def main():
    parser = argparse.ArgumentParser(description="Model Testing Agent")
    parser.add_argument('--model', required=True, help='Model file (.joblib)')
    parser.add_argument('--data', required=True, help='Data file (.csv, .parquet)')
    parser.add_argument('--label_col', default=None, help='Label column')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--sections', default=None, help='Sections to run')
    parser.add_argument('--columns', default=None, help='Columns for all sections (comma-separated)')
    parser.add_argument('--columns-effectiveness', default=None, help='Columns for effectiveness section')
    parser.add_argument('--columns-efficiency', default=None, help='Columns for efficiency section')
    parser.add_argument('--columns-stability', default=None, help='Columns for stability section')
    parser.add_argument('--columns-interpretability', default=None, help='Columns for interpretability section')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = ModelTestingAgent.load_model(args.model)
    print(f"Loading data: {args.data}")
    X, y, features = ModelTestingAgent.load_data(args.data, args.label_col)
    if y is None: print("Error: No label column"); sys.exit(1)

    if args.interactive:
        agent = InteractiveAgent(output_dir=args.output)
        agent.run_interactive(model=model, X=X, y=y, feature_names=features)
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
        )
        print(f"\nPDF: {agent.generate_report(results)}")
        print(f"JSON: {agent.save_results(results)}")
    print("\nDone!")


if __name__ == '__main__': main()
