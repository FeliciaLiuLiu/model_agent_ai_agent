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
        agent = ModelTestingAgent(output_dir=args.output)
        results = agent.run(model=model, X=X, y=y, feature_names=features, sections=sections, threshold=args.threshold)
        print(f"\nPDF: {agent.generate_report(results)}")
        print(f"JSON: {agent.save_results(results)}")
    print("\nDone!")


if __name__ == '__main__': main()
