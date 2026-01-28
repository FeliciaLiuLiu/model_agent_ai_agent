"""Main runner: Non-Interactive mode."""
import os, json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from ..core.report import ReportBuilder
from ..matrices.effectiveness import ModelEffectiveness
from ..matrices.efficiency import ModelEfficiency
from ..matrices.stability import ModelStability
from ..matrices.interpretability import ModelInterpretability


class ModelTestingAgent:
    """
    Main orchestrator for model testing (Non-Interactive Mode).

    Usage:
        from adm_central_utility.model_testing_agent import ModelTestingAgent
        agent = ModelTestingAgent(output_dir='./output')
        results = agent.run(model=model, X=X, y=y)
        agent.generate_report(results)
    """

    SECTIONS = ['effectiveness', 'efficiency', 'stability', 'interpretability']

    def __init__(self, output_dir="./output", experiment_tag="model_testing"):
        self.output_dir = output_dir
        self.experiment_tag = experiment_tag
        os.makedirs(output_dir, exist_ok=True)
        self.effectiveness = ModelEffectiveness(data_dir=output_dir)
        self.efficiency = ModelEfficiency(data_dir=output_dir)
        self.stability = ModelStability(data_dir=output_dir)
        self.interpretability = ModelInterpretability(data_dir=output_dir)
        self.report_builder = ReportBuilder(output_dir=output_dir, tag=experiment_tag)

    def run(self, model, X, y, feature_names=None, sections=None, threshold=0.5, **kwargs) -> Dict[str, Any]:
        """Run model evaluation on all (or specified) sections."""
        sections = sections or self.SECTIONS
        feature_names = feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else [f'f_{i}' for i in range(X.shape[1])])

        results = {}

        if 'effectiveness' in sections:
            print("Running Effectiveness evaluation...")
            metrics, plots = self.effectiveness.evaluate(model, X, y, threshold=threshold, **kwargs)
            results['effectiveness'] = {'metrics': metrics, 'plots': plots}

        if 'efficiency' in sections:
            print("Running Efficiency evaluation...")
            metrics, plots = self.efficiency.evaluate(model, X, y, threshold=threshold, **kwargs)
            results['efficiency'] = {'metrics': metrics, 'plots': plots}

        if 'stability' in sections:
            print("Running Stability evaluation...")
            metrics, plots, artifacts = self.stability.evaluate(model, X, y, feature_names=feature_names, **kwargs)
            results['stability'] = {'metrics': metrics, 'plots': plots, 'artifacts': artifacts}

        if 'interpretability' in sections:
            print("Running Interpretability evaluation...")
            interp = self.interpretability.evaluate(model, X, y, feature_names=feature_names, **kwargs)
            results['interpretability'] = interp

        return results

    def generate_report(self, results: Dict[str, Any], filename=None) -> str:
        """Generate PDF report."""
        return self.report_builder.build(results, filename=filename or "model_testing_agent_Model_Testing_Report.pdf")

    def save_results(self, results: Dict[str, Any], filename="results.json") -> str:
        """Save results to JSON."""
        path = os.path.join(self.output_dir, filename)
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        return path

    @staticmethod
    def load_model(path: str):
        """Load model from file."""
        return joblib.load(path)

    @staticmethod
    def load_data(path: str, label_col=None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """Load dataset from file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv': df = pd.read_csv(path)
        elif ext == '.parquet': df = pd.read_parquet(path)
        elif ext in ['.xlsx', '.xls']: df = pd.read_excel(path)
        else: raise ValueError(f"Unsupported: {ext}")

        if not label_col:
            for name in ['label', 'target', 'y', 'class', 'fraud', 'is_fraud']:
                if name in df.columns: label_col = name; break

        if label_col and label_col in df.columns:
            return df.drop(columns=[label_col]), df[label_col], list(df.drop(columns=[label_col]).columns)
        return df, None, list(df.columns)
