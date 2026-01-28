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

    def run(
        self,
        model,
        X,
        y,
        feature_names=None,
        sections=None,
        threshold=0.5,
        columns=None,
        section_columns=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run model evaluation on all (or specified) sections."""
        sections = sections or self.SECTIONS
        feature_names = feature_names or (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f'f_{i}' for i in range(X.shape[1])]
        )

        results = {}

        def select_columns(X_in, cols, feature_names_all):
            if not cols:
                return X_in, feature_names_all
            # Coerce to list if string provided
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(',') if c.strip()]

            if isinstance(X_in, pd.DataFrame):
                if all(isinstance(c, int) for c in cols):
                    X_sel = X_in.iloc[:, cols]
                else:
                    missing = [c for c in cols if c not in X_in.columns]
                    if missing:
                        raise ValueError(f"Missing columns in X: {missing}")
                    X_sel = X_in.loc[:, cols]
                return X_sel, list(X_sel.columns)

            X_arr = np.asarray(X_in)
            if X_arr.ndim == 1:
                raise ValueError("Cannot select columns from 1D array")
            if all(isinstance(c, int) for c in cols):
                idx = cols
            else:
                if feature_names_all is None:
                    raise ValueError("Column names provided but feature_names are not available for array inputs.")
                idx = []
                for c in cols:
                    if isinstance(c, int):
                        idx.append(c)
                    elif c in feature_names_all:
                        idx.append(feature_names_all.index(c))
                    else:
                        raise ValueError(f"Unknown column name: {c}")
            X_sel = X_arr[:, idx]
            names = [feature_names_all[i] for i in idx] if feature_names_all else [f'f_{i}' for i in idx]
            return X_sel, names

        section_columns = section_columns or {}

        if 'effectiveness' in sections:
            print("Running Effectiveness evaluation...")
            cols = section_columns.get('effectiveness') or columns
            X_eff, _ = select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots = self.effectiveness.evaluate(model, X_eff, y, threshold=threshold, **kwargs)
            results['effectiveness'] = {'metrics': metrics, 'plots': plots}

        if 'efficiency' in sections:
            print("Running Efficiency evaluation...")
            cols = section_columns.get('efficiency') or columns
            X_eff, _ = select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots = self.efficiency.evaluate(model, X_eff, y, threshold=threshold, **kwargs)
            results['efficiency'] = {'metrics': metrics, 'plots': plots}

        if 'stability' in sections:
            print("Running Stability evaluation...")
            cols = section_columns.get('stability') or columns
            X_stab, feature_names_stab = select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots, artifacts = self.stability.evaluate(
                model, X_stab, y, feature_names=feature_names_stab, **kwargs
            )
            results['stability'] = {'metrics': metrics, 'plots': plots, 'artifacts': artifacts}

        if 'interpretability' in sections:
            print("Running Interpretability evaluation...")
            cols = section_columns.get('interpretability') or columns
            X_int, feature_names_int = select_columns(X, cols, feature_names) if cols else (X, feature_names)
            interp = self.interpretability.evaluate(
                model, X_int, y, feature_names=feature_names_int, **kwargs
            )
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
