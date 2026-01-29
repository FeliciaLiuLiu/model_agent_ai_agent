"""Main runner: Non-Interactive mode (PySpark)."""
import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
from pyspark.sql import DataFrame

from ..core.report import ReportBuilder
from ..core.utils import load_model as _load_model, load_data as _load_data, get_spark
from ..matrices.effectiveness import ModelEffectivenessSpark
from ..matrices.efficiency import ModelEfficiencySpark
from ..matrices.stability import ModelStabilitySpark
from ..matrices.interpretability import ModelInterpretabilitySpark


class ModelTestingAgentSpark:
    """Main orchestrator for model testing (PySpark, non-interactive)."""

    SECTIONS = ["effectiveness", "efficiency", "stability", "interpretability"]

    def __init__(self, output_dir: str = "./output", experiment_tag: str = "model_testing_pyspark", spark=None):
        self.output_dir = output_dir
        self.experiment_tag = experiment_tag
        self.spark = spark or get_spark()
        os.makedirs(output_dir, exist_ok=True)
        self.effectiveness = ModelEffectivenessSpark(data_dir=output_dir)
        self.efficiency = ModelEfficiencySpark(data_dir=output_dir)
        self.stability = ModelStabilitySpark(data_dir=output_dir)
        self.interpretability = ModelInterpretabilitySpark(data_dir=output_dir)
        self.report_builder = ReportBuilder(output_dir=output_dir, tag=experiment_tag)

    def run(
        self,
        model,
        df: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        threshold: float = 0.5,
        columns: Optional[List[str]] = None,
        section_columns: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        sections = sections or self.SECTIONS
        feature_cols = feature_cols or [c for c in df.columns if c != label_col]

        def select_columns(df_in: DataFrame, cols: Optional[List[str]], feature_names_all: List[str]):
            if not cols:
                return df_in, feature_names_all
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",") if c.strip()]
            if all(isinstance(c, int) for c in cols):
                names = [feature_names_all[i] for i in cols]
            else:
                missing = [c for c in cols if c not in feature_names_all]
                if missing:
                    raise ValueError(f"Missing columns in dataset: {missing}")
                names = list(cols)
            df_sel = df_in.select(*(names + [label_col]))
            return df_sel, names

        results = {}
        section_columns = section_columns or {}

        if "effectiveness" in sections:
            cols = section_columns.get("effectiveness") or columns
            df_eff, feat_eff = select_columns(df, cols, feature_cols) if cols else (df, feature_cols)
            metrics, plots, explanations = self.effectiveness.evaluate(
                model, df_eff, label_col=label_col, feature_cols=feat_eff, threshold=threshold, **kwargs
            )
            results["effectiveness"] = {"metrics": metrics, "plots": plots, "explanations": explanations}

        if "efficiency" in sections:
            cols = section_columns.get("efficiency") or columns
            df_eff, feat_eff = select_columns(df, cols, feature_cols) if cols else (df, feature_cols)
            metrics, plots, explanations = self.efficiency.evaluate(
                model, df_eff, label_col=label_col, feature_cols=feat_eff, threshold=threshold, **kwargs
            )
            results["efficiency"] = {"metrics": metrics, "plots": plots, "explanations": explanations}

        if "stability" in sections:
            cols = section_columns.get("stability") or columns
            df_stab, feat_stab = select_columns(df, cols, feature_cols) if cols else (df, feature_cols)
            metrics, plots, artifacts, explanations = self.stability.evaluate(
                model, df_stab, label_col=label_col, feature_cols=feat_stab, **kwargs
            )
            results["stability"] = {"metrics": metrics, "plots": plots, "artifacts": artifacts, "explanations": explanations}

        if "interpretability" in sections:
            cols = section_columns.get("interpretability") or columns
            df_int, feat_int = select_columns(df, cols, feature_cols) if cols else (df, feature_cols)
            interp = self.interpretability.evaluate(
                model, df_int, label_col=label_col, feature_cols=feat_int, **kwargs
            )
            results["interpretability"] = interp

        return results

    def generate_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        return self.report_builder.build(results, filename=filename or "model_testing_agent_Model_Testing_Report.pdf")

    def save_results(self, results: Dict[str, Any], filename: str = "results.json") -> str:
        path = os.path.join(self.output_dir, filename)

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [convert(v) for v in obj]
            return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, indent=2)
        return path

    @staticmethod
    def load_model(path: str):
        return _load_model(path)

    @staticmethod
    def load_data(path: str, label_col: Optional[str] = None, spark=None):
        return _load_data(path, label_col=label_col, spark=spark)
