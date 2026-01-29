"""Interactive CLI mode (PySpark)."""
import os
from typing import Dict, Any, List, Optional

from ..core.report import ReportBuilder
from ..core.utils import get_numeric_columns
from ..matrices.effectiveness import ModelEffectivenessSpark
from ..matrices.efficiency import ModelEfficiencySpark
from ..matrices.stability import ModelStabilitySpark
from ..matrices.interpretability import ModelInterpretabilitySpark


class InteractiveAgentSpark:
    """Interactive mode with step-by-step selection (PySpark)."""

    MATRICES = {
        1: ("effectiveness", "Effectiveness (ROC, PR, CM, P/R/F1, KS, P@K/R@K)"),
        2: ("efficiency", "Efficiency (FPR Analysis)"),
        3: ("stability", "Stability (PSI, Data Drift, Concept Drift, CV, Bootstrap)"),
        4: ("interpretability", "Interpretability (Perm Imp, LIME, PDP, ICE)"),
    }

    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.effectiveness = ModelEffectivenessSpark(data_dir=output_dir)
        self.efficiency = ModelEfficiencySpark(data_dir=output_dir)
        self.stability = ModelStabilitySpark(data_dir=output_dir)
        self.interpretability = ModelInterpretabilitySpark(data_dir=output_dir)
        self.report_builder = ReportBuilder(output_dir=output_dir, tag="interactive_pyspark")

    def run_interactive(self, model, df, label_col: str, feature_cols: List[str]) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("MODEL TESTING AGENT (PYSPARK) - INTERACTIVE MODE")
        print("=" * 60)

        selected = self._select_matrices()
        if not selected:
            return {}

        results = {}
        numeric_cols = [c for c in feature_cols if c in get_numeric_columns(df)]

        for key in selected:
            name, desc = self.MATRICES[key]
            print(f"\n{'=' * 60}\nConfiguring: {desc}\n{'=' * 60}")
            if name == "interpretability" and numeric_cols:
                cols = self._select_columns(numeric_cols, name)
            else:
                cols = self._select_columns(feature_cols, name)

            df_sel = df.select(*(cols + [label_col])) if cols else df
            if cols:
                feat_sel = cols
            else:
                feat_sel = numeric_cols if (name == "interpretability" and numeric_cols) else feature_cols

            print(f"\nRunning {name}...")
            if name == "effectiveness":
                m, p, e = self.effectiveness.evaluate(model, df_sel, label_col=label_col, feature_cols=feat_sel)
                results["effectiveness"] = {"metrics": m, "plots": p, "explanations": e}
            elif name == "efficiency":
                m, p, e = self.efficiency.evaluate(model, df_sel, label_col=label_col, feature_cols=feat_sel)
                results["efficiency"] = {"metrics": m, "plots": p, "explanations": e}
            elif name == "stability":
                m, p, a, e = self.stability.evaluate(model, df_sel, label_col=label_col, feature_cols=feat_sel)
                results["stability"] = {"metrics": m, "plots": p, "artifacts": a, "explanations": e}
            elif name == "interpretability":
                results["interpretability"] = self.interpretability.evaluate(model, df_sel, label_col=label_col, feature_cols=feat_sel)

            print(f"Completed {name}!")

        print("\n" + "=" * 60 + "\nGENERATING REPORT\n" + "=" * 60)
        pdf = self.report_builder.build(results)
        print(f"\nPDF Report: {pdf}")
        self._print_summary(results)
        return results

    def _select_matrices(self) -> List[int]:
        print("\nAvailable Matrices:\n" + "-" * 40)
        for k, (_, d) in self.MATRICES.items():
            print(f"  {k}. {d}")
        print("  0. Select ALL\n" + "-" * 40)
        while True:
            inp = input("\nEnter matrix numbers (e.g., 1,2,4 or 0): ").strip()
            if inp == "0":
                return list(self.MATRICES.keys())
            try:
                sel = [int(x.strip()) for x in inp.split(",")]
                valid = [x for x in sel if x in self.MATRICES]
                if valid:
                    return valid
            except Exception:
                pass
            print("Invalid input.")

    def _select_columns(self, feature_cols, matrix_name) -> Optional[List[str]]:
        print(f"\nColumns for {matrix_name}:\n" + "-" * 40)
        for i, n in enumerate(feature_cols):
            print(f"  {i}. {n}")
        print("  a. ALL columns\n" + "-" * 40)
        while True:
            inp = input("\nEnter column numbers (e.g., 0,1,5) or 'a': ").strip().lower()
            if inp == "a":
                return None
            try:
                sel = [int(x.strip()) for x in inp.split(",")]
                valid = [x for x in sel if 0 <= x < len(feature_cols)]
                if valid:
                    return [feature_cols[i] for i in valid]
            except Exception:
                pass
            print("Invalid input.")

    def _print_summary(self, results):
        print("\n" + "=" * 60 + "\nRESULTS SUMMARY\n" + "=" * 60)
        if "effectiveness" in results:
            e = results["effectiveness"]["metrics"]
            print(f"\nEffectiveness: AUC-ROC={e.get('auc_roc', 0):.4f}, F1={e.get('f1', 0):.4f}, KS={e.get('ks_statistic', 0):.4f}")
        if "efficiency" in results:
            print(f"Efficiency: FPR={results['efficiency']['metrics'].get('fpr', 0):.4f}")
        if "stability" in results:
            s = results["stability"]["metrics"]
            print(f"Stability: PSI={s.get('psi', 0):.4f}, CV={s.get('cv_auc_roc_mean', 0):.4f}±{s.get('cv_auc_roc_std', 0):.4f}")
        if "interpretability" in results:
            top = results["interpretability"].get("metrics", {}).get("perm_top_features", [])[:5]
            print(f"Interpretability: Top features={top}")
        print("\n" + "=" * 60 + "\nDONE!\n" + "=" * 60)
