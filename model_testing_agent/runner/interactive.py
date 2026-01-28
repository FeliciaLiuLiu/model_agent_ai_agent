"""Interactive CLI mode."""
import os
from typing import Dict, Any, List, Optional
import numpy as np
from ..core.report import ReportBuilder
from ..core.utils import get_feature_names
from ..matrices.effectiveness import ModelEffectiveness
from ..matrices.efficiency import ModelEfficiency
from ..matrices.stability import ModelStability
from ..matrices.interpretability import ModelInterpretability


class InteractiveAgent:
    """Interactive mode with step-by-step selection."""

    MATRICES = {
        1: ('effectiveness', 'Effectiveness (ROC, PR, CM, P/R/F1, KS, P@K/R@K)'),
        2: ('efficiency', 'Efficiency (FPR Analysis)'),
        3: ('stability', 'Stability (PSI, Data Drift, Concept Drift, CV, Bootstrap)'),
        4: ('interpretability', 'Interpretability (Perm Imp, SHAP, LIME, PDP, ICE)'),
    }

    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.effectiveness = ModelEffectiveness(data_dir=output_dir)
        self.efficiency = ModelEfficiency(data_dir=output_dir)
        self.stability = ModelStability(data_dir=output_dir)
        self.interpretability = ModelInterpretability(data_dir=output_dir)
        self.report_builder = ReportBuilder(output_dir=output_dir, tag="interactive")

    def run_interactive(self, model, X, y, feature_names=None) -> Dict[str, Any]:
        """Run interactive mode."""
        feature_names = get_feature_names(X, feature_names)

        print("\n" + "="*60)
        print("MODEL TESTING AGENT - INTERACTIVE MODE")
        print("="*60)

        selected = self._select_matrices()
        if not selected: return {}

        results = {}
        for key in selected:
            name, desc = self.MATRICES[key]
            print(f"\n{'='*60}\nConfiguring: {desc}\n{'='*60}")
            cols = self._select_columns(feature_names, name)

            print(f"\nRunning {name}...")
            X_sel = self._filter_columns(X, cols, feature_names) if cols else X
            fn_sel = [feature_names[i] for i in cols] if cols else feature_names

            if name == 'effectiveness':
                m, p = self.effectiveness.evaluate(model, X_sel, y)
                results['effectiveness'] = {'metrics': m, 'plots': p}
            elif name == 'efficiency':
                m, p = self.efficiency.evaluate(model, X_sel, y)
                results['efficiency'] = {'metrics': m, 'plots': p}
            elif name == 'stability':
                m, p, a = self.stability.evaluate(model, X_sel, y, feature_names=fn_sel)
                results['stability'] = {'metrics': m, 'plots': p, 'artifacts': a}
            elif name == 'interpretability':
                results['interpretability'] = self.interpretability.evaluate(model, X_sel, y, feature_names=fn_sel)

            print(f"Completed {name}!")

        print("\n" + "="*60 + "\nGENERATING REPORT\n" + "="*60)
        pdf = self.report_builder.build(results)
        print(f"\nPDF Report: {pdf}")
        self._print_summary(results)
        return results

    def _select_matrices(self) -> List[int]:
        print("\nAvailable Matrices:\n" + "-"*40)
        for k, (n, d) in self.MATRICES.items(): print(f"  {k}. {d}")
        print(f"  0. Select ALL\n" + "-"*40)
        while True:
            inp = input("\nEnter matrix numbers (e.g., 1,2,4 or 0): ").strip()
            if inp == '0': return list(self.MATRICES.keys())
            try:
                sel = [int(x.strip()) for x in inp.split(',')]
                valid = [x for x in sel if x in self.MATRICES]
                if valid: return valid
            except: pass
            print("Invalid input.")

    def _select_columns(self, feature_names, matrix_name) -> Optional[List[int]]:
        print(f"\nColumns for {matrix_name}:\n" + "-"*40)
        for i, n in enumerate(feature_names): print(f"  {i}. {n}")
        print(f"  a. ALL columns\n" + "-"*40)
        while True:
            inp = input("\nEnter column numbers (e.g., 0,1,5) or 'a': ").strip().lower()
            if inp == 'a': return None
            try:
                sel = [int(x.strip()) for x in inp.split(',')]
                valid = [x for x in sel if 0 <= x < len(feature_names)]
                if valid: return valid
            except: pass
            print("Invalid input.")

    def _filter_columns(self, X, cols, feature_names):
        if isinstance(X, np.ndarray): return X[:, cols]
        return X[[feature_names[i] for i in cols]]

    def _print_summary(self, results):
        print("\n" + "="*60 + "\nRESULTS SUMMARY\n" + "="*60)
        if 'effectiveness' in results:
            e = results['effectiveness']['metrics']
            print(f"\nEffectiveness: AUC-ROC={e.get('auc_roc',0):.4f}, F1={e.get('f1',0):.4f}, KS={e.get('ks_statistic',0):.4f}")
        if 'efficiency' in results:
            print(f"Efficiency: FPR={results['efficiency']['metrics'].get('fpr',0):.4f}")
        if 'stability' in results:
            s = results['stability']['metrics']
            print(f"Stability: PSI={s.get('psi',0):.4f}, CV={s.get('cv_auc_roc_mean',0):.4f}±{s.get('cv_auc_roc_std',0):.4f}")
        if 'interpretability' in results:
            top = results['interpretability'].get('metrics',{}).get('perm_top_features',[])[:5]
            print(f"Interpretability: Top features={top}")
        print("\n" + "="*60 + "\nDONE!\n" + "="*60)
