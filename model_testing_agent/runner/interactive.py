"""Interactive CLI mode."""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.utils import get_feature_names
from .main import ModelTestingAgent


class InteractiveAgent:
    """Interactive mode with step-by-step selection."""

    MATRICES = {
        1: ("effectiveness", "Effectiveness (ROC, PR, CM, P/R/F1, KS, P@K/R@K)"),
        2: ("efficiency", "Efficiency (FPR Analysis)"),
        3: ("stability", "Stability (PSI, Data Drift, Concept Drift, CV, Bootstrap)"),
        4: ("interpretability", "Interpretability (Perm Imp, SHAP, LIME, PDP, ICE)"),
    }

    GROUPBY_MODES = {
        1: ("value", "Group by distinct values in a column"),
        2: ("time", "Group by time buckets derived from a datetime-like column"),
    }

    TIME_FREQS = {
        1: "day",
        2: "week",
        3: "month",
        4: "quarter",
        5: "year",
    }

    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.agent = ModelTestingAgent(output_dir=output_dir, experiment_tag="interactive")

    def run_interactive(self, model, X, y, feature_names=None, segmentation=None) -> Dict[str, Any]:
        """Run interactive mode."""
        feature_names = get_feature_names(X, feature_names)

        print("\n" + "=" * 60)
        print("MODEL TESTING AGENT - INTERACTIVE MODE")
        print("=" * 60)

        selected = self._select_matrices()
        if not selected:
            return {}

        sections = []
        section_columns = {}
        for key in selected:
            name, desc = self.MATRICES[key]
            sections.append(name)
            print(f"\n{'=' * 60}\nConfiguring: {desc}\n{'=' * 60}")
            cols = self._select_columns(feature_names, name)
            if cols:
                section_columns[name] = [feature_names[i] for i in cols]

        segmentation_cfg = self._select_segmentation(X, preset=segmentation)

        print("\n" + "=" * 60 + "\nRUNNING MODEL TESTING\n" + "=" * 60)
        results = self.agent.run(
            model=model,
            X=X,
            y=y,
            feature_names=feature_names,
            sections=sections,
            section_columns=section_columns or None,
            segmentation=segmentation_cfg,
        )

        print("\n" + "=" * 60 + "\nGENERATING REPORT\n" + "=" * 60)
        pdf = self.agent.generate_report(
            results, filename="model_testing_agent_Interactive_Model_Testing_Report.pdf"
        )
        json_path = self.agent.save_results(results, filename="interactive_results.json")
        print(f"\nPDF Report: {pdf}")
        print(f"JSON Results: {json_path}")
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

    def _select_columns(self, feature_names, matrix_name) -> Optional[List[int]]:
        print(f"\nColumns for {matrix_name}:\n" + "-" * 40)
        for i, name in enumerate(feature_names):
            print(f"  {i}. {name}")
        print("  a. ALL columns\n" + "-" * 40)
        while True:
            inp = input("\nEnter column numbers (e.g., 0,1,5) or 'a': ").strip().lower()
            if inp == "a":
                return None
            try:
                sel = [int(x.strip()) for x in inp.split(",")]
                valid = [x for x in sel if 0 <= x < len(feature_names)]
                if valid:
                    return valid
            except Exception:
                pass
            print("Invalid input.")

    def _prompt_yes_no(self, prompt: str, default: bool = False) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            inp = input(prompt + suffix).strip().lower()
            if not inp:
                return default
            if inp in {"y", "yes"}:
                return True
            if inp in {"n", "no"}:
                return False
            print("Invalid input.")

    def _prompt_int(self, prompt: str, default: int) -> int:
        while True:
            inp = input(f"{prompt} [{default}]: ").strip()
            if not inp:
                return default
            try:
                value = int(inp)
                if value > 0:
                    return value
            except Exception:
                pass
            print("Invalid input.")

    def _select_segmentation(self, X, preset=None) -> Optional[Dict[str, Any]]:
        if preset:
            print("\nUsing segmentation config provided by the caller.")
            return preset

        if not isinstance(X, pd.DataFrame):
            print("\nSegmentation is available only when X is a pandas DataFrame with named columns.")
            return None

        if not self._prompt_yes_no("Do you want to run segmented testing?", default=False):
            return None

        columns = list(X.columns)
        print("\nSegmentation columns:\n" + "-" * 40)
        for i, name in enumerate(columns):
            print(f"  {i}. {name}")
        print("-" * 40)
        while True:
            inp = input("\nEnter the segmentation column number: ").strip()
            try:
                idx = int(inp)
                if 0 <= idx < len(columns):
                    break
            except Exception:
                pass
            print("Invalid input.")

        column = columns[idx]
        include_overall = self._prompt_yes_no("Include overall results in the final report?", default=True)
        min_rows = self._prompt_int("Minimum rows required per segment", default=1)
        keep_column = self._prompt_yes_no(
            "Keep the segmentation column in model features?", default=False
        )

        print("\nSegmentation modes:\n" + "-" * 40)
        for key, (_, desc) in self.GROUPBY_MODES.items():
            print(f"  {key}. {desc}")
        print("-" * 40)
        while True:
            inp = input("\nEnter segmentation mode number: ").strip()
            try:
                mode_key = int(inp)
                if mode_key in self.GROUPBY_MODES:
                    break
            except Exception:
                pass
            print("Invalid input.")

        mode, _ = self.GROUPBY_MODES[mode_key]
        groupby_cfg: Dict[str, Any] = {"kind": mode}

        if mode == "time":
            print("\nTime grouping frequency:\n" + "-" * 40)
            for key, freq in self.TIME_FREQS.items():
                print(f"  {key}. {freq}")
            print("-" * 40)
            while True:
                inp = input("\nEnter frequency number: ").strip()
                try:
                    freq_key = int(inp)
                    if freq_key in self.TIME_FREQS:
                        break
                except Exception:
                    pass
                print("Invalid input.")
            groupby_cfg["freq"] = self.TIME_FREQS[freq_key]

        available_groups = self._available_groups(X, column, groupby_cfg)
        selected_groups = self._prompt_selected_groups(available_groups)
        if selected_groups:
            groupby_cfg["selected_groups"] = selected_groups

        return {
            "column": column,
            "mode": "groupby",
            "include_overall": include_overall,
            "min_rows": min_rows,
            "keep_column_in_features": keep_column,
            "groupby": groupby_cfg,
        }

    def _available_groups(self, X: pd.DataFrame, column: str, groupby_cfg: Dict[str, Any]) -> List[str]:
        if groupby_cfg["kind"] == "time":
            labels = self.agent._time_group_labels(X[column], groupby_cfg["freq"])
            groups = labels.dropna().drop_duplicates().astype("string").tolist()
        else:
            groups = X[column].dropna().astype("string").drop_duplicates().tolist()
        return sorted(str(group) for group in groups)

    def _prompt_selected_groups(self, available_groups: List[str]) -> Optional[List[str]]:
        if not available_groups:
            print("\nNo non-null groups were found for the selected segmentation column.")
            return []

        preview = ", ".join(available_groups[:10])
        suffix = " ..." if len(available_groups) > 10 else ""
        print(f"\nAvailable groups ({len(available_groups)} total): {preview}{suffix}")
        if self._prompt_yes_no("Run all available groups?", default=True):
            return None

        while True:
            raw = input(
                "Enter group labels to run, separated by commas (labels are case-sensitive): "
            ).strip()
            selected = [item.strip() for item in raw.split(",") if item.strip()]
            if selected:
                return selected
            print("Invalid input.")

    def _print_payload_summary(self, title: str, payload: Dict[str, Any]) -> None:
        if not payload:
            return

        print(f"\n{title}")
        if "effectiveness" in payload:
            metrics = payload["effectiveness"]["metrics"]
            print(
                f"  Effectiveness: AUC-ROC={metrics.get('auc_roc', 0):.4f}, "
                f"F1={metrics.get('f1', 0):.4f}, KS={metrics.get('ks_statistic', 0):.4f}"
            )
        if "efficiency" in payload:
            print(f"  Efficiency: FPR={payload['efficiency']['metrics'].get('fpr', 0):.4f}")
        if "stability" in payload:
            metrics = payload["stability"]["metrics"]
            print(
                f"  Stability: PSI={metrics.get('psi', 0):.4f}, "
                f"CV={metrics.get('cv_auc_roc_mean', 0):.4f}±{metrics.get('cv_auc_roc_std', 0):.4f}"
            )
        if "interpretability" in payload:
            top = payload["interpretability"].get("metrics", {}).get("perm_top_features", [])[:5]
            print(f"  Interpretability: Top features={top}")

    def _print_summary(self, results):
        print("\n" + "=" * 60 + "\nRESULTS SUMMARY\n" + "=" * 60)
        if "segmentation" not in results:
            self._print_payload_summary("Overall", results)
            print("\n" + "=" * 60 + "\nDONE!\n" + "=" * 60)
            return

        meta = results["segmentation"]
        print(
            f"\nSegmentation column: {meta.get('column')} "
            f"(mode={meta.get('mode', 'segments')}, "
            f"keep_column_in_features={meta.get('keep_column_in_features', False)})"
        )
        skipped = [item for item in meta.get("segments", []) if item.get("status") == "skipped"]
        if skipped:
            print("Skipped segments:")
            for item in skipped:
                print(f"  - {item['name']}: {item.get('reason', 'skipped')}")

        if "overall" in results:
            self._print_payload_summary("Overall", results["overall"])

        for name, payload in results.get("segments", {}).items():
            self._print_payload_summary(f"Segment: {name}", payload)

        print("\n" + "=" * 60 + "\nDONE!\n" + "=" * 60)
