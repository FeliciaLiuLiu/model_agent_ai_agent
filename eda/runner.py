"""EDA main runner (Pandas)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .report import EDAReportBuilder
from .utils import (
    detect_latest_dataset,
    ensure_datetime,
    infer_column_types,
    is_time_col_clean,
    load_data,
    pick_time_column,
)


class EDA:
    """Exploratory Data Analysis utility (Pandas)."""

    SECTION_INFO = [
        {
            "key": "data_quality",
            "title": "Data Quality",
            "description": "Rows, columns, duplicates, missingness, column type classification, and basic validity checks.",
            "applicable_columns": "all",
        },
        {
            "key": "target",
            "title": "Target / Label EDA",
            "description": "Target distribution, label rate over time, and label rate by categorical dimensions.",
            "applicable_columns": "target_col",
        },
        {
            "key": "univariate",
            "title": "Univariate",
            "description": "Numeric summary statistics, histograms, and categorical frequency tables.",
            "applicable_columns": "numeric/categorical",
        },
        {
            "key": "bivariate_target",
            "title": "Bivariate with Target",
            "description": "Numeric and categorical features vs target (binning and rates).",
            "applicable_columns": "numeric/categorical",
        },
        {
            "key": "feature_vs_feature",
            "title": "Feature vs Feature",
            "description": "Numeric correlation heatmap and high-correlation pairs.",
            "applicable_columns": "numeric",
        },
        {
            "key": "time_drift",
            "title": "Time Series and Drift",
            "description": "Time bucket trends, PSI drift, and categorical drift (requires time column).",
            "applicable_columns": "time_col/numeric/categorical",
        },
        {
            "key": "summary",
            "title": "Summary and Recommendations",
            "description": "Highlights of quality issues and recommended next steps.",
            "applicable_columns": "all",
        },
    ]

    SECTION_KEYS = [s["key"] for s in SECTION_INFO]

    def __init__(
        self,
        output_dir: str = "./output_eda",
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        tag: str = "eda",
        plot_dpi: int = 80,
        fig_size: Tuple[float, float] = (6.0, 3.6),
        heatmap_size: Tuple[float, float] = (6.5, 4.5),
        time_parse_min_ratio: float = 0.9,
        max_numeric_cols: int = 10,
        max_categorical_cols: int = 10,
        max_plots: int = 10,
        top_k_categories: int = 10,
        sample_frac: Optional[float] = None,
        sample_seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.target_col = target_col
        self.time_col = time_col
        self.id_cols = id_cols or []
        self.plot_dpi = plot_dpi
        self.fig_size = fig_size
        self.heatmap_size = heatmap_size
        self.time_parse_min_ratio = time_parse_min_ratio
        self.max_numeric_cols = max_numeric_cols
        self.max_categorical_cols = max_categorical_cols
        self.max_plots = max_plots
        self.top_k_categories = top_k_categories
        self.sample_frac = sample_frac
        self.sample_seed = sample_seed
        os.makedirs(output_dir, exist_ok=True)
        self.report_builder = EDAReportBuilder(output_dir=output_dir, tag=tag)

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        sections: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        section_columns: Optional[Dict[str, List[str]]] = None,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        max_rows: Optional[int] = None,
        save_json: bool = True,
        generate_report: bool = True,
        report_name: str = "EDA_Report.pdf",
        data_dir: str = "./data",
        return_payload: bool = False,
        sample_frac: Optional[float] = None,
        sample_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run EDA on a dataset and optionally generate a report."""
        if df is None:
            path = file_path or detect_latest_dataset(data_dir=data_dir)
            df = load_data(path)
        else:
            path = file_path

        rows_original = int(df.shape[0])
        sample_frac = sample_frac if sample_frac is not None else self.sample_frac
        sample_seed = sample_seed if sample_seed is not None else self.sample_seed
        if sample_frac is None and path and "synthetic_aml_200k_" in os.path.basename(path):
            sample_frac = 0.05
        if sample_frac is not None:
            if not (0 < sample_frac <= 1):
                raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
            df = df.sample(frac=sample_frac, random_state=sample_seed)
        if max_rows and rows_original > max_rows:
            df = df.head(max_rows).copy()

        target_col = target_col or self.target_col
        col_types = infer_column_types(df, id_cols=self.id_cols)
        time_col = time_col or self.time_col or pick_time_column(df, col_types, self.time_parse_min_ratio)
        section_columns = section_columns or {}

        time_clean, time_ratio = (False, 0.0)
        if time_col:
            time_clean, time_ratio = is_time_col_clean(df, time_col, min_valid_ratio=self.time_parse_min_ratio)

        context = {
            "df": df,
            "data_path": path or "",
            "rows_original": rows_original,
            "target_col": target_col,
            "time_col": time_col,
            "time_clean": time_clean,
            "time_ratio": time_ratio,
            "col_types": col_types,
        }

        results: Dict[str, Any] = {}
        skipped: List[Dict[str, Any]] = []
        run_sections = [s for s in (sections or self.SECTION_KEYS) if s in self.SECTION_KEYS]

        for sec in run_sections:
            req_ok, reason = self._check_prerequisites(sec, context)
            if not req_ok:
                skipped.append({"section": sec, "reason": reason})
                continue

            cols_requested = section_columns.get(sec) if sec in section_columns else columns
            applicable = self._applicable_columns_for_section(sec, context)
            cols = self._filter_selected_columns(cols_requested, applicable)
            if cols_requested is not None and not cols and applicable:
                skipped.append({"section": sec, "reason": "No applicable columns selected."})
                continue

            if sec == "data_quality":
                results[sec] = self._section_data_quality(context)
            elif sec == "target":
                results[sec] = self._section_target(context)
            elif sec == "univariate":
                results[sec] = self._section_univariate(context, cols)
            elif sec == "bivariate_target":
                results[sec] = self._section_bivariate_target(context, cols)
            elif sec == "feature_vs_feature":
                results[sec] = self._section_feature_vs_feature(context, cols)
            elif sec == "time_drift":
                results[sec] = self._section_time_drift(context, cols)
            elif sec == "summary":
                results[sec] = self._section_summary(context)

        payload = {
            "results": results,
            "skipped_sections": skipped,
            "config": {
                "data_path": path or "",
            "rows_original": rows_original,
            "rows_used": int(df.shape[0]),
            "target_col": target_col or "",
            "time_col": time_col or "",
            "time_parse_ratio": round(float(time_ratio), 4),
            "sample_frac": sample_frac if sample_frac is not None else "",
        },
    }

        if save_json:
            json_path = os.path.join(self.output_dir, "eda_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._convert(payload), f, indent=2)

        if generate_report:
            self.report_builder.build(
                results,
                skipped_sections=skipped,
                config=payload["config"],
                filename=report_name,
            )

        return payload if return_payload else results

    def run_interactive(
        self,
        df: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        max_rows: Optional[int] = None,
        save_json: bool = True,
        generate_report: bool = True,
        report_name: str = "EDA_Report.pdf",
        data_dir: str = "./data",
        return_payload: bool = False,
        sample_frac: Optional[float] = None,
        sample_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Interactive selection of sections and columns."""
        if df is None:
            path = file_path or detect_latest_dataset(data_dir=data_dir)
            df = load_data(path)
        else:
            path = file_path

        sample_frac = sample_frac if sample_frac is not None else self.sample_frac
        sample_seed = sample_seed if sample_seed is not None else self.sample_seed
        if sample_frac is None and path and "synthetic_aml_200k_" in os.path.basename(path):
            sample_frac = 0.05
        if sample_frac is not None:
            if not (0 < sample_frac <= 1):
                raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
            df = df.sample(frac=sample_frac, random_state=sample_seed)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows).copy()

        target_col = target_col or self.target_col
        col_types = infer_column_types(df, id_cols=self.id_cols)
        time_col = time_col or self.time_col or pick_time_column(df, col_types, self.time_parse_min_ratio)

        time_clean, time_ratio = (False, 0.0)
        if time_col:
            time_clean, time_ratio = is_time_col_clean(df, time_col, self.time_parse_min_ratio)

        context = {
            "df": df,
            "data_path": path or "",
            "rows_original": int(df.shape[0]),
            "target_col": target_col,
            "time_col": time_col,
            "time_clean": time_clean,
            "time_ratio": time_ratio,
            "col_types": col_types,
        }

        self.print_functions()
        selection = input("Select functions by number (e.g., 1,2,3) or 'all': ").strip()
        sections = self.parse_function_selection(selection)
        if not sections:
            sections = None

        section_columns: Dict[str, List[str]] = {}
        for sec in sections or self.SECTION_KEYS:
            applicable = self._applicable_columns_for_section(sec, context)
            if not applicable:
                continue
            print(f"\nSection '{sec}' columns ({len(applicable)}):")
            self._print_numbered_list(applicable, max_items=60)
            cols_raw = input("Select columns by number/name (comma-separated), or press Enter for default: ").strip()
            cols = self.parse_column_selection(cols_raw, applicable)
            if cols:
                section_columns[sec] = cols

        return self.run(
            df=df,
            file_path=path,
            sections=sections,
            section_columns=section_columns,
            target_col=target_col,
            time_col=time_col,
            max_rows=max_rows,
            save_json=save_json,
            generate_report=generate_report,
            report_name=report_name,
            data_dir=data_dir,
            return_payload=return_payload,
            sample_frac=sample_frac,
            sample_seed=sample_seed,
        )

    @classmethod
    def list_functions(cls) -> List[Dict[str, str]]:
        """Return EDA function catalog."""
        return [dict(info) for info in cls.SECTION_INFO]

    @classmethod
    def parse_function_selection(cls, selection: Optional[str]) -> Optional[List[str]]:
        """Parse section selection by number or name."""
        if not selection:
            return None
        if selection.strip().lower() == "all":
            return None
        items = [s.strip() for s in selection.split(",") if s.strip()]
        keys = [s["key"] for s in cls.SECTION_INFO]
        if all(item.isdigit() for item in items):
            idxs = [int(i) for i in items]
            return [keys[i - 1] for i in idxs if 1 <= i <= len(keys)]
        return [item for item in items if item in keys]

    @classmethod
    def parse_column_selection(cls, selection: str, options: List[str]) -> Optional[List[str]]:
        """Parse column selection by number or name."""
        if not selection:
            return None
        if selection.strip().lower() == "all":
            return options
        items = [s.strip() for s in selection.split(",") if s.strip()]
        if not items:
            return None
        if all(item.isdigit() for item in items):
            idxs = [int(i) for i in items]
            return [options[i - 1] for i in idxs if 1 <= i <= len(options)]
        return [item for item in items if item in options]

    def print_functions(self) -> None:
        """Print EDA functions with numbering."""
        for idx, info in enumerate(self.SECTION_INFO, start=1):
            print(f"{idx}. {info['title']} ({info['key']}): {info['description']} | columns: {info['applicable_columns']}")

    def _applicable_columns_for_section(self, section: str, context: Dict[str, Any]) -> List[str]:
        df = context["df"]
        col_types = context["col_types"]
        target_col = context.get("target_col")
        numeric = col_types["numeric"]
        categorical = col_types["categorical"] + col_types["boolean"]
        text = col_types.get("text", [])
        if section in ("data_quality", "summary"):
            return list(df.columns)
        if section in ("univariate", "bivariate_target", "time_drift"):
            cols = list(dict.fromkeys(numeric + categorical + text))
            return [c for c in cols if c != target_col]
        if section == "feature_vs_feature":
            return numeric
        return []

    def _filter_selected_columns(self, requested: Optional[List[str]], applicable: List[str]) -> Optional[List[str]]:
        if not requested:
            return None
        return [c for c in requested if c in applicable]

    def _check_prerequisites(self, section: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        col_types = context["col_types"]
        target_col = context.get("target_col")
        time_col = context.get("time_col")
        time_clean = context.get("time_clean")
        numeric = col_types["numeric"]
        categorical = col_types["categorical"] + col_types["boolean"]

        if section == "target" and (not target_col or target_col not in context["df"].columns):
            return False, "Target column is missing."
        if section == "bivariate_target" and (not target_col or target_col not in context["df"].columns):
            return False, "Target column is missing."
        if section == "feature_vs_feature" and len(numeric) < 2:
            return False, "Need at least 2 numeric columns for correlation."
        if section == "univariate" and not (numeric or categorical):
            return False, "No numeric or categorical columns available."
        if section == "time_drift":
            if not time_col:
                return False, "Time column is missing."
            if not time_clean:
                ratio = context.get("time_ratio", 0.0)
                return False, f"Time column parse ratio {ratio:.2%} below threshold."
        return True, ""

    def _select_numeric(self, df: pd.DataFrame, col_types: Dict[str, List[str]], cols: Optional[List[str]]) -> List[str]:
        numeric = col_types["numeric"]
        if cols:
            return [c for c in cols if c in numeric]
        if len(numeric) <= self.max_numeric_cols:
            return numeric
        variances = df[numeric].var().sort_values(ascending=False)
        return list(variances.head(self.max_numeric_cols).index)

    def _select_categorical(self, df: pd.DataFrame, col_types: Dict[str, List[str]], cols: Optional[List[str]]) -> List[str]:
        categorical = col_types["categorical"] + col_types["boolean"]
        if cols:
            return [c for c in cols if c in categorical]
        if len(categorical) <= self.max_categorical_cols:
            return categorical
        nunique = df[categorical].nunique(dropna=True).sort_values(ascending=False)
        return list(nunique.head(self.max_categorical_cols).index)

    def _section_data_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        duplicate_ratio = float(df.duplicated().mean())
        metrics["rows"] = int(df.shape[0])
        metrics["columns"] = int(df.shape[1])
        metrics["duplicate_ratio"] = round(duplicate_ratio, 6)

        if self.id_cols:
            id_cols = [c for c in self.id_cols if c in df.columns]
            if id_cols:
                dup_ids = df.duplicated(subset=id_cols).mean()
                metrics["duplicate_id_ratio"] = round(float(dup_ids), 6)

        missing_count = df.isna().sum()
        missing_rate = (missing_count / max(1, len(df))).round(6)
        missing_table = [
            [col, int(missing_count[col]), float(missing_rate[col])]
            for col in missing_rate.sort_values(ascending=False).index
        ]
        tables.append({
            "title": "Missingness",
            "headers": ["Column", "Missing Count", "Missing Rate"],
            "rows": missing_table,
        })

        if not missing_rate.empty:
            top_missing = missing_rate.head(min(20, len(missing_rate)))
            path = os.path.join(self.output_dir, "missingness.png")
            self._plot_bar(top_missing, path, title="Missing Rate (Top)", ylabel="Missing Rate")
            plots["missingness"] = path

        type_rows = []
        for type_name in ["numeric", "categorical", "datetime", "boolean", "text", "other"]:
            cols = col_types.get(type_name, [])
            for col in cols:
                type_rows.append([type_name, col])
        tables.append({
            "title": "Column Type Classification",
            "headers": ["Type", "Columns"],
            "rows": type_rows,
        })

        numeric_cols = col_types["numeric"]
        invalid_rows = []
        for col in numeric_cols:
            if any(tok in col.lower() for tok in ["amount", "balance", "count", "num"]):
                neg_ratio = float((df[col] < 0).mean())
                if neg_ratio > 0:
                    invalid_rows.append([col, round(neg_ratio, 6)])
        if invalid_rows:
            tables.append({
                "title": "Potential Invalid Values (Negative Ratios)",
                "headers": ["Column", "Negative Ratio"],
                "rows": invalid_rows,
            })
            summary.append("Some numeric columns contain negative values where non-negative is expected.")

        outlier_rows = []
        for col in self._select_numeric(df, col_types, None):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            ratio = float(((df[col] < lower) | (df[col] > upper)).mean())
            outlier_rows.append([col, round(ratio, 6)])
        if outlier_rows:
            outlier_rows.sort(key=lambda x: x[1], reverse=True)
            tables.append({
                "title": "Outlier Ratio (IQR)",
                "headers": ["Column", "Outlier Ratio"],
                "rows": outlier_rows[: min(20, len(outlier_rows))],
            })

        summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        summary.append(f"Duplicate row ratio: {duplicate_ratio:.2%}.")
        if rows := context.get("rows_original"):
            if rows > df.shape[0]:
                summary.append(f"Rows limited to {df.shape[0]} from {rows} for analysis.")
        if missing_rate.max() > 0:
            summary.append(
                f"Highest missing rate is {missing_rate.max():.2%} in '{missing_rate.idxmax()}'."
            )

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_target(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context["df"]
        target_col = context.get("target_col")
        time_col = context.get("time_col")
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if not target_col or target_col not in df.columns:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Target column not found."]}

        target = df[target_col]
        is_classification = target.nunique(dropna=True) <= 10
        if is_classification:
            counts = target.value_counts(dropna=False)
            rows = [[str(k), int(v), round(float(v / len(target)), 6)] for k, v in counts.items()]
            tables.append({
                "title": "Target Distribution",
                "headers": ["Value", "Count", "Rate"],
                "rows": rows,
            })
            summary.append(f"Target '{target_col}' has {target.nunique()} classes.")
        else:
            stats = target.describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
            rows = [[k, round(float(v), 6)] for k, v in stats.items()]
            tables.append({
                "title": "Target Summary Statistics",
                "headers": ["Metric", "Value"],
                "rows": rows,
            })
            summary.append(f"Target '{target_col}' appears continuous.")

        if time_col and context.get("time_clean"):
            ts = ensure_datetime(df, time_col)
            df_time = df.copy()
            df_time[time_col] = ts
            df_time = df_time.dropna(subset=[time_col])
            df_time["time_bucket"] = df_time[time_col].dt.to_period("M").dt.to_timestamp()
            if is_classification:
                rate = df_time.groupby("time_bucket")[target_col].mean().sort_index()
                path = os.path.join(self.output_dir, "target_rate_over_time.png")
                self._plot_line(rate, path, title="Target Rate Over Time", ylabel="Rate")
                plots["target_rate_over_time"] = path
                summary.append("Target rate over time plotted.")
            else:
                mean_target = df_time.groupby("time_bucket")[target_col].mean().sort_index()
                path = os.path.join(self.output_dir, "target_mean_over_time.png")
                self._plot_line(mean_target, path, title="Target Mean Over Time", ylabel="Mean")
                plots["target_mean_over_time"] = path

        col_types = context["col_types"]
        categorical = self._select_categorical(df, col_types, None)
        if categorical:
            for col in categorical[: min(3, len(categorical))]:
                rates = df.groupby(col)[target_col].mean().sort_values(ascending=False).head(self.top_k_categories)
                rows = [[str(idx), round(float(val), 6)] for idx, val in rates.items()]
                tables.append({
                    "title": f"Target Rate by {col}",
                    "headers": [col, "Target Rate"],
                    "rows": rows,
                })

        metrics["target_column"] = target_col
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_univariate(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        categorical_cols = self._select_categorical(df, col_types, selected_cols)

        if numeric_cols:
            desc = df[numeric_cols].describe().T.round(6)
            rows = [[idx] + [desc.loc[idx, col] for col in desc.columns] for idx in desc.index]
            tables.append({
                "title": "Numeric Summary Statistics",
                "headers": ["Column"] + list(desc.columns),
                "rows": rows,
            })
            for col in numeric_cols[: min(self.max_plots, len(numeric_cols))]:
                path = os.path.join(self.output_dir, f"hist_{col}.png")
                self._plot_hist(df[col], path, title=f"Distribution: {col}")
                plots[f"hist_{col}"] = path
            summary.append(f"Numeric columns analyzed: {numeric_cols}.")
        else:
            summary.append("No numeric columns selected for univariate analysis.")

        if categorical_cols:
            rows = []
            for col in categorical_cols:
                counts = df[col].value_counts(dropna=False).head(self.top_k_categories)
                for idx, val in counts.items():
                    rows.append([col, str(idx), int(val), round(float(val / len(df)), 6)])
                if col in categorical_cols[: min(self.max_plots, len(categorical_cols))]:
                    path = os.path.join(self.output_dir, f"cat_{col}.png")
                    self._plot_bar(counts, path, title=f"Top {self.top_k_categories}: {col}", ylabel="Count")
                    plots[f"cat_{col}"] = path
            tables.append({
                "title": "Categorical Frequency (Top K)",
                "headers": ["Column", "Category", "Count", "Rate"],
                "rows": rows,
            })
            summary.append(f"Categorical columns analyzed: {categorical_cols}.")
        else:
            summary.append("No categorical columns selected for univariate analysis.")

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_bivariate_target(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        df = context["df"]
        target_col = context.get("target_col")
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if not target_col or target_col not in df.columns:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Target column missing."]}

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        categorical_cols = self._select_categorical(df, col_types, selected_cols)
        target = df[target_col]
        is_classification = target.nunique(dropna=True) <= 10

        if numeric_cols:
            rows = []
            for col in numeric_cols:
                try:
                    bins = pd.qcut(df[col], q=5, duplicates="drop")
                except ValueError:
                    continue
                grouped = df.groupby(bins)[target_col].mean()
                for idx, val in grouped.items():
                    rows.append([col, str(idx), round(float(val), 6)])
                if col == numeric_cols[0]:
                    path = os.path.join(self.output_dir, f"target_rate_bins_{col}.png")
                    self._plot_bar(grouped, path, title=f"Target Rate by {col} bins", ylabel="Rate")
                    plots[f"target_rate_bins_{col}"] = path
            tables.append({
                "title": "Numeric vs Target (Binned)",
                "headers": ["Column", "Bin", "Target Rate" if is_classification else "Target Mean"],
                "rows": rows,
            })
        else:
            summary.append("No numeric columns available for bivariate target analysis.")

        if categorical_cols:
            rows = []
            for col in categorical_cols:
                rates = df.groupby(col)[target_col].mean().sort_values(ascending=False).head(self.top_k_categories)
                for idx, val in rates.items():
                    rows.append([col, str(idx), round(float(val), 6)])
                if col == categorical_cols[0]:
                    path = os.path.join(self.output_dir, f"target_rate_cat_{col}.png")
                    self._plot_bar(rates, path, title=f"Target Rate by {col}", ylabel="Rate")
                    plots[f"target_rate_cat_{col}"] = path
            tables.append({
                "title": "Categorical vs Target",
                "headers": ["Column", "Category", "Target Rate" if is_classification else "Target Mean"],
                "rows": rows,
            })
        else:
            summary.append("No categorical columns available for bivariate target analysis.")

        summary.append("Bivariate target analysis completed.")
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_feature_vs_feature(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        if len(numeric_cols) < 2:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Not enough numeric columns."]}

        corr = df[numeric_cols].corr().round(4)
        metrics["correlation"] = corr.to_dict()
        path = os.path.join(self.output_dir, "correlation_heatmap.png")
        self._plot_heatmap(corr.values, list(corr.columns), path, title="Correlation Heatmap")
        plots["correlation_heatmap"] = path

        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr.iloc[i, j])
                if abs(val) >= 0.98:
                    pairs.append([cols[i], cols[j], round(val, 4)])
        if pairs:
            tables.append({
                "title": "Highly Correlated Feature Pairs (|corr| >= 0.98)",
                "headers": ["Feature A", "Feature B", "Correlation"],
                "rows": pairs,
            })
            summary.append("Found highly correlated feature pairs; consider removing duplicates.")
        else:
            summary.append("No highly correlated feature pairs detected.")

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_time_drift(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        df = context["df"]
        col_types = context["col_types"]
        time_col = context.get("time_col")
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if not time_col or time_col not in df.columns:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Time column missing."]}

        ts = ensure_datetime(df, time_col)
        df_time = df.copy()
        df_time[time_col] = ts
        df_time = df_time.dropna(subset=[time_col])
        df_time = df_time.sort_values(time_col)
        df_time["time_bucket"] = df_time[time_col].dt.to_period("M").dt.to_timestamp()

        counts = df_time.groupby("time_bucket").size()
        path = os.path.join(self.output_dir, "time_volume.png")
        self._plot_line(counts, path, title="Transaction Volume Over Time", ylabel="Count")
        plots["time_volume"] = path

        amount_cols = [c for c in col_types["numeric"] if "amount" in c.lower()]
        if amount_cols:
            amount_col = amount_cols[0]
            amount = df_time.groupby("time_bucket")[amount_col].mean()
            path = os.path.join(self.output_dir, "time_amount_mean.png")
            self._plot_line(amount, path, title=f"{amount_col} Mean Over Time", ylabel="Mean")
            plots["time_amount_mean"] = path

        mid_point = df_time[time_col].quantile(0.5)
        df_a = df_time[df_time[time_col] <= mid_point]
        df_b = df_time[df_time[time_col] > mid_point]

        numeric_cols = self._select_numeric(df_time, col_types, selected_cols)
        psi_rows = []
        for col in numeric_cols:
            psi_val = self._psi(df_a[col], df_b[col])
            psi_rows.append([col, round(float(psi_val), 6)])
        if psi_rows:
            psi_rows.sort(key=lambda x: x[1], reverse=True)
            tables.append({
                "title": "PSI Drift (Numeric)",
                "headers": ["Column", "PSI"],
                "rows": psi_rows,
            })

        categorical_cols = self._select_categorical(df_time, col_types, selected_cols)
        cat_rows = []
        for col in categorical_cols:
            drift = self._categorical_drift(df_a[col], df_b[col], top_k=self.top_k_categories)
            cat_rows.append([col, round(float(drift), 6)])
        if cat_rows:
            cat_rows.sort(key=lambda x: x[1], reverse=True)
            tables.append({
                "title": "Categorical Drift (Total Variation)",
                "headers": ["Column", "Drift Score"],
                "rows": cat_rows,
            })

        summary.append("Time series and drift analysis completed.")
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context["df"]
        col_types = context["col_types"]
        summary: List[str] = []

        missing_rate = df.isna().mean().sort_values(ascending=False)
        high_missing = missing_rate[missing_rate > 0.2]
        if not high_missing.empty:
            summary.append(f"High missingness columns: {list(high_missing.index[:5])}.")

        categorical = col_types["categorical"]
        if categorical:
            nunique = df[categorical].nunique(dropna=True).sort_values(ascending=False)
            high_card = nunique[nunique > 50]
            if not high_card.empty:
                summary.append(f"High-cardinality categoricals: {list(high_card.index[:5])}.")

        numeric = col_types["numeric"]
        if numeric:
            skewness = df[numeric].skew().sort_values(ascending=False)
            heavy_tail = skewness[skewness > 2]
            if not heavy_tail.empty:
                summary.append(f"Heavy-tailed numeric features: {list(heavy_tail.index[:5])}.")

        if not summary:
            summary.append("No major data quality issues detected with current heuristics.")
        summary.append("Next steps: consider feature engineering, handling missing values, and monitoring drift.")
        return {"metrics": {}, "tables": [], "plots": {}, "summary": summary}

    def _psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        baseline = baseline.dropna()
        current = current.dropna()
        if baseline.empty or current.empty:
            return 0.0
        quantiles = np.linspace(0, 1, bins + 1)
        breaks = np.unique(np.quantile(baseline, quantiles))
        if len(breaks) < 3:
            return 0.0
        b_counts, _ = np.histogram(baseline, bins=breaks)
        c_counts, _ = np.histogram(current, bins=breaks)
        b_perc = b_counts / max(1, b_counts.sum())
        c_perc = c_counts / max(1, c_counts.sum())
        eps = 1e-6
        psi = np.sum((b_perc - c_perc) * np.log((b_perc + eps) / (c_perc + eps)))
        return float(psi)

    def _categorical_drift(self, base: pd.Series, curr: pd.Series, top_k: int = 10) -> float:
        base_counts = base.value_counts(normalize=True, dropna=False)
        curr_counts = curr.value_counts(normalize=True, dropna=False)
        categories = list(base_counts.head(top_k).index)
        drift = 0.0
        for cat in categories:
            drift += abs(base_counts.get(cat, 0.0) - curr_counts.get(cat, 0.0))
        return float(drift / 2.0)

    def _plot_bar(self, series: pd.Series, path: str, title: str, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.fig_size)
        series.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=self.plot_dpi)
        plt.close(fig)

    def _plot_hist(self, series: pd.Series, path: str, title: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.hist(series.dropna().values, bins=30, color="steelblue", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=self.plot_dpi)
        plt.close(fig)

    def _plot_heatmap(self, mat: np.ndarray, labels: List[str], path: str, title: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.heatmap_size)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=self.plot_dpi)
        plt.close(fig)

    def _plot_line(self, series: pd.Series, path: str, title: str, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(series.index, series.values, color="steelblue", marker="o", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=self.plot_dpi)
        plt.close(fig)

    def _convert(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._convert(v) for v in obj]
        return obj

    def _print_numbered_list(self, items: List[str], max_items: int = 50) -> None:
        if len(items) > max_items:
            print(f"(showing first {max_items} of {len(items)} columns)")
        for i, name in enumerate(items[:max_items], start=1):
            print(f"{i}. {name}")
