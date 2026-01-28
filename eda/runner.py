"""EDA main runner."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .report import EDAReportBuilder
from .utils import (
    auto_detect_data_path,
    detect_column_types,
    ensure_datetime,
    is_time_col_clean,
    load_data,
    safe_select_columns,
)


class EDA:
    """Exploratory Data Analysis utility."""

    SECTIONS = [
        "overview",
        "missingness",
        "numeric",
        "categorical",
        "correlation",
        "target",
        "outliers",
        "time",
    ]

    def __init__(
        self,
        output_dir: str = "./output_eda",
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        tag: str = "eda",
        plot_dpi: int = 120,
        fig_size: Tuple[float, float] = (6.5, 4.0),
        heatmap_size: Tuple[float, float] = (7.0, 5.0),
        time_parse_min_ratio: float = 0.9,
    ):
        self.output_dir = output_dir
        self.target_col = target_col
        self.time_col = time_col
        self.id_cols = id_cols or []
        self.plot_dpi = plot_dpi
        self.fig_size = fig_size
        self.heatmap_size = heatmap_size
        self.time_parse_min_ratio = time_parse_min_ratio
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
        save_json: bool = True,
        generate_report: bool = True,
        report_name: str = "EDA_Report.pdf",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run EDA on a dataset and optionally generate a report."""
        if df is None:
            path = file_path or auto_detect_data_path()
            df = load_data(path)
        else:
            path = file_path

        target_col = target_col or self.target_col
        time_col = time_col or self.time_col
        section_columns = section_columns or {}

        results: Dict[str, Any] = {}
        run_sections = sections or self.SECTIONS

        for sec in run_sections:
            cols = section_columns.get(sec) or columns
            df_sec = safe_select_columns(df, cols) if cols else df

            if sec == "overview":
                results["overview"] = self.overview(df_sec, target_col=target_col, data_path=path)
            elif sec == "missingness":
                results["missingness"] = self.missingness(df_sec)
            elif sec == "numeric":
                results["numeric"] = self.numeric(df_sec)
            elif sec == "categorical":
                results["categorical"] = self.categorical(df_sec)
            elif sec == "correlation":
                results["correlation"] = self.correlation(df_sec, target_col=target_col)
            elif sec == "target":
                if target_col:
                    results["target"] = self.target_analysis(df, target_col=target_col, feature_cols=cols)
            elif sec == "outliers":
                results["outliers"] = self.outliers(df_sec)
            elif sec == "time":
                if time_col:
                    clean, ratio = is_time_col_clean(df, time_col, min_valid_ratio=self.time_parse_min_ratio)
                    if clean:
                        results["time"] = self.time_analysis(df, time_col=time_col, target_col=target_col)
                    else:
                        results["time"] = {
                            "metrics": {},
                            "plots": {},
                            "summary": [
                                f"Time analysis skipped: '{time_col}' parse success ratio {ratio:.2%} "
                                f"is below threshold {self.time_parse_min_ratio:.0%}."
                            ],
                        }

        if save_json:
            json_path = os.path.join(self.output_dir, "eda_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._convert(results), f, indent=2)

        if generate_report:
            self.report_builder.build(results, filename=report_name)

        return results

    # Section methods
    def overview(self, df: pd.DataFrame, target_col: Optional[str], data_path: Optional[str] = None) -> Dict[str, Any]:
        """Overview of dataset and column types."""
        types = detect_column_types(df, id_cols=self.id_cols)
        duplicate_ratio = float(df.duplicated().mean())
        unique_counts = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

        metrics = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "data_path": data_path or "",
            "duplicate_ratio": round(duplicate_ratio, 6),
            "numeric_columns": types["numeric"],
            "categorical_columns": types["categorical"],
            "datetime_columns": types["datetime"],
            "text_columns": types["text"],
            "boolean_columns": types["boolean"],
            "other_columns": types["other"],
            "unique_counts": unique_counts,
        }

        summary = []
        summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        summary.append(f"Duplicate row ratio is {duplicate_ratio:.2%}.")
        summary.append(f"Numeric columns: {types['numeric']}.")
        summary.append(f"Categorical columns: {types['categorical']}.")
        if target_col and target_col in df.columns:
            target = df[target_col]
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
                summary.append(
                    f"Target '{target_col}' appears continuous with mean {target.mean():.4f} and std {target.std():.4f}."
                )
            else:
                counts = target.value_counts(dropna=False).to_dict()
                summary.append(f"Target '{target_col}' distribution: {counts}.")

        return {"metrics": metrics, "plots": {}, "summary": summary}

    def missingness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Missingness analysis."""
        miss = df.isna().mean().sort_values(ascending=False)
        metrics = {"missing_rate": miss.round(6).to_dict()}

        # Plot top 20 missing columns
        top = miss.head(20)
        plot_path = os.path.join(self.output_dir, "missingness.png")
        self._plot_bar(top, plot_path, title="Missing Rate (Top 20)", ylabel="Missing Rate")

        summary = []
        if not top.empty:
            summary.append(f"Highest missing rate is {top.iloc[0]:.2%} in column '{top.index[0]}'.")
        else:
            summary.append("No missing values detected.")

        return {"metrics": metrics, "plots": {"missingness": plot_path}, "summary": summary}

    def numeric(self, df: pd.DataFrame, max_plots: int = 12) -> Dict[str, Any]:
        """Numeric feature statistics and distributions."""
        num = df.select_dtypes(include=[np.number])
        metrics = {"numeric_stats": num.describe().round(6).to_dict()}
        summary = []

        if num.empty:
            summary.append("No numeric columns found.")
            return {"metrics": metrics, "plots": {}, "summary": summary}

        summary.append(f"Numeric columns analyzed: {list(num.columns)}.")

        # Plot histograms for up to max_plots columns
        plot_paths = {}
        for col in list(num.columns)[:max_plots]:
            path = os.path.join(self.output_dir, f"hist_{col}.png")
            self._plot_hist(num[col], path, title=f"Distribution: {col}")
            plot_paths[f"hist_{col}"] = path

        return {"metrics": metrics, "plots": plot_paths, "summary": summary}

    def categorical(self, df: pd.DataFrame, top_k: int = 10, max_plots: int = 12) -> Dict[str, Any]:
        """Categorical feature distributions."""
        cat = df.select_dtypes(include=["object", "string", "category", "bool"])
        metrics = {}
        summary = []

        if cat.empty:
            summary.append("No categorical columns found.")
            return {"metrics": metrics, "plots": {}, "summary": summary}

        plot_paths = {}
        for col in cat.columns:
            counts = cat[col].value_counts(dropna=False).head(top_k)
            metrics[col] = counts.to_dict()
        summary.append(f"Categorical columns analyzed: {list(cat.columns)}.")

        for col in list(cat.columns)[:max_plots]:
            counts = cat[col].value_counts(dropna=False).head(top_k)
            path = os.path.join(self.output_dir, f"cat_{col}.png")
            self._plot_bar(counts, path, title=f"Top {top_k} Categories: {col}", ylabel="Count")
            plot_paths[f"cat_{col}"] = path

        return {"metrics": {"top_categories": metrics}, "plots": plot_paths, "summary": summary}

    def correlation(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Correlation analysis for numeric features."""
        num = df.select_dtypes(include=[np.number])
        metrics = {}
        summary = []

        if num.shape[1] < 2:
            summary.append("Not enough numeric columns for correlation.")
            return {"metrics": metrics, "plots": {}, "summary": summary}

        corr = num.corr().round(4)
        metrics["correlation"] = corr.to_dict()

        plot_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        self._plot_heatmap(corr.values, list(corr.columns), plot_path, title="Correlation Heatmap")

        summary.append("Correlation heatmap shows linear relationships among numeric features.")

        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_corr = num.corrwith(df[target_col]).sort_values(ascending=False).round(4)
            metrics["target_correlation"] = target_corr.to_dict()
            summary.append(f"Top correlated features with target '{target_col}': {list(target_corr.head(5).index)}.")

        return {"metrics": metrics, "plots": {"correlation_heatmap": plot_path}, "summary": summary}

    def target_analysis(self, df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze relationship between features and target."""
        if target_col not in df.columns:
            return {"metrics": {}, "plots": {}, "summary": [f"Target column not found: {target_col}."]}

        target = df[target_col]
        df_feat = safe_select_columns(df.drop(columns=[target_col]), feature_cols)
        num = df_feat.select_dtypes(include=[np.number])
        cat = df_feat.select_dtypes(include=["object", "string", "category", "bool"])

        metrics: Dict[str, Any] = {}
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if target.nunique() <= 10:
            # Classification-style analysis
            summary.append(f"Target '{target_col}' appears categorical with classes: {target.unique().tolist()}.")
            if not num.empty:
                means = num.groupby(target).mean().round(4)
                metrics["numeric_mean_by_target"] = means.to_dict()
            if not cat.empty:
                rates = {}
                for col in cat.columns:
                    ct = pd.crosstab(cat[col], target, normalize="index")
                    rates[col] = ct.round(4).to_dict()
                metrics["categorical_target_rate"] = rates
        else:
            summary.append(f"Target '{target_col}' appears continuous; showing correlations.")
            if not num.empty:
                metrics["target_correlation"] = num.corrwith(target).sort_values(ascending=False).round(4).to_dict()

        return {"metrics": metrics, "plots": plots, "summary": summary}

    def outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Outlier ratio per numeric feature using IQR."""
        num = df.select_dtypes(include=[np.number])
        metrics = {}
        summary = []
        if num.empty:
            summary.append("No numeric columns found for outlier detection.")
            return {"metrics": metrics, "plots": {}, "summary": summary}

        ratios = {}
        for col in num.columns:
            q1 = num[col].quantile(0.25)
            q3 = num[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                ratios[col] = 0.0
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((num[col] < lower) | (num[col] > upper)).mean()
            ratios[col] = round(float(outliers), 6)

        metrics["outlier_ratio"] = ratios
        top = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:10]
        summary.append(f"Top outlier ratios: {top}.")

        if top:
            series = pd.Series({k: v for k, v in top})
            plot_path = os.path.join(self.output_dir, "outliers.png")
            self._plot_bar(series, plot_path, title="Top Outlier Ratios", ylabel="Outlier Ratio")
            return {"metrics": metrics, "plots": {"outliers": plot_path}, "summary": summary}

        return {"metrics": metrics, "plots": {}, "summary": summary}

    def time_analysis(self, df: pd.DataFrame, time_col: str, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Time series analysis for volume and target rate."""
        if time_col not in df.columns:
            return {"metrics": {}, "plots": {}, "summary": [f"Time column not found: {time_col}."]}

        ts = ensure_datetime(df, time_col)
        df_time = df.copy()
        df_time[time_col] = ts
        df_time = df_time.dropna(subset=[time_col])
        df_time["date"] = df_time[time_col].dt.date

        counts = df_time.groupby("date").size()
        metrics = {"daily_volume": counts.to_dict()}
        plots: Dict[str, str] = {}
        summary = [f"Time span: {df_time['date'].min()} to {df_time['date'].max()}."]

        plot_path = os.path.join(self.output_dir, "time_volume.png")
        self._plot_line(counts, plot_path, title="Daily Volume", ylabel="Count")
        plots["time_volume"] = plot_path

        if target_col and target_col in df_time.columns:
            target = df_time[target_col]
            if target.nunique() <= 10:
                rate = df_time.groupby("date")[target_col].mean()
                metrics["daily_target_rate"] = rate.to_dict()
                rate_path = os.path.join(self.output_dir, "time_target_rate.png")
                self._plot_line(rate, rate_path, title="Daily Target Rate", ylabel="Rate")
                plots["time_target_rate"] = rate_path
                summary.append("Daily target rate plotted.")

        return {"metrics": metrics, "plots": plots, "summary": summary}

    # Plot helpers
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
