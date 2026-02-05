"""EDA main runner (PySpark)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .report import EDAReportBuilder
    from .utils import (
        DEFAULT_NULL_LIKE_VALUES,
        detect_latest_dataset,
        detect_null_like_values,
        infer_column_types,
        load_data_spark,
        pick_target_column,
        pick_time_column,
        time_parse_ratio,
    )
except ImportError:  # allow running as a script: python eda_spark/runner.py
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from eda_spark.report import EDAReportBuilder
    from eda_spark.utils import (
        DEFAULT_NULL_LIKE_VALUES,
        detect_latest_dataset,
        detect_null_like_values,
        infer_column_types,
        load_data_spark,
        pick_target_column,
        pick_time_column,
        time_parse_ratio,
    )


class EDASpark:
    """Exploratory Data Analysis utility (Spark)."""

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
        output_dir: str = "./output_eda_spark",
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        tag: str = "eda_spark",
        plot_dpi: int = 80,
        fig_size: Tuple[float, float] = (6.0, 3.6),
        heatmap_size: Tuple[float, float] = (6.5, 4.5),
        time_parse_min_ratio: float = 0.9,
        max_numeric_cols: int = 10,
        max_categorical_cols: int = 10,
        max_plots: int = 10,
        top_k_categories: int = 10,
        sample_size: int = 5000,
        spark=None,
        spark_master: Optional[str] = None,
        spark_app_name: str = "EDASpark",
        spark_conf: Optional[Dict[str, str]] = None,
    ) -> None:
        from pyspark.sql import SparkSession

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
        self.sample_size = sample_size
        os.makedirs(output_dir, exist_ok=True)
        self.report_builder = EDAReportBuilder(output_dir=output_dir, tag=tag)
        if spark is not None:
            self.spark = spark
        else:
            builder = SparkSession.builder.appName(spark_app_name)
            if spark_master:
                builder = builder.master(spark_master)
            if spark_conf:
                for key, value in spark_conf.items():
                    builder = builder.config(key, value)
            self.spark = builder.getOrCreate()

    def run(
        self,
        df: Optional[Any] = None,
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
    ) -> Dict[str, Any]:
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType

        if df is None:
            path = file_path or detect_latest_dataset(data_dir=data_dir)
            df = load_data_spark(self.spark, path)
        else:
            path = file_path

        if max_rows:
            df = df.limit(max_rows)

        col_types = infer_column_types(df, id_cols=self.id_cols, sample_size=self.sample_size)
        target_col = target_col or self.target_col or pick_target_column(df, col_types, id_cols=self.id_cols)
        time_col = time_col or self.time_col or pick_time_column(
            df,
            col_types,
            min_valid_ratio=self.time_parse_min_ratio,
        )

        time_clean = False
        time_ratio = 0.0
        if time_col:
            time_clean, time_ratio = time_parse_ratio(df, time_col, min_valid_ratio=self.time_parse_min_ratio)

        context = {
            "df": df,
            "data_path": path or "",
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

            cols_requested = section_columns.get(sec) if section_columns and sec in section_columns else columns
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

        rows_used = int(df.count())
        payload = {
            "results": results,
            "skipped_sections": skipped,
            "config": {
                "data_path": path or "",
                "rows_used": rows_used,
                "target_col": target_col or "",
                "time_col": time_col or "",
                "time_parse_ratio": round(float(time_ratio), 4),
            },
        }

        if save_json:
            json_path = os.path.join(self.output_dir, "eda_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                import json

                json.dump(payload, f, indent=2)

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
        df: Optional[Any] = None,
        file_path: Optional[str] = None,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        max_rows: Optional[int] = None,
        save_json: bool = True,
        generate_report: bool = True,
        report_name: str = "EDA_Report.pdf",
        data_dir: str = "./data",
        return_payload: bool = False,
    ) -> Dict[str, Any]:
        """Interactive selection of sections and columns (Spark)."""
        if df is None:
            path = file_path or detect_latest_dataset(data_dir=data_dir)
            df = load_data_spark(self.spark, path)
        else:
            path = file_path

        if max_rows:
            df = df.limit(max_rows)

        col_types = infer_column_types(df, id_cols=self.id_cols, sample_size=self.sample_size)
        target_col = target_col or self.target_col or pick_target_column(df, col_types, id_cols=self.id_cols)
        time_col = time_col or self.time_col or pick_time_column(
            df,
            col_types,
            min_valid_ratio=self.time_parse_min_ratio,
        )

        self.print_functions()
        selection = input("Select functions by number (e.g., 1,2,3) or 'all': ").strip()
        sections = self.parse_function_selection(selection)
        if not sections:
            sections = None

        section_columns: Dict[str, List[str]] = {}
        time_clean = False
        time_ratio = 0.0
        if time_col:
            time_clean, time_ratio = time_parse_ratio(df, time_col, min_valid_ratio=self.time_parse_min_ratio)

        context = {
            "df": df,
            "data_path": path or "",
            "target_col": target_col,
            "time_col": time_col,
            "time_clean": time_clean,
            "time_ratio": time_ratio,
            "col_types": col_types,
        }
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

    def _print_numbered_list(self, items: List[str], max_items: int = 50) -> None:
        if len(items) > max_items:
            print(f"(showing first {max_items} of {len(items)} columns)")
        for i, name in enumerate(items[:max_items], start=1):
            print(f"{i}. {name}")


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

    def _select_numeric(self, df, col_types: Dict[str, List[str]], cols: Optional[List[str]]) -> List[str]:
        from pyspark.sql import functions as F

        numeric = col_types["numeric"]
        if cols:
            return [c for c in cols if c in numeric]
        if len(numeric) <= self.max_numeric_cols:
            return numeric
        vars_row = df.agg(*[F.var_samp(c).alias(c) for c in numeric]).collect()[0].asDict()
        ranked = sorted(vars_row.items(), key=lambda x: (x[1] or 0), reverse=True)
        return [name for name, _ in ranked[: self.max_numeric_cols]]

    def _select_categorical(self, df, col_types: Dict[str, List[str]], cols: Optional[List[str]]) -> List[str]:
        from pyspark.sql import functions as F

        categorical = col_types["categorical"] + col_types["boolean"]
        if cols:
            return [c for c in cols if c in categorical]
        if len(categorical) <= self.max_categorical_cols:
            return categorical
        counts = df.agg(*[F.countDistinct(c).alias(c) for c in categorical]).collect()[0].asDict()
        ranked = sorted(counts.items(), key=lambda x: (x[1] or 0), reverse=True)
        return [name for name, _ in ranked[: self.max_categorical_cols]]

    def _section_data_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType
        import pandas as pd

        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        rows = df.count()
        cols = len(df.columns)
        dup_rows = df.dropDuplicates().count()
        duplicate_ratio = 1.0 - (dup_rows / max(1, rows))
        metrics["rows"] = int(rows)
        metrics["columns"] = int(cols)
        metrics["duplicate_ratio"] = round(float(duplicate_ratio), 6)

        null_set = [str(v).strip().lower() for v in DEFAULT_NULL_LIKE_VALUES]
        string_cols = {f.name for f in df.schema.fields if isinstance(f.dataType, StringType)}
        missing_exprs = []
        for c in df.columns:
            if c in string_cols:
                norm = F.lower(F.trim(F.col(c)))
                missing_exprs.append(
                    F.sum(F.when(F.col(c).isNull() | norm.isin(null_set), 1).otherwise(0)).alias(c)
                )
            else:
                missing_exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c))
        missing_counts = df.agg(*missing_exprs).collect()[0].asDict()
        missing_columns = []
        non_missing_columns = []
        for col, cnt in missing_counts.items():
            if cnt > 0:
                rate = float(cnt) / max(1, rows)
                missing_columns.append(
                    {
                        "column": col,
                        "missing_count": int(cnt),
                        "missing_rate": round(rate, 6),
                    }
                )
            else:
                non_missing_columns.append(col)
        metrics["missingness_payload"] = {
            "missing_columns": missing_columns,
            "non_missing_columns": non_missing_columns,
        }
        if missing_columns:
            missing_rows = [
                [row["column"], row["missing_count"], row["missing_rate"]]
                for row in missing_columns
            ]
            tables.append({
                "title": "Missingness",
                "headers": ["Column", "Missing Count", "Missing Rate"],
                "rows": missing_rows,
                "style": "missingness",
            })
            missing_columns.sort(key=lambda x: x["missing_rate"], reverse=True)
            missing_series = pd.Series({row["column"]: row["missing_rate"] for row in missing_columns})
            top_missing = missing_series.head(min(20, len(missing_series)))
            if not top_missing.empty:
                path = os.path.join(self.output_dir, "missingness.png")
                self._plot_bar(top_missing, path, title="Missing Rate (Top)", ylabel="Missing Rate")
                plots["missingness"] = path
        else:
            metrics["missingness_skipped_reason"] = "No columns with missing values."

        type_rows = []
        for type_name in ["numeric", "categorical", "datetime", "boolean", "text", "other"]:
            cols_list = col_types.get(type_name, [])
            for col in cols_list:
                type_rows.append([type_name, col])
        tables.append({
            "title": "Column Type Classification",
            "headers": ["Type", "Columns"],
            "rows": type_rows,
        })

        summary.append(f"Dataset has {rows} rows and {cols} columns.")
        summary.append(f"Duplicate row ratio: {duplicate_ratio:.2%}.")

        null_like_payload = detect_null_like_values(
            df,
            null_like_values=DEFAULT_NULL_LIKE_VALUES,
        )
        if not null_like_payload:
            metrics["null_like_skipped_reason"] = "No string-like columns available or no null-like values detected."
        metrics["null_like_payload"] = null_like_payload

        outlier_rows = []
        numeric_cols = self._select_numeric(df, col_types, None)
        for col in numeric_cols:
            try:
                col_series = F.col(col).cast("double")
                q1, q3 = df.select(col_series.alias(col)).approxQuantile(col, [0.25, 0.75], 0.005)
            except Exception:
                continue
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_cnt = df.filter((col_series < lower) | (col_series > upper)).count()
            ratio = float(out_cnt) / max(1, rows)
            outlier_rows.append([col, round(ratio, 6)])
        if outlier_rows:
            outlier_rows.sort(key=lambda x: x[1], reverse=True)
            tables.append({
                "title": "Outlier Ratio (IQR)",
                "headers": ["Column", "Outlier Ratio"],
                "rows": outlier_rows[: min(20, len(outlier_rows))],
            })
            top_outliers = outlier_rows[: min(20, len(outlier_rows))]
            outlier_series = pd.Series({row[0]: row[1] for row in top_outliers})
            path = os.path.join(self.output_dir, "outlier_iqr.png")
            self._plot_bar(outlier_series, path, title="Outlier Ratio (IQR)", ylabel="Outlier Ratio")
            plots["outlier_iqr"] = path

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_target(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from pyspark.sql import functions as F
        import pandas as pd

        df = context["df"]
        target_col = context.get("target_col")
        time_col = context.get("time_col")
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if not target_col or target_col not in df.columns:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Target column not found."]}

        distinct_count = df.select(target_col).distinct().count()
        is_classification = distinct_count <= 10
        if is_classification:
            counts = df.groupBy(target_col).count().orderBy(F.desc("count"))
            total = df.count()
            rows = [[str(r[target_col]), int(r["count"]), round(r["count"] / total, 6)] for r in counts.collect()]
            tables.append({
                "title": "Target Distribution",
                "headers": ["Value", "Count", "Rate"],
                "rows": rows,
            })
        else:
            stats = df.select(
                F.mean(target_col).alias("mean"),
                F.stddev(target_col).alias("std"),
                F.min(target_col).alias("min"),
                F.max(target_col).alias("max"),
            ).collect()[0]
            rows = [[k, round(float(v), 6)] for k, v in stats.asDict().items()]
            tables.append({
                "title": "Target Summary Statistics",
                "headers": ["Metric", "Value"],
                "rows": rows,
            })

        if time_col and context.get("time_clean"):
            df_time = df.withColumn("time_ts", F.to_timestamp(F.col(time_col)))
            df_time = df_time.dropna(subset=["time_ts"])
            df_time = df_time.withColumn("time_bucket", F.date_trunc("month", F.col("time_ts")))
            if is_classification:
                rate = df_time.groupBy("time_bucket").agg(F.mean(target_col).alias("rate")).orderBy("time_bucket")
                rate = rate.withColumn("time_bucket", F.col("time_bucket").cast("string"))
                pdf = rate.toPandas()
                if "time_bucket" in pdf.columns:
                    pdf["time_bucket"] = pd.to_datetime(pdf["time_bucket"], errors="coerce")
                if not pdf.empty:
                    path = os.path.join(self.output_dir, "target_rate_over_time.png")
                    self._plot_line(pdf.set_index("time_bucket")["rate"], path, "Target Rate Over Time", "Rate")
                    plots["target_rate_over_time"] = path

        categorical = self._select_categorical(df, col_types, None)
        if categorical:
            for col in categorical[: min(3, len(categorical))]:
                rates = (
                    df.groupBy(col)
                    .agg(F.mean(target_col).alias("rate"))
                    .orderBy(F.desc("rate"))
                    .limit(self.top_k_categories)
                    .collect()
                )
                rows = [[str(r[col]), round(float(r["rate"]), 6)] for r in rates]
                tables.append({
                    "title": f"Target Rate by {col}",
                    "headers": [col, "Target Rate" if is_classification else "Target Mean"],
                    "rows": rows,
                })
                series = pd.Series(
                    [float(r["rate"]) for r in rates],
                    index=[str(r[col]) for r in rates],
                )
                path = os.path.join(self.output_dir, f"target_rate_by_{col}.png")
                self._plot_bar(series, path, title=f"Target Rate by {col}", ylabel="Rate")
                plots[f"target_rate_by_{col}"] = path

        summary.append("Target analysis completed.")
        metrics["target_column"] = target_col
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_univariate(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        from pyspark.sql import functions as F
        import pandas as pd

        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        categorical_cols = self._select_categorical(df, col_types, selected_cols)
        chart_paths: List[Dict[str, str]] = []

        def summarize_cols(cols: List[str], limit: int = 12) -> str:
            if not cols:
                return "None"
            if len(cols) <= limit:
                return ", ".join(cols)
            extra = len(cols) - limit
            return ", ".join(cols[:limit]) + f" (+{extra} more)"

        summary = [
            f"Numeric columns analyzed: {summarize_cols(numeric_cols)}",
            f"Categorical columns analyzed: {summarize_cols(categorical_cols)}",
        ]

        stats = None
        if numeric_cols:
            stats = df.select(numeric_cols).describe().toPandas()
            sample = df.select(numeric_cols).limit(self.sample_size).toPandas()
            for col in numeric_cols[: min(self.max_plots, len(numeric_cols))]:
                path = os.path.join(self.output_dir, f"hist_{col}.png")
                self._plot_hist(sample[col], path, title=f"Distribution: {col}")
                plots[f"hist_{col}"] = path
                chart_paths.append({"title": f"{col} distribution", "path": path, "kind": "hist"})

        topk_by_col: Dict[str, List[Dict[str, Any]]] = {}
        if categorical_cols:
            total = df.count()
            for col in categorical_cols:
                counts = (
                    df.groupBy(col)
                    .count()
                    .orderBy(F.desc("count"))
                    .limit(self.top_k_categories)
                    .toPandas()
                )
                topk_by_col[col] = []
                for _, r in counts.iterrows():
                    topk_by_col[col].append(
                        {"category": str(r[col]), "count": int(r["count"]), "rate": float(r["count"] / total)}
                    )
                if col in categorical_cols[: min(self.max_plots, len(categorical_cols))]:
                    path = os.path.join(self.output_dir, f"cat_{col}.png")
                    series = counts.set_index(col)["count"]
                    self._plot_bar(series, path, title=f"Top {self.top_k_categories}: {col}", ylabel="Count")
                    plots[f"cat_{col}"] = path
                    chart_paths.append({"title": f"{col} top {self.top_k_categories}", "path": path, "kind": "bar"})
            for col, rows in topk_by_col.items():
                table_rows = [[r["category"], r["count"], r["rate"]] for r in rows]
                tables.append({
                    "title": f"Top K: {col}",
                    "headers": ["Category", "Count", "Rate"],
                    "rows": table_rows,
                    "style": "categorical_topk",
                    "col_widths": [3.0, 0.9, 0.9],
                })

        numeric_summary_rows = []
        if numeric_cols and stats is not None:
            if "summary" in stats.columns:
                stats = stats.set_index("summary")
            for col in numeric_cols:
                if col not in stats.columns:
                    continue
                series = pd.to_numeric(stats[col], errors="coerce")
                stat_dict = series.to_dict()
                numeric_summary_rows.append(
                    {
                        "column": col,
                        "count": float(stat_dict.get("count", 0) or 0),
                        "mean": float(stat_dict.get("mean", 0) or 0),
                        "std": float(stat_dict.get("stddev", 0) or 0),
                        "min": float(stat_dict.get("min", 0) or 0),
                        "p25": float(stat_dict.get("25%", 0) or 0),
                        "p50": float(stat_dict.get("50%", 0) or 0),
                        "p75": float(stat_dict.get("75%", 0) or 0),
                        "max": float(stat_dict.get("max", 0) or 0),
                    }
                )

        if numeric_summary_rows:
            table_rows = [
                [
                    r.get("column", ""),
                    r.get("count", None),
                    r.get("mean", None),
                    r.get("std", None),
                    r.get("min", None),
                    r.get("p25", None),
                    r.get("p50", None),
                    r.get("p75", None),
                    r.get("max", None),
                ]
                for r in numeric_summary_rows
            ]
            tables.append({
                "title": "Numeric Summary Statistics",
                "headers": ["Column", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                "rows": table_rows,
                "style": "wide_numeric_stats",
                "col_widths": [1.45, 0.65, 0.75, 0.75, 0.70, 0.65, 0.65, 0.65, 0.75],
            })

        univariate_payload = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary_rows": numeric_summary_rows,
            "categorical_topk_by_column": topk_by_col if categorical_cols else {},
            "chart_paths": chart_paths,
        }

        return {
            "metrics": metrics,
            "tables": tables,
            "plots": plots,
            "summary": summary,
            "univariate_payload": univariate_payload,
        }

    def _section_bivariate_target(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        from pyspark.sql import functions as F
        from pyspark.ml.feature import QuantileDiscretizer
        import pandas as pd

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

        rows = []
        for col in numeric_cols:
            try:
                discretizer = QuantileDiscretizer(
                    numBuckets=5,
                    inputCol=col,
                    outputCol="bucket",
                    handleInvalid="skip",
                )
                model = discretizer.fit(df)
                binned = model.transform(df)
            except Exception:
                continue

            grouped = (
                binned.groupBy("bucket")
                .agg(F.mean(target_col).alias("rate"))
                .orderBy("bucket")
            )

            if col == numeric_cols[0]:
                pdf = grouped.toPandas()
                if not pdf.empty:
                    pdf["bucket"] = pdf["bucket"].astype(int)
                    series = pd.Series(
                        pdf["rate"].values,
                        index=[f"bin_{i}" for i in pdf["bucket"].tolist()],
                    )
                    path = os.path.join(self.output_dir, f"target_rate_bins_{col}.png")
                    self._plot_bar(series, path, title=f"Target Rate by {col} bins", ylabel="Rate")
                    plots[f"target_rate_bins_{col}"] = path

            for g in grouped.collect():
                if g["bucket"] is None:
                    continue
                rows.append([col, f"bin_{int(g['bucket'])}", round(float(g["rate"]), 6)])

        if rows:
            tables.append({
                "title": "Numeric vs Target (Binned)",
                "headers": ["Column", "Bin", "Target Rate"],
                "rows": rows,
            })

        rows = []
        for col in categorical_cols:
            rates = (
                df.groupBy(col)
                .agg(F.mean(target_col).alias("rate"))
                .orderBy(F.desc("rate"))
                .limit(self.top_k_categories)
                .collect()
            )
            for r in rates:
                rows.append([col, str(r[col]), round(float(r["rate"]), 6)])
        if rows:
            tables.append({
                "title": "Categorical vs Target",
                "headers": ["Column", "Category", "Target Rate"],
                "rows": rows,
            })

        summary.append("Bivariate target analysis completed.")
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_feature_vs_feature(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.stat import Correlation
        from pyspark.sql import functions as F

        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        if len(numeric_cols) < 2:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Not enough numeric columns."]}

        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
        vec_df = assembler.transform(df.select(numeric_cols).dropna())
        corr_mat = Correlation.corr(vec_df, "features").head()[0]
        corr_array = np.array(corr_mat.toArray())
        metrics["correlation"] = corr_array.tolist()

        path = os.path.join(self.output_dir, "correlation_heatmap.png")
        self._plot_heatmap(corr_array, numeric_cols, path, title="Correlation Heatmap")
        plots["correlation_heatmap"] = path

        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = float(corr_array[i, j])
                if abs(val) >= 0.98:
                    pairs.append([numeric_cols[i], numeric_cols[j], round(val, 4)])
        if pairs:
            tables.append({
                "title": "Highly Correlated Feature Pairs (|corr| >= 0.98)",
                "headers": ["Feature A", "Feature B", "Correlation"],
                "rows": pairs,
            })

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_time_drift(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        from pyspark.ml.feature import Bucketizer
        from pyspark.sql import functions as F
        import pandas as pd

        df = context["df"]
        col_types = context["col_types"]
        time_col = context.get("time_col")
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        if not time_col or time_col not in df.columns:
            return {"metrics": metrics, "tables": tables, "plots": plots, "summary": ["Time column missing."]}

        df_time = df.withColumn("time_ts", F.to_timestamp(F.col(time_col)))
        df_time = df_time.dropna(subset=["time_ts"])
        df_time = df_time.withColumn("time_bucket", F.date_trunc("month", F.col("time_ts")))
        counts = df_time.groupBy("time_bucket").count().orderBy("time_bucket")
        counts = counts.withColumn("time_bucket", F.col("time_bucket").cast("string")).toPandas()
        if "time_bucket" in counts.columns:
            counts["time_bucket"] = pd.to_datetime(counts["time_bucket"], errors="coerce")
        if not counts.empty:
            path = os.path.join(self.output_dir, "time_volume.png")
            self._plot_line(counts.set_index("time_bucket")["count"], path, "Transaction Volume Over Time", "Count")
            plots["time_volume"] = path

        amount_cols = [c for c in col_types["numeric"] if "amount" in c.lower()]
        if amount_cols:
            amount_col = amount_cols[0]
            amount = (
                df_time.groupBy("time_bucket")
                .agg(F.mean(amount_col).alias("mean_amount"))
                .orderBy("time_bucket")
            )
            amount = amount.withColumn("time_bucket", F.col("time_bucket").cast("string")).toPandas()
            if "time_bucket" in amount.columns:
                amount["time_bucket"] = pd.to_datetime(amount["time_bucket"], errors="coerce")
            if not amount.empty:
                path = os.path.join(self.output_dir, "time_amount_mean.png")
                self._plot_line(amount.set_index("time_bucket")["mean_amount"], path, f"{amount_col} Mean Over Time", "Mean")
                plots["time_amount_mean"] = path

        median_ts = df_time.select(F.col("time_ts").cast("long")).approxQuantile("time_ts", [0.5], 0.01)[0]
        df_a = df_time.filter(F.col("time_ts").cast("long") <= median_ts)
        df_b = df_time.filter(F.col("time_ts").cast("long") > median_ts)

        numeric_cols = self._select_numeric(df_time, col_types, selected_cols)
        psi_rows = []
        for col in numeric_cols:
            splits = df_a.approxQuantile(col, [i / 10 for i in range(11)], 0.01)
            splits = sorted(set(splits))
            if len(splits) < 3:
                continue
            splits[0] = float("-inf")
            splits[-1] = float("inf")
            bucketizer = Bucketizer(splits=splits, inputCol=col, outputCol="bucket")
            a_counts = bucketizer.transform(df_a).groupBy("bucket").count().toPandas()
            b_counts = bucketizer.transform(df_b).groupBy("bucket").count().toPandas()
            psi_val = self._psi_from_counts(a_counts, b_counts)
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
            base = df_a.groupBy(col).count().toPandas()
            curr = df_b.groupBy(col).count().toPandas()
            drift = self._categorical_drift(base, curr, top_k=self.top_k_categories)
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

        numeric = col_types["numeric"]
        categorical = col_types["categorical"]
        if numeric:
            summary.append(f"Numeric columns: {numeric[:5]}.")
        if categorical:
            summary.append(f"Categorical columns: {categorical[:5]}.")
        if not summary:
            summary.append("No major data quality issues detected with current heuristics.")
        summary.append("Next steps: consider feature engineering, handling missing values, and monitoring drift.")
        return {"metrics": {}, "tables": [], "plots": {}, "summary": summary}

    def _psi_from_counts(self, base_df, curr_df) -> float:
        import pandas as pd

        base = base_df.set_index("bucket")["count"] if not base_df.empty else pd.Series(dtype=float)
        curr = curr_df.set_index("bucket")["count"] if not curr_df.empty else pd.Series(dtype=float)
        buckets = sorted(set(base.index).union(curr.index))
        b = np.array([base.get(b, 0) for b in buckets], dtype=float)
        c = np.array([curr.get(b, 0) for b in buckets], dtype=float)
        b_perc = b / max(1.0, b.sum())
        c_perc = c / max(1.0, c.sum())
        eps = 1e-6
        return float(np.sum((b_perc - c_perc) * np.log((b_perc + eps) / (c_perc + eps))))

    def _categorical_drift(self, base_df, curr_df, top_k: int = 10) -> float:
        import pandas as pd

        base = base_df.set_index(base_df.columns[0])["count"] if not base_df.empty else pd.Series(dtype=float)
        curr = curr_df.set_index(curr_df.columns[0])["count"] if not curr_df.empty else pd.Series(dtype=float)
        base = base / max(1.0, base.sum())
        curr = curr / max(1.0, curr.sum())
        categories = list(base.sort_values(ascending=False).head(top_k).index)
        drift = 0.0
        for cat in categories:
            drift += abs(base.get(cat, 0.0) - curr.get(cat, 0.0))
        return float(drift / 2.0)

    def _plot_bar(self, series, path: str, title: str, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.fig_size)
        series.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=self.plot_dpi)
        plt.close(fig)

    def _plot_hist(self, series, path: str, title: str) -> None:
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

    def _plot_line(self, series, path: str, title: str, ylabel: str) -> None:
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
