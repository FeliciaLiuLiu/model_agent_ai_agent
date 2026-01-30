"""EDA main runner (PySpark)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .report import EDAReportBuilder
from .utils import detect_latest_dataset


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
        output_dir: str = "./output_eda",
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
        self.spark = spark or SparkSession.builder.getOrCreate()

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

        if df is None:
            path = file_path or detect_latest_dataset(data_dir=data_dir)
            df = self._load_data(path)
        else:
            path = file_path

        if max_rows:
            df = df.limit(max_rows)

        target_col = target_col or self.target_col
        col_types = self._infer_column_types(df)
        time_col = time_col or self.time_col or self._pick_time_column(col_types)

        time_clean = False
        time_ratio = 0.0
        if time_col:
            time_clean, time_ratio = self._time_parse_ratio(df, time_col)

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
            df = self._load_data(path)
        else:
            path = file_path

        if max_rows:
            df = df.limit(max_rows)

        target_col = target_col or self.target_col
        col_types = self._infer_column_types(df)
        time_col = time_col or self.time_col or self._pick_time_column(col_types)

        self.print_functions()
        selection = input("Select functions by number (e.g., 1,2,3) or 'all': ").strip()
        sections = self.parse_function_selection(selection)
        if not sections:
            sections = None

        section_columns: Dict[str, List[str]] = {}
        context = {
            "df": df,
            "data_path": path or "",
            "target_col": target_col,
            "time_col": time_col,
            "time_clean": True,
            "time_ratio": 1.0,
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

    def _load_data(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return self.spark.read.option("header", True).option("inferSchema", True).csv(path)
        if ext == ".parquet":
            return self.spark.read.parquet(path)
        raise ValueError(f"Unsupported file extension: {ext}")

    def _infer_column_types(self, df) -> Dict[str, List[str]]:
        from pyspark.sql.types import BooleanType, DateType, NumericType, StringType, TimestampType
        from pyspark.sql import functions as F

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        boolean_cols: List[str] = []
        text_cols: List[str] = []
        other_cols: List[str] = []

        string_cols: List[str] = []
        for field in df.schema.fields:
            name = field.name
            if isinstance(field.dataType, BooleanType):
                boolean_cols.append(name)
            elif isinstance(field.dataType, (TimestampType, DateType)):
                datetime_cols.append(name)
            elif isinstance(field.dataType, NumericType):
                numeric_cols.append(name)
            elif isinstance(field.dataType, StringType):
                string_cols.append(name)
            else:
                other_cols.append(name)

        if string_cols:
            sample = df.select(string_cols).limit(self.sample_size).toPandas()
            for col in string_cols:
                series = sample[col].dropna().astype(str)
                if series.empty:
                    categorical_cols.append(col)
                    continue
                avg_len = float(series.str.len().mean())
                unique_ratio = float(series.nunique() / max(1, len(series)))
                if avg_len >= 30 or unique_ratio >= 0.5:
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "text": text_cols,
            "boolean": boolean_cols,
            "other": other_cols,
        }

    def _pick_time_column(self, col_types: Dict[str, List[str]]) -> Optional[str]:
        if col_types.get("datetime"):
            return col_types["datetime"][0]
        return None

    def _time_parse_ratio(self, df, time_col: str) -> Tuple[bool, float]:
        from pyspark.sql import functions as F

        if time_col not in df.columns:
            return False, 0.0
        parsed = F.to_timestamp(F.col(time_col))
        ratio = (
            df.select(F.mean(F.when(parsed.isNotNull(), F.lit(1)).otherwise(F.lit(0)))).collect()[0][0]
            or 0.0
        )
        return ratio >= self.time_parse_min_ratio, float(ratio)

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

        missing_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
        missing_counts = df.agg(*missing_exprs).collect()[0].asDict()
        missing_rows = []
        for col, cnt in missing_counts.items():
            rate = float(cnt) / max(1, rows)
            missing_rows.append([col, int(cnt), round(rate, 6)])
        missing_rows.sort(key=lambda x: x[2], reverse=True)
        tables.append({
            "title": "Missingness",
            "headers": ["Column", "Missing Count", "Missing Rate"],
            "rows": missing_rows,
        })

        type_rows = [
            ["numeric", ", ".join(col_types["numeric"]) or "-"],
            ["categorical", ", ".join(col_types["categorical"]) or "-"],
            ["datetime", ", ".join(col_types["datetime"]) or "-"],
            ["boolean", ", ".join(col_types["boolean"]) or "-"],
            ["text", ", ".join(col_types["text"]) or "-"],
            ["other", ", ".join(col_types["other"]) or "-"],
        ]
        tables.append({
            "title": "Column Type Classification",
            "headers": ["Type", "Columns"],
            "rows": type_rows,
        })

        summary.append(f"Dataset has {rows} rows and {cols} columns.")
        summary.append(f"Duplicate row ratio: {duplicate_ratio:.2%}.")

        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_target(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from pyspark.sql import functions as F

        df = context["df"]
        target_col = context.get("target_col")
        time_col = context.get("time_col")
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
                pdf = rate.toPandas()
                if not pdf.empty:
                    path = os.path.join(self.output_dir, "target_rate_over_time.png")
                    self._plot_line(pdf.set_index("time_bucket")["rate"], path, "Target Rate Over Time", "Rate")
                    plots["target_rate_over_time"] = path

        summary.append("Target analysis completed.")
        metrics["target_column"] = target_col
        return {"metrics": metrics, "tables": tables, "plots": plots, "summary": summary}

    def _section_univariate(self, context: Dict[str, Any], selected_cols: Optional[List[str]]) -> Dict[str, Any]:
        from pyspark.sql import functions as F

        df = context["df"]
        col_types = context["col_types"]
        metrics: Dict[str, Any] = {}
        tables: List[Dict[str, Any]] = []
        plots: Dict[str, str] = {}
        summary: List[str] = []

        numeric_cols = self._select_numeric(df, col_types, selected_cols)
        categorical_cols = self._select_categorical(df, col_types, selected_cols)

        if numeric_cols:
            stats = df.select(numeric_cols).describe().toPandas()
            headers = ["stat"] + numeric_cols
            rows = stats.values.tolist()
            tables.append({
                "title": "Numeric Summary Statistics",
                "headers": headers,
                "rows": rows,
            })
            sample = df.select(numeric_cols).limit(self.sample_size).toPandas()
            for col in numeric_cols[: min(self.max_plots, len(numeric_cols))]:
                path = os.path.join(self.output_dir, f"hist_{col}.png")
                self._plot_hist(sample[col], path, title=f"Distribution: {col}")
                plots[f"hist_{col}"] = path
            summary.append(f"Numeric columns analyzed: {numeric_cols}.")
        else:
            summary.append("No numeric columns selected for univariate analysis.")

        if categorical_cols:
            rows = []
            total = df.count()
            for col in categorical_cols:
                counts = (
                    df.groupBy(col)
                    .count()
                    .orderBy(F.desc("count"))
                    .limit(self.top_k_categories)
                    .toPandas()
                )
                for _, r in counts.iterrows():
                    rows.append([col, str(r[col]), int(r["count"]), round(float(r["count"] / total), 6)])
                if col in categorical_cols[: min(self.max_plots, len(categorical_cols))]:
                    path = os.path.join(self.output_dir, f"cat_{col}.png")
                    series = counts.set_index(col)["count"]
                    self._plot_bar(series, path, title=f"Top {self.top_k_categories}: {col}", ylabel="Count")
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
        from pyspark.sql import functions as F

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
            quantiles = df.approxQuantile(col, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.01)
            quantiles = sorted(set(quantiles))
            if len(quantiles) < 3:
                continue
            buckets = df.withColumn("bucket", F.when(F.col(col) <= quantiles[1], f"<= {quantiles[1]:.4f}"))
            for idx in range(1, len(quantiles) - 1):
                lower = quantiles[idx]
                upper = quantiles[idx + 1]
                buckets = buckets.withColumn(
                    "bucket",
                    F.when(
                        (F.col(col) > lower) & (F.col(col) <= upper),
                        f"({lower:.4f}, {upper:.4f}]",
                    ).otherwise(F.col("bucket")),
                )
            grouped = buckets.groupBy("bucket").agg(F.mean(target_col).alias("rate")).collect()
            for g in grouped:
                rows.append([col, g["bucket"], round(float(g["rate"]), 6)])
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
        counts = df_time.groupBy("time_bucket").count().orderBy("time_bucket").toPandas()
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
                .toPandas()
            )
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
