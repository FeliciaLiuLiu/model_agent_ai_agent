"""Main runner: Non-Interactive mode (PySpark)."""
import os
import json
import re
from typing import Dict, Any, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..core.report import ReportBuilder
from ..core.utils import load_model as _load_model, load_data as _load_data, get_spark, get_numeric_columns
from ..matrices.effectiveness import ModelEffectivenessSpark
from ..matrices.efficiency import ModelEfficiencySpark
from ..matrices.stability import ModelStabilitySpark
from ..matrices.interpretability import ModelInterpretabilitySpark


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug for per-run artifact directories."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return slug or "segment"


class ModelTestingAgentSpark:
    """Main orchestrator for model testing (PySpark, non-interactive)."""

    SECTIONS = ["effectiveness", "efficiency", "stability", "interpretability"]

    def __init__(self, output_dir: str = "./output", experiment_tag: str = "model_testing_pyspark", spark=None):
        self.output_dir = output_dir
        self.experiment_tag = experiment_tag
        self.spark = spark or get_spark()
        os.makedirs(output_dir, exist_ok=True)
        self.report_builder = ReportBuilder(output_dir=output_dir, tag=experiment_tag)

    def _build_components(self, data_dir: Optional[str] = None):
        """Create fresh matrix evaluators for a single run scope."""
        target_dir = data_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        return {
            "effectiveness": ModelEffectivenessSpark(data_dir=target_dir),
            "efficiency": ModelEfficiencySpark(data_dir=target_dir),
            "stability": ModelStabilitySpark(data_dir=target_dir),
            "interpretability": ModelInterpretabilitySpark(data_dir=target_dir),
        }

    def _select_columns(self, df_in: DataFrame, cols: Optional[List[str]], feature_names_all: List[str], label_col: str):
        """Select a subset of feature columns for one section."""
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

    def _normalize_segmentation(self, segmentation) -> Dict[str, Any]:
        """Normalize segmentation input from dict, JSON string, or JSON file."""
        if segmentation is None:
            return {}
        if isinstance(segmentation, str):
            seg_path = os.path.abspath(segmentation)
            if os.path.exists(seg_path) and os.path.isfile(seg_path):
                with open(seg_path, "r", encoding="utf-8") as f:
                    segmentation = json.load(f)
            else:
                segmentation = json.loads(segmentation)
        if not isinstance(segmentation, dict):
            raise ValueError("Segmentation must be a dict, JSON string, or path to a JSON file.")
        return segmentation

    def _segment_dataset(self, df: DataFrame, label_col: str, segmentation: Dict[str, Any]):
        """Split a Spark DataFrame into user-defined segments."""
        column = segmentation.get("column")
        if not column:
            raise ValueError("Segmentation requires a 'column'.")
        if column not in df.columns:
            raise ValueError(f"Segmentation column '{column}' not found in dataset.")

        include_overall = bool(segmentation.get("include_overall", True))
        min_rows = int(segmentation.get("min_rows", 1))
        dropna = bool(segmentation.get("dropna", True))
        default_end_inclusive = bool(segmentation.get("end_inclusive", False))
        segments_cfg = segmentation.get("segments")

        if not segments_cfg:
            distinct_df = df.select(column).distinct()
            if dropna:
                distinct_df = distinct_df.where(F.col(column).isNotNull())
            values = [row[column] for row in distinct_df.collect()]
            segments_cfg = [
                {"name": f"{column}={value}", "values": [value]}
                for value in values
            ]

        segments = []
        meta_segments = []

        for idx, segment in enumerate(segments_cfg):
            if not isinstance(segment, dict):
                raise ValueError("Each segmentation entry must be a dict.")

            if "values" in segment:
                raw_values = segment["values"]
                values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
                mask = F.col(column).isin(list(values))
                criteria = {"values": list(values)}
                default_name = f"{column}={'|'.join(str(v) for v in values)}"
            else:
                start = segment.get("start")
                end = segment.get("end")
                if start is None and end is None:
                    raise ValueError("Range-based segmentation requires at least 'start' or 'end'.")
                ts_col = F.to_timestamp(F.col(column))
                mask = F.lit(True)
                if start is not None:
                    mask = mask & (ts_col >= F.lit(str(start)).cast("timestamp"))
                if end is not None:
                    end_expr = F.lit(str(end)).cast("timestamp")
                    if segment.get("end_inclusive", default_end_inclusive):
                        mask = mask & (ts_col <= end_expr)
                    else:
                        mask = mask & (ts_col < end_expr)
                criteria = {
                    "start": str(start) if start is not None else None,
                    "end": str(end) if end is not None else None,
                    "end_inclusive": bool(segment.get("end_inclusive", default_end_inclusive)),
                }
                default_name = f"{column}_range_{idx + 1}"

            name = segment.get("name") or default_name
            df_seg = df.where(mask)
            row_count = df_seg.count()
            meta = {"name": name, "row_count": int(row_count), "criteria": criteria}
            if row_count < min_rows:
                meta["status"] = "skipped"
                meta["reason"] = f"row_count < min_rows ({min_rows})"
                meta_segments.append(meta)
                continue

            meta["status"] = "completed"
            meta_segments.append(meta)
            segments.append((name, df_seg))

        return {
            "column": column,
            "include_overall": include_overall,
            "min_rows": min_rows,
            "segments": meta_segments,
        }, segments

    def _run_single(
        self,
        model,
        df: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        threshold: float = 0.5,
        columns: Optional[List[str]] = None,
        section_columns: Optional[Dict[str, List[str]]] = None,
        artifact_scope: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run model testing once for a single dataset scope."""
        sections = sections or self.SECTIONS
        feature_cols = feature_cols or [c for c in df.columns if c != label_col]
        numeric_cols = [c for c in feature_cols if c in get_numeric_columns(df)]
        components = self._build_components(
            os.path.join(self.output_dir, "_segments", _slugify(artifact_scope))
            if artifact_scope
            else self.output_dir
        )

        results = {}
        section_columns = section_columns or {}

        if "effectiveness" in sections:
            cols = section_columns.get("effectiveness") or columns
            df_eff, feat_eff = self._select_columns(df, cols, feature_cols, label_col) if cols else (df, feature_cols)
            metrics, plots, explanations = components["effectiveness"].evaluate(
                model, df_eff, label_col=label_col, feature_cols=feat_eff, threshold=threshold, **kwargs
            )
            results["effectiveness"] = {"metrics": metrics, "plots": plots, "explanations": explanations}

        if "efficiency" in sections:
            cols = section_columns.get("efficiency") or columns
            df_eff, feat_eff = self._select_columns(df, cols, feature_cols, label_col) if cols else (df, feature_cols)
            metrics, plots, explanations = components["efficiency"].evaluate(
                model, df_eff, label_col=label_col, feature_cols=feat_eff, threshold=threshold, **kwargs
            )
            results["efficiency"] = {"metrics": metrics, "plots": plots, "explanations": explanations}

        if "stability" in sections:
            cols = section_columns.get("stability") or columns
            df_stab, feat_stab = self._select_columns(df, cols, feature_cols, label_col) if cols else (df, feature_cols)
            metrics, plots, artifacts, explanations = components["stability"].evaluate(
                model, df_stab, label_col=label_col, feature_cols=feat_stab, **kwargs
            )
            results["stability"] = {"metrics": metrics, "plots": plots, "artifacts": artifacts, "explanations": explanations}

        if "interpretability" in sections:
            cols = section_columns.get("interpretability") or columns
            if cols:
                cols = [c for c in cols if c in numeric_cols] if numeric_cols else cols
                if cols:
                    df_int, feat_int = self._select_columns(df, cols, feature_cols, label_col)
                else:
                    feat_int = numeric_cols if numeric_cols else feature_cols
                    df_int = df.select(*(feat_int + [label_col]))
            else:
                feat_int = numeric_cols if numeric_cols else feature_cols
                df_int = df.select(*(feat_int + [label_col]))
            interp = components["interpretability"].evaluate(
                model, df_int, label_col=label_col, feature_cols=feat_int, **kwargs
            )
            results["interpretability"] = interp

        return results

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
        segmentation=None,
        **kwargs,
    ) -> Dict[str, Any]:
        segmentation_cfg = self._normalize_segmentation(segmentation) if segmentation is not None else None

        if not segmentation_cfg:
            return self._run_single(
                model=model,
                df=df,
                label_col=label_col,
                feature_cols=feature_cols,
                sections=sections,
                threshold=threshold,
                columns=columns,
                section_columns=section_columns,
                **kwargs,
            )

        seg_meta, segments = self._segment_dataset(df, label_col, segmentation_cfg)
        results = {"segmentation": seg_meta, "segments": {}}

        if seg_meta.get("include_overall", True):
            results["overall"] = self._run_single(
                model=model,
                df=df,
                label_col=label_col,
                feature_cols=feature_cols,
                sections=sections,
                threshold=threshold,
                columns=columns,
                section_columns=section_columns,
                artifact_scope="overall",
                **kwargs,
            )

        for segment_name, df_seg in segments:
            print(f"Running segmented model testing for: {segment_name}")
            segment_feature_cols = feature_cols or [c for c in df_seg.columns if c != label_col]
            results["segments"][segment_name] = self._run_single(
                model=model,
                df=df_seg,
                label_col=label_col,
                feature_cols=segment_feature_cols,
                sections=sections,
                threshold=threshold,
                columns=columns,
                section_columns=section_columns,
                artifact_scope=segment_name,
                **kwargs,
            )

        return results

    def generate_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        return self.report_builder.build(results, filename=filename or "model_testing_agent_Model_Testing_Report_pyspark.pdf")

    def save_results(self, results: Dict[str, Any], filename: str = "results.json") -> str:
        path = os.path.join(self.output_dir, filename)

        def convert(obj):
            if hasattr(obj, "tolist"):
                try:
                    return obj.tolist()
                except Exception:
                    pass
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
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
    def load_data(
        path: Optional[str] = None,
        label_col: Optional[str] = None,
        spark=None,
        sql: Optional[str] = None,
        conn: Optional[str] = None,
        loader_py: Optional[str] = None,
        loader_fn: str = "load_data",
        jdbc_options: Optional[Dict[str, str]] = None,
    ):
        return _load_data(
            path=path,
            label_col=label_col,
            spark=spark,
            sql=sql,
            conn=conn,
            loader_py=loader_py,
            loader_fn=loader_fn,
            jdbc_options=jdbc_options,
        )
