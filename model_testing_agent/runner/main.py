"""Main runner: Non-Interactive mode."""
import importlib.util
import inspect
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from ..core.report import ReportBuilder
from ..matrices.effectiveness import ModelEffectiveness
from ..matrices.efficiency import ModelEfficiency
from ..matrices.stability import ModelStability
from ..matrices.interpretability import ModelInterpretability


LABEL_CANDIDATES = ["label", "target", "y", "class", "fraud", "is_fraud", "is_suspicious"]


def _guess_label_col(df: pd.DataFrame, label_col: Optional[str] = None) -> Optional[str]:
    """Infer a label column from a DataFrame when one is not provided."""
    if label_col and label_col in df.columns:
        return label_col

    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name

    return None


def _split_dataframe(df: pd.DataFrame, label_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Split a DataFrame into X / y if a label column is available."""
    inferred = _guess_label_col(df, label_col)
    if inferred and inferred in df.columns:
        X = df.drop(columns=[inferred])
        return X, df[inferred], list(X.columns)
    return df, None, list(df.columns)


def _read_sql_text(sql: str) -> str:
    """Read SQL from a .sql file path or return the query text directly."""
    sql_path = Path(sql)
    if sql_path.exists() and sql_path.is_file() and sql_path.suffix.lower() == ".sql":
        return sql_path.read_text(encoding="utf-8")
    return sql


def _load_sql_dataframe(sql: str, conn: str) -> pd.DataFrame:
    """Load a pandas DataFrame from SQL text using a connection string."""
    sql_text = _read_sql_text(sql)

    if conn.startswith("sqlite:///"):
        db_path = conn.replace("sqlite:///", "", 1)
        with sqlite3.connect(db_path) as db_conn:
            return pd.read_sql_query(sql_text, db_conn)

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for non-sqlite SQL connections. "
            "Install sqlalchemy or use a sqlite:/// connection string."
        ) from exc

    engine = create_engine(conn)
    with engine.connect() as db_conn:
        return pd.read_sql_query(sql_text, db_conn)


def _load_python_callable(loader_py: str, loader_fn: str):
    """Load a callable from a Python loader file."""
    loader_path = Path(loader_py)
    if not loader_path.exists():
        raise FileNotFoundError(f"Python loader file not found: {loader_py}")

    spec = importlib.util.spec_from_file_location(f"model_testing_loader_{loader_path.stem}", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Python loader file: {loader_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, loader_fn):
        raise AttributeError(f"Loader function '{loader_fn}' not found in {loader_py}")
    return getattr(module, loader_fn)


def _invoke_loader(loader_py: str, loader_fn: str, **kwargs):
    """Invoke a Python loader function, passing supported kwargs only."""
    fn = _load_python_callable(loader_py, loader_fn)
    signature = inspect.signature(fn)
    params = signature.parameters.values()
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    filtered_kwargs = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**filtered_kwargs)


def _normalize_python_payload(payload, label_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Normalize Python loader outputs into X / y / feature_names."""
    if isinstance(payload, pd.DataFrame):
        return _split_dataframe(payload, label_col)

    if isinstance(payload, dict):
        if "X" in payload and "y" in payload:
            X = payload["X"]
            y = payload["y"]
            feature_names = payload.get("feature_names")
            if feature_names is None:
                feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f_{i}" for i in range(np.asarray(X).shape[1])]
            return X, y, list(feature_names)
        if "df" in payload:
            df = payload["df"]
            inferred_label = payload.get("label_col", label_col)
            feature_names = payload.get("feature_names")
            if feature_names:
                X_df = df.loc[:, [c for c in feature_names if c in df.columns]]
                if inferred_label and inferred_label in X_df.columns:
                    X_df = X_df.drop(columns=[inferred_label])
                y = df[inferred_label] if inferred_label and inferred_label in df.columns else None
                return X_df, y, list(X_df.columns)
            return _split_dataframe(df, inferred_label)

    if isinstance(payload, tuple):
        if len(payload) == 3:
            X, y, feature_names = payload
            if isinstance(X, pd.DataFrame) and (isinstance(y, str) or y is None):
                df = X
                inferred_label = y or label_col
                if feature_names:
                    X_df = df.loc[:, [c for c in feature_names if c in df.columns]]
                    if inferred_label and inferred_label in X_df.columns:
                        X_df = X_df.drop(columns=[inferred_label])
                else:
                    X_df = df.drop(columns=[inferred_label]) if inferred_label and inferred_label in df.columns else df
                target = df[inferred_label] if inferred_label and inferred_label in df.columns else None
                return X_df, target, list(X_df.columns)
            if feature_names is None:
                feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f_{i}" for i in range(np.asarray(X).shape[1])]
            return X, y, list(feature_names)
        if len(payload) == 2 and isinstance(payload[0], pd.DataFrame):
            df, inferred_label = payload
            return _split_dataframe(df, inferred_label or label_col)

    raise ValueError(
        "Python loader must return a pandas DataFrame, "
        "(X, y, feature_names), or a dict containing either df/label_col or X/y."
    )


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug for per-run artifact directories."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return slug or "segment"


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
        self.report_builder = ReportBuilder(output_dir=output_dir, tag=experiment_tag)

    def _build_components(self, data_dir: Optional[str] = None):
        """Create fresh matrix evaluators for a single run scope."""
        target_dir = data_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        return {
            "effectiveness": ModelEffectiveness(data_dir=target_dir),
            "efficiency": ModelEfficiency(data_dir=target_dir),
            "stability": ModelStability(data_dir=target_dir),
            "interpretability": ModelInterpretability(data_dir=target_dir),
        }

    def _select_columns(self, X_in, cols, feature_names_all):
        """Select a subset of feature columns for one section."""
        if not cols:
            return X_in, feature_names_all
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

    def _normalize_segmentation(self, segmentation) -> Dict[str, Any]:
        """Normalize segmentation input from dict, JSON string, or JSON file."""
        if segmentation is None:
            return {}
        if isinstance(segmentation, str):
            seg_path = Path(segmentation)
            if seg_path.exists() and seg_path.is_file():
                segmentation = json.loads(seg_path.read_text(encoding="utf-8"))
            else:
                segmentation = json.loads(segmentation)
        if not isinstance(segmentation, dict):
            raise ValueError("Segmentation must be a dict, JSON string, or path to a JSON file.")
        return segmentation

    def _prepare_segment_features(
        self,
        X,
        feature_names,
        columns,
        section_columns,
        segment_column: str,
        keep_column_in_features: bool,
    ):
        """Drop a segmentation-only column from model inputs unless explicitly retained."""
        if keep_column_in_features:
            return X, feature_names, columns, section_columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Segmentation requires X to be a pandas DataFrame with named columns.")
        if segment_column not in X.columns:
            return X, feature_names, columns, section_columns

        names_all = list(feature_names) if feature_names else list(X.columns)
        segment_index = names_all.index(segment_column) if segment_column in names_all else None

        def sanitize(selected_cols, scope_name):
            if not selected_cols:
                return None
            if isinstance(selected_cols, str):
                selected_cols = [c.strip() for c in selected_cols.split(",") if c.strip()]

            filtered = []
            for item in selected_cols:
                if isinstance(item, int):
                    if segment_index is not None and item == segment_index:
                        continue
                    if segment_index is not None and item > segment_index:
                        filtered.append(item - 1)
                    else:
                        filtered.append(item)
                    continue
                if item != segment_column:
                    filtered.append(item)

            if not filtered:
                raise ValueError(
                    f"Selected columns for {scope_name} contain only the segmentation column "
                    f"'{segment_column}'. Set keep_column_in_features=true to evaluate it."
                )
            return filtered

        X_eval = X.drop(columns=[segment_column]).copy()
        feature_names_eval = [name for name in names_all if name != segment_column]
        columns_eval = sanitize(columns, "all sections")
        section_columns_eval = {}
        for section, selected in (section_columns or {}).items():
            if selected:
                section_columns_eval[section] = sanitize(selected, f"section '{section}'")
        return X_eval, feature_names_eval, columns_eval, section_columns_eval

    def _time_group_labels(self, base_series: pd.Series, freq: str) -> pd.Series:
        """Derive group labels from a datetime-like pandas Series."""
        dt_series = pd.to_datetime(base_series, errors="coerce")
        freq = (freq or "month").lower()
        if freq == "month":
            return dt_series.dt.to_period("M").astype("string")
        if freq == "quarter":
            return dt_series.dt.to_period("Q").astype("string")
        if freq == "year":
            return dt_series.dt.to_period("Y").astype("string")
        if freq == "day":
            return dt_series.dt.strftime("%Y-%m-%d").astype("string")
        if freq == "week":
            iso = dt_series.dt.isocalendar()
            return (
                iso["year"].astype("string")
                + "-W"
                + iso["week"].astype("string").str.zfill(2)
            )
        raise ValueError("Unsupported groupby time frequency. Use day, week, month, quarter, or year.")

    def _segment_dataset_groupby(self, X, y, column: str, base_series: pd.Series, segmentation: Dict[str, Any]):
        """Split a pandas dataset by derived or direct group labels."""
        include_overall = bool(segmentation.get("include_overall", True))
        min_rows = int(segmentation.get("min_rows", 1))
        dropna = bool(segmentation.get("dropna", True))
        groupby_cfg = segmentation.get("groupby") or {}
        kind = (groupby_cfg.get("kind") or ("time" if groupby_cfg.get("freq") else "value")).lower()
        selected_groups = groupby_cfg.get("selected_groups") or segmentation.get("selected_groups")
        name_prefix = groupby_cfg.get("name_prefix")

        if kind == "time":
            freq = groupby_cfg.get("freq", "month")
            labels = self._time_group_labels(base_series, freq)
            default_prefix = name_prefix or f"{column}_{freq}"
            criteria_base = {"mode": "groupby", "kind": "time", "freq": freq}
        elif kind == "value":
            labels = base_series.astype("string")
            default_prefix = name_prefix or column
            criteria_base = {"mode": "groupby", "kind": "value"}
        else:
            raise ValueError("Groupby segmentation kind must be either 'time' or 'value'.")

        available_groups = labels.dropna().drop_duplicates().tolist() if dropna else labels.drop_duplicates().tolist()
        if selected_groups:
            groups_to_run = list(selected_groups)
        else:
            groups_to_run = sorted(str(group) for group in available_groups)

        segments = []
        meta_segments = []
        for group_value in groups_to_run:
            mask = labels == str(group_value)
            row_count = int(mask.sum())
            meta = {
                "name": f"{default_prefix}={group_value}",
                "row_count": row_count,
                "criteria": {**criteria_base, "group": str(group_value)},
            }
            if row_count == 0:
                meta["status"] = "skipped"
                meta["reason"] = "group not found in dataset"
                meta_segments.append(meta)
                continue
            if row_count < min_rows:
                meta["status"] = "skipped"
                meta["reason"] = f"row_count < min_rows ({min_rows})"
                meta_segments.append(meta)
                continue

            X_seg = X.loc[mask].copy()
            if isinstance(y, pd.Series):
                y_seg = y.loc[mask]
            else:
                y_seg = np.asarray(y)[mask.to_numpy()]
            meta["status"] = "completed"
            meta_segments.append(meta)
            segments.append((meta["name"], X_seg, y_seg))

        return {
            "column": column,
            "include_overall": include_overall,
            "min_rows": min_rows,
            "mode": "groupby",
            "segments": meta_segments,
        }, segments

    def _segment_dataset(self, X, y, segmentation: Dict[str, Any]):
        """Split a pandas dataset into user-defined segments."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Segmentation requires X to be a pandas DataFrame with named columns.")

        column = segmentation.get("column")
        if not column:
            raise ValueError("Segmentation requires a 'column'.")
        if column not in X.columns:
            raise ValueError(f"Segmentation column '{column}' not found in X.")

        mode = (segmentation.get("mode") or ("groupby" if segmentation.get("groupby") else "segments")).lower()
        base_series = X[column]
        if mode == "groupby":
            return self._segment_dataset_groupby(X, y, column, base_series, segmentation)

        include_overall = bool(segmentation.get("include_overall", True))
        min_rows = int(segmentation.get("min_rows", 1))
        dropna = bool(segmentation.get("dropna", True))
        default_end_inclusive = bool(segmentation.get("end_inclusive", False))
        segments_cfg = segmentation.get("segments")

        if not segments_cfg:
            values = base_series.unique()
            if dropna:
                values = [v for v in values if pd.notna(v)]
            segments_cfg = [
                {"name": f"{column}={value}", "values": [value]}
                for value in values
            ]

        dt_series = None
        if any(("start" in segment or "end" in segment) for segment in segments_cfg):
            dt_series = pd.to_datetime(base_series, errors="coerce")

        segments = []
        meta_segments = []

        for idx, segment in enumerate(segments_cfg):
            if not isinstance(segment, dict):
                raise ValueError("Each segmentation entry must be a dict.")

            if "values" in segment:
                raw_values = segment["values"]
                values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
                mask = base_series.isin(list(values))
                criteria = {"values": list(values)}
                default_name = f"{column}={'|'.join(str(v) for v in values)}"
            else:
                if dt_series is None:
                    dt_series = pd.to_datetime(base_series, errors="coerce")
                start = segment.get("start")
                end = segment.get("end")
                if start is None and end is None:
                    raise ValueError("Range-based segmentation requires at least 'start' or 'end'.")
                mask = pd.Series(True, index=X.index)
                if start is not None:
                    mask &= dt_series >= pd.Timestamp(start)
                if end is not None:
                    end_ts = pd.Timestamp(end)
                    if segment.get("end_inclusive", default_end_inclusive):
                        mask &= dt_series <= end_ts
                    else:
                        mask &= dt_series < end_ts
                criteria = {
                    "start": str(start) if start is not None else None,
                    "end": str(end) if end is not None else None,
                    "end_inclusive": bool(segment.get("end_inclusive", default_end_inclusive)),
                }
                default_name = f"{column}_range_{idx + 1}"

            name = segment.get("name") or default_name
            row_count = int(mask.sum())
            meta = {"name": name, "row_count": row_count, "criteria": criteria}

            if row_count < min_rows:
                meta["status"] = "skipped"
                meta["reason"] = f"row_count < min_rows ({min_rows})"
                meta_segments.append(meta)
                continue

            X_seg = X.loc[mask].copy()
            if isinstance(y, pd.Series):
                y_seg = y.loc[mask]
            else:
                y_seg = np.asarray(y)[mask.to_numpy()]
            meta["status"] = "completed"
            meta_segments.append(meta)
            segments.append((name, X_seg, y_seg))

        return {
            "column": column,
            "include_overall": include_overall,
            "min_rows": min_rows,
            "segments": meta_segments,
        }, segments

    def _run_single(
        self,
        model,
        X,
        y,
        feature_names=None,
        sections=None,
        threshold=0.5,
        columns=None,
        section_columns=None,
        artifact_scope: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run model testing once for a single dataset scope."""
        sections = sections or self.SECTIONS
        feature_names = feature_names or (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f'f_{i}' for i in range(X.shape[1])]
        )
        components = self._build_components(
            os.path.join(self.output_dir, "_segments", _slugify(artifact_scope))
            if artifact_scope
            else self.output_dir
        )

        results = {}
        section_columns = section_columns or {}

        if 'effectiveness' in sections:
            print("Running Effectiveness evaluation...")
            cols = section_columns.get('effectiveness') or columns
            X_eff, _ = self._select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots, explanations = components["effectiveness"].evaluate(model, X_eff, y, threshold=threshold, **kwargs)
            results['effectiveness'] = {'metrics': metrics, 'plots': plots, 'explanations': explanations}

        if 'efficiency' in sections:
            print("Running Efficiency evaluation...")
            cols = section_columns.get('efficiency') or columns
            X_eff, _ = self._select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots, explanations = components["efficiency"].evaluate(model, X_eff, y, threshold=threshold, **kwargs)
            results['efficiency'] = {'metrics': metrics, 'plots': plots, 'explanations': explanations}

        if 'stability' in sections:
            print("Running Stability evaluation...")
            cols = section_columns.get('stability') or columns
            X_stab, feature_names_stab = self._select_columns(X, cols, feature_names) if cols else (X, feature_names)
            metrics, plots, artifacts, explanations = components["stability"].evaluate(
                model, X_stab, y, feature_names=feature_names_stab, **kwargs
            )
            results['stability'] = {'metrics': metrics, 'plots': plots, 'artifacts': artifacts, 'explanations': explanations}

        if 'interpretability' in sections:
            print("Running Interpretability evaluation...")
            cols = section_columns.get('interpretability') or columns
            X_int, feature_names_int = self._select_columns(X, cols, feature_names) if cols else (X, feature_names)
            interp = components["interpretability"].evaluate(
                model, X_int, y, feature_names=feature_names_int, **kwargs
            )
            results['interpretability'] = interp

        return results

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
        segmentation=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run model evaluation on all (or specified) sections."""
        segmentation_cfg = self._normalize_segmentation(segmentation) if segmentation is not None else None

        if not segmentation_cfg:
            return self._run_single(
                model=model,
                X=X,
                y=y,
                feature_names=feature_names,
                sections=sections,
                threshold=threshold,
                columns=columns,
                section_columns=section_columns,
                **kwargs,
            )

        seg_meta, segments = self._segment_dataset(X, y, segmentation_cfg)
        keep_column_in_features = bool(segmentation_cfg.get("keep_column_in_features", False))
        seg_meta["keep_column_in_features"] = keep_column_in_features
        results = {"segmentation": seg_meta, "segments": {}}

        X_eval, feature_names_eval, columns_eval, section_columns_eval = self._prepare_segment_features(
            X=X,
            feature_names=feature_names,
            columns=columns,
            section_columns=section_columns,
            segment_column=seg_meta["column"],
            keep_column_in_features=keep_column_in_features,
        )

        if seg_meta.get("include_overall", True):
            results["overall"] = self._run_single(
                model=model,
                X=X_eval,
                y=y,
                feature_names=feature_names_eval,
                sections=sections,
                threshold=threshold,
                columns=columns_eval,
                section_columns=section_columns_eval,
                artifact_scope="overall",
                **kwargs,
            )

        for segment_name, X_seg, y_seg in segments:
            print(f"Running segmented model testing for: {segment_name}")
            X_seg_eval, feature_names_seg, columns_seg, section_columns_seg = self._prepare_segment_features(
                X=X_seg,
                feature_names=list(X_seg.columns),
                columns=columns,
                section_columns=section_columns,
                segment_column=seg_meta["column"],
                keep_column_in_features=keep_column_in_features,
            )
            results["segments"][segment_name] = self._run_single(
                model=model,
                X=X_seg_eval,
                y=y_seg,
                feature_names=feature_names_seg,
                sections=sections,
                threshold=threshold,
                columns=columns_seg,
                section_columns=section_columns_seg,
                artifact_scope=segment_name,
                **kwargs,
            )

        return results

    def generate_report(self, results: Dict[str, Any], filename=None) -> str:
        """Generate PDF report."""
        return self.report_builder.build(results, filename=filename or "model_testing_agent_Model_Testing_Report.pdf")

    def save_results(self, results: Dict[str, Any], filename="results.json") -> str:
        """Save results to JSON."""
        path = os.path.join(self.output_dir, filename)
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
            if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)): return [convert(v) for v in obj]
            return obj
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        return path

    @staticmethod
    def load_model(path: str):
        """Load model from file."""
        return joblib.load(path)

    @staticmethod
    def load_data(
        path: Optional[str] = None,
        label_col=None,
        sql: Optional[str] = None,
        conn: Optional[str] = None,
        loader_py: Optional[str] = None,
        loader_fn: str = "load_data",
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """Load data from a file, SQL query, or Python loader."""
        provided = [value is not None for value in (path, sql, loader_py)]
        if sum(provided) != 1:
            raise ValueError("Specify exactly one data source: path, sql, or loader_py.")

        if path is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext == '.parquet':
                df = pd.read_parquet(path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported: {ext}")
            return _split_dataframe(df, label_col)

        if sql is not None:
            if not conn:
                raise ValueError("A SQL connection string is required when sql is provided.")
            df = _load_sql_dataframe(sql, conn)
            return _split_dataframe(df, label_col)

        payload = _invoke_loader(loader_py, loader_fn, label_col=label_col)
        return _normalize_python_payload(payload, label_col)
