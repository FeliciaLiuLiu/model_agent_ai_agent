"""Unified Spark DataFrame loader for EDA Spark inputs."""
from __future__ import annotations

import glob
import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .utils import detect_latest_dataset, to_local_file_uri


SUPPORTED_EXTS = {".csv", ".tsv", ".parquet", ".json", ".xlsx", ".xls", ".feather"}
AUTO_DB_EXTS = {".db", ".sqlite", ".sqlite3"}
PREFERRED_SQL_TABLES = ("aml_dataset", "eda_dataset", "eda_input")


def _is_glob(path: str) -> bool:
    return any(ch in path for ch in ["*", "?", "["])


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _expand_paths(data: Sequence[str], recursive: bool = False) -> List[Path]:
    paths: List[Path] = []
    for item in data:
        if not item:
            continue
        if _is_glob(item):
            for match in glob.glob(item, recursive=recursive):
                paths.append(Path(match))
            continue
        p = Path(item)
        if p.is_dir():
            for ext in SUPPORTED_EXTS:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                paths.extend(p.glob(pattern))
            continue
        paths.append(p)
    existing = [p for p in paths if p.exists()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    pkg_root = Path(__file__).resolve().parents[1]
    candidate = pkg_root / data_dir
    return candidate if candidate.exists() else p


def _scan_dir_for_exts(data_dir: str, exts: Sequence[str], recursive: bool) -> List[Path]:
    base = _resolve_data_dir(data_dir)
    if not base.exists():
        return []
    paths: List[Path] = []
    for ext in exts:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        paths.extend(base.glob(pattern))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_excel_with_spark(spark, path: str):
    try:
        return (
            spark.read.format("com.crealytics.spark.excel")
            .option("header", True)
            .option("inferSchema", True)
            .load(path)
        )
    except Exception:
        return None


def _read_single_spark(spark, path: Path):
    ext = path.suffix.lower()
    local_uri = to_local_file_uri(str(path))
    if ext == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(local_uri)
    if ext == ".tsv":
        return spark.read.option("header", True).option("inferSchema", True).option("sep", "\t").csv(local_uri)
    if ext == ".parquet":
        return spark.read.parquet(local_uri)
    if ext == ".json":
        return spark.read.option("multiLine", True).json(local_uri)
    if ext in {".xlsx", ".xls"}:
        df = _read_excel_with_spark(spark, local_uri)
        if df is not None:
            return df
        try:
            pdf = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError("Reading Excel requires openpyxl. Install openpyxl to proceed.") from exc
        return spark.createDataFrame(pdf)
    if ext == ".feather":
        try:
            pdf = pd.read_feather(path)
        except ImportError as exc:
            raise ImportError("Reading Feather requires pyarrow. Install pyarrow to proceed.") from exc
        return spark.createDataFrame(pdf)
    raise ValueError(f"Unsupported file extension: {ext}")


def _concat_frames(frames: List):
    if not frames:
        raise ValueError("No input files matched.")
    df = frames[0]
    for frame in frames[1:]:
        df = df.unionByName(frame, allowMissingColumns=True)
    return df


def _load_python_file(path: str, spark):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    spec = importlib.util.spec_from_file_location("eda_spark_loader_module", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    if hasattr(module, "load") and callable(module.load):
        df = module.load()
    elif hasattr(module, "df"):
        df = getattr(module, "df")
    else:
        raise ValueError("Python file must define `load()` or `df`.")
    return _ensure_spark_df(df, spark)


def _load_python_code(code: str, spark):
    local_ns: dict = {}
    exec(code, {}, local_ns)
    if "load" in local_ns and callable(local_ns["load"]):
        df = local_ns["load"]()
    elif "df" in local_ns:
        df = local_ns["df"]
    else:
        raise ValueError("Python code must define `load()` or `df`.")
    return _ensure_spark_df(df, spark)


def _load_notebook(path: str, spark):
    nb_path = Path(path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with open(nb_path, "r", encoding="utf-8-sig") as f:
        nb = json.load(f)
    local_ns: dict = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else str(source)
        exec(code, {}, local_ns)
    if "load" in local_ns and callable(local_ns["load"]):
        df = local_ns["load"]()
    elif "df" in local_ns:
        df = local_ns["df"]
    else:
        raise ValueError("Notebook must define `load()` or `df` in code cells.")
    return _ensure_spark_df(df, spark)


def _ensure_spark_df(df, spark):
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except Exception:
        SparkDataFrame = None
    if SparkDataFrame is not None and isinstance(df, SparkDataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return spark.createDataFrame(df)
    raise ValueError("Python/Notebook loader did not return a pandas or Spark DataFrame.")


def _clean_sql_lines(sql_text: str) -> List[str]:
    lines: List[str] = []
    for raw in sql_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            continue
        lines.append(line)
    return lines


def _is_select_only(sql_text: str) -> bool:
    lines = _clean_sql_lines(sql_text)
    if not lines:
        return False
    head = lines[0].lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False
    return ";" not in sql_text.strip().rstrip(";")


def _sqlite_objects(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [(str(name), str(obj_type)) for name, obj_type in rows]


def _pick_sql_table(objects: List[Tuple[str, str]]) -> Optional[str]:
    names = [name for name, _ in objects]
    for preferred in PREFERRED_SQL_TABLES:
        if preferred in names:
            return preferred
    if len(names) == 1:
        return names[0]
    return None


def _load_sql_with_db(spark, sql: str, db: str):
    if db.startswith("jdbc:"):
        return (
            spark.read.format("jdbc")
            .option("url", db)
            .option("dbtable", f"({sql}) t")
            .load()
        )

    db_path = None
    if db.startswith("sqlite:///"):
        db_path = db.replace("sqlite:///", "", 1)
    elif db.startswith("sqlite://"):
        db_path = db.replace("sqlite://", "", 1)
    else:
        p = Path(db)
        if p.exists() and p.suffix.lower() in AUTO_DB_EXTS:
            db_path = str(p)

    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            pdf = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        return spark.createDataFrame(pdf)

    try:
        import sqlalchemy  # type: ignore
    except ImportError as exc:
        raise ImportError("SQLAlchemy is required for non-sqlite DB connections.") from exc
    engine = sqlalchemy.create_engine(db)
    pdf = pd.read_sql_query(sql, engine)
    return spark.createDataFrame(pdf)


def _load_sql_auto(spark, sql_files: List[Path], db_files: List[Path]) -> List:
    if not sql_files:
        return []
    texts = [p.read_text(encoding="utf-8") for p in sql_files]
    if db_files and all(_is_select_only(t) for t in texts):
        db_path = str(db_files[0])
        conn = sqlite3.connect(db_path)
        try:
            frames = [spark.createDataFrame(pd.read_sql_query(t, conn)) for t in texts]
        finally:
            conn.close()
        return frames

    conn = sqlite3.connect(":memory:")
    try:
        for text in texts:
            conn.executescript(text)
        objects = _sqlite_objects(conn)
        table = _pick_sql_table(objects)
        if not table:
            raise ValueError(
                "SQL auto-exec produced multiple tables/views. "
                "Create a single table/view or name it one of: "
                f"{', '.join(PREFERRED_SQL_TABLES)}."
            )
        pdf = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        return [spark.createDataFrame(pdf)]
    finally:
        conn.close()


class DataLoader:
    """Unified loader that returns a Spark DataFrame and a data_source string."""

    def __init__(
        self,
        spark,
        data: Optional[Sequence[str]] = None,
        sql: Optional[str] = None,
        db: Optional[str] = None,
        py: Optional[str] = None,
        py_code: Optional[str] = None,
        nb: Optional[str] = None,
        data_dir: str = "./data",
        recursive: bool = False,
        auto_exec: bool = False,
    ) -> None:
        self.spark = spark
        self.data = _as_list(data)
        self.sql = sql
        self.db = db
        self.py = py
        self.py_code = py_code
        self.nb = nb
        self.data_dir = data_dir
        self.recursive = recursive
        self.auto_exec = auto_exec

    def load(self) -> Tuple[Any, str]:
        mode = self._detect_mode()

        if mode == "sql":
            if not self.db:
                raise ValueError("SQL mode requires --db.")
            df = _load_sql_with_db(self.spark, self.sql or "", self.db)
            return df, f"sql:{self.sql}"

        if mode == "py":
            df = _load_python_file(self.py or "", self.spark)
            return df, f"py:{self.py}"

        if mode == "py_code":
            df = _load_python_code(self.py_code or "", self.spark)
            return df, "py_code"

        if mode == "nb":
            df = _load_notebook(self.nb or "", self.spark)
            return df, f"nb:{self.nb}"

        if mode == "data":
            if not self.data:
                path = detect_latest_dataset(data_dir=self.data_dir)
                df = _read_single_spark(self.spark, Path(path))
                return df, path

            paths = _expand_paths(self.data, recursive=self.recursive)
            if not paths:
                raise FileNotFoundError("No matching data files found.")
            frames = [_read_single_spark(self.spark, p) for p in paths]
            df = _concat_frames(frames)
            data_source = ";".join(str(p) for p in paths)
            return df, data_source

        if mode == "auto":
            data_paths = _scan_dir_for_exts(self.data_dir, list(SUPPORTED_EXTS), self.recursive)
            sql_files = _scan_dir_for_exts(self.data_dir, [".sql"], self.recursive)
            py_files = _scan_dir_for_exts(self.data_dir, [".py"], self.recursive)
            nb_files = _scan_dir_for_exts(self.data_dir, [".ipynb"], self.recursive)
            db_files = _scan_dir_for_exts(self.data_dir, list(AUTO_DB_EXTS), self.recursive)

            frames: List[Any] = []
            sources: List[str] = []
            if data_paths:
                frames.extend([_read_single_spark(self.spark, p) for p in data_paths])
                sources.extend(str(p) for p in data_paths)
            if sql_files:
                sql_frames = _load_sql_auto(self.spark, sql_files, db_files)
                frames.extend(sql_frames)
                sources.extend(str(p) for p in sql_files)
            if py_files:
                for p in py_files:
                    frames.append(_load_python_file(str(p), self.spark))
                    sources.append(str(p))
            if nb_files:
                for p in nb_files:
                    frames.append(_load_notebook(str(p), self.spark))
                    sources.append(str(p))

            if not frames:
                raise FileNotFoundError("No supported data/auto-exec files found in ./data.")

            df = _concat_frames(frames)
            return df, ";".join(sources)

        raise ValueError("No valid data input provided.")

    def _detect_mode(self) -> str:
        provided = {
            "data": bool(self.data),
            "sql": bool(self.sql),
            "py": bool(self.py),
            "py_code": bool(self.py_code),
            "nb": bool(self.nb),
        }
        active = [k for k, v in provided.items() if v]
        if len(active) > 1:
            raise ValueError(f"Only one input mode is allowed. Provided: {active}")
        if not active and self.auto_exec:
            return "auto"
        if self.sql:
            return "sql"
        if self.py:
            return "py"
        if self.py_code:
            return "py_code"
        if self.nb:
            return "nb"
        return "data"
