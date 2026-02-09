"""Unified pandas DataFrame loader for EDA inputs."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .utils import detect_latest_dataset


SUPPORTED_EXTS = {".csv", ".tsv", ".parquet", ".json", ".xlsx", ".xls", ".feather"}


def _flatten(items: Iterable[Iterable[str]]) -> List[str]:
    out: List[str] = []
    for group in items:
        out.extend(list(group))
    return out


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
    return [p for p in paths if p.exists()]


def _read_single(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError("Reading Excel requires openpyxl. Install openpyxl to proceed.") from exc
    if ext == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def _concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No input files matched.")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def _load_python_file(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    spec = importlib.util.spec_from_file_location("eda_loader_module", str(file_path))
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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Python loader did not return a pandas DataFrame.")
    return df


def _load_python_code(code: str) -> pd.DataFrame:
    local_ns: dict = {}
    exec(code, {}, local_ns)
    if "load" in local_ns and callable(local_ns["load"]):
        df = local_ns["load"]()
    elif "df" in local_ns:
        df = local_ns["df"]
    else:
        raise ValueError("Python code must define `load()` or `df`.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Python code did not return a pandas DataFrame.")
    return df


def _load_notebook(path: str) -> pd.DataFrame:
    nb_path = Path(path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with open(nb_path, "r", encoding="utf-8") as f:
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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Notebook execution did not return a pandas DataFrame.")
    return df


def _load_sql(sql: str, db: str) -> pd.DataFrame:
    if db.startswith("sqlite:///"):
        db_path = db.replace("sqlite:///", "", 1)
        conn = sqlite3.connect(db_path)
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()
    if db.startswith("sqlite://"):
        db_path = db.replace("sqlite://", "", 1)
        conn = sqlite3.connect(db_path)
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()
    try:
        import sqlalchemy  # type: ignore
    except ImportError as exc:
        raise ImportError("SQLAlchemy is required for non-sqlite DB connections.") from exc
    engine = sqlalchemy.create_engine(db)
    return pd.read_sql_query(sql, engine)


class DataLoader:
    """Unified loader that returns a pandas DataFrame and a data_source string."""

    def __init__(
        self,
        data: Optional[Sequence[str]] = None,
        sql: Optional[str] = None,
        db: Optional[str] = None,
        py: Optional[str] = None,
        py_code: Optional[str] = None,
        nb: Optional[str] = None,
        data_dir: str = "./data",
        recursive: bool = False,
    ) -> None:
        self.data = _as_list(data)
        self.sql = sql
        self.db = db
        self.py = py
        self.py_code = py_code
        self.nb = nb
        self.data_dir = data_dir
        self.recursive = recursive

    def load(self) -> Tuple[pd.DataFrame, str]:
        mode = self._detect_mode()

        if mode == "sql":
            if not self.db:
                raise ValueError("SQL mode requires --db.")
            df = _load_sql(self.sql or "", self.db)
            return df, f"sql:{self.sql}"

        if mode == "py":
            df = _load_python_file(self.py or "")
            return df, f"py:{self.py}"

        if mode == "py_code":
            df = _load_python_code(self.py_code or "")
            return df, "py_code"

        if mode == "nb":
            df = _load_notebook(self.nb or "")
            return df, f"nb:{self.nb}"

        if mode == "data":
            if not self.data:
                path = detect_latest_dataset(data_dir=self.data_dir)
                df = _read_single(Path(path))
                return df, path

            paths = _expand_paths(self.data, recursive=self.recursive)
            if not paths:
                raise FileNotFoundError("No matching data files found.")
            frames = [_read_single(p) for p in paths]
            df = _concat_frames(frames)
            data_source = ";".join(str(p) for p in paths)
            return df, data_source

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
        if self.sql:
            return "sql"
        if self.py:
            return "py"
        if self.py_code:
            return "py_code"
        if self.nb:
            return "nb"
        return "data"
