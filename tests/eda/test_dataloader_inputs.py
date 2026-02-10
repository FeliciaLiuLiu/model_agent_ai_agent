import json
import sqlite3
from pathlib import Path

import pandas as pd

from eda.dataloader import DataLoader


def _write_csv(tmp_path: Path, name: str = "input.csv", rows: int = 3) -> Path:
    df = pd.DataFrame({"a": list(range(1, rows + 1)), "b": ["x"] * rows})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _write_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "demo.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.executemany("INSERT INTO t (a, b) VALUES (?, ?)", [(1, "x"), (2, "y")])
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_py_loader(tmp_path: Path) -> Path:
    path = tmp_path / "loader.py"
    path.write_text(
        "import pandas as pd\n"
        "def load():\n"
        "    return pd.DataFrame({'a':[1,2],'b':['x','y']})\n",
        encoding="utf-8",
    )
    return path


def _write_sql_file(tmp_path: Path, name: str = "query.sql") -> Path:
    path = tmp_path / name
    path.write_text("SELECT * FROM t", encoding="utf-8")
    return path


def _write_nb_loader(tmp_path: Path) -> Path:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "\n",
                    "def load():\n",
                    "    values = np.array([1, 2, 3])\n",
                    "    return pd.DataFrame({'a': values, 'b': ['x', 'y', 'z']})\n",
                ],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "loader.ipynb"
    payload = json.dumps(nb)
    path.write_bytes(payload.encode("utf-8-sig"))
    return path


def test_dataloader_csv(local_tmp_path: Path):
    csv_path = _write_csv(local_tmp_path)
    loader = DataLoader(data=[str(csv_path)])
    df, source = loader.load()
    assert df.shape == (3, 2)
    assert "input.csv" in source


def test_dataloader_sql(local_tmp_path: Path):
    db_path = _write_sqlite_db(local_tmp_path)
    loader = DataLoader(sql="SELECT * FROM t", db=f"sqlite:///{db_path}")
    df, source = loader.load()
    assert df.shape == (2, 2)
    assert source.startswith("sql:")


def test_dataloader_py(local_tmp_path: Path):
    py_path = _write_py_loader(local_tmp_path)
    loader = DataLoader(py=str(py_path))
    df, source = loader.load()
    assert df.shape == (2, 2)
    assert source.startswith("py:")


def test_dataloader_notebook(local_tmp_path: Path):
    nb_path = _write_nb_loader(local_tmp_path)
    loader = DataLoader(nb=str(nb_path))
    df, source = loader.load()
    assert df.shape == (3, 2)
    assert source.startswith("nb:")


def test_dataloader_py_code():
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "def load():\n"
        "    values = np.array([1, 2, 3])\n"
        "    return pd.DataFrame({'a': values, 'b': ['x', 'y', 'z']})\n"
    )
    loader = DataLoader(py_code=code)
    df, source = loader.load()
    assert df.shape == (3, 2)
    assert source == "py_code"


def test_dataloader_data_dir_and_glob(local_tmp_path: Path):
    _write_csv(local_tmp_path, name="part1.csv", rows=2)
    _write_csv(local_tmp_path, name="part2.csv", rows=3)
    nested_dir = local_tmp_path / "nested"
    nested_dir.mkdir()
    _write_csv(nested_dir, name="nested.csv", rows=1)

    loader_dir = DataLoader(data=[str(local_tmp_path)])
    df_dir, source_dir = loader_dir.load()
    assert df_dir.shape[0] == 5
    assert "part1.csv" in source_dir
    assert "part2.csv" in source_dir

    loader_glob = DataLoader(data=[str(local_tmp_path / "*.csv")])
    df_glob, source_glob = loader_glob.load()
    assert df_glob.shape[0] == 5
    assert "part1.csv" in source_glob
    assert "part2.csv" in source_glob

    loader_recursive = DataLoader(data=[str(local_tmp_path)], recursive=True)
    df_recursive, source_recursive = loader_recursive.load()
    assert df_recursive.shape[0] == 6
    assert "nested.csv" in source_recursive


def test_dataloader_auto_exec(local_tmp_path: Path):
    _write_csv(local_tmp_path, name="data.csv", rows=3)
    _write_sqlite_db(local_tmp_path)
    _write_sql_file(local_tmp_path)
    _write_py_loader(local_tmp_path)
    _write_nb_loader(local_tmp_path)

    loader = DataLoader(data_dir=str(local_tmp_path), auto_exec=True)
    df, source = loader.load()
    assert df.shape[0] == 10
    assert "data.csv" in source
    assert "query.sql" in source
    assert "loader.py" in source
    assert "loader.ipynb" in source
