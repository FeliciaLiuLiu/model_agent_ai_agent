import json
import sqlite3
from pathlib import Path

import pandas as pd

from adm_central_utility.eda.runner import EDA


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


def _write_sql_file(tmp_path: Path, name: str = "query.sql") -> Path:
    path = tmp_path / name
    path.write_text("SELECT * FROM t", encoding="utf-8")
    return path


def _write_py_loader(tmp_path: Path) -> Path:
    path = tmp_path / "loader.py"
    path.write_text(
        "import pandas as pd\n"
        "def load():\n"
        "    return pd.DataFrame({'a':[1,2],'b':['x','y']})\n",
        encoding="utf-8",
    )
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


def _run_eda(output_dir: Path, **kwargs):
    eda = EDA(output_dir=str(output_dir))
    return eda.run(
        sections=["data_quality"],
        generate_report=False,
        save_json=False,
        return_payload=True,
        **kwargs,
    )


def test_eda_run_csv(local_tmp_path: Path):
    csv_path = _write_csv(local_tmp_path)
    payload = _run_eda(local_tmp_path / "out_csv", data=[str(csv_path)])
    assert payload["config"]["rows_used"] == 3
    assert "input.csv" in payload["config"]["data_path"]


def test_eda_run_data_dir_and_glob(local_tmp_path: Path):
    _write_csv(local_tmp_path, name="part1.csv", rows=2)
    _write_csv(local_tmp_path, name="part2.csv", rows=3)

    payload_dir = _run_eda(local_tmp_path / "out_dir", data=[str(local_tmp_path)])
    assert payload_dir["config"]["rows_used"] == 5

    payload_glob = _run_eda(local_tmp_path / "out_glob", data=[str(local_tmp_path / "*.csv")])
    assert payload_glob["config"]["rows_used"] == 5


def test_eda_run_sql(local_tmp_path: Path):
    db_path = _write_sqlite_db(local_tmp_path)
    payload = _run_eda(
        local_tmp_path / "out_sql",
        sql="SELECT * FROM t",
        db=f"sqlite:///{db_path}",
    )
    assert payload["config"]["rows_used"] == 2
    assert payload["config"]["data_path"].startswith("sql:")


def test_eda_run_py(local_tmp_path: Path):
    py_path = _write_py_loader(local_tmp_path)
    payload = _run_eda(local_tmp_path / "out_py", py=str(py_path))
    assert payload["config"]["rows_used"] == 2
    assert payload["config"]["data_path"].startswith("py:")


def test_eda_run_py_code(local_tmp_path: Path):
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "def load():\n"
        "    values = np.array([1, 2, 3])\n"
        "    return pd.DataFrame({'a': values, 'b': ['x', 'y', 'z']})\n"
    )
    payload = _run_eda(local_tmp_path / "out_py_code", py_code=code)
    assert payload["config"]["rows_used"] == 3
    assert payload["config"]["data_path"] == "py_code"


def test_eda_run_notebook(local_tmp_path: Path):
    nb_path = _write_nb_loader(local_tmp_path)
    payload = _run_eda(local_tmp_path / "out_nb", nb=str(nb_path))
    assert payload["config"]["rows_used"] == 3
    assert payload["config"]["data_path"].startswith("nb:")


def test_eda_run_auto_exec(local_tmp_path: Path):
    _write_csv(local_tmp_path, name="data.csv", rows=3)
    _write_sqlite_db(local_tmp_path)
    _write_sql_file(local_tmp_path)
    _write_py_loader(local_tmp_path)
    _write_nb_loader(local_tmp_path)

    payload = _run_eda(local_tmp_path / "out_auto", data_dir=str(local_tmp_path), auto_exec=True)
    assert payload["config"]["rows_used"] == 10
    data_path = payload["config"]["data_path"]
    assert "data.csv" in data_path
    assert "query.sql" in data_path
    assert "loader.py" in data_path
    assert "loader.ipynb" in data_path
