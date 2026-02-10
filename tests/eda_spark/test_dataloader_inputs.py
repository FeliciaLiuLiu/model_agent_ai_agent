import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

from adm_central_utility.eda_spark.dataloader import DataLoader


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("eda_spark_tests")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def _write_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "input.csv"
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


def test_spark_dataloader_csv(local_tmp_path: Path, spark):
    csv_path = _write_csv(local_tmp_path)
    loader = DataLoader(spark=spark, data=[str(csv_path)])
    df, source = loader.load()
    assert df.count() == 3
    assert "input.csv" in source


def test_spark_dataloader_sql(local_tmp_path: Path, spark):
    db_path = _write_sqlite_db(local_tmp_path)
    loader = DataLoader(spark=spark, sql="SELECT * FROM t", db=f"sqlite:///{db_path}")
    df, source = loader.load()
    assert df.count() == 2
    assert source.startswith("sql:")


def test_spark_dataloader_py(local_tmp_path: Path, spark):
    py_path = _write_py_loader(local_tmp_path)
    loader = DataLoader(spark=spark, py=str(py_path))
    df, source = loader.load()
    assert df.count() == 2
    assert source.startswith("py:")


def test_spark_dataloader_notebook(local_tmp_path: Path, spark):
    nb_path = _write_nb_loader(local_tmp_path)
    loader = DataLoader(spark=spark, nb=str(nb_path))
    df, source = loader.load()
    assert df.count() == 3
    assert source.startswith("nb:")
