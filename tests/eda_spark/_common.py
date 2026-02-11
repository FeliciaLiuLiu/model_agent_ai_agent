from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd


CML_BASE_DIR = Path("/home/cdsw")
TEST_TMP_ROOT = CML_BASE_DIR / ".tmp_eda_spark_unittest"
SPARK_TMP_ROOT = TEST_TMP_ROOT / "spark"
SPARK_WAREHOUSE_DIR = SPARK_TMP_ROOT / "warehouse"
SPARK_LOCAL_DIR = SPARK_TMP_ROOT / "local"


def _ensure_test_dirs() -> None:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    SPARK_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    SPARK_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    SPARK_LOCAL_DIR.mkdir(parents=True, exist_ok=True)


class SparkTestCase(unittest.TestCase):
    spark = None

    @classmethod
    def setUpClass(cls):
        try:
            from pyspark.sql import SparkSession
        except Exception as exc:
            raise unittest.SkipTest(f"pyspark not available: {exc}")

        try:
            active = SparkSession.getActiveSession()
            if active is not None:
                active.stop()
            if hasattr(SparkSession, "_instantiatedSession"):
                SparkSession._instantiatedSession = None
            if hasattr(SparkSession, "_activeSession"):
                SparkSession._activeSession = None
        except Exception:
            pass

        _ensure_test_dirs()
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName(f"{cls.__name__}_unittest")
            .config("spark.sql.warehouse.dir", f"file://{SPARK_WAREHOUSE_DIR}")
            .config("spark.local.dir", str(SPARK_LOCAL_DIR))
            .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={SPARK_TMP_ROOT}")
            .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={SPARK_TMP_ROOT}")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        if cls.spark is not None:
            cls.spark.stop()
            cls.spark = None

    def setUp(self):
        _ensure_test_dirs()
        self._tmp = tempfile.TemporaryDirectory(prefix="eda_spark_unittest_", dir=str(TEST_TMP_ROOT))
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


def write_csv(tmp_path: Path, name: str = "input.csv", rows: int = 3) -> Path:
    df = pd.DataFrame({"a": list(range(1, rows + 1)), "b": ["x"] * rows})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def write_tsv(tmp_path: Path, name: str = "input.tsv", rows: int = 3) -> Path:
    df = pd.DataFrame({"a": list(range(1, rows + 1)), "b": ["x"] * rows})
    path = tmp_path / name
    df.to_csv(path, index=False, sep="\t")
    return path


def write_json_array(tmp_path: Path, name: str = "input.json", rows: int = 3) -> Path:
    rows_data = [{"a": i + 1, "b": "x"} for i in range(rows)]
    path = tmp_path / name
    path.write_text(json.dumps(rows_data), encoding="utf-8")
    return path


def write_parquet_with_spark(tmp_path: Path, spark, name: str = "input.parquet", rows: int = 3) -> Path:
    path = tmp_path / name
    pdf = pd.DataFrame({"a": list(range(1, rows + 1)), "b": ["x"] * rows})
    # Force local filesystem path in CML/HDFS environments.
    spark.createDataFrame(pdf).write.mode("overwrite").parquet(path.resolve().as_uri())
    return path


def write_sqlite_db(tmp_path: Path, name: str = "demo.db") -> Path:
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.executemany("INSERT INTO t (a, b) VALUES (?, ?)", [(1, "x"), (2, "y")])
        conn.commit()
    finally:
        conn.close()
    return db_path


def write_sql_file(tmp_path: Path, name: str = "query.sql", sql_text: str = "SELECT * FROM t") -> Path:
    path = tmp_path / name
    path.write_text(sql_text, encoding="utf-8")
    return path


def write_py_loader(tmp_path: Path, name: str = "loader.py") -> Path:
    path = tmp_path / name
    path.write_text(
        "import pandas as pd\n"
        "def load():\n"
        "    return pd.DataFrame({'a':[1,2],'b':['x','y']})\n",
        encoding="utf-8",
    )
    return path


def write_py_df_loader(tmp_path: Path, name: str = "loader_df.py") -> Path:
    path = tmp_path / name
    path.write_text(
        "import pandas as pd\n"
        "df = pd.DataFrame({'a':[1,2,3],'b':['x','y','z']})\n",
        encoding="utf-8",
    )
    return path


def write_nb_loader(tmp_path: Path, name: str = "loader.ipynb") -> Path:
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
    path = tmp_path / name
    path.write_bytes(json.dumps(nb).encode("utf-8-sig"))
    return path


def write_nb_df_loader(tmp_path: Path, name: str = "loader_df.ipynb") -> Path:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})\n",
                ],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / name
    path.write_bytes(json.dumps(nb).encode("utf-8-sig"))
    return path


def make_runner_dataframe(rows: int = 240) -> pd.DataFrame:
    channels = ["card", "wire", "crypto", "cash"]
    segments = ["retail", "smb", "vip"]
    notes = [
        "Customer reported unusual transfer pattern requiring manual review.",
        "Normal payment.",
    ]
    data = {
        "txn_id": list(range(1, rows + 1)),
        "txn_datetime": [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d} 10:00:00" for i in range(rows)],
        "amount": [1000 + (i % 60) * 90 for i in range(rows)],
        "amount_dup": [1000 + (i % 60) * 90 for i in range(rows)],
        "risk_score": [float((i % 10) + 1) for i in range(rows)],
        "channel": [channels[i % len(channels)] for i in range(rows)],
        "segment": [segments[i % len(segments)] for i in range(rows)],
        "notes": [notes[i % 2] for i in range(rows)],
        "is_pep": [(i % 2) == 0 for i in range(rows)],
    }
    df = pd.DataFrame(data)
    df["sar_actual"] = (
        (df["amount"] >= 4200)
        | ((df["channel"] == "crypto") & (df["amount"] >= 1800))
        | ((df["channel"] == "wire") & (df["risk_score"] >= 7))
    ).astype(int)
    return df


def write_png(path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1.2))
    ax.plot([0, 1], [0, 1], color="steelblue")
    ax.set_title("demo")
    fig.tight_layout()
    fig.savefig(path, dpi=72)
    plt.close(fig)
