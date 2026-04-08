import sqlite3
from pathlib import Path
import sys

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_testing_agent.data_analysis.agent import DataAnalysisAgent
from model_testing_agent.runner.main import ModelTestingAgent


def _write_labeled_csv(tmp_path: Path) -> Path:
    path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [5.0, 6.0, 7.0],
            "label": [0, 1, 0],
        }
    ).to_csv(path, index=False)
    return path


def _write_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "testing.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE scored_data (feature_a REAL, feature_b REAL, label INTEGER)")
        conn.executemany(
            "INSERT INTO scored_data (feature_a, feature_b, label) VALUES (?, ?, ?)",
            [(1.0, 5.0, 0), (2.0, 6.0, 1), (3.0, 7.0, 0)],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_python_loader(tmp_path: Path) -> Path:
    path = tmp_path / "loader.py"
    path.write_text(
        "import pandas as pd\n"
        "\n"
        "def load_data(label_col=None):\n"
        "    return pd.DataFrame({\n"
        "        'feature_a': [10.0, 20.0],\n"
        "        'feature_b': [30.0, 40.0],\n"
        "        'label': [1, 0],\n"
        "    })\n",
        encoding="utf-8",
    )
    return path


def test_model_testing_load_data_from_file(local_tmp_path: Path):
    csv_path = _write_labeled_csv(local_tmp_path)

    X, y, feature_names = ModelTestingAgent.load_data(path=str(csv_path))

    assert list(feature_names) == ["feature_a", "feature_b"]
    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0]


def test_model_testing_load_data_from_sql(local_tmp_path: Path):
    db_path = _write_sqlite_db(local_tmp_path)

    X, y, feature_names = ModelTestingAgent.load_data(
        sql="SELECT * FROM scored_data",
        conn=f"sqlite:///{db_path}",
    )

    assert list(feature_names) == ["feature_a", "feature_b"]
    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0]


def test_model_testing_load_data_from_python_loader(local_tmp_path: Path):
    loader_path = _write_python_loader(local_tmp_path)

    X, y, feature_names = ModelTestingAgent.load_data(loader_py=str(loader_path))

    assert list(feature_names) == ["feature_a", "feature_b"]
    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [1, 0]


def test_data_analysis_agent_supports_sql_and_python_loader(local_tmp_path: Path):
    db_path = _write_sqlite_db(local_tmp_path)
    loader_path = _write_python_loader(local_tmp_path)
    agent = DataAnalysisAgent()

    sql_df = agent.load_dataset(sql="SELECT * FROM scored_data", conn=f"sqlite:///{db_path}")
    loader_df = agent.load_dataset(loader_py=str(loader_path))

    assert sql_df.shape == (3, 3)
    assert loader_df.shape == (2, 3)
