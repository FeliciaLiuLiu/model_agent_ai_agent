from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_testing_agent.runner.main import ModelTestingAgent


class ScoreModel:
    """Simple deterministic model for exercising segmented runs."""

    def _scores(self, X):
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].astype(float).to_numpy()
        else:
            values = np.asarray(X)[:, 0].astype(float)
        return np.clip(values, 0.0, 1.0)

    def predict_proba(self, X):
        scores = self._scores(X)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X):
        scores = self._scores(X)
        return (scores >= 0.5).astype(int)


class StrictScoreModel(ScoreModel):
    """Model that fails if a segmentation-only column leaks into model features."""

    def _scores(self, X):
        if isinstance(X, pd.DataFrame):
            assert list(X.columns) == ["score_feature"]
        else:
            assert np.asarray(X).shape[1] == 1
        return super()._scores(X)


def test_model_testing_segmentation_by_values(local_tmp_path: Path):
    X = pd.DataFrame(
        {
            "score_feature": [0.1, 0.9, 0.2, 0.8],
            "segment": ["A", "A", "B", "B"],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="label")
    agent = ModelTestingAgent(output_dir=str(local_tmp_path / "segmented_values"))

    results = agent.run(
        model=ScoreModel(),
        X=X,
        y=y,
        sections=["effectiveness"],
        columns=["score_feature"],
        segmentation={
            "column": "segment",
            "segments": [
                {"name": "group_A", "values": ["A"]},
                {"name": "group_B", "values": ["B"]},
            ],
        },
    )

    assert "overall" in results
    assert set(results["segments"].keys()) == {"group_A", "group_B"}
    assert results["segmentation"]["column"] == "segment"
    assert results["segmentation"]["segments"][0]["row_count"] == 2
    assert Path(results["segments"]["group_A"]["effectiveness"]["plots"]["roc_curve"]).exists()


def test_model_testing_segmentation_by_time_range(local_tmp_path: Path):
    X = pd.DataFrame(
        {
            "score_feature": [0.1, 0.8, 0.3, 0.7],
            "event_time": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-18"]),
        }
    )
    y = pd.Series([0, 1, 0, 1], name="label")
    agent = ModelTestingAgent(output_dir=str(local_tmp_path / "segmented_time"))

    results = agent.run(
        model=ScoreModel(),
        X=X,
        y=y,
        sections=["effectiveness"],
        columns=["score_feature"],
        segmentation={
            "column": "event_time",
            "segments": [
                {"name": "jan_window", "start": "2024-01-01", "end": "2024-02-01"},
                {"name": "feb_window", "start": "2024-02-01", "end": "2024-03-01"},
            ],
        },
    )

    assert set(results["segments"].keys()) == {"jan_window", "feb_window"}
    row_counts = {item["name"]: item["row_count"] for item in results["segmentation"]["segments"]}
    assert row_counts["jan_window"] == 2
    assert row_counts["feb_window"] == 2
    assert Path(results["segments"]["feb_window"]["effectiveness"]["plots"]["pr_curve"]).exists()


def test_model_testing_segmentation_groupby_month(local_tmp_path: Path):
    X = pd.DataFrame(
        {
            "score_feature": [0.1, 0.8, 0.3, 0.7],
            "event_time": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-18"]),
        }
    )
    y = pd.Series([0, 1, 0, 1], name="label")
    agent = ModelTestingAgent(output_dir=str(local_tmp_path / "groupby_month"))

    results = agent.run(
        model=ScoreModel(),
        X=X,
        y=y,
        sections=["effectiveness"],
        columns=["score_feature"],
        segmentation={
            "column": "event_time",
            "mode": "groupby",
            "groupby": {
                "kind": "time",
                "freq": "month",
            },
        },
    )

    assert results["segmentation"]["mode"] == "groupby"
    assert set(results["segments"].keys()) == {"event_time_month=2024-01", "event_time_month=2024-02"}


def test_model_testing_segmentation_groupby_value_excludes_segment_column(local_tmp_path: Path):
    X = pd.DataFrame(
        {
            "score_feature": [0.1, 0.9, 0.2, 0.8],
            "txn_type": ["ACH", "wire", "ACH", "wire"],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="label")
    agent = ModelTestingAgent(output_dir=str(local_tmp_path / "groupby_value"))

    results = agent.run(
        model=StrictScoreModel(),
        X=X,
        y=y,
        sections=["effectiveness"],
        segmentation={
            "column": "txn_type",
            "mode": "groupby",
            "groupby": {
                "kind": "value",
            },
        },
    )

    assert results["segmentation"]["keep_column_in_features"] is False
    assert set(results["segments"].keys()) == {"txn_type=ACH", "txn_type=wire"}


def test_model_testing_segmentation_groupby_selected_groups(local_tmp_path: Path):
    X = pd.DataFrame(
        {
            "score_feature": [0.1, 0.8, 0.3, 0.7, 0.4],
            "event_time": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-18", "2024-03-02"]),
        }
    )
    y = pd.Series([0, 1, 0, 1, 0], name="label")
    agent = ModelTestingAgent(output_dir=str(local_tmp_path / "groupby_selected"))

    results = agent.run(
        model=ScoreModel(),
        X=X,
        y=y,
        sections=["effectiveness"],
        columns=["score_feature"],
        segmentation={
            "column": "event_time",
            "mode": "groupby",
            "groupby": {
                "kind": "time",
                "freq": "month",
                "selected_groups": ["2024-01", "2024-03", "2024-04"],
            },
        },
    )

    assert set(results["segments"].keys()) == {"event_time_month=2024-01", "event_time_month=2024-03"}
    meta = {item["name"]: item for item in results["segmentation"]["segments"]}
    assert meta["event_time_month=2024-04"]["status"] == "skipped"
    assert meta["event_time_month=2024-04"]["reason"] == "group not found in dataset"
