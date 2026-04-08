from pathlib import Path
import importlib.util
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_testing_agent.core.report import ReportBuilder as PandasReportBuilder


def _load_spark_report_builder():
    report_path = ROOT_DIR / "model_testing_agent_pyspark" / "core" / "report.py"
    spec = importlib.util.spec_from_file_location("model_testing_agent_pyspark_report", report_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ReportBuilder


SparkReportBuilder = _load_spark_report_builder()


def _sample_results():
    return {
        "overall": {
            "effectiveness": {
                "metrics": {
                    "auc_roc": 0.8470,
                    "auc_pr": 0.5343,
                    "precision": 0.6121,
                    "recall": 0.3701,
                    "f1": 0.4613,
                    "ks_statistic": 0.5528,
                    "ks_threshold": 0.15,
                    "confusion_matrix": {"TN": 40126, "FP": 1876, "FN": 5038, "TP": 2960},
                    "precision_at_k": {"10": 0.9},
                    "recall_at_k": {"10": 0.0011},
                },
                "plots": {},
            },
            "efficiency": {
                "metrics": {
                    "fpr": 0.0447,
                    "tn": 40126,
                    "fp": 1876,
                    "threshold": 0.5,
                    "fpr_at_thresholds": {"t_0.05": 0.3221, "t_0.50": 0.0447},
                },
                "plots": {},
            },
            "stability": {
                "metrics": {
                    "psi": 0.5645,
                    "cv_auc_roc_mean": 0.8471,
                    "cv_auc_roc_std": 0.0082,
                    "cv_auc_pr_mean": 0.5350,
                    "cv_auc_pr_std": 0.0155,
                    "bootstrap_auc_roc_mean": 0.8219,
                    "bootstrap_auc_roc_ci_lower": 0.8154,
                    "bootstrap_auc_roc_ci_upper": 0.8279,
                    "concept_drift_detected": False,
                    "concept_drift_score": 0.0058,
                },
                "plots": {},
            },
            "interpretability": {
                "metrics": {
                    "model_type": "tree",
                    "methods_used": ["permutation", "pdp", "ice", "shap", "lime"],
                    "perm_top_features": ["num_txn_24h", "txn_amount", "txn_type_code"],
                    "shap_top_features": ["num_txn_24h", "txn_amount", "txn_type_code"],
                    "lime_instances": 3,
                    "pdp_features": ["txn_amount", "avg_amount_7d"],
                    "ice_features": ["txn_amount", "avg_amount_7d"],
                },
                "plots": {},
            },
        }
    }


def test_report_build_metric_rows_include_explanations(local_tmp_path: Path):
    builder = PandasReportBuilder(output_dir=str(local_tmp_path / "out"))
    metrics = _sample_results()["overall"]["effectiveness"]["metrics"]

    rows = builder._build_metric_rows("effectiveness", metrics)
    rows_by_name = {row[0]: row for row in rows}

    assert "auc_roc" in rows_by_name
    assert "Next:" in rows_by_name["auc_roc"][2]
    assert "precision_at_k[10]" in rows_by_name
    assert "recall_at_k[10]" in rows_by_name
    assert "confusion_matrix[TN]" in rows_by_name


def test_report_builds_pdf_for_pandas_and_spark(local_tmp_path: Path):
    results = _sample_results()

    pandas_builder = PandasReportBuilder(output_dir=str(local_tmp_path / "pandas"))
    spark_builder = SparkReportBuilder(output_dir=str(local_tmp_path / "spark"))

    pandas_pdf = Path(pandas_builder.build(results, filename="pandas_report.pdf"))
    spark_pdf = Path(spark_builder.build(results, filename="spark_report.pdf"))

    assert pandas_pdf.exists()
    assert spark_pdf.exists()
    assert pandas_pdf.stat().st_size > 0
    assert spark_pdf.stat().st_size > 0
