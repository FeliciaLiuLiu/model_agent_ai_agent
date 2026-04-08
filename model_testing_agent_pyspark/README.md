# Model Testing Agent (PySpark)

This module provides a PySpark-based version of the Model Testing Agent. It uses Spark DataFrames for data processing and evaluation, and a scikit-learn joblib model for scoring. The output PDF structure matches the pandas version so you can compare results.

## What It Does

- Effectiveness: ROC/PR curves, AUC, confusion matrix, precision/recall/F1, KS, Precision@K
- Efficiency: FPR analysis and threshold tradeoffs
- Stability: PSI, data drift, concept drift, cross-validation stability, bootstrap confidence intervals
- Interpretability: Permutation Importance, LIME, PDP, ICE (SHAP removed)

## Inputs

- A Spark DataFrame or a labeled testing dataset provided through one of the following input modes:
  - file input: CSV / Parquet
  - SQL input: Spark SQL text / `.sql` file or JDBC-backed query
  - Python loader input: `.py` file that returns a Spark DataFrame, pandas DataFrame, or `(df, label_col, feature_cols)`
- A scikit-learn compatible model saved via `joblib`
- Label column name
- Optional: segmentation config to run the full testing pipeline by segment or time window

## Outputs

- `model_testing_agent_Model_Testing_Report.pdf`
- `results.json`
- `.png` plot files

## API Usage

```python
from adm_central_utility.model_testing_agent_pyspark import ModelTestingAgentSpark

model = ModelTestingAgentSpark.load_model("./path/to/your_model.joblib")

# Load data into Spark DataFrame
spark_df, label_col, feature_cols = ModelTestingAgentSpark.load_data(
    "./path/to/your_dataset.csv",
    label_col="your_label",
)

agent = ModelTestingAgentSpark(output_dir="./output")
results = agent.run(
    model=model,
    df=spark_df,
    label_col=label_col,
    feature_cols=feature_cols,
)
agent.generate_report(results)
```

### Load Data from SQL

```python
from adm_central_utility.model_testing_agent_pyspark import ModelTestingAgentSpark

model = ModelTestingAgentSpark.load_model("./path/to/your_model.joblib")
spark_df, label_col, feature_cols = ModelTestingAgentSpark.load_data(
    sql="./queries/model_testing_query.sql",      # or raw Spark SQL text
    conn="jdbc:postgresql://host:5432/db_name",   # omit conn to use spark.sql(...)
    label_col="your_label",
    jdbc_options={
        "user": "your_user",
        "password": "your_password",
        "driver": "org.postgresql.Driver",
    },
)
```

### Load Data from a Python Loader

```python
from adm_central_utility.model_testing_agent_pyspark import ModelTestingAgentSpark

model = ModelTestingAgentSpark.load_model("./path/to/your_model.joblib")
spark_df, label_col, feature_cols = ModelTestingAgentSpark.load_data(
    loader_py="./loaders/custom_testing_input_spark.py",
    loader_fn="load_data",  # optional, defaults to load_data
    spark=spark,
    label_col="your_label",
)
```

### Run the Full Pipeline by Segment or Time Window

```python
results = agent.run(
    model=model,
    df=spark_df,
    label_col=label_col,
    feature_cols=feature_cols,
    columns=["score_feature", "amount", "balance"],
    segmentation={
        "column": "event_time",
        "include_overall": True,
        "min_rows": 1000,
        "segments": [
            {"name": "jan_2024", "start": "2024-01-01", "end": "2024-02-01"},
            {"name": "feb_2024", "start": "2024-02-01", "end": "2024-03-01"},
        ],
    },
)

agent.generate_report(results, filename="segmented_model_testing_report_pyspark.pdf")
agent.save_results(results, filename="segmented_results_pyspark.json")
```

For value-based segmentation, use `values` instead of `start` / `end`:

```python
segmentation = {
    "column": "customer_segment",
    "segments": [
        {"name": "retail", "values": ["retail"]},
        {"name": "commercial", "values": ["commercial"]},
    ],
}
```

The PySpark report filename is:

- `model_testing_agent_Model_Testing_Report_pyspark.pdf`

## Interactive Mode

```python
from adm_central_utility.model_testing_agent_pyspark import InteractiveAgentSpark

agent = InteractiveAgentSpark(output_dir="./output")
agent.run_interactive(model=model, df=spark_df, label_col=label_col, feature_cols=feature_cols)
```

## CLI Usage

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output
```

SQL input:

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --sql ./queries/model_testing_query.sql \
  --conn jdbc:postgresql://host:5432/db_name \
  --jdbc-option user=your_user \
  --jdbc-option password=your_password \
  --jdbc-option driver=org.postgresql.Driver \
  --label_col your_label \
  --output ./output
```

Python loader input:

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --loader-py ./loaders/custom_testing_input_spark.py \
  --loader-fn load_data \
  --label_col your_label \
  --output ./output
```

Interactive CLI:

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output \
  --interactive
```

Select matrices and columns:

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --sections effectiveness,stability,interpretability \
  --columns-stability col_a,col_b \
  --columns-interpretability col_a,col_b \
  --output ./output
```

Segmented execution:

Use the shared template at [examples/model_testing_segmentation.json](/Users/felicia/Desktop/Felicia/DB_work/adm_central_utility/model_agent_ai_agent/examples/model_testing_segmentation.json) and edit it for your own grouping or time-window logic.

```bash
model-testing-agent-spark \
  --model ./models/bank_aml_gbt.joblib \
  --sql ./queries/model_testing_query.sql \
  --conn jdbc:postgresql://host:5432/db_name \
  --jdbc-option user=your_user \
  --jdbc-option password=your_password \
  --jdbc-option driver=org.postgresql.Driver \
  --label_col is_suspicious \
  --segmentation ./examples/model_testing_segmentation.json \
  --output ./output
```

## Example (Using Scripts 03/04)

```bash
python scripts/03_generate_bank_aml_dataset.py \
  --out-dir ./data \
  --rows 200000 \
  --suspicious-rate 0.04 \
  --label-noise 0.02 \
  --seed 7
```

```bash
python scripts/04_train_bank_aml_gbt_pipeline.py \
  --data ./data/synthetic_bank_aml_200k.csv \
  --model ./models/bank_aml_gbt.joblib \
  --label-col is_suspicious \
  --test-size 0.3 \
  --seed 42
```

```bash
model-testing-agent-spark \
  --model ./models/bank_aml_gbt.joblib \
  --data ./data/synthetic_bank_aml_200k_test.csv \
  --label_col is_suspicious \
  --output ./output
```

## Notes

- This PySpark version uses a Python UDF to score with a scikit-learn model.
- The default training script uses encoded categorical columns, which are compatible with the Spark scoring UDF.
- If your model expects raw string categorical columns with a ColumnTransformer, you must ensure consistent feature preprocessing.
- SHAP is removed in the PySpark interpretability module.
- `joblib` is required for loading `.joblib` models. If `joblib` is not available in CML, either install it or export the model as `.pkl`.
- Segmentation requires a named column in the dataset and runs the full pipeline once per segment.
- Segmented runs write plot artifacts into per-segment subdirectories to avoid overwriting outputs.

## Column Selection Rules

- If you do not specify columns, all features are used for effectiveness/efficiency/stability.
- For interpretability (Permutation/LIME/PDP/ICE), numeric columns are used by default.
- You can pass `--columns` or `--columns-interpretability` to override defaults.

## CML Execution (Spark Cluster)

Use `spark-submit` or a CML session with Spark enabled. Ensure temp dirs are writable.

```bash
export SPARK_LOCAL_DIRS=/tmp/spark
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=/tmp/spark"
export MPLCONFIGDIR=./.mpl_cache
export MPLBACKEND=Agg

spark-submit \
  --master yarn \
  -m model_testing_agent_pyspark.runner.cli \
  --model /path/to/model.joblib \
  --data /path/to/data.csv \
  --label_col your_label \
  --output /path/to/output
```

### CML Job Template

Use the provided script to run the job in CML:

```bash
bash cml_job_template_pyspark.sh /path/to/model.joblib /path/to/data.csv your_label /path/to/output
```

If you run inside a CML Python session, you can also use:

```bash
python -m model_testing_agent_pyspark.runner.cli \
  --model /path/to/model.joblib \
  --data /path/to/data.csv \
  --label_col your_label \
  --output /path/to/output
```
