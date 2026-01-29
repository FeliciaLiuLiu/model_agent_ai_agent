# Model Testing Agent (PySpark)

This module provides a PySpark-based version of the Model Testing Agent. It uses Spark DataFrames for data processing and evaluation, and a scikit-learn joblib model for scoring. The output PDF structure matches the pandas version so you can compare results.

## What It Does

- Effectiveness: ROC/PR curves, AUC, confusion matrix, precision/recall/F1, KS, Precision@K
- Efficiency: FPR analysis and threshold tradeoffs
- Stability: PSI, data drift, concept drift, cross-validation stability, bootstrap confidence intervals
- Interpretability: Permutation Importance, LIME, PDP, ICE (SHAP removed)

## Inputs

- A Spark DataFrame or a dataset path (CSV/Parquet)
- A scikit-learn compatible model saved via `joblib`
- Label column name

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

Interactive CLI:

```bash
model-testing-agent-spark \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output \
  --interactive
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
