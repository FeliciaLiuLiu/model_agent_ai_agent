# Model Testing Agent

The Model Testing Agent provides a comprehensive evaluation pipeline for classification models. It generates metrics, plots, and a PDF report that covers effectiveness, efficiency, stability, and interpretability.

## What It Does

- Effectiveness: ROC/PR curves, AUC, confusion matrix, precision/recall/F1, KS, Precision@K
- Efficiency: threshold analysis and FPR diagnostics
- Stability: PSI, data drift, concept drift, CV stability, bootstrap CI
- Interpretability: permutation importance, SHAP, LIME, PDP, ICE

## Inputs

- A scikit-learn compatible model or pipeline saved via `joblib`
- A dataset file (CSV/Parquet) with a label column
- Optional: section and column selection

## Outputs

- `model_testing_agent_Model_Testing_Report.pdf`
- `results.json`
- `.png` plots saved under the output directory

## API Usage

### Non-Interactive (Full Report)

```python
from adm_central_utility.model_testing_agent import ModelTestingAgent

model = ModelTestingAgent.load_model("./path/to/your_model.joblib")
X, y, feature_names = ModelTestingAgent.load_data(
    "./path/to/your_dataset.csv",
    label_col="your_label",
)

agent = ModelTestingAgent(output_dir="./output")
results = agent.run(model=model, X=X, y=y, feature_names=feature_names)
agent.generate_report(results)
```

### Interactive Mode (Pick Matrices and Columns)

```python
from adm_central_utility.model_testing_agent import InteractiveAgent

agent = InteractiveAgent(output_dir="./output")
agent.run_interactive(model=model, X=X, y=y, feature_names=list(X.columns))
```

### Choose Sections and Columns Programmatically

```python
results = agent.run(
    model=model,
    X=X,
    y=y,
    feature_names=feature_names,
    sections=["effectiveness", "stability"],
    columns=["col_a", "col_b"],
    section_columns={
        "stability": ["col_a", "col_c"],
        "interpretability": ["col_a", "col_b"],
    },
)
```

## CLI Usage

### Non-Interactive

```bash
model-testing-agent \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output
```

### Interactive

```bash
model-testing-agent \
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
model-testing-agent \
  --model ./models/bank_aml_gbt.joblib \
  --data ./data/synthetic_bank_aml_200k_test.csv \
  --label_col is_suspicious \
  --output ./output
```

## Notes

- The model should support `predict_proba` for full metrics and interpretability.
- If only `decision_function` exists, AUC metrics still work but LIME may be limited.
- The feature schema must match what the model expects.
