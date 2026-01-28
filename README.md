# ADM Central Utility - Model Testing Agent

Comprehensive ML model evaluation toolkit for classification models. It generates metrics, plots, and a PDF report.

## Installation

```bash
# From Bitbucket (SSH)
pip install git+ssh://git@bitbucket.org/YOUR_COMPANY/adm_central_utility.git

# From Bitbucket (HTTPS)
pip install git+https://bitbucket.org/YOUR_COMPANY/adm_central_utility.git

# For full features (SHAP, LIME)
pip install "adm_central_utility[full] @ git+ssh://git@bitbucket.org/YOUR_COMPANY/adm_central_utility.git"
```

## Model Requirements

- The joblib must load a scikit-learn compatible classifier or pipeline.
- `predict_proba` is strongly recommended for full metrics and interpretability.
- If only `decision_function` exists, AUC metrics still work but LIME will not.
- Input features must match the training schema (names, order, types).

## Quick Start (API)

```python
from adm_central_utility import model_testing_agent

model = model_testing_agent.ModelTestingAgent.load_model("./models/bank_aml_gbt.joblib")
X, y, feature_names = model_testing_agent.ModelTestingAgent.load_data(
    "./data/synthetic_bank_aml_200k_test.csv",
    label_col="is_suspicious",
)

agent = model_testing_agent.ModelTestingAgent(output_dir="./output")
results = agent.run(model=model, X=X, y=y, feature_names=feature_names)
agent.generate_report(results)
```

## Quick Start (CLI)

```bash
model-testing-agent \
  --model ./models/bank_aml_gbt.joblib \
  --data ./data/synthetic_bank_aml_200k_test.csv \
  --label_col is_suspicious \
  --output ./output
```

## Synthetic AML Example (Scripts 03 & 04)

1) Generate dataset (with label noise):

```bash
python scripts/03_generate_bank_aml_dataset.py \
  --out-dir ./data \
  --rows 200000 \
  --suspicious-rate 0.04 \
  --label-noise 0.02 \
  --seed 7
```

2) Train model and save test set:

```bash
python scripts/04_train_bank_aml_gbt_pipeline.py \
  --data ./data/synthetic_bank_aml_200k.csv \
  --model ./models/bank_aml_gbt.joblib \
  --label-col is_suspicious \
  --test-size 0.3 \
  --seed 42
```

This creates:
- `./models/bank_aml_gbt.joblib`
- `./data/synthetic_bank_aml_200k_test.csv` (saved by default)

3) Run evaluation:

```bash
model-testing-agent \
  --model ./models/bank_aml_gbt.joblib \
  --data ./data/synthetic_bank_aml_200k_test.csv \
  --label_col is_suspicious \
  --output ./output \
  --sections effectiveness,efficiency,stability,interpretability
```

## Choose Matrices and Columns

### Sections (matrices)

```python
results = agent.run(
    model=model,
    X=X,
    y=y,
    sections=["effectiveness", "stability"]
)
```

CLI:
```bash
--sections effectiveness,stability
```

### Columns (features)

You can pass a global column list or per-section columns. Only do this if your model accepts that subset.

```python
results = agent.run(
    model=model,
    X=X,
    y=y,
    columns=["txn_amount", "num_txn_24h"],
    section_columns={
        "stability": ["txn_amount", "kyc_risk_score"],
        "interpretability": ["txn_amount", "num_txn_24h"],
    },
)
```

CLI:
```bash
--columns txn_amount,num_txn_24h \
--columns-stability txn_amount,kyc_risk_score \
--columns-interpretability txn_amount,num_txn_24h
```

## Evaluation Matrices

### 1. Effectiveness
- ROC Curve & AUC-ROC
- PR Curve & AUC-PR
- Confusion Matrix (raw & normalized)
- Precision, Recall, F1 Score
- KS Statistic & KS Curve
- Precision@K & Recall@K

### 2. Efficiency
- False Positive Rate (FPR)
- FPR vs Threshold Analysis
- True Negatives & False Positives

### 3. Stability
- Population Stability Index (PSI)
- Data Drift Detection (per feature)
- Concept Drift Detection
- Cross-Validation Stability
- Bootstrap Stability & Confidence Intervals

### 4. Interpretability
- Permutation Importance
- SHAP Values & Summary Plot
- LIME Explanations
- Partial Dependence Plot (PDP)
- Individual Conditional Expectation (ICE)

## Output
- `model_testing_agent_Model_Testing_Report.pdf`
- `results.json`
- Various `.png` plot files
