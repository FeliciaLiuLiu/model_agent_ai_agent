# ADM Central Utility - Model Testing Agent

Comprehensive ML model evaluation toolkit.

## Installation

```bash
# From Bitbucket (SSH)
pip install git+ssh://git@bitbucket.org/YOUR_COMPANY/adm_central_utility.git

# From Bitbucket (HTTPS)
pip install git+https://bitbucket.org/YOUR_COMPANY/adm_central_utility.git

# For full features (SHAP, LIME)
pip install "adm_central_utility[full] @ git+ssh://git@bitbucket.org/YOUR_COMPANY/adm_central_utility.git"
```

## Quick Start

```python
from adm_central_utility.model_testing_agent import ModelTestingAgent

agent = ModelTestingAgent(output_dir='./output')
results = agent.run(model=model, X=X, y=y)
agent.generate_report(results)
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
