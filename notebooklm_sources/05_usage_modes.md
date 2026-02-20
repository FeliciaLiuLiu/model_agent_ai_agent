# 05 Usage Modes (Method First, Cases Later)

## 1) Method Slide Content (No Dataset-Specific Paths)
Use this as the consolidated usage slide content.

### CLI Non-Interactive (Template)
```bash
python -m eda.cli \
  --data ./path/to/input.csv \
  --sections data_quality,target,univariate,feature_vs_feature,time_drift \
  --output ./output_eda
```

### CLI Interactive (Template)
```bash
python -m eda.cli \
  --data ./path/to/input.csv \
  --interactive \
  --output ./output_eda
```

### API Non-Interactive (Template)
```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda')
results = eda.run(
    data=['./path/to/input.csv'],
    sections=['data_quality', 'univariate', 'feature_vs_feature'],
)
```

### API Interactive (Template)
```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda')
payload = eda.run_interactive(
    data=['./path/to/input.csv'],
    return_payload=True,
)
```

Spark adaptation (same mode logic):
- CLI: replace `eda.cli` with `eda_spark.cli`.
- API: replace `EDA` with `EDASpark` and import from `eda_spark.runner`.

## 2) Case Slides (Concrete Execution Examples)

### Case A - EDA file-input execution example
CLI non-interactive:
```bash
python -m eda.cli \
  --data ./data/synthetic_bank_aml_200k.csv \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --output ./output_eda
```

CLI interactive:
```bash
python -m eda.cli \
  --data ./data/synthetic_bank_aml_200k.csv \
  --target-col is_suspicious \
  --interactive \
  --output ./output_eda
```

API non-interactive:
```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda', target_col='is_suspicious')
results = eda.run(
    data=['./data/synthetic_bank_aml_200k.csv'],
    sections=['data_quality', 'target', 'univariate', 'bivariate_target', 'feature_vs_feature', 'time_drift'],
)
```

API interactive:
```python
payload = eda.run_interactive(
    data=['./data/synthetic_bank_aml_200k.csv'],
    return_payload=True,
)
```

### Case B - EDA Spark demo with `Paypal_data.py`
Required CLI non-interactive example (top 5000 rows):

```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

CLI interactive:
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

API non-interactive:
```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir='./output_eda_spark', spark_master='local[*]')
results = eda.run(
    py='./data/Paypal_data.py',
    sections=['data_quality', 'univariate', 'feature_vs_feature', 'time_drift'],
    max_rows=5000,
)
```

API interactive:
```python
payload = eda.run_interactive(
    py='./data/Paypal_data.py',
    max_rows=5000,
    return_payload=True,
)
```

## 3) Output Verification
Expected output files:
- `./output_eda/EDA_Report.pdf`
- `./output_eda/eda_results.json`
- `./output_eda_spark/EDA_Report.pdf`
- `./output_eda_spark/eda_results.json`
