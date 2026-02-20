# 06 Demo Paths and Runbook for This Deck

## Demo Inputs to Reference in Slides

### EDA demo input
- Data file: `./data/synthetic_bank_aml_200k.csv`

### EDA Spark demo input
- Python loader: `./data/Paypal_data.py`
- Contract: file must define `load()` or `df`.

## Quick Validation Commands

### Validate EDA data file
```bash
ls -l ./data/synthetic_bank_aml_200k.csv
head -n 5 ./data/synthetic_bank_aml_200k.csv
```

### Validate EDA Spark Python loader
```bash
ls -l ./data/Paypal_data.py
head -n 40 ./data/Paypal_data.py
```

## Canonical Commands to Put into Slides

### EDA CLI non-interactive
```bash
python -m eda.cli \
  --data ./data/synthetic_bank_aml_200k.csv \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --output ./output_eda
```

### EDA CLI interactive
```bash
python -m eda.cli \
  --data ./data/synthetic_bank_aml_200k.csv \
  --target-col is_suspicious \
  --interactive \
  --output ./output_eda
```

### EDA Spark CLI non-interactive (required)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

### EDA Spark CLI interactive
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## Common CML Path Pitfalls to Mention
1. Missing `.py` extension:
- `--py ./data/Paypal_data` fails.
- Use `--py ./data/Paypal_data.py`.

2. Loader contract mismatch:
- If `load()` or `df` is missing, run fails.

3. Wrong data path:
- Validate the file path before running CLI/API.

## Output Paths to Show
- EDA: `./output_eda/EDA_Report.pdf`, `./output_eda/eda_results.json`
- EDA Spark: `./output_eda_spark/EDA_Report.pdf`, `./output_eda_spark/eda_results.json`
