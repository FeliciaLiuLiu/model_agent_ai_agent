# 05 How Model Developers Use CLI and API

## What This Document Covers
- Exact CLI and API usage for both `eda` and `eda_spark`.
- How to select functions and columns.
- How to focus on output **results** for model-development decisions.

## A. CLI Usage (Batch, Non-Interactive)

### EDA (Pandas)
```bash
python -m eda.cli \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --target-col is_suspicious \
  --sections data_quality,target,univariate,time_drift \
  --output ./output_eda
```

### EDA Spark (PySpark)
```bash
python -m eda_spark.cli \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --target-col sar_actual \
  --sections data_quality,target,univariate,time_drift \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

### Output artifacts to verify after each run
- `<output_dir>/eda_results.json`
- `<output_dir>/EDA_Report.pdf`

## B. CLI Usage (Interactive)

```bash
# show available functions
python -m eda.cli --list-functions
python -m eda_spark.cli --list-functions

# interactive selection
python -m eda.cli --interactive --data ./data/synthetic_aml_200k_20260130_135951.csv --output ./output_eda
python -m eda_spark.cli --interactive --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv --spark-master "local[*]" --output ./output_eda_spark
```

Interactive flow:
- choose function keys by number or name,
- choose columns for each selected function,
- runner executes with chosen scope.

## C. API Usage (Batch, Non-Interactive)

### EDA API
```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="is_suspicious")
results = eda.run(
    data=["./data/synthetic_aml_200k_20260130_135951.csv"],
    sections=["data_quality", "target", "univariate", "time_drift"],
    section_columns={"univariate": ["txn_amount", "velocity_score", "origin_country"]},
)
```

### EDA Spark API
```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", spark_master="local[*]", target_col="sar_actual")
results = eda.run(
    data=["./data/synthetic_aml_mixed_50k_20260205_094055.csv"],
    sections=["data_quality", "target", "univariate", "time_drift"],
    section_columns={"univariate": ["txn_amount", "velocity_score", "merchant_category"]},
)
```

## D. API Usage (Return Full Payload)
Use this when you need `config` and `skipped_sections` in addition to section results.

```python
payload = eda.run(
    data=["./data/synthetic_aml_200k_20260130_135951.csv"],
    return_payload=True,
)

print(payload.keys())
# dict_keys(["results", "skipped_sections", "config"])
```

## E. Function Keys You Can Run
- `data_quality`
- `target`
- `univariate`
- `bivariate_target`
- `feature_vs_feature`
- `time_drift`

## F. How Model Developers Should Read Results
1. Start from `eda_results.json`:
- programmatic checks,
- section-level metrics,
- run config and skipped sections (if `return_payload=True`).
2. Use `EDA_Report.pdf` for discussion and review.
3. Convert findings into modeling actions:
- missingness handling,
- feature filtering,
- split strategy,
- drift and monitoring setup.
