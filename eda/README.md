# EDA Pipeline

This module generates a professional PDF EDA report using ReportLab (Platypus). It supports both Pandas and Spark implementations, auto-detects the latest dataset in `./data`, and provides interactive or non-interactive modes.

## Quick Start (Pandas)

```bash
# 1) Generate a synthetic AML dataset (optional)
python scripts/05_generate_synthetic_aml_200k_timeseries.py --out-dir ./data --rows 200000 --seed 7 --suspicious-rate 0.04 --label-noise 0.02

# 2) Run EDA (auto-detect latest dataset)
python -m eda.cli --output ./output_eda --target-col is_suspicious
```

The PDF will be saved to:

```
./output_eda/EDA_Report.pdf
```

## Auto-Detection + Sampling

- If `--data` is not provided, the pipeline finds the latest file in `./data` (by timestamp in filename, otherwise by mtime).
- If the dataset filename matches `synthetic_aml_200k_*.csv`, the pipeline **auto-samples 5%** by default to speed up PDF generation.
- You can override the sample ratio:

```bash
python -m eda.cli --output ./output_eda --target-col is_suspicious --sample-frac 0.10 --sample-seed 42
```

## API Usage (Pandas)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run(file_path="./path/to/your_dataset.csv")
```

Interactive mode:

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.run_interactive(file_path="./path/to/your_dataset.csv")
```

## API Usage (Spark)

```python
from adm_central_utility import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", target_col="your_target")
eda.run(file_path="./path/to/your_dataset.csv")
```

> Spark mode requires `pyspark` to be installed in the runtime.

## CLI Usage

### Non-Interactive (Pandas)

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --target-col your_target
```

### Interactive (Pandas)

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --interactive
```

### Spark Mode

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda_spark --spark
```

### Sampling (5% or custom)

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --sample-frac 0.05 --sample-seed 42
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --sample-frac 0.10 --sample-seed 42
```

## Select Sections and Columns

```bash
python -m eda.cli \
  --data ./path/to/your_dataset.csv \
  --sections data_quality,univariate,time_drift \
  --columns-univariate txn_amount,channel,origin_country \
  --output ./output_eda
```

## Output Files

- `EDA_Report.pdf`
- `eda_results.json`
- Plot images under the output directory

## Notes

- Time-series and drift sections run only if a time column is detected and parse success is >= 90%.
- Column type classification is shown as a table (numeric, categorical, datetime, boolean, text).
- Tables are styled with headers, zebra striping, and numeric alignment.
