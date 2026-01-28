# EDA Module

The EDA module provides automated exploratory data analysis with PDF and JSON outputs. It is designed to be used as a simple API or CLI so users can run EDA with their own datasets without changing code.

## What It Does

- Overview: row/column counts, column types, unique counts, duplicate ratio
- Missingness: missing value rates and plots
- Numeric: summary stats and distributions
- Categorical: top categories and frequency plots
- Correlation: correlation heatmap and target correlations (if numeric target)
- Target: feature vs target summaries
- Outliers: IQR-based outlier ratios
- Time: time-series volume and target rate (if time column provided)

## Inputs

- Dataset file path (CSV, Parquet, or XLSX) or a pandas DataFrame
- Optional: `target_col`, `time_col`, `id_cols`
- Optional: section and column selection

## Outputs

- `EDA_Report.pdf`
- `eda_results.json`
- `.png` plots saved under the output directory

## API Usage

### Non-Interactive (Full Report)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run(file_path="./path/to/your_dataset.csv")
```

### Interactive Selection (Choose Sections and Columns)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run(
    file_path="./path/to/your_dataset.csv",
    sections=["overview", "missingness", "numeric"],
    columns=["col_a", "col_b", "col_c"],
    section_columns={
        "numeric": ["col_a", "col_b"],
        "categorical": ["col_c"],
    },
)
```

## CLI Usage

### Basic

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --target-col your_target
```

### Auto-Detect Dataset

If `--data` is omitted, EDA will use the most recent dataset in `./data`.

```bash
eda-agent --output ./output_eda --target-col your_target
```

### Select Sections and Columns

```bash
eda-agent \
  --data ./path/to/your_dataset.csv \
  --sections overview,missingness,numeric \
  --columns-numeric col_a,col_b \
  --output ./output_eda
```

## Example (Using Scripts 03/04 Dataset)

```bash
python scripts/03_generate_bank_aml_dataset.py \
  --out-dir ./data \
  --rows 200000 \
  --suspicious-rate 0.04 \
  --label-noise 0.02 \
  --seed 7
```

```bash
eda-agent --data ./data/synthetic_bank_aml_200k.csv --target-col is_suspicious --output ./output_eda
```

## Notes

- You can also pass a pandas DataFrame to `EDA.run(df=...)`.
- You can set `EDA_DATA_PATH` to point to a dataset and omit `--data`.
- The EDA runner auto-detects column types and will list numeric/categorical/time columns in the report.
