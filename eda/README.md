# EDA Module

The EDA module provides automated exploratory data analysis with PDF and JSON outputs. It supports both Pandas and Spark implementations and is designed to run on unknown datasets with auto detection, skippable sections, and interactive selection.

## What It Does

Sections (skippable if prerequisites are not met):

- Data Quality
- Target / Label EDA (requires target column)
- Univariate (numeric + categorical)
- Bivariate with Target (requires target column)
- Feature vs Feature (requires >= 2 numeric columns)
- Time Series and Drift (requires parseable time column)
- Summary and Recommendations

Key features:

- Auto-detect latest dataset in `./data` by filename timestamp (fallback to mtime)
- Column type classification tables (numeric/categorical/datetime/boolean/text)
- Interactive selection of sections and columns
- Skipped sections summary with reasons in PDF and JSON
- ReportLab PDF layout with tables and charts

## Inputs

- Dataset file path (CSV or Parquet), or a DataFrame
- Optional: `target_col`, `time_col`, `id_cols`
- Optional: section and column selection

## Outputs

- `EDA_Report.pdf`
- `eda_results.json`
- Plot PNGs under the output directory

## API Usage (Pandas)

### Non-Interactive (Full Report)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.run(file_path="./path/to/your_dataset.csv")
```

### Auto-Detect Latest Dataset

If `file_path` is omitted, EDA finds the latest dataset in `./data`:

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.run()  # auto-detect latest ./data/*.csv or *.parquet
```

### Interactive Selection (Choose Sections and Columns)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.run_interactive(file_path="./path/to/your_dataset.csv")
```

### Programmatic Selection (Notebook)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.print_functions()  # numbered list

sections = EDA.parse_function_selection("1,3,6")
selected_cols = ["txn_amount", "channel", "origin_country"]

eda.run(
    file_path="./path/to/your_dataset.csv",
    sections=sections,
    section_columns={
        "univariate": selected_cols,
        "time_drift": ["txn_amount"],
    },
)
```

## API Usage (Spark)

Spark version is available via `EDASpark` and avoids scikit-learn.

```python
from adm_central_utility import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", target_col="your_target")
eda.run(file_path="./path/to/your_dataset.csv")
```

## CLI Usage (Mac/Linux)

### Non-Interactive (Pandas)

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --target-col your_target
```

### Interactive (Pandas)

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --interactive
```

### Spark Implementation

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda_spark --spark
```

### Auto-Detect Latest Dataset

```bash
eda-agent --output ./output_eda --target-col your_target
```

## CLI Usage (Windows PowerShell)

### Non-Interactive (Pandas)

```powershell
eda-agent --data C:\path\to\your_dataset.csv --output .\output_eda --target-col your_target
```

### Interactive (Pandas)

```powershell
eda-agent --data C:\path\to\your_dataset.csv --output .\output_eda --interactive
```

### Spark Implementation

```powershell
eda-agent --data C:\path\to\your_dataset.csv --output .\output_eda_spark --spark
```

## Section Selection and Column Filters (CLI)

```bash
eda-agent \
  --data ./path/to/your_dataset.csv \
  --sections data_quality,univariate,time_drift \
  --columns-univariate txn_amount,channel,origin_country \
  --columns-time-drift txn_amount \
  --output ./output_eda
```

## Example: Generate Synthetic AML Dataset (Script 05)

```bash
python scripts/05_generate_synthetic_aml_200k_timeseries.py --out-dir ./data --rows 200000 --seed 7 --suspicious-rate 0.04 --label-noise 0.02 --parquet
```

```bash
eda-agent --output ./output_eda --target-col is_suspicious
```

## Notes

- If you do not specify columns, the runner selects top numeric columns by variance and top categorical columns by unique count.
- Time series and drift sections only run if time parsing success >= 90%.
- Set `EDA_DATA_PATH` to point to a dataset and omit `--data`.
- Spark version requires `pyspark` to be installed and accessible in your environment.
