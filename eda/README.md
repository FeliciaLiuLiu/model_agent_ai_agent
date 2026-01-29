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

## Functions (Detailed)

### 1) Overview
- Purpose: dataset size, column types, uniqueness, duplicates, and target distribution (if provided).
- Applicable columns: all.
- Metrics: rows, columns, duplicate_ratio, unique_counts, typed column lists.
- Plots: none.

### 2) Missingness
- Purpose: missing rate per column.
- Applicable columns: all.
- Metrics: missing_rate per column.
- Plots: `missingness.png` (top 20 missing columns).

### 3) Numeric
- Purpose: numeric summary statistics and distributions.
- Applicable columns: numeric.
- Metrics: count/mean/std/min/percentiles/max per numeric column.
- Plots: `hist_<col>.png` (up to top 5 numeric columns by variance when no columns are specified).

### 4) Categorical
- Purpose: category frequency summaries.
- Applicable columns: categorical/boolean.
- Metrics: top category counts per column.
- Plots: `cat_<col>.png` (up to top 5 categorical columns by unique count when no columns are specified).

### 5) Correlation
- Purpose: correlation structure of numeric features.
- Applicable columns: numeric.
- Metrics: correlation matrix; optional target correlation if target is numeric.
- Plots: `correlation_heatmap.png`.

### 6) Target
- Purpose: relate features to target.
- Applicable columns: numeric/categorical (requires `target_col`).
- Metrics: class-wise numeric means and categorical rates (classification) or numeric correlations (regression).
- Plots: none.

### 7) Outliers
- Purpose: IQR-based outlier ratios per numeric feature.
- Applicable columns: numeric.
- Metrics: outlier_ratio per numeric column.
- Plots: `outliers.png` (top 10 outlier ratios).

### 8) Time
- Purpose: time-series volume and target rate over time.
- Applicable columns: time column (requires `time_col`).
- Metrics: daily volume and optional daily target rate.
- Plots: `time_volume.png`, `time_target_rate.png` (if target is categorical).

Notes:
- If you pass columns that do not apply to a function (e.g., strings to Numeric), the function is skipped with a summary note.
- Time analysis runs only if the time column parses successfully (>= 90% valid).

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
# or: from adm_central_utility import eda

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run(file_path="./path/to/your_dataset.csv")
```

### Interactive Selection (Choose Sections and Columns)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run_interactive(file_path="./path/to/your_dataset.csv")
```

### Choose Functions and Columns by Number (Notebook)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.print_functions()  # Shows numbered list

# Example: select 1,2,3 (overview, missingness, numeric)
sections = EDA.parse_function_selection("1,2,3")

numeric_cols = ["col_a", "col_b", "col_c"]
selected_numeric = EDA.parse_column_selection("1,3", numeric_cols)  # -> ["col_a", "col_c"]

results = eda.run(
    file_path="./path/to/your_dataset.csv",
    sections=sections,
    section_columns={
        "numeric": selected_numeric,
        "categorical": ["col_c"],
    },
)
```

## CLI Usage

### Basic

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --target-col your_target
```

### List Functions

```bash
eda-agent --list-functions
```

### Interactive Mode (Select by 1,2,3)

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --interactive
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

### Limit Rows for Faster Reports

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --max-rows 5000
```

For testing datasets, use `--max-rows 5000` to generate a quicker PDF.

### Skip JSON Output (Faster)

```bash
eda-agent --data ./path/to/your_dataset.csv --output ./output_eda --no-json
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
- If you do not specify columns, EDA analyzes the top 5 numeric columns (by variance) and top 5 categorical columns (by unique count).
- Plots are generated at 80 DPI with smaller figure sizes by default to speed up rendering.
- Time-series plots run only if the time column can be parsed successfully (>= 90% valid).
