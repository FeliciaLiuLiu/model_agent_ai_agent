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
