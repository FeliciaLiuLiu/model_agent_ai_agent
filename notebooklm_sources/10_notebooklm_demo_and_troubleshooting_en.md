# 10 NotebookLM Prompt, Validation, and Troubleshooting

## A) Prompt to Generate the Deck in NotebookLM

```text
Create a 16-slide English technical deck for model developers about EDA (Pandas) and EDA Spark (PySpark) in this repository.

Visual style requirements:
- White background.
- Simple and clean layout.
- Minimal visual decoration.
- Prioritize architecture diagrams, comparison tables, and command blocks.

Content requirements:
1) Focus on three primary themes:
   - Input flexibility and input constraints,
   - Full system design (Input -> Engine -> Output) for EDA and EDA Spark,
   - Concrete usage via CLI and API.
2) Include explicit weaknesses:
   - exactly one input mode per run,
   - multi-table row-level merge needs joinable keys, else fail-fast by default.
3) Show section keys:
   data_quality, target, univariate, bivariate_target, feature_vs_feature, time_drift.
4) Show runner-level functions:
   run(), run_interactive(), list_functions(), parse_function_selection(), parse_column_selection(), print_functions().
5) Create one dedicated "Usage Methods" slide that contains all four modes together:
   - CLI non-interactive,
   - CLI interactive,
   - API non-interactive,
   - API interactive.
6) In that usage slide, use generic placeholders only (no dataset-specific paths).
7) Put dataset-specific commands in later case slides.
8) Include EDA case using ./data/aml_synthetic_20k.sql (materialize to SQLite then query aml_synthetic).
9) Include EDA Spark required CLI case:
   python -m eda_spark.cli --py ./data/Paypal_data.py --sections data_quality,univariate,feature_vs_feature,time_drift --max-rows 5000 --output ./output_eda_spark
10) Emphasize outputs:
   <output_dir>/eda_results.json and <output_dir>/EDA_Report.pdf.
11) Add slides mapping EDA findings to next actions for Data Cleaning and Feature Engineering.
```

## B) Generic Usage Slide Snippets (No Dataset-Specific Paths)

CLI non-interactive template:
```bash
python -m eda.cli --data ./path/to/input.csv --sections data_quality,univariate --output ./output_eda
```

CLI interactive template:
```bash
python -m eda.cli --data ./path/to/input.csv --interactive --output ./output_eda
```

API non-interactive template:
```python
from adm_central_utility import EDA
eda = EDA(output_dir='./output_eda')
results = eda.run(data=['./path/to/input.csv'], sections=['data_quality','univariate'])
```

API interactive template:
```python
payload = eda.run_interactive(data=['./path/to/input.csv'], return_payload=True)
```

Spark adaptation note:
- CLI: `python -m eda_spark.cli ...`
- API: `from eda_spark.runner import EDASpark`

## C) Mandatory Case Commands to Keep in Later Slides

### EDA case (CLI)
```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

### EDA Spark case (CLI, required)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## D) Troubleshooting Notes for Demo
1. `ImportError: Unable to import python file: ./data/Paypal_data`
- Cause: missing `.py` extension or wrong path.
- Fix: use `--py ./data/Paypal_data.py`.

2. `ValueError: Python file must define load() or df`
- Cause: loader contract not satisfied.
- Fix: add `load()` or `df`.

3. `TABLE OR VIEW NOT FOUND`
- Cause: SQL references missing schema/table.
- Fix: materialize correct table into SQLite or fix schema/table names.

4. `Only one input mode is allowed`
- Cause: mixed modes in one run.
- Fix: keep one mode per run.

5. `time_drift` skipped
- Cause: no parseable time column.
- Fix: pass `--time-col` and normalize datetime values.

## E) Validation Checklist for Generated Slides
- One consolidated usage slide contains CLI/API + interactive/non-interactive together.
- Usage slide uses placeholders (no fixed dataset).
- Later slides contain concrete EDA and EDA Spark cases.
- Both architectures are shown from Input -> Engine -> Output.
- Output artifacts (`eda_results.json`, `EDA_Report.pdf`) are highlighted.
- Data Cleaning and Feature Engineering actions are explicit.
