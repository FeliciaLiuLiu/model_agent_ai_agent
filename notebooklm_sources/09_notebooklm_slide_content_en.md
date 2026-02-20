# 09 Slide-by-Slide Content (Detailed)

## Slide 1 - Title
**EDA and EDA Spark for Model Developers**
- Focus: Input flexibility, system design, and practical usage.

## Slide 2 - Agenda
- Input modes and constraints.
- System design (EDA + EDA Spark).
- Section blocks and functions.
- One-slide usage methods (CLI/API, interactive/non-interactive).
- Case slides.
- Output interpretation and next modeling actions.

## Slide 3 - Input Flexibility Overview
- Supported modes: `data`, `sql+db`, `py`, `py_code`, `nb`.
- Supported file types: csv/tsv/parquet/json/xlsx/xls/feather.
- Single mode per run rule.

## Slide 4 - Input Weaknesses and Failure Conditions
- Constraint 1: only one input mode allowed.
- Constraint 2: row-level multi-table merge requires joinable keys.
- Constraint 3: SQL mode requires `--db`.
- Constraint 4: `--py`/`--nb` must expose `load()` or `df`.

## Slide 5 - EDA (Pandas) Architecture
- Show flow: Input -> `eda/dataloader.py` -> `eda/runner.py` -> outputs.
- Explain prerequisite checks and section skip logic.

## Slide 6 - EDA Spark (PySpark) Architecture
- Show flow: Input -> `eda_spark/dataloader.py` -> `eda_spark/runner.py` -> distributed compute -> outputs.
- Clarify driver vs executor roles.

## Slide 7 - Section Blocks and Runner Functions
Section keys:
- `data_quality`, `target`, `univariate`, `bivariate_target`, `feature_vs_feature`, `time_drift`

Runner functions:
- `run()`, `run_interactive()`, `list_functions()`, `parse_function_selection()`, `parse_column_selection()`, `print_functions()`

## Slide 8 - Unified Usage Methods (One Slide, No Dataset-Specific Paths)
Show all four modes in one page:

CLI non-interactive (template):
```bash
python -m eda.cli \
  --data ./path/to/input_file.csv \
  --sections data_quality,target,univariate,feature_vs_feature,time_drift \
  --output ./output_eda
```

CLI interactive (template):
```bash
python -m eda.cli \
  --data ./path/to/input_file.csv \
  --interactive \
  --output ./output_eda
```

API non-interactive (template):
```python
from adm_central_utility import EDA
eda = EDA(output_dir='./output_eda')
results = eda.run(data=['./path/to/input_file.csv'], sections=['data_quality','univariate'])
```

API interactive (template):
```python
from adm_central_utility import EDA
eda = EDA(output_dir='./output_eda')
payload = eda.run_interactive(data=['./path/to/input_file.csv'], return_payload=True)
```

Note on this slide:
- For Spark, keep same mode logic and replace module/class:
- `python -m eda_spark.cli ...`
- `from eda_spark.runner import EDASpark`

## Slide 9 - EDA Case Slide (SQL Script Example)
- Demo source: `./data/aml_synthetic_20k.sql` (materialize to SQLite first).

```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

## Slide 10 - EDA Case Slide (API Example)
```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda', target_col='is_suspicious', time_col='txn_datetime')
results = eda.run(
    sql='SELECT * FROM aml_synthetic',
    db='sqlite:///./data/aml_synthetic_20k.db',
    sections=['data_quality','target','univariate','bivariate_target','feature_vs_feature','time_drift'],
)

payload = eda.run_interactive(
    sql='SELECT * FROM aml_synthetic',
    db='sqlite:///./data/aml_synthetic_20k.db',
    return_payload=True,
)
```

## Slide 11 - EDA Spark Case Slide (Required CLI Example)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

Interactive CLI:
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## Slide 12 - EDA Spark Case Slide (API Example)
```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir='./output_eda_spark', spark_master='local[*]')
results = eda.run(
    py='./data/Paypal_data.py',
    sections=['data_quality','univariate','feature_vs_feature','time_drift'],
    max_rows=5000,
)

payload = eda.run_interactive(
    py='./data/Paypal_data.py',
    max_rows=5000,
    return_payload=True,
)
```

## Slide 13 - Output Contract and How to Read It
- File outputs:
- `output_eda/eda_results.json`
- `output_eda/EDA_Report.pdf`
- `output_eda_spark/eda_results.json`
- `output_eda_spark/EDA_Report.pdf`
- API payload keys:
- `results`, `skipped_sections`, `config`

## Slide 14 - From EDA Findings to Data Cleaning Tasks
- Missingness high -> define imputation/drop rules.
- Duplicate ratio high -> dedup strategy and primary-key validation.
- Null-like strings -> value normalization map.
- Time parse issues -> datetime parsing standardization.

## Slide 15 - From EDA Findings to Feature Engineering Tasks
- Strong feature correlations -> prune/reduce multicollinearity.
- Drift signals -> robust temporal split and monitoring features.
- Categorical imbalance -> encoding + frequency bucketing.
- Target imbalance -> class weight or resampling strategy.

## Slide 16 - Execution Checklist for Model Developers
1. Confirm input mode and loader contract.
2. Start with the one-slide usage method and choose CLI/API + interactive mode.
3. Run case command.
4. Review `eda_results.json` first, then PDF.
5. Convert findings into cleaning/feature tasks and re-run.
