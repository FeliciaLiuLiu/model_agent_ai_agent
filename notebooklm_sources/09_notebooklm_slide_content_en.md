# 09 Slide-by-Slide Content (Detailed)

## Slide 1 - Title
**EDA and EDA Spark for Model Developers**
- Focus: Input flexibility, system design, and practical usage.

## Slide 2 - Agenda
- Input modes and constraints.
- EDA system design.
- EDA Spark system design.
- CLI/API usage in interactive and non-interactive modes.
- How to convert report outputs into next modeling actions.

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
- Show flow diagram: Input -> `eda/dataloader.py` -> `eda/runner.py` -> outputs.
- Explain runner responsibilities and skip logic.

## Slide 6 - EDA Section Blocks and Functions
- `data_quality`: duplicates, missingness, null-like checks, type summary.
- `target`: target distribution/stats, target trend, target-by-category rates.
- `univariate`: numeric stats/histograms, categorical top-k.
- `bivariate_target`: bins/groups vs target.
- `feature_vs_feature`: correlation matrix + high-correlation pairs.
- `time_drift`: trend and drift signals.

## Slide 7 - EDA CLI Non-Interactive Example (SQL Demo)
- Use `aml_synthetic_20k.sql` materialized into SQLite.

```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

## Slide 8 - EDA CLI Interactive Example

```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --interactive \
  --output ./output_eda
```

## Slide 9 - EDA API (Non-Interactive + Interactive)
Non-interactive:
```python
from adm_central_utility import EDA
eda = EDA(output_dir='./output_eda', target_col='is_suspicious', time_col='txn_datetime')
results = eda.run(sql='SELECT * FROM aml_synthetic', db='sqlite:///./data/aml_synthetic_20k.db', sections=['data_quality','target','univariate','bivariate_target','feature_vs_feature','time_drift'])
```

Interactive:
```python
payload = eda.run_interactive(sql='SELECT * FROM aml_synthetic', db='sqlite:///./data/aml_synthetic_20k.db', return_payload=True)
```

## Slide 10 - EDA Spark Architecture
- Show flow diagram: Input -> `eda_spark/dataloader.py` -> `eda_spark/runner.py` -> distributed compute -> outputs.
- Clarify driver/executor responsibilities.

## Slide 11 - EDA Spark Section Blocks and Functions
- Same section keys as EDA.
- Explain that semantics are aligned, compute backend is different.

## Slide 12 - EDA Spark CLI Non-Interactive (Required Example)

```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## Slide 13 - EDA Spark CLI Interactive

```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## Slide 14 - EDA Spark API (Non-Interactive + Interactive)
Non-interactive:
```python
from eda_spark.runner import EDASpark
eda = EDASpark(output_dir='./output_eda_spark', spark_master='local[*]')
results = eda.run(py='./data/Paypal_data.py', sections=['data_quality','univariate','feature_vs_feature','time_drift'], max_rows=5000)
```

Interactive:
```python
payload = eda.run_interactive(py='./data/Paypal_data.py', max_rows=5000, return_payload=True)
```

## Slide 15 - Output Contract and How to Read It
- File outputs:
- `output_eda/eda_results.json`
- `output_eda/EDA_Report.pdf`
- `output_eda_spark/eda_results.json`
- `output_eda_spark/EDA_Report.pdf`
- API payload keys:
- `results`, `skipped_sections`, `config`

## Slide 16 - From EDA Findings to Data Cleaning Tasks
- Missingness high -> define imputation/drop rules.
- Duplicate ratio high -> dedup strategy and primary-key validation.
- Null-like strings -> value normalization map.
- Time parse issues -> datetime parsing standardization.

## Slide 17 - From EDA Findings to Feature Engineering Tasks
- Strong feature correlations -> prune/reduce multicollinearity.
- Drift signals -> robust temporal split and monitoring features.
- Categorical imbalance -> encoding + frequency bucketing.
- Target imbalance -> class weight or resampling strategy.

## Slide 18 - Execution Checklist for Model Developers
1. Confirm input mode and loader contract.
2. Run non-interactive baseline command.
3. Review `eda_results.json` first, then PDF.
4. Convert findings into cleaning/feature tasks.
5. Re-run after data updates and compare outputs.
