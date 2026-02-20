# 10 NotebookLM Prompt, Validation, and Troubleshooting

## A) Prompt to Generate the Deck in NotebookLM

```text
Create a 16-18 slide English technical deck for model developers about EDA (Pandas) and EDA Spark (PySpark) in this repository.

Visual style requirements:
- White background.
- Simple and clean layout.
- Minimal visual decoration.
- Prioritize architecture diagrams, comparison tables, and command blocks.

Content requirements:
1) Focus on three primary themes:
   - Input flexibility and input constraints,
   - Full system design (Input -> Engine -> Output) for EDA and EDA Spark,
   - Concrete usage via CLI and API, both non-interactive and interactive.
2) Include explicit weaknesses:
   - exactly one input mode per run,
   - multi-table row-level merge needs joinable keys, else fail-fast by default.
3) Show the section keys and explain what each section analyzes:
   data_quality, target, univariate, bivariate_target, feature_vs_feature, time_drift.
4) Show runner-level functions:
   run(), run_interactive(), list_functions(), parse_function_selection(), parse_column_selection(), print_functions().
5) Include exact EDA (Pandas) examples using ./data/aml_synthetic_20k.sql (materialize to SQLite then query aml_synthetic).
6) Include exact EDA Spark CLI example:
   python -m eda_spark.cli --py ./data/Paypal_data.py --sections data_quality,univariate,feature_vs_feature,time_drift --max-rows 5000 --output ./output_eda_spark
7) Include API examples for both projects in non-interactive and interactive modes.
8) Emphasize output artifacts:
   <output_dir>/eda_results.json and <output_dir>/EDA_Report.pdf.
9) Add slides that convert EDA findings into next actions for Data Cleaning and Feature Engineering.
10) End with an execution checklist for model developers.
```

## B) Mandatory Command Examples to Keep in Deck

### EDA (CLI non-interactive)
```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

### EDA (CLI interactive)
```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --interactive \
  --output ./output_eda
```

### EDA Spark (CLI non-interactive, required)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

### EDA Spark (CLI interactive)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## C) Troubleshooting Notes for Demo
1. `ImportError: Unable to import python file: ./data/Paypal_data`
- Cause: missing `.py` extension or wrong path.
- Fix: use `--py ./data/Paypal_data.py`.

2. `ValueError: Python file must define load() or df`
- Cause: loader file contract not satisfied.
- Fix: add `def load(): return <DataFrame>` or define `df = <DataFrame>`.

3. `TABLE OR VIEW NOT FOUND`
- Cause: SQL references missing schema/table.
- Fix: materialize correct table into SQLite or fix schema-qualified names.

4. `Only one input mode is allowed`
- Cause: mixed `--data` with `--sql`/`--py`/`--nb` in one run.
- Fix: keep one mode per run.

5. `time_drift` skipped
- Cause: no parseable time column.
- Fix: pass `--time-col` and standardize datetime parsing.

## D) Validation Checklist for Generated Slides
- Both architectures clearly shown from Input -> Engine -> Output.
- Input flexibility and weaknesses clearly separated.
- CLI and API both shown for both projects.
- Interactive and non-interactive both shown for both projects.
- Section keys and their analysis functions are explicitly described.
- Output artifacts (`eda_results.json`, `EDA_Report.pdf`) are central in the narrative.
- Data Cleaning and Feature Engineering guidance is actionable.
