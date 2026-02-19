# 10 NotebookLM Demo and Troubleshooting (EN)

## A. Copy/Paste Command Sheet

### 1) Generate demo datasets
```bash
python scripts/05_generate_synthetic_aml_200k_timeseries.py --out-dir ./data
python scripts/07_generate_synthetic_aml_mixed_bank_fintech.py --out-dir ./data
```

### 2) EDA (Pandas) one-shot
```bash
python -m eda.cli --output ./output_eda
```

### 3) EDA Spark (PySpark) one-shot
```bash
python -m eda_spark.cli --spark-master "local[*]" --output ./output_eda_spark
```

### 4) EDA (Pandas) explicit dataset
```bash
python -m eda.cli \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --target-col is_suspicious \
  --output ./output_eda
```

### 5) EDA Spark explicit dataset
```bash
python -m eda_spark.cli \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --target-col sar_actual \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

### 6) Section-scoped runs
```bash
python -m eda.cli \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --sections data_quality,univariate,time_drift \
  --columns-univariate txn_amount,velocity_score,origin_country \
  --target-col is_suspicious \
  --output ./output_eda

python -m eda_spark.cli \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --sections data_quality,target,univariate \
  --columns-univariate txn_amount,velocity_score,merchant_category \
  --columns-target sar_actual \
  --target-col sar_actual \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

### 7) Interactive runs
```bash
python -m eda.cli --list-functions
python -m eda_spark.cli --list-functions

python -m eda.cli \
  --interactive \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --target-col is_suspicious \
  --output ./output_eda

python -m eda_spark.cli \
  --interactive \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --target-col sar_actual \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

### 8) Output files to showcase in deck
- `./output_eda/EDA_Report.pdf`
- `./output_eda/eda_results.json`
- `./output_eda_spark/EDA_Report.pdf`
- `./output_eda_spark/eda_results.json`

## B. Parameters and Section Names (Do Not Rename)
- Input modes:
- `--data --sql --db --py --py-code --nb --data-recursive --compose-spec --no-key-policy --auto-exec`
- Selection controls:
- `--sections --columns --columns-data-quality --columns-target --columns-univariate --columns-bivariate-target --columns-feature-vs-feature --columns-time-drift`
- Run modes:
- `--interactive --list-functions --max-rows --no-report --no-json`
- Spark options:
- `--spark-master --spark-app-name --spark-conf`
- Section keys:
- `data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift`

## C. Frequent Errors and Fast Fixes

### 1) Error: "Only one input mode is allowed"
- Cause: more than one of `--data/--sql/--py/--py-code/--nb` is set.
- Fix: keep exactly one mode per command.

### 2) Error: SQL mode requires DB
- Cause: using `--sql` without `--db`.
- Fix: pass `--db`, for example `sqlite:///./data/aml.db`.

### 3) Error: mixed named and unnamed `--data`
- Cause: command mixes `--data a.csv` and `--data table=b.csv`.
- Fix: use either all unnamed or all named bindings.

### 4) Multi-table join composition failure
- Cause: no reliable join keys inferred.
- Fix:
- pass `--compose-spec` with explicit joins, or
- set `--no-key-policy aggregate_only` if table-level fallback is acceptable.

### 5) Spark runtime/config issues
- Cause: local Spark environment not initialized correctly.
- Fix:
- pass `--spark-master "local[*]"`,
- verify PySpark installation,
- in CML/headless contexts set `MPLBACKEND=Agg` and `MPLCONFIGDIR`.

## D. NotebookLM Upload Order (Recommended)
- `00_overview.md`
- `08_notebooklm_deck_story_en.md`
- `09_notebooklm_slide_content_en.md`
- `10_notebooklm_demo_and_troubleshooting_en.md`
- `01_inputs_catalog.md`
- `02_eda_pipeline.md`
- `03_eda_spark_pipeline.md`
- `05_usage_modes.md`
- `06_demo_paths_in_data_folder.md`

## E. NotebookLM Deck Generation Prompt (English)
Use this text as your Slide Deck prompt inside NotebookLM:

```text
Create a 12-14 slide English technical deck for model developers on how to use EDA (Pandas) and EDA Spark (PySpark) in this repository.

Requirements:
- Focus on practical usage and decisions, not generic EDA theory.
- Include exact runnable commands from the sources.
- Include one comparison slide: when to choose EDA vs EDA Spark.
- Include one slide mapping each section (data_quality, target, univariate, bivariate_target, feature_vs_feature, time_drift) to modeling decisions.
- Include one troubleshooting slide with common errors and fixes.
- Keep all parameter names and file paths exactly as in the sources.
- End with a rollout checklist and next steps for model developers.
```
