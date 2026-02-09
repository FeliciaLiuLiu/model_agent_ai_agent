# EDA Spark (CML)

Spark-first EDA for large datasets and CML environments. This version uses PySpark for computations where possible, only converting small aggregates to pandas for plotting.

## Minimal Files to Copy into a Fresh CML Project

You only need:
- `eda_spark/` (this folder)
- `data/` (your datasets)
- `eda_spark/requirements.txt` (dependencies)

Important: keep the folder name **lowercase** `eda_spark` (Linux is case-sensitive).

## Quick Start (One-Shot Report)

1) Generate or place a dataset under `./data`:
```bash
python scripts/07_generate_synthetic_aml_mixed_bank_fintech.py
```

2) Run EDA Spark and export the PDF + JSON:
```bash
export MPLCONFIGDIR=./.mpl_cache
export MPLBACKEND=Agg

python -m eda_spark.cli \
  --output ./output_eda_spark
```

Outputs:
- `./output_eda_spark/EDA_Report.pdf`
- `./output_eda_spark/eda_results.json`

By default, the CLI auto-loads from `./data`. You can override the dataset:
```bash
python -m eda_spark.cli --data ./data/synthetic_aml_mixed_50k_YYYYMMDD_HHMMSS.csv
```
If you want parquet, pass it explicitly via `--data`. Local paths are forced to `file://` URIs internally to avoid HDFS defaults in CML.
Local paths are forced to `file://` URIs internally to avoid HDFS defaults in CML.

## Interactive Mode (Choose Sections + Columns)

```bash
python -m eda_spark.cli --interactive
```

How interactive mode works:
- You will be prompted to select sections by number (e.g., `1,3,5`) or `all`.
- For each selected section, you can choose columns by number or name.
- Press Enter to accept the defaults.

You can list all sections and descriptions:
```bash
python -m eda_spark.cli --list-functions
```

## Notebook Usage (Non-Interactive + Interactive)

Non-interactive (Notebook):
```python
import sys
sys.path.append(".")

from eda_spark.runner import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", spark_master="local[*]")
eda.run()  # auto-loads from ./data
```

Interactive (Notebook):
```python
import sys
sys.path.append(".")

from eda_spark.runner import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", spark_master="local[*]")
eda.run_interactive()
```

## CLI Usage (Module or Script)

Both styles work:
```bash
python -m eda_spark.cli --output ./output_eda_spark
```
```bash
python eda_spark/cli.py --output ./output_eda_spark
```

## Unified Data Loader (Spark)

EDA Spark accepts exactly one input mode and loads everything into a Spark DataFrame:

1) Files/dirs/globs: `data=[...]` or CLI `--data` (repeatable). Multiple files are unioned by name.
2) SQL: `sql` + `db` (SQLite path/URL or JDBC URL). For JDBC, Spark uses `spark.read.format("jdbc")`.
3) Python file: `py` that defines `load()` or `df` (pandas or Spark DataFrame).
4) Inline Python: `py_code` that defines `load()` or `df`.
5) Notebook: `nb` (`.ipynb`) with `load()` or `df` in code cells.

Supported file types: `.csv`, `.tsv`, `.parquet`, `.json`, `.xlsx`, `.xls`, `.feather`.
Excel uses `spark-excel` if available, otherwise falls back to pandas (`openpyxl` required).

### Auto-Exec (no input flags)

If you run `python -m eda_spark.cli --output ./output_eda_spark` with no input flags, EDA will auto-scan `./data` and:

- Load any supported data files and union them.
- Execute any `.py` / `.ipynb` files that define `df` or `load()` and union them.
- Execute any `.sql` files:
  - If a `.db`/`.sqlite` file exists and the SQL is a single `SELECT`/`WITH` query, it runs against that DB.
  - Otherwise, the SQL is executed in a temporary SQLite DB and must create a single table/view (or name it `aml_dataset`, `eda_dataset`, or `eda_input`).

Use `--auto-exec` to explicitly enable this behavior when desired.

## Section and Column Overrides (Non-Interactive)

Run only specific sections:
```bash
python -m eda_spark.cli --sections data_quality,target,univariate
```

Provide columns globally:
```bash
python -m eda_spark.cli --columns txn_amount,velocity_score,origin_country
```

Or per section:
```bash
python -m eda_spark.cli \
  --columns-univariate txn_amount,velocity_score \
  --columns-target sar_actual
```

## CML Notes

Typical CML Spark session environment configuration:
```bash
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=./spark_tmp"
export SPARK_LOCAL_DIRS="./spark_tmp"
export SPARK_DRIVER_OPTS="-Djava.io.tmpdir=./spark_tmp"
export SPARK_EXECUTOR_OPTS="-Djava.io.tmpdir=./spark_tmp"
export MPLCONFIGDIR=./.mpl_cache
export MPLBACKEND=Agg

python -m eda_spark.cli --output ./output_eda_spark --spark-master "local[*]"
```

You can also pass extra Spark configs:
```bash
python -m eda_spark.cli \
  --spark-conf spark.sql.shuffle.partitions=200 \
  --spark-conf spark.driver.memory=4g
```

## Input Examples

Multiple files:
```bash
python -m eda_spark.cli --data ./data/part1.csv --data ./data/part2.parquet --output ./output_eda_spark
```

SQL:
```bash
python -m eda_spark.cli --sql "SELECT * FROM aml_dataset" --db "sqlite:///./data/aml.db" --output ./output_eda_spark
```

Python / notebook:
```bash
python -m eda_spark.cli --py ./load_data.py --output ./output_eda_spark
python -m eda_spark.cli --nb ./load_data.ipynb --output ./output_eda_spark
```

## Environment Overrides

- `EDA_SPARK_DATA_PATH`: force a specific dataset path.
- `MPLCONFIGDIR`: Matplotlib cache path (recommended for CML).
- `MPLBACKEND=Agg`: headless PDF rendering.
