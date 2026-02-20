# 05 How Model Developers Run EDA and EDA Spark

## Output Guarantee
Unless `--no-report` is used, CLI runs generate:
- `EDA_Report.pdf`
- `eda_results.json`

API runs also generate PDF/JSON when `generate_report=True` and `save_json=True` (both default to True in runner methods).

## A) EDA (Pandas) Demo Using `./data/aml_synthetic_20k.sql`

### Step A1: Materialize SQL script into SQLite DB
`aml_synthetic_20k.sql` is a SQL script (DDL + data). Create a DB first:

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

sql_text = Path('./data/aml_synthetic_20k.sql').read_text(encoding='utf-8')
conn = sqlite3.connect('./data/aml_synthetic_20k.db')
conn.executescript(sql_text)
conn.commit()
conn.close()
print('created ./data/aml_synthetic_20k.db')
PY
```

### Step A2: CLI Non-Interactive (EDA)

```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

### Step A3: CLI Interactive (EDA)

```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --interactive \
  --output ./output_eda
```

### Step A4: API Non-Interactive (EDA)

```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda', target_col='is_suspicious', time_col='txn_datetime')
results = eda.run(
    sql='SELECT * FROM aml_synthetic',
    db='sqlite:///./data/aml_synthetic_20k.db',
    sections=['data_quality', 'target', 'univariate', 'bivariate_target', 'feature_vs_feature', 'time_drift'],
    generate_report=True,
    save_json=True,
)
```

### Step A5: API Interactive (EDA)

```python
from adm_central_utility import EDA

eda = EDA(output_dir='./output_eda', target_col='is_suspicious', time_col='txn_datetime')
payload = eda.run_interactive(
    sql='SELECT * FROM aml_synthetic',
    db='sqlite:///./data/aml_synthetic_20k.db',
    generate_report=True,
    save_json=True,
    return_payload=True,
)
```

## B) EDA Spark Demo Using `--py ./data/Paypal_data.py`

### Required Non-Interactive Example (Top 5000 rows)

```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

### CLI Interactive (EDA Spark, Top 5000 rows)

```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

### API Non-Interactive (EDA Spark, Top 5000 rows)

```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir='./output_eda_spark', spark_master='local[*]')
results = eda.run(
    py='./data/Paypal_data.py',
    sections=['data_quality', 'univariate', 'feature_vs_feature', 'time_drift'],
    max_rows=5000,
    generate_report=True,
    save_json=True,
)
```

### API Interactive (EDA Spark, Top 5000 rows)

```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir='./output_eda_spark', spark_master='local[*]')
payload = eda.run_interactive(
    py='./data/Paypal_data.py',
    max_rows=5000,
    generate_report=True,
    save_json=True,
    return_payload=True,
)
```

## C) Non-Interactive vs Interactive Summary

| Mode | How it works | Best for |
|---|---|---|
| Non-interactive | User pre-defines sections/columns via args or code | CI jobs, reproducible pipelines |
| Interactive | Runner prompts section/column choices at runtime | Exploration, ad-hoc analyst flow |

## D) Verify PDF + JSON Were Generated

```bash
ls -lh ./output_eda
ls -lh ./output_eda_spark
```

Expected:
- `./output_eda/EDA_Report.pdf`
- `./output_eda/eda_results.json`
- `./output_eda_spark/EDA_Report.pdf`
- `./output_eda_spark/eda_results.json`
