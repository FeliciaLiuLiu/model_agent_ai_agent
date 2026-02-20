# 06 Demo Paths and Runbook for This Deck

## Demo Inputs to Reference in Slides

### EDA demo input
- Source script: `./data/aml_synthetic_20k.sql`
- Materialized DB: `./data/aml_synthetic_20k.db`
- Query target table: `aml_synthetic`

### EDA Spark demo input
- Python loader: `./data/Paypal_data.py`
- Contract: file must define `load()` or `df`.

## Quick Validation Commands

### Validate EDA SQL demo files
```bash
ls -l ./data/aml_synthetic_20k.sql
head -n 20 ./data/aml_synthetic_20k.sql
```

### Validate EDA Spark Python loader
```bash
ls -l ./data/Paypal_data.py
head -n 40 ./data/Paypal_data.py
```

## Build DB from SQL Script (for EDA demo)

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

sql_text = Path('./data/aml_synthetic_20k.sql').read_text(encoding='utf-8')
conn = sqlite3.connect('./data/aml_synthetic_20k.db')
conn.executescript(sql_text)
conn.commit()
conn.close()
print('ready: ./data/aml_synthetic_20k.db')
PY
```

## Canonical Commands to Put into Slides

### EDA CLI non-interactive
```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --sections data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --output ./output_eda
```

### EDA CLI interactive
```bash
python -m eda.cli \
  --sql "SELECT * FROM aml_synthetic" \
  --db "sqlite:///./data/aml_synthetic_20k.db" \
  --target-col is_suspicious \
  --time-col txn_datetime \
  --interactive \
  --output ./output_eda
```

### EDA Spark CLI non-interactive (required)
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --sections data_quality,univariate,feature_vs_feature,time_drift \
  --max-rows 5000 \
  --output ./output_eda_spark
```

### EDA Spark CLI interactive
```bash
python -m eda_spark.cli \
  --py ./data/Paypal_data.py \
  --interactive \
  --max-rows 5000 \
  --output ./output_eda_spark
```

## Common CML Path Pitfalls to Mention
1. Missing `.py` extension:
- `--py ./data/Paypal_data` fails.
- Use `--py ./data/Paypal_data.py`.

2. Loader contract mismatch:
- If `load()` or `df` is missing, run fails.

3. SQL table not found:
- Validate schema/table in SQL text and DB before running `--sql`.

## Output Paths to Show
- EDA: `./output_eda/EDA_Report.pdf`, `./output_eda/eda_results.json`
- EDA Spark: `./output_eda_spark/EDA_Report.pdf`, `./output_eda_spark/eda_results.json`
