# 06 Input-to-Result Recipes (Model Developer Playbook)

## Recipe 1: Single File Input

### EDA
```bash
python -m eda.cli --data ./data/transactions.csv --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli --data ./data/transactions.csv --spark-master "local[*]" --output ./output_eda_spark
```

Expected result files:
- `eda_results.json`
- `EDA_Report.pdf`

## Recipe 2: Multiple Files as Input

### EDA
```bash
python -m eda.cli --data ./data/part1.csv --data ./data/part2.csv --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli --data ./data/part1.csv --data ./data/part2.csv --spark-master "local[*]" --output ./output_eda_spark
```

## Recipe 3: Directory and Glob Input

### EDA
```bash
python -m eda.cli --data ./data/shards --output ./output_eda
python -m eda.cli --data './data/shards/**/*.csv' --data-recursive --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli --data ./data/shards --spark-master "local[*]" --output ./output_eda_spark
python -m eda_spark.cli --data './data/shards/**/*.csv' --data-recursive --spark-master "local[*]" --output ./output_eda_spark
```

## Recipe 4: Named Multi-Table Input

### EDA
```bash
python -m eda.cli \
  --data transaction=./data/transaction.csv \
  --data customer=./data/customer.csv \
  --data account=./data/account.csv \
  --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli \
  --data transaction=./data/transaction.csv \
  --data customer=./data/customer.csv \
  --data account=./data/account.csv \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

## Recipe 5: Multi-Table with Explicit Join Spec

### EDA
```bash
python -m eda.cli \
  --data transaction=./data/transaction.csv \
  --data customer=./data/customer.csv \
  --compose-spec ./data/compose_spec.json \
  --no-key-policy error \
  --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli \
  --data transaction=./data/transaction.csv \
  --data customer=./data/customer.csv \
  --compose-spec ./data/compose_spec.json \
  --no-key-policy error \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

## Recipe 6: SQL Input

### EDA
```bash
python -m eda.cli --sql "SELECT * FROM aml_dataset" --db "sqlite:///./data/aml.db" --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli --sql "SELECT * FROM aml_dataset" --db "sqlite:///./data/aml.db" --spark-master "local[*]" --output ./output_eda_spark
```

## Recipe 7: Python and Notebook Loader Inputs

### EDA
```bash
python -m eda.cli --py ./data/eda_input_loader.py --output ./output_eda
python -m eda.cli --nb ./data/eda_input_loader.ipynb --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli --py ./data/eda_spark_input_loader.py --spark-master "local[*]" --output ./output_eda_spark
python -m eda_spark.cli --nb ./data/eda_spark_input_loader.ipynb --spark-master "local[*]" --output ./output_eda_spark
```

## Recipe 8: Function-Scoped Runs (Result-Focused)

### EDA
```bash
python -m eda.cli \
  --data ./data/transactions.csv \
  --sections data_quality,target,time_drift \
  --output ./output_eda
```

### EDA Spark
```bash
python -m eda_spark.cli \
  --data ./data/transactions.parquet \
  --sections data_quality,target,time_drift \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

## Result-Reading Checklist
- Confirm files exist in output directory.
- Open `eda_results.json` first for section-level machine-readable results.
- Open `EDA_Report.pdf` second for review and communication.
- Translate section findings into modeling tasks.
