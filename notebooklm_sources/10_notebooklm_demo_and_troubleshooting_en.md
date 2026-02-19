# 10 NotebookLM Generation Prompt + Validation Checklist

## A. Prompt for NotebookLM Slide Deck (Use As-Is)

```text
Create a 12-14 slide English technical deck for model developers about the EDA (Pandas) and EDA Spark (PySpark) projects in this repository.

Hard requirements:
1) Focus only on three themes:
   - input data types and input modes,
   - system design of EDA and EDA Spark,
   - concrete CLI/API usage and output results interpretation.
2) Do not include legacy workflow history.
3) Show exact function keys used by both designs:
   data_quality, target, univariate, bivariate_target, feature_vs_feature, time_drift.
4) Show runner-level functions:
   run(), run_interactive(), list_functions(), parse_function_selection(), parse_column_selection(), print_functions().
5) Include exact runnable command examples for both EDA and EDA Spark.
6) Include API examples for both EDA and EDA Spark.
7) Emphasize outputs and results:
   - <output_dir>/eda_results.json
   - <output_dir>/EDA_Report.pdf
   - API default return (results) and return_payload=True payload structure.
8) Include one slide mapping section results to model-development decisions.
9) Keep all parameter names and paths exactly as in sources.
10) End with an immediate action checklist for model developers.
```

## B. Command Snippets NotebookLM Should Prefer

### EDA CLI
```bash
python -m eda.cli \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --target-col is_suspicious \
  --sections data_quality,target,univariate,time_drift \
  --output ./output_eda
```

### EDA Spark CLI
```bash
python -m eda_spark.cli \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --target-col sar_actual \
  --sections data_quality,target,univariate,time_drift \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```

### EDA API
```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="is_suspicious")
results = eda.run(data=["./data/synthetic_aml_200k_20260130_135951.csv"])
```

### EDA Spark API
```python
from eda_spark.runner import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", spark_master="local[*]", target_col="sar_actual")
results = eda.run(data=["./data/synthetic_aml_mixed_50k_20260205_094055.csv"])
```

## C. Validation Checklist for Generated Slides
- Input types are explicitly listed.
- Both system designs are shown as architecture/data-flow.
- CLI and API usage are both covered for each engine.
- Function keys and runner functions are present and spelled correctly.
- Output artifacts (`eda_results.json`, `EDA_Report.pdf`) are emphasized.
- Results interpretation for model-development actions is explicit.
