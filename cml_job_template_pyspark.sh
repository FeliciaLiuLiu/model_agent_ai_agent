#!/usr/bin/env bash
set -euo pipefail

# CML Spark Job Template for model_testing_agent_pyspark
# Usage example (edit paths):
#   bash cml_job_template_pyspark.sh \
#     /path/to/model.joblib \
#     /path/to/data.csv \
#     is_suspicious \
#     /path/to/output

MODEL_PATH=${1:-"/path/to/model.joblib"}
DATA_PATH=${2:-"/path/to/data.csv"}
LABEL_COL=${3:-"label"}
OUTPUT_DIR=${4:-"/path/to/output"}

# Spark temp locations (must be writable in CML)
export SPARK_LOCAL_DIRS=${SPARK_LOCAL_DIRS:-/tmp/spark}
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-"-Djava.io.tmpdir=/tmp/spark"}

# Matplotlib cache
export MPLCONFIGDIR=${MPLCONFIGDIR:-./.mpl_cache}
export MPLBACKEND=${MPLBACKEND:-Agg}

mkdir -p "$SPARK_LOCAL_DIRS" "$OUTPUT_DIR" "$MPLCONFIGDIR"

# Optional: install joblib if your CML image does not include it
# python -m pip install joblib

# Run with spark-submit (recommended in CML Spark jobs)
spark-submit \
  --master yarn \
  -m model_testing_agent_pyspark.runner.cli \
  --model "$MODEL_PATH" \
  --data "$DATA_PATH" \
  --label_col "$LABEL_COL" \
  --output "$OUTPUT_DIR"
