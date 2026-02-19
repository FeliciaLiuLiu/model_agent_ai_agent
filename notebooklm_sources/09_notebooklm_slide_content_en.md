# 09 NotebookLM Slide Content (EN)

## Slide 1 - Title and Outcome
**Title:** How Model Developers Use `eda` and `eda_spark` in `model_agent_ai_agent`  
**Key points:**
- Goal: produce a repeatable pre-modeling EDA baseline in minutes.
- Scope: Pandas and Spark implementations with shared analysis semantics.
- Output: JSON + PDF artifacts for technical and business review.
**Model developer action:** adopt one EDA baseline command per project.

## Slide 2 - Why Standardized EDA Before Modeling
**Key points:**
- Manual profiling is inconsistent across notebooks and team members.
- Missingness, leakage risk, drift, and target imbalance are often found late.
- Standardized sections reduce rework in feature engineering and model debugging.
**Model developer action:** treat EDA outputs as a required gate before training.

## Slide 3 - Architecture at a Glance
**Key points:**
- `eda/cli.py -> eda/runner.py::EDA -> eda/dataloader.py`
- `eda_spark/cli.py -> eda_spark/runner.py::EDASpark -> eda_spark/dataloader.py`
- Shared intent, different execution backend (Pandas vs Spark).
**Model developer action:** pick the engine by data/runtime constraints, not by report format.

## Slide 4 - EDA vs EDA Spark Decision Matrix
**Key points:**
- Use `eda` when local iteration speed is the top priority.
- Use `eda_spark` when data volume or runtime environment requires distributed execution.
- Both produce the same section taxonomy and final artifacts.
**Suggested table columns:**
- Criteria
- `eda` (Pandas)
- `eda_spark` (PySpark)
- Recommended choice
**Model developer action:** define engine choice in your project kickoff checklist.

## Slide 5 - 5-Minute Pandas Quick Start
**Commands to show:**
```bash
python scripts/05_generate_synthetic_aml_200k_timeseries.py --out-dir ./data

python -m eda.cli \
  --data ./data/synthetic_aml_200k_20260130_135951.csv \
  --target-col is_suspicious \
  --output ./output_eda
```
**Key points:**
- Explicit `--data` keeps demo deterministic.
- `is_suspicious` is the target in this synthetic dataset.
- Outputs:
- `./output_eda/EDA_Report.pdf`
- `./output_eda/eda_results.json`
**Model developer action:** run this command as your local baseline template.

## Slide 6 - 5-Minute Spark Quick Start
**Commands to show:**
```bash
python scripts/07_generate_synthetic_aml_mixed_bank_fintech.py --out-dir ./data

python -m eda_spark.cli \
  --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv \
  --target-col sar_actual \
  --spark-master "local[*]" \
  --output ./output_eda_spark
```
**Key points:**
- `sar_actual` is the target in the mixed bank + fintech dataset.
- Spark mode keeps heavy aggregation distributed.
- Output files match Pandas naming convention.
**Model developer action:** keep one Spark command preset for large datasets.

## Slide 7 - What Each Section Means for Modeling
**Section list (exact):**
- `data_quality`
- `target`
- `univariate`
- `bivariate_target`
- `feature_vs_feature`
- `time_drift`
**Mapping idea:**
- `data_quality` -> missingness/duplicates cleanup priority.
- `target` -> imbalance strategy and threshold planning.
- `feature_vs_feature` -> multicollinearity and feature pruning.
- `time_drift` -> split strategy and monitoring expectations.
**Model developer action:** map each section to one modeling decision.

## Slide 8 - Non-Interactive vs Interactive Usage
**Commands to show:**
```bash
python -m eda.cli --list-functions
python -m eda_spark.cli --list-functions

python -m eda.cli --interactive --data ./data/synthetic_aml_200k_20260130_135951.csv --target-col is_suspicious --output ./output_eda
python -m eda_spark.cli --interactive --data ./data/synthetic_aml_mixed_50k_20260205_094055.csv --target-col sar_actual --spark-master "local[*]" --output ./output_eda_spark
```
**Key points:**
- Interactive mode is ideal for live exploration and scoped deep dives.
- Non-interactive mode is preferred for CI/reproducible runs.
**Model developer action:** use interactive for diagnosis, non-interactive for pipeline runs.

## Slide 9 - Input Modes and Multi-Table Composition
**Key points:**
- Exactly one mode per run: `data`, `sql`, `py`, `py_code`, `nb`.
- Supports named table bindings in `--data` (for join composition).
- `--compose-spec` allows explicit base/joins/checks.
- Default `--no-key-policy error`; use `aggregate_only` only when needed.
**Command example:**
```bash
python -m eda_spark.cli \
  --data transaction=./data/transaction.csv \
  --data customer=./data/customer.csv \
  --compose-spec ./data/compose_spec.json \
  --no-key-policy error \
  --output ./output_eda_spark
```
**Model developer action:** set join policy explicitly when working with multi-table inputs.

## Slide 10 - API Pattern for Pipeline Integration
**Code to show:**
```python
from adm_central_utility import EDA
from eda_spark.runner import EDASpark

eda_pd = EDA(output_dir="./output_eda", target_col="is_suspicious")
eda_pd.run(
    data=["./data/synthetic_aml_200k_20260130_135951.csv"],
    sections=["data_quality", "univariate", "time_drift"],
)

eda_sp = EDASpark(output_dir="./output_eda_spark", spark_master="local[*]", target_col="sar_actual")
eda_sp.run(
    data=["./data/synthetic_aml_mixed_50k_20260205_094055.csv"],
    sections=["data_quality", "target", "univariate"],
)
```
**Key points:**
- API mode is better for fixed workflows and version-controlled runs.
- Section scoping reduces runtime and report noise.
**Model developer action:** wrap EDA calls in your model training pre-check stage.

## Slide 11 - Artifact Consumption Pattern
**Key points:**
- `EDA_Report.pdf`: human-readable summary for analysts/reviewers.
- `eda_results.json`: machine-readable payload for automation or QC checks.
- Store both artifacts with model experiment metadata.
**Model developer action:** archive both files with every modeling experiment.

## Slide 12 - Common Failure Patterns and Fixes
**Common issues:**
- Wrong target/time column assumptions.
- Mixed named and unnamed `--data` inputs in the same run.
- Missing Spark runtime setup for `eda_spark`.
- Join inference failure in multi-table scenarios.
**Fixes:**
- Pass `--target-col`/`--time-col` explicitly.
- Use only named or only unnamed `--data` syntax.
- Validate `--spark-master` and Spark environment first.
- Provide explicit `--compose-spec` and review key mappings.
**Model developer action:** keep a one-page troubleshooting checklist in project docs.

## Slide 13 - Rollout Checklist and Next Steps
**Checklist:**
- Standard command template per project type (Pandas vs Spark).
- Required outputs committed to experiment artifacts.
- Section-to-decision mapping included in model review.
**Next steps:**
- Create a team EDA command catalog.
- Add EDA run to pre-training CI jobs.
- Define a minimum EDA review rubric for every new dataset.
**Model developer action:** implement one team-level EDA standard this sprint.
