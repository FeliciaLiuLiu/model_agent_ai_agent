# 08 NotebookLM Deck Story (EN)

## Presentation Positioning
- Audience: model developers with Python/SQL/Spark fundamentals.
- Scenario: first-pass dataset understanding before feature engineering and model training.
- Time budget: 15-20 minutes, action-oriented technical walkthrough.

## Core Promise
- Replace ad-hoc notebook profiling with a reusable EDA workflow that has:
- shared section semantics,
- repeatable CLI/API execution,
- standardized outputs for review (`eda_results.json`, `EDA_Report.pdf`).

## Story Arc (Recommended)
- Act 1: Problem and value
- Teams repeatedly rebuild exploratory checks in different notebooks.
- Result: low comparability, inconsistent review quality, slow handoff.
- Act 2: Engine selection
- `eda` for fast local/smaller workflows.
- `eda_spark` for large/distributed or CML Spark environments.
- Act 3: Fast execution path
- One-shot CLI run, section-focused run, and interactive run.
- API integration for non-interactive pipelines.
- Act 4: Team operationalization
- Command templates, artifact conventions, and troubleshooting playbook.

## Accuracy Guardrails for NotebookLM
- Use only these sections:
- `data_quality,target,univariate,bivariate_target,feature_vs_feature,time_drift`
- Keep CLI parameters exact:
- `--data --sql --db --py --py-code --nb --data-recursive --compose-spec --no-key-policy --auto-exec --sections --columns --interactive --list-functions`
- Spark-only options:
- `--spark-master --spark-app-name --spark-conf`
- Artifact paths:
- `<output_dir>/eda_results.json`
- `<output_dir>/EDA_Report.pdf`
- Multi-table policy default:
- `no_key_policy=error` (fail fast)
- optional fallback: `aggregate_only`

## Demo Assets Already in Repo
- Pandas target column: `is_suspicious`
- Spark target column: `sar_actual`
- Recommended files:
- `./data/synthetic_aml_200k_20260130_135951.csv`
- `./data/synthetic_aml_mixed_50k_20260205_094055.csv`

## Required Slide Tone
- English only.
- Technical and concise.
- Every slide should include one practical action for model developers.
