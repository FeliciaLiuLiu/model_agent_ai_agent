# 00 Overview for NotebookLM Deck (EDA + EDA Spark)

## Deck Objective
Create a fully English slide deck for model developers that focuses on:
1. Full system design for `eda` and `eda_spark` from **Input -> Engine -> Output**.
2. Input data flexibility based on real dataloader behavior.
3. Input weaknesses and failure conditions.
4. Section blocks and what functions/analyses each block executes.
5. How users run both projects via **CLI** and **API**, in **interactive** and **non-interactive** modes.
6. How to use the EDA results to drive **Data Cleaning** and **Feature Engineering**.

## Hard Focus Areas (Must Be Visible in Slides)
- Input flexibility and real constraints.
- End-to-end design visibility for both engines.
- Usage mode clarity (CLI/API + interactive/non-interactive).
- Output-first mindset: `eda_results.json` and `EDA_Report.pdf`.

## Required Example Anchors
- EDA Spark must include this command pattern with row cap:
- `python -m eda_spark.cli --py ./data/Paypal_data.py --sections data_quality,univariate,feature_vs_feature,time_drift --max-rows 5000 --output ./output_eda_spark`
- EDA must use `./data/aml_synthetic_20k.sql` as the demo source (materialize SQL script -> query DB).

## Required Weakness Statements
- Exactly one input mode is allowed per run (`data` or `sql` or `py` or `py_code` or `nb`).
- Multi-table row-level composition requires joinable key columns; otherwise default behavior is fail-fast error.
- Fallback mode `no_key_policy=aggregate_only` exists, but it loses row-level linkage.

## Output Contract (Both Engines)
- `<output_dir>/eda_results.json`
- `<output_dir>/EDA_Report.pdf`

## Slide Style Constraint
- White background.
- Clean, simple style.
- Minimal decorative elements.

## Suggested NotebookLM Upload Order
1. `00_overview.md`
2. `01_inputs_catalog.md`
3. `02_eda_pipeline.md`
4. `03_eda_spark_pipeline.md`
5. `05_usage_modes.md`
6. `06_demo_paths_in_data_folder.md`
7. `08_notebooklm_deck_story_en.md`
8. `09_notebooklm_slide_content_en.md`
9. `10_notebooklm_demo_and_troubleshooting_en.md`
