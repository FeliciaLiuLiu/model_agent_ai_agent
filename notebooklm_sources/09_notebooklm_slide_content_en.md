# 09 Slide-by-Slide Content (Model Developer Focus)

## Slide 1 - Title
**How Model Developers Use EDA and EDA Spark**
- Scope: input types, system design, CLI/API usage, and output results.

## Slide 2 - Why This Matters for Model Development
- Standardized pre-modeling analysis reduces rework and hidden data risks.
- Output artifacts make results reproducible and reviewable.

## Slide 3 - Input Data Types You Can Use
- `--data` (single file, multiple files, directory, glob, named tables)
- `--sql` + `--db`
- `--py`
- `--py-code`
- `--nb`
- `--auto-exec` and `--compose-spec` / `--no-key-policy`

## Slide 4 - Input Mode Rule
- Exactly one input mode per run.
- If no explicit mode is given, loader resolves default/auto behavior.
- Multi-table inputs are composed by key rules.

## Slide 5 - EDA (Pandas) System Design
- CLI -> `EDA.run()` -> `DataLoader.load()` -> section functions -> output.
- Show component responsibilities and data flow.

## Slide 6 - EDA Spark (PySpark) System Design
- Driver orchestration + executor-side distributed computation.
- Same section semantics and output contract as Pandas.

## Slide 7 - Function Catalog in Both Designs
- Runner orchestration functions:
- `run()`, `run_interactive()`, `list_functions()`, `parse_function_selection()`, `parse_column_selection()`, `print_functions()`
- Section function keys:
- `data_quality`, `target`, `univariate`, `bivariate_target`, `feature_vs_feature`, `time_drift`

## Slide 8 - CLI Usage (EDA)
- Show an exact runnable command.
- Emphasize `--sections`, `--columns-*`, and output path.

## Slide 9 - CLI Usage (EDA Spark)
- Show an exact runnable command with `--spark-master`.
- Emphasize same function keys, different compute backend.

## Slide 10 - API Usage (EDA + EDA Spark)
- Show minimal API examples.
- Explain default return (`results`) and `return_payload=True`.

## Slide 11 - Output Results Contract
- File outputs:
- `eda_results.json`
- `EDA_Report.pdf`
- API payload structure:
- `results`
- `skipped_sections`
- `config`

## Slide 12 - How to Interpret Results as a Model Developer
- `data_quality` -> cleaning priorities.
- `target` -> imbalance/threshold strategy.
- `feature_vs_feature` -> redundancy and feature pruning.
- `time_drift` -> split and monitoring decisions.

## Slide 13 - Practical Input-to-Result Recipes
- Single file
- Multi-file
- SQL
- Python/Notebook loader
- Multi-table with compose spec

## Slide 14 - Actionable Next Step
- Pick one dataset.
- Run one EDA command.
- Read `eda_results.json` first.
- Convert findings into a concrete modeling checklist.
