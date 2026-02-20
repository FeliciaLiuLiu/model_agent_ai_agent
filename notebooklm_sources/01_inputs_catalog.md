# 01 Input Data Flexibility and Constraints

## Input Modes Supported by Both Projects
Both `eda` and `eda_spark` support these modes:
- `--data`
- `--sql` with `--db`
- `--py`
- `--py-code`
- `--nb`

Only one mode can be active in a single run.

## File Types Supported in `--data`
- `.csv`, `.tsv`, `.parquet`, `.json`, `.xlsx`, `.xls`, `.feather`

## Flexibility Matrix

| Input pattern | EDA (Pandas) | EDA Spark (PySpark) | Typical usage |
|---|---|---|---|
| Single file | Yes | Yes | `--data ./data/file.csv` |
| Multiple files | Yes | Yes | `--data ./data/a.csv --data ./data/b.csv` |
| Directory | Yes | Yes | `--data ./data/folder` |
| Glob + recursive | Yes | Yes | `--data './data/**/*.csv' --data-recursive` |
| Named table bindings | Yes | Yes | `--data trans=... --data cust=...` |
| SQL query | Yes | Yes | `--sql "SELECT ..." --db "sqlite:///..."` |
| Python loader file | Yes | Yes | `--py ./data/loader.py` |
| Inline Python | Yes | Yes | `--py-code "..."` |
| Notebook loader | Yes | Yes | `--nb ./data/loader.ipynb` |

## Weaknesses and Risks to Show Clearly
1. Single-mode limitation:
- If multiple modes are provided together, run fails.
- Typical error: `Only one input mode is allowed.`

2. Multi-table composition dependency:
- Row-level merge needs usable join keys.
- If join inference fails and policy is `error`, run fails.
- Typical error: missing join keys / unable to compose all tables at row level.

3. SQL mode dependency:
- `--sql` requires `--db` in direct SQL mode.
- SQL scripts containing DDL + inserts are better materialized first into a DB, then queried.

4. Python/Notebook loader contract:
- Loader must define `load()` or `df`.
- Returned object must be DataFrame-compatible (pandas or Spark depending on engine conversion path).

## Mitigations
- Keep one input mode per run.
- Use `--compose-spec` for explicit joins when multiple tables exist.
- Keep `no_key_policy=error` for strict quality; switch to `aggregate_only` only when row-level merge is impossible and acceptable.
- For SQL script files, materialize to SQLite first and use `--sql "SELECT ..." --db ...`.
