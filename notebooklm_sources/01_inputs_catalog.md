# 01 Inputs Catalog (Code-Verified)

## Scope
- EDA (Pandas): `eda/cli.py` -> `eda/runner.py::EDA` -> `eda/dataloader.py::DataLoader`
- EDA Spark (PySpark): `eda_spark/cli.py` -> `eda_spark/runner.py::EDASpark` -> `eda_spark/dataloader.py::DataLoader`

## Mode Selection Rules (Both Engines)
- Exactly one input mode is allowed: `data`, `sql`, `py`, `py_code`, `nb`.
- If multiple modes are set together, loader raises `ValueError("Only one input mode is allowed...")`.
- If no explicit input mode is provided:
- `auto_exec=True` -> mode `auto` (scan `./data`).
- `auto_exec=False` -> mode `data` (latest dataset from `./data`).
- CLI default behavior with no explicit input flags effectively runs auto mode via runner defaulting.

## Supported Data File Extensions (data mode)
- `.csv`, `.tsv`, `.parquet`, `.json`, `.xlsx`, `.xls`, `.feather`
- Pandas loader constant: `eda/dataloader.py::SUPPORTED_EXTS`
- Spark loader constant: `eda_spark/dataloader.py::SUPPORTED_EXTS`

## Input Types and How Developers Specify Them

| Input type | EDA | EDA Spark | How user specifies | Example under `./data` | Runtime behavior |
|---|---|---|---|---|---|
| Single data file | Yes | Yes | CLI: `--data`; API: `run(data=[...])` | `./data/transactions.csv` | Load one logical table; run EDA |
| Multiple data files (unnamed) | Yes | Yes | CLI repeatable `--data`; comma-separated also accepted | `--data ./data/p1.csv --data ./data/p2.csv` | Files are grouped by normalized stem into logical tables, then composed by keys |
| Directory | Yes | Yes | `--data ./data/shards` | `./data/shards/` | Expands supported files in directory |
| Glob | Yes | Yes | `--data './data/**/*.csv'` | `./data/**/*.parquet` | Glob expansion; use `--data-recursive` for recursive matching |
| Named table binding | Yes | Yes | `--data transaction=... --data customer=...` | `transaction=./data/transaction.csv` | Explicit logical table names, then compose by keys |
| SQL query string | Yes | Yes | `--sql 'SELECT ...' --db ...` | `--db sqlite:///./data/aml.db` | Query result loaded as table |
| SQL file path | Yes | Yes | `--sql ./data/query.sql --db ...` | `./data/query.sql` | SQL file content executed |
| Multiple SQL files | Yes | Yes | `--sql './data/sql/*.sql' --db ...` or comma list | `./data/sql/*.sql` | Each SQL source becomes table (name from file stem), then compose by keys |
| Named SQL map | Yes | Yes | `--sql '{"t1":"SELECT...","t2":"SELECT..."}' --db ...` | `--sql ./data/sql_map.json` | Build multiple tables, then compose |
| Python loader file | Yes | Yes | `--py ./data/loader.py` | `./data/eda_input_loader.py` | Must expose `load()` or `df`; can return DataFrame or table dict |
| Multiple Python loader files | Yes | Yes | `--py './data/loaders/*.py'` or comma list | `./data/loaders/*.py` | Execute each file, merge into table map, compose |
| Inline Python code | Yes | Yes | `--py-code '...python...'` | N/A | Must define `load()` or `df`; can return table dict |
| Notebook loader | Yes | Yes | `--nb ./data/loader.ipynb` | `./data/eda_input_loader.ipynb` | Execute code cells; require `load()` or `df` |
| Multiple notebooks | Yes | Yes | `--nb './data/loaders/*.ipynb'` or comma list | `./data/loaders/*.ipynb` | Execute each notebook, compose outputs |
| Auto scan from `./data` | Yes | Yes | No explicit mode flags, or `--auto-exec` | mixed `csv/sql/py/ipynb` in `./data` | Scans and loads/executed sources, then composes |

## Key-Based Multi-Table Composition Rules (Current Behavior)
- Multi-table composition is handled in:
- Pandas: `eda/dataloader.py::_compose_tables_pandas`
- Spark: `eda_spark/dataloader.py::_compose_tables_spark`
- Join mapping inference includes name similarity, type compatibility, uniqueness, and value overlap.
- Default behavior when joins cannot be established:
- `no_key_policy=error` (default in CLI/runner/dataloader).
- Optional fallback: `no_key_policy=aggregate_only`.

## How a Developer Chooses Specific Input Docs in `./data`
- Pick exact files:
- `--data ./data/a.csv --data ./data/b.csv`
- Pick by folder:
- `--data ./data/raw_inputs`
- Pick by glob:
- `--data './data/2026/**/*.parquet' --data-recursive`
- Pick and name tables explicitly:
- `--data transaction=./data/transaction.csv --data customer=./data/customer.csv`
- For SQL/Python/Notebook loaders:
- `--sql`, `--py`, `--nb` accept a single path, comma list, directory, or glob-compatible expression (loader resolves files internally).

## Input Resolution Decision Tree

```mermaid
flowchart TD
  A[Start CLI/API] --> B{Explicit input mode set?\n--data/--sql/--py/--py-code/--nb}
  B -- No --> C{auto_exec?}
  C -- Yes --> D[Mode=auto\nscan ./data for data/sql/py/ipynb]
  C -- No --> E[Mode=data\nload latest dataset from ./data]
  B -- Yes --> F{Exactly one mode?}
  F -- No --> X[Error: only one mode allowed]
  F -- Yes --> G{Mode selected}

  G -->|data| H[Resolve file/dir/glob/list\n(optional recursive)]
  H --> H2[Build logical table map\n(group by table alias/stem)]
  H2 --> H3[Compose tables by keys\n(default no_key_policy=error)]

  G -->|sql| I[Run SQL string/file/map\nrequires --db]
  I --> I2[Build table map if multi-source]
  I2 --> I3[Compose tables by keys]

  G -->|py/py_code/nb| J[Execute loaders\nread DataFrame or dict]
  J --> J2[Build table map if multi-source]
  J2 --> J3[Compose tables by keys]

  D --> K[Compose loaded sources by keys]
  E --> Z[Single table output]
  H3 --> Z
  I3 --> Z
  J3 --> Z
  K --> Z
  Z[DataFrame + source + composition metadata]
```
