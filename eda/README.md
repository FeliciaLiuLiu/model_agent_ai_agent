# EDA Pipeline

This module generates a professional PDF EDA report using ReportLab (Platypus). It supports both Pandas and Spark implementations, auto-detects the latest dataset in `./data`, and provides interactive or non-interactive modes.

## Installation (dbConda + IDE)

### Step 1 - Create a dbConda environment

1) Open a dbConda terminal.
2) Create a virtual environment:

```bash
conda create --name <env_name> python=<version>
# Example
conda create --name adm_central_utility python=3.11
```

3) Activate the environment:

```bash
conda activate adm_central_utility
```

4) If activation succeeds, the prompt should show:

```
(adm_central_utility) C:\>
```

5) Install pip inside the environment:

```bash
conda install pip
```

### Step 2 - Configure the IDE interpreter (example: PyCharm)

1) In the dbConda terminal, activate the environment and run:

```bash
where python
```

Note the full path to `python.exe` for this environment.

2) Open your IDE (e.g., PyCharm), then go to:

```
Settings | Project: <Path to Project location> | Project Interpreter
```

3) Click the gear icon and choose **Add Interpreter**.
4) Select **Conda Environment** and set the **Conda Executable** path so the IDE can load dbConda. A common location is:

```
C:\Users\<username>\dev\dbConda-2023_09-py311-r36\Scripts\conda.exe
```

Alternatively:

```
C:\Users\<username>\dev\dbConda-2023_09-py311-r36\condabin\conda.bat
```

5) Click **Load Environments** to read the dbConda environment list.
6) Choose **Existing environment** and select your environment (e.g., `adm_central_utility`).
7) If it does not appear, set the Interpreter path manually, for example:

```
C:\Users\<username>\dev\dbConda-2023_09-py311-r36\envs\adm_central_utility\python.exe
```

8) Apply the interpreter settings to the current project.

### Step 3 - Activate the environment in the IDE terminal

1) Open the IDE terminal.
2) If conda is already initialized, activate the environment:

```bash
conda activate adm_central_utility
```

3) If the terminal cannot find `conda`, run this once in system PowerShell:

```powershell
C:\Users\<username>\dev\dbConda-2023_09-py311-r36\condabin\conda.bat init powershell
```

Then reopen the IDE terminal and activate the environment again:

```bash
conda activate adm_central_utility
```

### Step 4 - Install project dependencies

After the environment and interpreter are set up, confirm the terminal prefix is the active environment, then:

1) In the IDE terminal, `cd` to the project root (e.g., `adm_central_utility`).
2) Install required packages:

```bash
pip install -r requirements.txt
```

3) Optional: if `pip install` fails, try one of the following in the **IDE terminal**:

```bat
for /f %i in (requirements.txt) do conda install %i -y
```

```bat
conda install -y -c conda-forge ^
  numpy pandas scikit-learn matplotlib reportlab joblib scipy shap numba llvmlite lime pytest
```

## Quick Start (Pandas)

```bash
# 1) Optional: generate a synthetic AML dataset (example only)
python scripts/05_generate_synthetic_aml_200k_timeseries.py --out-dir ./data
```

`05_generate_synthetic_aml_200k_timeseries.py` is only an example generator. You can use other data generators
or place your dataset directly under `./data`.

```bash
# 2) Run EDA (auto-detect latest dataset)
python -m eda.cli --output ./output_eda
```

Specify a dataset explicitly:

```bash
python -m eda.cli --data ./data/your_dataset.csv --output ./output_eda
```

The PDF will be saved to:

```
./output_eda/EDA_Report.pdf
```

Target column notes:
- The target column can be any label you want EDA to analyze (binary or numeric).
- If not provided, EDA will attempt to auto-detect a likely target by name and binary values.
- If auto-detection fails, target-dependent sections are skipped.
- For the mixed bank + fintech dataset from `scripts/07_generate_synthetic_aml_mixed_bank_fintech.py`, the target is `sar_actual`.

## Auto-Detection

- If no input is provided, the pipeline scans `./data` and auto-loads any supported files.
- Supported file types: `.csv`, `.tsv`, `.parquet`, `.json`, `.xlsx`, `.xls`, `.feather` (Excel requires `openpyxl`).
- You can override the auto-detected path with `EDA_DATA_PATH`.

## Unified Data Loader (Pandas)

EDA accepts exactly one input mode and loads everything into a pandas DataFrame:

1) Files/dirs/globs: `data=[...]` or CLI `--data` (repeatable). Multiple files are concatenated row-wise.
2) SQL: `sql` + `db` (SQLite URL or SQLAlchemy connection string; non-sqlite requires `sqlalchemy`).
3) Python file: `py` that defines `load()` or `df`.
4) Inline Python: `py_code` that defines `load()` or `df`.
5) Notebook: `nb` (`.ipynb`) with `load()` or `df` in code cells.

Note: Python/Notebook modes execute code. Use trusted inputs only.

### Auto-Exec (no input flags)

If you run `python -m eda.cli --output ./output_eda` with no input flags, EDA will auto-scan `./data` and:

- Load any supported data files and concatenate them.
- Execute any `.py` / `.ipynb` files that define `df` or `load()` and concatenate them.
- Execute any `.sql` files:
  - If a `.db`/`.sqlite` file exists and the SQL is a single `SELECT`/`WITH` query, it runs against that DB.
  - Otherwise, the SQL is executed in a temporary SQLite DB and must create a single table/view (or name it `aml_dataset`, `eda_dataset`, or `eda_input`).

Use `--auto-exec` to explicitly enable this behavior when desired.

## API Usage (Pandas)

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
results = eda.run(data=["./path/to/your_dataset.csv"])
```

Multiple files:

```python
eda.run(data=["./data/part1.csv", "./data/part2.csv"])
```

SQL:

```python
eda.run(sql="SELECT * FROM transactions", db="sqlite:///./data/demo.db")
```

Python file / notebook:

```python
eda.run(py="./load_data.py")
eda.run(nb="./load_data.ipynb")
```

Inline Python:

```python
eda.run(py_code="import pandas as pd\n\ndf = pd.read_csv('./data/demo.csv')")
```

Interactive mode:

```python
from adm_central_utility import EDA

eda = EDA(output_dir="./output_eda", target_col="your_target")
eda.run_interactive(data=["./path/to/your_dataset.csv"])
```

## API Usage (Spark)

```python
from adm_central_utility import EDASpark

eda = EDASpark(output_dir="./output_eda_spark", target_col="your_target")
eda.run(file_path="./path/to/your_dataset.csv")
```

> Spark mode requires `pyspark` to be installed in the runtime.

## CLI Usage

### One-Line (Auto-Exec from ./data)

```bash
python -m eda.cli --output ./output_eda
```

### Specify File(s)

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --target-col your_target
python -m eda.cli --data ./data/part1.csv --data ./data/part2.csv --output ./output_eda
```

Directory or glob:

```bash
python -m eda.cli --data ./data --output ./output_eda
python -m eda.cli --data "./data/*.csv" --output ./output_eda
python -m eda.cli --data ./data --data-recursive --output ./output_eda
```

### SQL Input

```bash
python -m eda.cli --sql "SELECT * FROM transactions" --db "sqlite:///./data/demo.db" --output ./output_eda
```

### Python / Notebook Input

```bash
python -m eda.cli --py ./load_data.py --output ./output_eda
python -m eda.cli --nb ./load_data.ipynb --output ./output_eda
```

Example loaders in this repo:
- `data/eda_input_loader.py`
- `data/eda_input_loader.ipynb`

General usage:
1. Copy an example loader to your own file, for example `data/eda_input_loader.py` to `data/user.py`.
2. Edit `load()` (or define a `df`) in your file.
3. Run with `--py data/user.py` or `--nb data/user.ipynb`.

## Run EDA Tests

From the repo root:
```bash
python -m pytest tests/eda
```

Run a single file:
```bash
python -m pytest tests/eda/test_report_payload_shapes.py
```

If you see import errors, install the package in editable mode first:
```bash
pip install -e .
python -m pytest tests/eda
```

### Interactive (Pandas)

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda --interactive
```

### Spark Mode

```bash
python -m eda.cli --data ./path/to/your_dataset.csv --output ./output_eda_spark --spark
```

## Select Sections and Columns

```bash
python -m eda.cli \
  --data ./path/to/your_dataset.csv \
  --sections data_quality,univariate,time_drift \
  --columns-univariate txn_amount,channel,origin_country \
  --output ./output_eda
```

## Output Files

- `EDA_Report.pdf`
- `eda_results.json`
- Plot images under the output directory

## Notes

- Time-series and drift sections run only if a time column is detected and parse success is >= 90%.
- Column type classification is shown as a table (numeric, categorical, datetime, boolean, text).
- Tables are styled with headers, zebra striping, and numeric alignment.
- Missingness table includes only columns with missing values; zero-missing columns are summarized in text.
- Null-like placeholder values (e.g., "NA", "N/A", "NULL", empty strings, "UNKNOWN") are detected and reported.
