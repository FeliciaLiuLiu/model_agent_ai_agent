# Model Testing Agent

The Model Testing Agent provides a comprehensive evaluation pipeline for classification models. It generates metrics, plots, and a PDF report that covers effectiveness, efficiency, stability, and interpretability.

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

## What It Does

- Effectiveness: ROC/PR curves, AUC, confusion matrix, precision/recall/F1, KS, Precision@K
- Efficiency: threshold analysis and FPR diagnostics
- Stability: PSI, data drift, concept drift, CV stability, bootstrap CI
- Interpretability: permutation importance, SHAP, LIME, PDP, ICE

## Inputs

- A scikit-learn compatible model or pipeline saved via `joblib`
- A labeled testing dataset provided through one of the following input modes:
  - file input: CSV / Parquet / Excel
  - SQL input: query text or `.sql` file plus connection string
  - Python loader input: `.py` file that returns a DataFrame or `(X, y, feature_names)`
- Optional: section and column selection
- Optional: segmentation config to run the full testing pipeline by segment or time window

## Outputs

- `model_testing_agent_Model_Testing_Report.pdf`
- `results.json`
- `.png` plots saved under the output directory

## API Usage

### Non-Interactive (Full Report)

```python
from adm_central_utility.model_testing_agent import ModelTestingAgent

model = ModelTestingAgent.load_model("./path/to/your_model.joblib")
X, y, feature_names = ModelTestingAgent.load_data(
    "./path/to/your_dataset.csv",
    label_col="your_label",
)

agent = ModelTestingAgent(output_dir="./output")
results = agent.run(model=model, X=X, y=y, feature_names=feature_names)
agent.generate_report(results)
```

### Load Data from SQL

```python
from adm_central_utility.model_testing_agent import ModelTestingAgent

model = ModelTestingAgent.load_model("./path/to/your_model.joblib")
X, y, feature_names = ModelTestingAgent.load_data(
    sql="./queries/model_testing_query.sql",  # or raw SQL text
    conn="sqlite:///./data/demo.db",          # or another SQLAlchemy connection string
    label_col="your_label",
)
```

### Load Data from a Python Loader

```python
from adm_central_utility.model_testing_agent import ModelTestingAgent

model = ModelTestingAgent.load_model("./path/to/your_model.joblib")
X, y, feature_names = ModelTestingAgent.load_data(
    loader_py="./loaders/custom_testing_input.py",
    loader_fn="load_data",  # optional, defaults to load_data
    label_col="your_label",
)
```

### Interactive Mode (Pick Matrices and Columns)

```python
from adm_central_utility.model_testing_agent import InteractiveAgent

agent = InteractiveAgent(output_dir="./output")
agent.run_interactive(model=model, X=X, y=y, feature_names=list(X.columns))
```

### Choose Sections and Columns Programmatically

```python
results = agent.run(
    model=model,
    X=X,
    y=y,
    feature_names=feature_names,
    sections=["effectiveness", "stability"],
    columns=["col_a", "col_b"],
    section_columns={
        "stability": ["col_a", "col_c"],
        "interpretability": ["col_a", "col_b"],
    },
)
```

### Run the Full Pipeline by Segment or Time Window

```python
results = agent.run(
    model=model,
    X=X,
    y=y,
    feature_names=feature_names,
    columns=["score_feature", "amount", "balance"],
    segmentation={
        "column": "event_time",
        "include_overall": True,
        "min_rows": 100,
        "segments": [
            {"name": "jan_2024", "start": "2024-01-01", "end": "2024-02-01"},
            {"name": "feb_2024", "start": "2024-02-01", "end": "2024-03-01"},
        ],
    },
)

agent.generate_report(results, filename="segmented_model_testing_report.pdf")
agent.save_results(results, filename="segmented_results.json")
```

For value-based segmentation, use `values` instead of `start` / `end`:

```python
segmentation = {
    "column": "customer_segment",
    "segments": [
        {"name": "retail", "values": ["retail"]},
        {"name": "commercial", "values": ["commercial"]},
    ],
}
```

## CLI Usage

### Non-Interactive

```bash
model-testing-agent \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output
```

### SQL Input

```bash
model-testing-agent \
  --model ./path/to/your_model.joblib \
  --sql ./queries/model_testing_query.sql \
  --conn sqlite:///./data/demo.db \
  --label_col your_label \
  --output ./output
```

### Python Loader Input

```bash
model-testing-agent \
  --model ./path/to/your_model.joblib \
  --loader-py ./loaders/custom_testing_input.py \
  --loader-fn load_data \
  --label_col your_label \
  --output ./output
```

### Interactive

```bash
model-testing-agent \
  --model ./path/to/your_model.joblib \
  --data ./path/to/your_dataset.csv \
  --label_col your_label \
  --output ./output \
  --interactive
```

### Segmented Execution

Use the shared template at [examples/model_testing_segmentation.json](/Users/felicia/Desktop/Felicia/DB_work/adm_central_utility/model_agent_ai_agent/examples/model_testing_segmentation.json) and edit it for your own grouping or time-window logic.

```bash
model-testing-agent \
  --model ./models/bank_aml_gbt.joblib \
  --sql ./queries/model_testing_query.sql \
  --conn sqlite:///./data/demo.db \
  --label_col is_suspicious \
  --segmentation ./examples/model_testing_segmentation.json \
  --output ./output
```

## Example (Using Scripts 03/04)

```bash
python scripts/03_generate_bank_aml_dataset.py \
  --out-dir ./data \
  --rows 200000 \
  --suspicious-rate 0.04 \
  --label-noise 0.02 \
  --seed 7
```

```bash
python scripts/04_train_bank_aml_gbt_pipeline.py \
  --data ./data/synthetic_bank_aml_200k.csv \
  --model ./models/bank_aml_gbt.joblib \
  --label-col is_suspicious \
  --test-size 0.3 \
  --seed 42
```

```bash
model-testing-agent \
  --model ./models/bank_aml_gbt.joblib \
  --data ./data/synthetic_bank_aml_200k_test.csv \
  --label_col is_suspicious \
  --output ./output
```

## Notes

- The model should support `predict_proba` for full metrics and interpretability.
- If only `decision_function` exists, AUC metrics still work but LIME may be limited.
- The feature schema must match what the model expects.
- Segmentation requires a named column in the dataset and runs the full pipeline once per segment.
- Segmented runs write plot artifacts into per-segment subdirectories to avoid overwriting outputs.
