# 08 Deck Story (Input + Design + Usage + Results)

## Audience
- Model developers who need a practical way to understand data before training.

## Single Narrative
- Start with **input flexibility**,
- explain **system design** for EDA and EDA Spark,
- show **exact CLI/API usage**,
- end with **how to read results and act**.

## What to Avoid
- Do not spend time on historical process or legacy workflows.
- Do not present generic EDA theory without usage context.

## Core Messages
- Message 1: Input is not a blocker.
- You can start from files, SQL, Python loaders, or notebooks.
- Message 2: Design is predictable.
- CLI/API always goes through runner + dataloader + section functions.
- Message 3: Usage is operational.
- Use CLI for fast runs and API for pipeline integration.
- Message 4: Results are the product.
- `eda_results.json` is the machine-readable source of truth.
- `EDA_Report.pdf` is the review and communication artifact.

## Function Coverage Requirement
The deck must explicitly show the function keys:
- `data_quality`
- `target`
- `univariate`
- `bivariate_target`
- `feature_vs_feature`
- `time_drift`

## Desired Closing
- The audience should leave with one immediate action:
- run one command on their own dataset and review `eda_results.json` first.
