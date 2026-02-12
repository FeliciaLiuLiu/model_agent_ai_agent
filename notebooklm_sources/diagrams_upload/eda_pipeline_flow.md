# eda pipeline flow

```mermaid
flowchart LR
  A[CLI/API\neda/cli.py::main\nEDA.run()] --> B[DataLoader\neda/dataloader.py::DataLoader.load]
  B --> C[Input resolution\nmode + source expansion]
  C --> D[Logical table map\n+ key-based composition]
  D --> E[Type inference\neda/utils.py::infer_column_types]
  E --> F[Auto target/time detection\npick_target_column + pick_time_column]
  F --> G[Section orchestration\nEDA._check_prerequisites]
  G --> H1[data_quality]
  G --> H2[target]
  G --> H3[univariate]
  G --> H4[bivariate_target]
  G --> H5[feature_vs_feature]
  G --> H6[time_drift]
  H1 --> I[Assemble payload]
  H2 --> I
  H3 --> I
  H4 --> I
  H5 --> I
  H6 --> I
  I --> J[Write JSON\noutput/eda_results.json]
  I --> K[Build PDF\noutput/EDA_Report.pdf]
```
