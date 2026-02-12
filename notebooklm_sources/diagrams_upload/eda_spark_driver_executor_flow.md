# eda spark driver executor flow

```mermaid
flowchart LR
  subgraph D[Driver]
    A[CLI/API\neda_spark/cli.py::main\nEDASpark.run()] --> B[Create SparkSession\nEDASpark.__init__]
    B --> C[DataLoader.load\nmode + input resolution]
    C --> D1[Logical table map\n+ key-based composition]
    D1 --> E[Infer column types\neda_spark/utils.py::infer_column_types]
    E --> F[Run sections\n_check_prerequisites + _section_*]
    F --> G[Assemble payload]
    G --> H[Write JSON]
    G --> I[Build PDF]
  end

  subgraph X[Executors]
    K[Distributed computations\ncount/groupBy/approxQuantile\ncorrelation/drift prep]
  end

  F --> K
  K --> F
  F --> L[toPandas for plotting]
```
