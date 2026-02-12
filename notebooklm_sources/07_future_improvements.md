# 07 Future Improvements (Slide-Ready)

## 1) Input Adapters
- Add Delta Lake adapter for transaction-scale lakehouse workflows.
- Add Hive table adapter for metastore-native ingestion.
- Expand JDBC presets and secure credential injection patterns.
- Add schema registry / contract adapter (e.g., schema JSON/YAML validation before run).
- Add first-class cloud object storage URI profiles (S3/GCS/ABFS auth helpers).

## 2) Performance and Scalability
- Introduce adaptive sampling profiles by section and dataset size.
- Add approximate statistics options (quantiles/distincts) tunable per section.
- Add cache/persist planning in Spark runner for repeated section scans.
- Add execution telemetry (stage timings + IO footprint) in JSON payload.
- Add parallel plot rendering for large univariate chart sets.

## 3) Extensibility and Plugin Architecture
- Add plugin API for custom sections (`register_section(...)`).
- Add pluggable key-inference strategies per domain (banking, retail, telecom).
- Add custom report template hooks (corporate branding + compliance sections).
- Add rule packs for organization-specific data quality checks.

## 4) Data Quality and Drift Depth
- Add cross-table referential integrity diagnostics in output payload.
- Add robust time-series drift decomposition (seasonality, trend shift, regime changes).
- Add label leakage checks and pre-model warning heuristics.
- Add richer null-like and semantic anomaly detection by domain dictionary.

## 5) Production Readiness
- Add structured logging + trace IDs for every run.
- Add lineage metadata in JSON (source paths, compose joins, inferred keys, versions).
- Add deterministic run manifest (`run_id`, git SHA, config hash).
- Add quality gates for CI/CD (fail thresholds for missingness/drift/invalid values).
