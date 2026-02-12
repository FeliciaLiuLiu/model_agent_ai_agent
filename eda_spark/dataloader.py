"""Unified Spark DataFrame loader for EDA Spark inputs."""
from __future__ import annotations

import glob
import importlib.util
import json
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .utils import detect_latest_dataset, to_local_file_uri


SUPPORTED_EXTS = {".csv", ".tsv", ".parquet", ".json", ".xlsx", ".xls", ".feather"}
AUTO_DB_EXTS = {".db", ".sqlite", ".sqlite3"}
PREFERRED_SQL_TABLES = ("aml_dataset", "eda_dataset", "eda_input")
KEY_TOKEN_ALIASES = {
    "cust": "customer",
    "customer": "customer",
    "acct": "account",
    "acc": "account",
    "account": "account",
    "txn": "transaction",
    "trans": "transaction",
    "tran": "transaction",
    "client": "customer",
    "member": "customer",
    "usr": "user",
    "uid": "user",
    "no": "number",
    "num": "number",
    "nbr": "number",
    "identifier": "id",
    "key": "id",
}
KEY_NAME_HINTS = {
    "id",
    "account",
    "customer",
    "transaction",
    "user",
    "client",
    "member",
    "number",
    "code",
    "key",
}
TABLE_SUFFIX_RE = re.compile(r"(?:[_-]?(?:part|chunk|shard|split|batch|seg))?[_-]?\d+$", re.IGNORECASE)


def _is_glob(path: str) -> bool:
    return any(ch in path for ch in ["*", "?", "["])


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _looks_like_alias_binding(token: str) -> bool:
    if "=" not in token:
        return False
    left, _ = token.split("=", 1)
    left = left.strip()
    if not left:
        return False
    if "/" in left or "\\" in left or left.startswith("."):
        return False
    return all(ch.isalnum() or ch in {"_", "-"} for ch in left)


def _parse_data_bindings(data: Sequence[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    named: Dict[str, List[str]] = {}
    unnamed: List[str] = []
    for raw in data:
        token = str(raw).strip()
        if not token:
            continue
        if _looks_like_alias_binding(token):
            alias, path_expr = token.split("=", 1)
            alias = alias.strip()
            path_expr = path_expr.strip()
            if not path_expr:
                continue
            named.setdefault(alias, []).append(path_expr)
        else:
            unnamed.append(token)
    return named, unnamed


def _load_compose_spec(spec: Optional[Any]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return dict(spec)

    raw = str(spec).strip()
    if not raw:
        return None

    payload = raw
    path = Path(raw)
    if path.exists() and path.is_file():
        payload = path.read_text(encoding="utf-8")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("compose_spec must be a JSON object string or a JSON file path.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("compose_spec must be a JSON object.")
    return parsed


def _parse_named_sql(sql: Optional[str]) -> Optional[Dict[str, str]]:
    if not sql:
        return None
    raw = sql.strip()
    if not raw:
        return None

    payload = raw
    path = Path(raw)
    if path.exists() and path.is_file() and path.suffix.lower() == ".json":
        payload = path.read_text(encoding="utf-8")

    if not payload.lstrip().startswith("{"):
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        raise ValueError("Named SQL payload must be a JSON object: {table_name: sql_query}.")

    out: Dict[str, str] = {}
    for key, value in parsed.items():
        table_name = str(key).strip()
        query = str(value).strip()
        if not table_name or not query:
            raise ValueError("Named SQL payload contains empty table name or query.")
        out[table_name] = query
    return out or None


def _normalize_keys(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        keys = [value]
    elif isinstance(value, (list, tuple)):
        keys = [str(v) for v in value]
    else:
        raise ValueError(f"{field_name} must be a string or list of strings.")
    return [k.strip() for k in keys if str(k).strip()]


def _normalize_no_key_policy(policy: Optional[str]) -> str:
    value = (policy or "error").strip().lower()
    if value not in {"aggregate_only", "error"}:
        raise ValueError("no_key_policy must be one of: aggregate_only, error")
    return value


def _normalize_table_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower()).strip("_")
    if not normalized:
        return "table"
    normalized = TABLE_SUFFIX_RE.sub("", normalized).strip("_")
    return normalized or "table"


def _table_name_from_path(path: Path) -> str:
    return _normalize_table_name(path.stem)


def _tokenize_name(name: str) -> List[str]:
    tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", str(name).lower()) if tok]
    return [KEY_TOKEN_ALIASES.get(tok, tok) for tok in tokens]


def _is_id_like_name(name: str) -> bool:
    tokens = set(_tokenize_name(name))
    if "id" in tokens:
        return True
    return bool(tokens.intersection(KEY_NAME_HINTS))


def _name_similarity(left_col: str, right_col: str) -> float:
    left_tokens = set(_tokenize_name(left_col))
    right_tokens = set(_tokenize_name(right_col))
    if not left_tokens or not right_tokens:
        return 0.0
    jaccard = len(left_tokens.intersection(right_tokens)) / max(1, len(left_tokens.union(right_tokens)))
    seq = SequenceMatcher(None, left_col.lower(), right_col.lower()).ratio()
    id_bonus = 0.1 if ("id" in left_tokens and "id" in right_tokens) else 0.0
    return min(1.0, max(jaccard, 0.8 * seq) + id_bonus)


def _type_compatibility(left_type: str, right_type: str) -> float:
    if left_type == right_type:
        return 1.0
    compatible = {
        ("bool", "numeric"),
        ("numeric", "bool"),
        ("string", "numeric"),
        ("numeric", "string"),
    }
    if (left_type, right_type) in compatible:
        return 0.6
    if "other" in {left_type, right_type}:
        return 0.3
    return 0.0


def _series_unique_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return float(non_null.nunique(dropna=True) / max(1, len(non_null)))


def _value_overlap_score(left: pd.Series, right: pd.Series, max_unique: int = 2000) -> float:
    left_vals = left.dropna().astype(str).str.strip()
    right_vals = right.dropna().astype(str).str.strip()
    if left_vals.empty or right_vals.empty:
        return 0.0
    left_set = set(left_vals.drop_duplicates().head(max_unique).tolist())
    right_set = set(right_vals.drop_duplicates().head(max_unique).tolist())
    if not left_set or not right_set:
        return 0.0
    inter = left_set.intersection(right_set)
    return float(len(inter) / max(1, min(len(left_set), len(right_set))))


def _spark_type_family(df, col: str) -> str:
    from pyspark.sql.types import BooleanType, DateType, NumericType, StringType, TimestampType

    field_map = {field.name: field.dataType for field in df.schema.fields}
    dtype = field_map.get(col)
    if dtype is None:
        return "other"
    if isinstance(dtype, BooleanType):
        return "bool"
    if isinstance(dtype, NumericType):
        return "numeric"
    if isinstance(dtype, (DateType, TimestampType)):
        return "datetime"
    if isinstance(dtype, StringType):
        return "string"
    return "other"


def _candidate_key_columns_spark(df, sample_size: int = 5000) -> List[str]:
    usable = [col for col in df.columns if _spark_type_family(df, col) not in {"datetime", "other"}]
    if not usable:
        return []

    id_like = [col for col in usable if _is_id_like_name(col)]
    sample_cols = usable[: min(40, len(usable))]
    sample_pdf = df.select(*sample_cols).limit(sample_size).toPandas()
    for col in sample_cols:
        if col in id_like:
            continue
        if _series_unique_ratio(sample_pdf[col]) >= 0.4:
            id_like.append(col)
    if not id_like:
        id_like = sample_cols
    return id_like[:40]


def _infer_join_mapping_spark(
    left_df,
    right_df,
    left_table: str,
    right_table: str,
    sample_size: int = 5000,
    min_confidence: float = 0.6,
) -> Optional[Dict[str, Any]]:
    left_candidates = _candidate_key_columns_spark(left_df, sample_size=sample_size)
    right_candidates = _candidate_key_columns_spark(right_df, sample_size=sample_size)
    if not left_candidates or not right_candidates:
        return None

    left_pdf = left_df.select(*left_candidates).limit(sample_size).toPandas()
    right_pdf = right_df.select(*right_candidates).limit(sample_size).toPandas()
    if left_pdf.empty or right_pdf.empty:
        return None

    scored: List[Tuple[float, str, str, Dict[str, float]]] = []
    for left_col in left_candidates:
        left_series = left_pdf[left_col]
        left_type = _spark_type_family(left_df, left_col)
        left_unique = _series_unique_ratio(left_series)
        left_id_like = 1.0 if _is_id_like_name(left_col) else 0.0
        for right_col in right_candidates:
            right_series = right_pdf[right_col]
            right_type = _spark_type_family(right_df, right_col)
            type_score = _type_compatibility(left_type, right_type)
            if type_score <= 0.0:
                continue
            right_unique = _series_unique_ratio(right_series)
            name_score = _name_similarity(left_col, right_col)
            overlap_score = _value_overlap_score(left_series, right_series)
            if overlap_score <= 0.0:
                continue
            right_id_like = 1.0 if _is_id_like_name(right_col) else 0.0
            uniqueness_score = min(left_unique, right_unique)
            id_hint_score = 0.5 * (left_id_like + right_id_like)
            score = (
                0.30 * name_score
                + 0.20 * type_score
                + 0.25 * overlap_score
                + 0.15 * uniqueness_score
                + 0.10 * id_hint_score
            )
            details = {
                "name_similarity": round(float(name_score), 6),
                "type_score": round(float(type_score), 6),
                "overlap_score": round(float(overlap_score), 6),
                "uniqueness_score": round(float(uniqueness_score), 6),
                "id_hint_score": round(float(id_hint_score), 6),
            }
            scored.append((score, left_col, right_col, details))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_left, best_right, best_details = scored[0]
    if best_score < min_confidence:
        return None
    return {
        "right": right_table,
        "left_on": [best_left],
        "right_on": [best_right],
        "how": "left",
        "inference": {
            "left_table": left_table,
            "right_table": right_table,
            "confidence": round(float(best_score), 6),
            "details": best_details,
        },
    }


def _infer_base_table(tables: Dict[str, Any]) -> str:
    preferred = ("transaction", "transactions", "txn", "fact_transaction", "fact")
    for name in preferred:
        if name in tables:
            return name
    return max(tables.items(), key=lambda x: int(x[1].count()))[0]


def _infer_join_keys(base_cols: Sequence[str], right_cols: Sequence[str], right_name: str) -> List[str]:
    shared = [c for c in base_cols if c in set(right_cols)]
    if not shared:
        return []

    right_lower = right_name.lower()
    if "customer" in right_lower and "customer_id" in shared:
        return ["customer_id"]
    if "account" in right_lower and "account_id" in shared:
        return ["account_id"]

    id_like = [c for c in shared if c.lower() == "id" or c.lower().endswith("_id") or "id" in c.lower()]
    if id_like:
        return [id_like[0]]
    return []


def _prepare_right_for_join_spark(right_df, right_on: Sequence[str], table_name: str):
    prepared = right_df
    renamed_keys: List[str] = []
    for col in right_df.columns:
        if col in right_on:
            new_name = f"__rk__{table_name}__{col}"
            renamed_keys.append(new_name)
            prepared = prepared.withColumnRenamed(col, new_name)
        else:
            prepared = prepared.withColumnRenamed(col, f"{table_name}__{col}")
    return prepared, renamed_keys


def _aggregate_tables_spark(spark, tables: Dict[str, Any]):
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, NumericType

    rows: List[Dict[str, Any]] = []
    for name, df in tables.items():
        row_count = int(df.count())
        column_count = len(df.columns)
        numeric_column_count = int(
            sum(1 for field in df.schema.fields if isinstance(field.dataType, (NumericType, BooleanType)))
        )
        if df.columns:
            missing_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
            missing_counts = df.agg(*missing_exprs).collect()[0].asDict()
            missing_cell_count = int(sum(int(v or 0) for v in missing_counts.values()))
        else:
            missing_cell_count = 0
        rows.append(
            {
                "table_name": name,
                "row_count": row_count,
                "column_count": int(column_count),
                "numeric_column_count": numeric_column_count,
                "non_numeric_column_count": int(column_count - numeric_column_count),
                "missing_cell_count": missing_cell_count,
            }
        )
    pdf = pd.DataFrame(rows)
    return spark.createDataFrame(pdf)


def _evaluate_consistency_checks_spark(df, checks: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    from pyspark.sql import functions as F

    reports: List[Dict[str, Any]] = []
    for check in checks:
        left_col = check.get("left", "")
        right_col = check.get("right", "")
        name = check.get("name", f"{left_col}_vs_{right_col}")
        if left_col not in df.columns or right_col not in df.columns:
            reports.append(
                {
                    "name": name,
                    "left": left_col,
                    "right": right_col,
                    "status": "skipped_missing_column",
                }
            )
            continue

        both = F.col(left_col).isNotNull() & F.col(right_col).isNotNull()
        comparable = int(df.filter(both).count())
        if comparable == 0:
            reports.append(
                {
                    "name": name,
                    "left": left_col,
                    "right": right_col,
                    "status": "skipped_no_overlap",
                    "comparable_rows": 0,
                }
            )
            continue

        mismatch = int(df.filter(both & (F.col(left_col) != F.col(right_col))).count())
        reports.append(
            {
                "name": name,
                "left": left_col,
                "right": right_col,
                "status": "ok",
                "comparable_rows": comparable,
                "mismatch_rows": mismatch,
                "mismatch_rate": round(float(mismatch / comparable), 6),
            }
        )
    return reports


def _compose_tables_spark(
    spark,
    tables: Dict[str, Any],
    compose_spec: Optional[Dict[str, Any]],
    no_key_policy: str,
):
    from pyspark.sql import functions as F

    if not tables:
        raise ValueError("No tables available for composition.")

    if len(tables) == 1:
        only_name = next(iter(tables.keys()))
        return tables[only_name], {"mode": "single_table", "base_table": only_name, "tables": list(tables.keys())}

    policy = _normalize_no_key_policy(no_key_policy)
    spec = compose_spec or {}
    base = str(spec.get("base", "")).strip() or _infer_base_table(tables)
    if base not in tables:
        raise ValueError(f"compose_spec base table '{base}' not found in inputs.")

    joins: List[Dict[str, Any]] = []
    unjoinable: List[str] = []

    raw_joins = spec.get("joins")
    if raw_joins is None:
        for table_name in tables.keys():
            if table_name == base:
                continue
            inferred = _infer_join_mapping_spark(
                left_df=tables[base],
                right_df=tables[table_name],
                left_table=base,
                right_table=table_name,
            )
            if not inferred:
                unjoinable.append(table_name)
                continue
            joins.append(inferred)
    else:
        if not isinstance(raw_joins, list) or not raw_joins:
            raise ValueError("compose_spec.joins must be a non-empty list.")
        seen: set = set()
        for raw in raw_joins:
            if not isinstance(raw, Mapping):
                raise ValueError("Each compose_spec.joins entry must be an object.")
            right = str(raw.get("right", "")).strip()
            if not right:
                raise ValueError("Each compose_spec.joins entry requires 'right'.")
            if right == base:
                continue
            if right not in tables:
                raise ValueError(f"compose_spec references unknown table '{right}'.")
            left_on = _normalize_keys(raw.get("left_on"), "left_on")
            right_on = _normalize_keys(raw.get("right_on", left_on), "right_on")
            if not left_on:
                raise ValueError(f"Join to '{right}' requires left_on.")
            if len(left_on) != len(right_on):
                raise ValueError(f"Join to '{right}' has different left_on/right_on lengths.")
            joins.append(
                {
                    "right": right,
                    "left_on": left_on,
                    "right_on": right_on,
                    "how": str(raw.get("how", "left")).strip() or "left",
                }
            )
            seen.add(right)
        unjoinable = [name for name in tables.keys() if name != base and name not in seen]

    if not joins or unjoinable:
        if policy == "aggregate_only":
            return _aggregate_tables_spark(spark, tables), {
                "mode": "aggregate_only",
                "reason": "missing_join_keys",
                "base_table": base,
                "tables_without_join_keys": unjoinable or [t for t in tables if t != base],
                "tables": list(tables.keys()),
            }
        raise ValueError(
            "Unable to compose all tables at row level due to missing join keys: "
            + ", ".join(unjoinable or [t for t in tables if t != base])
        )

    composed = tables[base]
    join_reports: List[Dict[str, Any]] = []
    for join in joins:
        right_name = join["right"]
        left_on = list(join["left_on"])
        right_on = list(join["right_on"])
        right_df = tables[right_name]

        missing_left = [c for c in left_on if c not in composed.columns]
        missing_right = [c for c in right_on if c not in right_df.columns]
        if missing_left or missing_right:
            if policy == "aggregate_only":
                return _aggregate_tables_spark(spark, tables), {
                    "mode": "aggregate_only",
                    "reason": "join_key_not_found",
                    "base_table": base,
                    "join_table": right_name,
                    "missing_left_keys": missing_left,
                    "missing_right_keys": missing_right,
                    "tables": list(tables.keys()),
                }
            raise ValueError(
                f"Join keys missing for table '{right_name}'. "
                f"left missing={missing_left}, right missing={missing_right}"
            )

        prepared_right, renamed_keys = _prepare_right_for_join_spark(right_df, right_on, right_name)
        marker = f"__joined__{right_name}"
        prepared_right = prepared_right.withColumn(marker, F.lit(1))

        condition = None
        for left_key, right_key in zip(left_on, renamed_keys):
            clause = composed[left_key] == prepared_right[right_key]
            condition = clause if condition is None else (condition & clause)

        try:
            merged = composed.join(prepared_right, on=condition, how=join["how"])
        except Exception as exc:
            if policy == "aggregate_only":
                return _aggregate_tables_spark(spark, tables), {
                    "mode": "aggregate_only",
                    "reason": "join_failed",
                    "base_table": base,
                    "join_table": right_name,
                    "error": str(exc),
                    "tables": list(tables.keys()),
                }
            raise

        unmatched_rows = int(merged.filter(F.col(marker).isNull()).count())
        composed = merged.drop(marker, *renamed_keys)
        rows_after_join = int(composed.count())
        join_reports.append(
            {
                "right_table": right_name,
                "how": join["how"],
                "left_on": left_on,
                "right_on": right_on,
                "rows_after_join": rows_after_join,
                "unmatched_left_rows": unmatched_rows,
                "inference": join.get("inference"),
            }
        )

    explicit_checks = spec.get("checks", [])
    if explicit_checks and not isinstance(explicit_checks, list):
        raise ValueError("compose_spec.checks must be a list of objects.")

    checks: List[Dict[str, str]] = []
    if isinstance(explicit_checks, list):
        for raw in explicit_checks:
            if not isinstance(raw, Mapping):
                raise ValueError("Each compose_spec.checks entry must be an object.")
            left_col = str(raw.get("left", "")).strip()
            right_col = str(raw.get("right", "")).strip()
            name = str(raw.get("name", f"{left_col}_vs_{right_col}")).strip()
            if not left_col or not right_col:
                continue
            checks.append({"name": name, "left": left_col, "right": right_col})

    if "customer_id" in composed.columns:
        for join in joins:
            candidate = f"{join['right']}__customer_id"
            if candidate in composed.columns:
                checks.append(
                    {
                        "name": f"customer_consistency_{join['right']}",
                        "left": "customer_id",
                        "right": candidate,
                    }
                )

    consistency = _evaluate_consistency_checks_spark(composed, checks) if checks else []
    meta = {
        "mode": "row_level",
        "base_table": base,
        "tables": list(tables.keys()),
        "join_reports": join_reports,
        "consistency_checks": consistency,
    }
    return composed, meta


def _ensure_table_dict(obj: Any, spark, source_name: str) -> Dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{source_name} must return a dict[str, DataFrame] for multi-table composition.")
    tables: Dict[str, Any] = {}
    for key, value in obj.items():
        name = str(key).strip()
        if not name:
            continue
        tables[name] = _ensure_spark_df(value, spark)
    if not tables:
        raise ValueError(f"{source_name} returned an empty table dictionary.")
    return tables


def _expand_paths(data: Sequence[str], recursive: bool = False) -> List[Path]:
    paths: List[Path] = []
    for item in data:
        if not item:
            continue
        if _is_glob(item):
            for match in glob.glob(item, recursive=recursive):
                paths.append(Path(match))
            continue
        p = Path(item)
        if p.is_dir():
            for ext in SUPPORTED_EXTS:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                paths.extend(p.glob(pattern))
            continue
        paths.append(p)
    existing = [p for p in paths if p.exists()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    pkg_root = Path(__file__).resolve().parents[1]
    candidate = pkg_root / data_dir
    return candidate if candidate.exists() else p


def _scan_dir_for_exts(data_dir: str, exts: Sequence[str], recursive: bool) -> List[Path]:
    base = _resolve_data_dir(data_dir)
    if not base.exists():
        return []
    paths: List[Path] = []
    for ext in exts:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        paths.extend(base.glob(pattern))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_excel_with_spark(spark, path: str):
    try:
        return (
            spark.read.format("com.crealytics.spark.excel")
            .option("header", True)
            .option("inferSchema", True)
            .load(path)
        )
    except Exception:
        return None


def _read_single_spark(spark, path: Path):
    ext = path.suffix.lower()
    local_uri = to_local_file_uri(str(path))
    if ext == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(local_uri)
    if ext == ".tsv":
        return spark.read.option("header", True).option("inferSchema", True).option("sep", "\t").csv(local_uri)
    if ext == ".parquet":
        return spark.read.parquet(local_uri)
    if ext == ".json":
        return spark.read.option("multiLine", True).json(local_uri)
    if ext in {".xlsx", ".xls"}:
        df = _read_excel_with_spark(spark, local_uri)
        if df is not None:
            return df
        try:
            pdf = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError("Reading Excel requires openpyxl. Install openpyxl to proceed.") from exc
        return spark.createDataFrame(pdf)
    if ext == ".feather":
        try:
            pdf = pd.read_feather(path)
        except ImportError as exc:
            raise ImportError("Reading Feather requires pyarrow. Install pyarrow to proceed.") from exc
        return spark.createDataFrame(pdf)
    raise ValueError(f"Unsupported file extension: {ext}")


def _concat_frames(frames: List):
    if not frames:
        raise ValueError("No input files matched.")
    df = frames[0]
    for frame in frames[1:]:
        df = df.unionByName(frame, allowMissingColumns=True)
    return df


def _merge_table_frame(tables: Dict[str, Any], table_name: str, frame) -> None:
    if table_name in tables:
        tables[table_name] = _concat_frames([tables[table_name], frame])
    else:
        tables[table_name] = frame


def _merge_loaded_object_into_tables(
    obj: Any,
    spark,
    tables: Dict[str, Any],
    source_name: str,
    default_table_name: str,
) -> None:
    if isinstance(obj, Mapping):
        table_dict = _ensure_table_dict(obj, spark, source_name)
        for table_name, frame in table_dict.items():
            _merge_table_frame(tables, _normalize_table_name(table_name), frame)
        return
    frame = _ensure_spark_df(obj, spark)
    _merge_table_frame(tables, _normalize_table_name(default_table_name), frame)


def _load_python_file_object(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    spec = importlib.util.spec_from_file_location("eda_spark_loader_module", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import python file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    if hasattr(module, "load") and callable(module.load):
        df = module.load()
    elif hasattr(module, "df"):
        df = getattr(module, "df")
    else:
        raise ValueError("Python file must define `load()` or `df`.")
    return df


def _load_python_file(path: str, spark):
    df = _load_python_file_object(path)
    return _ensure_spark_df(df, spark)


def _load_python_code_object(code: str) -> Any:
    local_ns: dict = {}
    exec(code, local_ns, local_ns)
    if "load" in local_ns and callable(local_ns["load"]):
        df = local_ns["load"]()
    elif "df" in local_ns:
        df = local_ns["df"]
    else:
        raise ValueError("Python code must define `load()` or `df`.")
    return df


def _load_python_code(code: str, spark):
    df = _load_python_code_object(code)
    return _ensure_spark_df(df, spark)


def _load_notebook_object(path: str) -> Any:
    nb_path = Path(path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with open(nb_path, "r", encoding="utf-8-sig") as f:
        nb = json.load(f)
    local_ns: dict = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else str(source)
        exec(code, local_ns, local_ns)
    if "load" in local_ns and callable(local_ns["load"]):
        df = local_ns["load"]()
    elif "df" in local_ns:
        df = local_ns["df"]
    else:
        raise ValueError("Notebook must define `load()` or `df` in code cells.")
    return df


def _load_notebook(path: str, spark):
    df = _load_notebook_object(path)
    return _ensure_spark_df(df, spark)


def _ensure_spark_df(df, spark):
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except Exception:
        SparkDataFrame = None
    if SparkDataFrame is not None and isinstance(df, SparkDataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return spark.createDataFrame(df)
    raise ValueError("Python/Notebook loader did not return a pandas or Spark DataFrame.")


def _clean_sql_lines(sql_text: str) -> List[str]:
    lines: List[str] = []
    for raw in sql_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            continue
        lines.append(line)
    return lines


def _is_select_only(sql_text: str) -> bool:
    lines = _clean_sql_lines(sql_text)
    if not lines:
        return False
    head = lines[0].lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False
    return ";" not in sql_text.strip().rstrip(";")


def _sqlite_objects(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [(str(name), str(obj_type)) for name, obj_type in rows]


def _pick_sql_table(objects: List[Tuple[str, str]]) -> Optional[str]:
    names = [name for name, _ in objects]
    for preferred in PREFERRED_SQL_TABLES:
        if preferred in names:
            return preferred
    if len(names) == 1:
        return names[0]
    return None


def _resolve_file_inputs(
    value: Optional[str],
    allowed_exts: Sequence[str],
    recursive: bool = False,
) -> List[Path]:
    if not value:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        return []
    matches: List[Path] = []
    for token in tokens:
        token_matches: List[Path] = []
        if _is_glob(token):
            token_matches.extend(Path(p) for p in glob.glob(token, recursive=recursive))
        else:
            p = Path(token)
            if p.is_dir():
                for ext in allowed_exts:
                    pattern = f"**/*{ext}" if recursive else f"*{ext}"
                    token_matches.extend(p.glob(pattern))
            elif p.exists():
                token_matches.append(p)
        if not token_matches:
            return []
        matches.extend(token_matches)
    allow_set = {ext.lower() for ext in allowed_exts}
    filtered = [p for p in matches if p.exists() and p.suffix.lower() in allow_set]
    return sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_sql_with_db(spark, sql: str, db: str):
    if db.startswith("jdbc:"):
        return (
            spark.read.format("jdbc")
            .option("url", db)
            .option("dbtable", f"({sql}) t")
            .load()
        )

    db_path = None
    if db.startswith("sqlite:///"):
        db_path = db.replace("sqlite:///", "", 1)
    elif db.startswith("sqlite://"):
        db_path = db.replace("sqlite://", "", 1)
    else:
        p = Path(db)
        if p.exists() and p.suffix.lower() in AUTO_DB_EXTS:
            db_path = str(p)

    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            pdf = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        return spark.createDataFrame(pdf)

    try:
        import sqlalchemy  # type: ignore
    except ImportError as exc:
        raise ImportError("SQLAlchemy is required for non-sqlite DB connections.") from exc
    engine = sqlalchemy.create_engine(db)
    pdf = pd.read_sql_query(sql, engine)
    return spark.createDataFrame(pdf)


def _load_sql_auto(spark, sql_files: List[Path], db_files: List[Path]) -> List[Tuple[str, Any]]:
    if not sql_files:
        return []
    texts = [(p, p.read_text(encoding="utf-8")) for p in sql_files]
    if db_files and all(_is_select_only(text) for _, text in texts):
        db_path = str(db_files[0])
        conn = sqlite3.connect(db_path)
        try:
            frames = [
                (_table_name_from_path(path), spark.createDataFrame(pd.read_sql_query(text, conn)))
                for path, text in texts
            ]
        finally:
            conn.close()
        return frames

    conn = sqlite3.connect(":memory:")
    try:
        for _, text in texts:
            conn.executescript(text)
        objects = _sqlite_objects(conn)
        table = _pick_sql_table(objects)
        if not table:
            raise ValueError(
                "SQL auto-exec produced multiple tables/views. "
                "Create a single table/view or name it one of: "
                f"{', '.join(PREFERRED_SQL_TABLES)}."
            )
        pdf = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        return [(_normalize_table_name(table), spark.createDataFrame(pdf))]
    finally:
        conn.close()


class DataLoader:
    """Unified loader that returns a Spark DataFrame and a data_source string."""

    def __init__(
        self,
        spark,
        data: Optional[Sequence[str]] = None,
        sql: Optional[str] = None,
        db: Optional[str] = None,
        py: Optional[str] = None,
        py_code: Optional[str] = None,
        nb: Optional[str] = None,
        data_dir: str = "./data",
        recursive: bool = False,
        auto_exec: bool = False,
        compose_spec: Optional[Any] = None,
        no_key_policy: str = "error",
    ) -> None:
        self.spark = spark
        self.data = _as_list(data)
        self.sql = sql
        self.db = db
        self.py = py
        self.py_code = py_code
        self.nb = nb
        self.data_dir = data_dir
        self.recursive = recursive
        self.auto_exec = auto_exec
        self.compose_spec = compose_spec
        self.no_key_policy = _normalize_no_key_policy(no_key_policy)
        self.last_compose_meta: Dict[str, Any] = {}

    def load(self) -> Tuple[Any, str]:
        self.last_compose_meta = {}
        mode = self._detect_mode()
        compose_spec = _load_compose_spec(self.compose_spec)

        if mode == "sql":
            if not self.db:
                raise ValueError("SQL mode requires --db.")
            named_sql = _parse_named_sql(self.sql)
            if named_sql:
                tables = {name: _load_sql_with_db(self.spark, query, self.db) for name, query in named_sql.items()}
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, "sql_map:" + ",".join(named_sql.keys())
            sql_files = _resolve_file_inputs(self.sql, [".sql"], recursive=self.recursive)
            if len(sql_files) > 1:
                tables: Dict[str, Any] = {}
                sources: List[str] = []
                for path in sql_files:
                    query = path.read_text(encoding="utf-8")
                    frame = _load_sql_with_db(self.spark, query, self.db)
                    _merge_table_frame(tables, _table_name_from_path(path), frame)
                    sources.append(str(path))
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, "sql_files:" + ",".join(sources)

            if (
                len(sql_files) == 1
                and self.sql
                and Path(str(self.sql).strip()).exists()
                and Path(str(self.sql).strip()).is_file()
            ):
                sql_path = sql_files[0]
                query = sql_path.read_text(encoding="utf-8")
                df = _load_sql_with_db(self.spark, query, self.db)
                return df, f"sql_file:{sql_path}"

            df = _load_sql_with_db(self.spark, self.sql or "", self.db)
            return df, f"sql:{self.sql}"

        if mode == "py":
            py_files = _resolve_file_inputs(self.py, [".py"], recursive=self.recursive)
            if len(py_files) > 1:
                tables: Dict[str, Any] = {}
                sources: List[str] = []
                for path in py_files:
                    obj = _load_python_file_object(str(path))
                    _merge_loaded_object_into_tables(
                        obj=obj,
                        spark=self.spark,
                        tables=tables,
                        source_name=f"Python loader file '{path}'",
                        default_table_name=_table_name_from_path(path),
                    )
                    sources.append(str(path))
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, "py_files:" + ",".join(sources)

            obj = _load_python_file_object(self.py or "")
            if isinstance(obj, Mapping):
                tables = _ensure_table_dict(obj, self.spark, "Python loader")
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, f"py:{self.py}"
            return _ensure_spark_df(obj, self.spark), f"py:{self.py}"

        if mode == "py_code":
            obj = _load_python_code_object(self.py_code or "")
            if isinstance(obj, Mapping):
                tables = _ensure_table_dict(obj, self.spark, "Python code")
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, "py_code"
            return _ensure_spark_df(obj, self.spark), "py_code"

        if mode == "nb":
            nb_files = _resolve_file_inputs(self.nb, [".ipynb"], recursive=self.recursive)
            if len(nb_files) > 1:
                tables: Dict[str, Any] = {}
                sources: List[str] = []
                for path in nb_files:
                    obj = _load_notebook_object(str(path))
                    _merge_loaded_object_into_tables(
                        obj=obj,
                        spark=self.spark,
                        tables=tables,
                        source_name=f"Notebook loader file '{path}'",
                        default_table_name=_table_name_from_path(path),
                    )
                    sources.append(str(path))
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, "nb_files:" + ",".join(sources)

            obj = _load_notebook_object(self.nb or "")
            if isinstance(obj, Mapping):
                tables = _ensure_table_dict(obj, self.spark, "Notebook loader")
                df, meta = _compose_tables_spark(
                    self.spark,
                    tables,
                    compose_spec=compose_spec,
                    no_key_policy=self.no_key_policy,
                )
                self.last_compose_meta = meta
                return df, f"nb:{self.nb}"
            return _ensure_spark_df(obj, self.spark), f"nb:{self.nb}"

        if mode == "data":
            if not self.data:
                path = detect_latest_dataset(data_dir=self.data_dir)
                df = _read_single_spark(self.spark, Path(path))
                return df, path

            named_data, unnamed_data = _parse_data_bindings(self.data)
            if named_data and unnamed_data:
                raise ValueError("Do not mix named and unnamed --data inputs when composing tables.")

            tables, sources = self._load_tables_for_data_mode(named_data, unnamed_data)
            df, meta = _compose_tables_spark(
                self.spark,
                tables,
                compose_spec=compose_spec,
                no_key_policy=self.no_key_policy,
            )
            self.last_compose_meta = meta
            return df, ";".join(sources)

        if mode == "auto":
            data_paths = _scan_dir_for_exts(self.data_dir, list(SUPPORTED_EXTS), self.recursive)
            sql_files = _scan_dir_for_exts(self.data_dir, [".sql"], self.recursive)
            py_files = _scan_dir_for_exts(self.data_dir, [".py"], self.recursive)
            nb_files = _scan_dir_for_exts(self.data_dir, [".ipynb"], self.recursive)
            db_files = _scan_dir_for_exts(self.data_dir, list(AUTO_DB_EXTS), self.recursive)

            tables: Dict[str, Any] = {}
            sources: List[str] = []
            if data_paths:
                grouped_frames: Dict[str, List[Any]] = {}
                for p in data_paths:
                    table_name = _table_name_from_path(p)
                    grouped_frames.setdefault(table_name, []).append(_read_single_spark(self.spark, p))
                for table_name, frames in grouped_frames.items():
                    _merge_table_frame(tables, table_name, _concat_frames(frames))
                sources.extend(str(p) for p in data_paths)
            if sql_files:
                sql_tables = _load_sql_auto(self.spark, sql_files, db_files)
                for table_name, frame in sql_tables:
                    _merge_table_frame(tables, table_name, frame)
                sources.extend(str(p) for p in sql_files)
            if py_files:
                for p in py_files:
                    obj = _load_python_file_object(str(p))
                    _merge_loaded_object_into_tables(
                        obj=obj,
                        spark=self.spark,
                        tables=tables,
                        source_name=f"Python loader file '{p}'",
                        default_table_name=_table_name_from_path(p),
                    )
                    sources.append(str(p))
            if nb_files:
                for p in nb_files:
                    obj = _load_notebook_object(str(p))
                    _merge_loaded_object_into_tables(
                        obj=obj,
                        spark=self.spark,
                        tables=tables,
                        source_name=f"Notebook loader file '{p}'",
                        default_table_name=_table_name_from_path(p),
                    )
                    sources.append(str(p))

            if not tables:
                raise FileNotFoundError("No supported data/auto-exec files found in ./data.")

            df, meta = _compose_tables_spark(
                self.spark,
                tables,
                compose_spec=compose_spec,
                no_key_policy=self.no_key_policy,
            )
            self.last_compose_meta = meta
            return df, ";".join(sources)

        raise ValueError("No valid data input provided.")

    def _load_tables_for_data_mode(
        self,
        named_data: Dict[str, List[str]],
        unnamed_data: List[str],
    ) -> Tuple[Dict[str, Any], List[str]]:
        tables: Dict[str, Any] = {}
        sources: List[str] = []

        if named_data:
            for table_name, inputs in named_data.items():
                paths = _expand_paths(inputs, recursive=self.recursive)
                if not paths:
                    raise FileNotFoundError(f"No matching files found for table '{table_name}'.")
                frames = [_read_single_spark(self.spark, p) for p in paths]
                normalized = _normalize_table_name(table_name)
                _merge_table_frame(tables, normalized, _concat_frames(frames))
                sources.append(f"{normalized}=" + ",".join(str(p) for p in paths))
            return tables, sources

        paths = _expand_paths(unnamed_data, recursive=self.recursive)
        if not paths:
            raise FileNotFoundError("No matching data files found.")

        grouped_frames: Dict[str, List[Any]] = {}
        grouped_sources: Dict[str, List[str]] = {}
        for path in paths:
            table_name = _table_name_from_path(path)
            grouped_frames.setdefault(table_name, []).append(_read_single_spark(self.spark, path))
            grouped_sources.setdefault(table_name, []).append(str(path))

        for table_name, frames in grouped_frames.items():
            tables[table_name] = _concat_frames(frames)
            sources.append(f"{table_name}=" + ",".join(grouped_sources.get(table_name, [])))
        return tables, sources

    def _detect_mode(self) -> str:
        provided = {
            "data": bool(self.data),
            "sql": bool(self.sql),
            "py": bool(self.py),
            "py_code": bool(self.py_code),
            "nb": bool(self.nb),
        }
        active = [k for k, v in provided.items() if v]
        if len(active) > 1:
            raise ValueError(f"Only one input mode is allowed. Provided: {active}")
        if not active and self.auto_exec:
            return "auto"
        if self.sql:
            return "sql"
        if self.py:
            return "py"
        if self.py_code:
            return "py_code"
        if self.nb:
            return "nb"
        return "data"
