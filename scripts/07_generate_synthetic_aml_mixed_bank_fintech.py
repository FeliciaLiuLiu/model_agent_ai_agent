"""Generate a mixed bank + fintech AML dataset for EDA and modeling demos."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _make_timestamps(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 7, 1)
    end = start + timedelta(days=180)
    seconds = rng.integers(0, int((end - start).total_seconds()), size=rows)
    return np.array([start + timedelta(seconds=int(s)) for s in seconds], dtype="datetime64[ns]")


def _generate_ip_addresses(rng: np.random.Generator, rows: int) -> np.ndarray:
    a = rng.integers(1, 255, size=rows)
    b = rng.integers(0, 255, size=rows)
    c = rng.integers(0, 255, size=rows)
    d = rng.integers(1, 255, size=rows)
    return np.array([f"10.{x}.{y}.{z}" for x, y, z in zip(a, b, c)], dtype=object)


def _generate_text_notes(rng: np.random.Generator, rows: int, prefix: str) -> np.ndarray:
    templates = np.array(
        [
            "Customer reported unexpected transfer pattern after device change.",
            "Transaction flagged for manual review due to high velocity behavior.",
            "Recurring payments observed with merchant mismatch to profile.",
            "Multiple small transfers aggregated into a single payout request.",
            "Counterparty linked to prior alerts; escalating for review.",
            "New account activity with elevated risk score and overseas destination.",
            "Chargeback history detected; verify supporting documentation.",
            "Device location shift not consistent with historical behavior.",
        ],
        dtype=object,
    )
    base = rng.choice(templates, size=rows)
    refs = rng.integers(100000, 999999, size=rows)
    return np.array([f"{prefix}: {note} Ref-{ref}" for note, ref in zip(base, refs)], dtype=object)


def _inject_numeric_missing(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> None:
    numeric_cols = [
        "account_age_days",
        "customer_tenure_days",
        "txn_amount",
        "fee_amount",
        "account_balance",
        "num_txn_1h",
        "num_txn_24h",
        "sum_amount_24h",
        "velocity_score",
        "ip_risk_score",
        "geo_lat",
        "geo_lon",
    ]
    for col in numeric_cols:
        mask = rng.random(len(df)) < rate
        if mask.any():
            df.loc[mask, col] = np.nan


def _inject_boolean_missing(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> None:
    bool_cols = [
        "is_international",
        "is_new_device",
        "is_high_risk_country",
        "is_crypto_related",
        "is_pep",
        "sanctions_match",
        "is_business_account",
    ]
    for col in bool_cols:
        df[col] = df[col].astype("boolean")
        mask = rng.random(len(df)) < rate
        if mask.any():
            df.loc[mask, col] = pd.NA


def _inject_string_missing(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> None:
    string_cols = [
        "merchant_name",
        "merchant_category",
        "merchant_description",
        "payment_memo",
        "case_status",
        "case_notes",
        "device_fingerprint",
        "payment_reference",
        "device_id",
        "app_version",
        "counterparty_id",
        "origin_country",
        "dest_country",
        "risk_segment",
        "kyc_level",
    ]
    placeholders = np.array(["", "N/A", "NULL", "UNKNOWN"], dtype=object)
    for col in string_cols:
        mask_placeholder = rng.random(len(df)) < rate
        if mask_placeholder.any():
            df.loc[mask_placeholder, col] = rng.choice(placeholders, size=int(mask_placeholder.sum()))
        mask_missing = rng.random(len(df)) < (rate / 2)
        if mask_missing.any():
            df.loc[mask_missing, col] = None


def _build_label_probability(
    features: Dict[str, np.ndarray],
    base_rate: float,
) -> np.ndarray:
    prob = np.full(features["rows"], base_rate, dtype=float)
    prob += 0.03 * features["high_value"]
    prob += 0.02 * features["high_velocity"]
    prob += 0.02 * features["high_risk_country"]
    prob += 0.015 * features["crypto_related"]
    prob += 0.01 * features["new_device"]
    prob += 0.01 * features["fintech_platform"]
    return np.clip(prob, 0.0, 0.8)


def generate_dataset(
    rows: int = 50_000,
    seed: int = 42,
    sar_rate: float = 0.04,
    label_noise: float = 0.01,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    txn_ts = _make_timestamps(rows, seed)

    platforms = np.array(["bank", "venmo", "paypal", "cashapp", "zelle"], dtype=object)
    platform = rng.choice(platforms, size=rows, p=[0.6, 0.15, 0.12, 0.08, 0.05])

    bank_mask = platform == "bank"
    fintech_mask = ~bank_mask

    channel = np.empty(rows, dtype=object)
    channel[bank_mask] = rng.choice(
        ["branch", "online", "atm", "mobile"],
        size=int(bank_mask.sum()),
        p=[0.15, 0.45, 0.1, 0.3],
    )
    channel[fintech_mask] = rng.choice(
        ["mobile", "web", "api"],
        size=int(fintech_mask.sum()),
        p=[0.7, 0.25, 0.05],
    )

    payment_rail = np.empty(rows, dtype=object)
    payment_rail[bank_mask] = rng.choice(
        ["ach", "wire", "card", "cash"],
        size=int(bank_mask.sum()),
        p=[0.45, 0.2, 0.25, 0.1],
    )
    payment_rail[fintech_mask] = rng.choice(
        ["p2p", "card", "crypto"],
        size=int(fintech_mask.sum()),
        p=[0.7, 0.2, 0.1],
    )

    txn_type = np.empty(rows, dtype=object)
    txn_type[bank_mask] = rng.choice(
        ["wire", "ach", "bill_pay", "cash_withdrawal", "card_purchase", "deposit"],
        size=int(bank_mask.sum()),
        p=[0.1, 0.35, 0.15, 0.1, 0.2, 0.1],
    )
    txn_type[fintech_mask] = rng.choice(
        ["p2p_transfer", "merchant_payment", "cash_out", "crypto_trade", "refund"],
        size=int(fintech_mask.sum()),
        p=[0.5, 0.2, 0.15, 0.1, 0.05],
    )

    txn_direction = rng.choice(["outbound", "inbound"], size=rows, p=[0.7, 0.3])

    account_type = rng.choice(
        ["checking", "savings", "credit_card", "prepaid", "wallet"],
        size=rows,
        p=[0.35, 0.2, 0.2, 0.1, 0.15],
    )

    countries = np.array(["US", "GB", "DE", "CA", "SG", "HK", "AE", "BR", "NG", "RU", "PK"], dtype=object)
    origin_country = rng.choice(countries, size=rows, p=[0.45, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03])
    dest_country = rng.choice(countries, size=rows, replace=True)
    currency_map = {"US": "USD", "GB": "GBP", "DE": "EUR", "CA": "CAD", "SG": "SGD", "HK": "HKD", "AE": "AED", "BR": "BRL", "NG": "NGN", "RU": "RUB", "PK": "PKR"}
    currency = np.array([currency_map.get(c, "USD") for c in origin_country], dtype=object)

    device_type = rng.choice(["ios", "android", "web", "pos", "atm"], size=rows, p=[0.35, 0.35, 0.2, 0.05, 0.05])
    device_id = np.array([f"DEV{idx:08d}" for idx in rng.integers(0, 5_000_000, size=rows)], dtype=object)
    ip_address = _generate_ip_addresses(rng, rows)

    app_version = np.empty(rows, dtype=object)
    app_version[bank_mask] = ""
    app_version[fintech_mask] = rng.choice(["1.2.0", "1.4.3", "2.0.1", "2.1.5"], size=int(fintech_mask.sum()))

    merchant_names = np.array(
        ["Alpha Market", "Bright Electronics", "Cobalt Services", "Delta Travel", "Echo Gaming", "Foxtrot Groceries", "P2P Contact"],
        dtype=object,
    )
    merchant_category = np.array(
        ["grocery", "electronics", "travel", "gaming", "utilities", "crypto", "marketplace", "services"],
        dtype=object,
    )
    payment_memo = np.array(
        ["rent", "invoice", "salary", "gift", "family", "loan", "subscription", "refund", "crypto", "tips"],
        dtype=object,
    )

    risk_segment = rng.choice(["low", "mid", "high"], size=rows, p=[0.6, 0.3, 0.1])
    kyc_level = rng.choice(["basic", "standard", "enhanced"], size=rows, p=[0.4, 0.45, 0.15])

    case_status = rng.choice(["open", "closed", "escalated", "monitoring"], size=rows, p=[0.4, 0.35, 0.15, 0.1])
    case_notes = _generate_text_notes(rng, rows, prefix="CaseNote")
    merchant_description = _generate_text_notes(rng, rows, prefix="MerchantDesc")
    device_fingerprint = np.array(
        [f"fp_{rng.integers(0, 16**12):012x}" for _ in range(rows)],
        dtype=object,
    )
    payment_reference = np.array([f"REF{rng.integers(0, 99999999):08d}" for _ in range(rows)], dtype=object)

    account_age_days = rng.integers(30, 3650, size=rows)
    customer_tenure_days = rng.integers(10, 5000, size=rows)

    txn_amount = rng.lognormal(mean=3.2 + fintech_mask.astype(float) * 0.2, sigma=1.0, size=rows)
    fee_amount = np.maximum(0.05, txn_amount * rng.uniform(0.001, 0.02, size=rows))
    account_balance = rng.lognormal(mean=8.0, sigma=0.8, size=rows)

    num_txn_1h = rng.poisson(lam=1.4 + fintech_mask.astype(float) * 0.4, size=rows)
    num_txn_24h = rng.poisson(lam=4.5 + fintech_mask.astype(float) * 0.8, size=rows)
    sum_amount_24h = txn_amount * np.maximum(1, num_txn_24h) * rng.uniform(0.6, 1.5, size=rows)
    velocity_score = np.clip((num_txn_24h * 0.35 + sum_amount_24h / 6000) + rng.normal(0, 0.35, size=rows), 0, None)
    ip_risk_score = np.clip(rng.normal(loc=0.4, scale=0.2, size=rows), 0, 1)
    geo_lat = rng.uniform(-85.0, 85.0, size=rows)
    geo_lon = rng.uniform(-170.0, 170.0, size=rows)

    is_international = origin_country != dest_country
    is_new_device = rng.random(size=rows) < 0.12
    high_risk_countries = {"NG", "RU", "PK"}
    is_high_risk_country = np.array([(o in high_risk_countries) or (d in high_risk_countries) for o, d in zip(origin_country, dest_country)])
    is_crypto_related = np.array([t == "crypto_trade" or r == "crypto" for t, r in zip(txn_type, payment_rail)])
    is_pep = rng.random(size=rows) < 0.03
    sanctions_match = rng.random(size=rows) < 0.01
    is_business_account = rng.random(size=rows) < 0.25

    features = {
        "rows": rows,
        "high_value": txn_amount > 5000,
        "high_velocity": num_txn_24h >= 8,
        "high_risk_country": is_high_risk_country,
        "crypto_related": is_crypto_related,
        "new_device": is_new_device,
        "fintech_platform": fintech_mask,
    }
    prob = _build_label_probability(features, sar_rate)
    sar_actual = (rng.random(size=rows) < prob).astype(int)
    if label_noise > 0:
        flip = rng.random(size=rows) < label_noise
        sar_actual = np.where(flip, 1 - sar_actual, sar_actual)

    settle_offsets = rng.integers(60, 3600 * 48, size=rows).astype("timedelta64[s]")
    settlement_time = txn_ts + settle_offsets
    dob_start = datetime(1950, 1, 1)
    dob_offsets = rng.integers(0, 56 * 365, size=rows)
    customer_birth_date = np.array(
        [dob_start + timedelta(days=int(d)) for d in dob_offsets],
        dtype="datetime64[ns]",
    )

    df = pd.DataFrame(
        {
            "txn_id": [f"TXN{idx:010d}" for idx in range(rows)],
            "txn_ts": txn_ts,
            "settlement_time": settlement_time,
            "platform": platform,
            "channel": channel,
            "payment_rail": payment_rail,
            "txn_type": txn_type,
            "txn_direction": txn_direction,
            "account_id": [f"ACC{idx % 40000:07d}" for idx in range(rows)],
            "customer_id": [f"CUST{idx % 50000:07d}" for idx in range(rows)],
            "counterparty_id": [f"CP{idx % 60000:07d}" for idx in range(rows)],
            "merchant_name": rng.choice(merchant_names, size=rows),
            "merchant_category": rng.choice(merchant_category, size=rows),
            "merchant_description": merchant_description,
            "payment_memo": rng.choice(payment_memo, size=rows),
            "payment_reference": payment_reference,
            "origin_country": origin_country,
            "dest_country": dest_country,
            "currency": currency,
            "device_type": device_type,
            "device_id": device_id,
            "device_fingerprint": device_fingerprint,
            "ip_address": ip_address,
            "app_version": app_version,
            "account_type": account_type,
            "kyc_level": kyc_level,
            "risk_segment": risk_segment,
            "case_status": case_status,
            "case_notes": case_notes,
            "customer_birth_date": customer_birth_date,
            "account_age_days": account_age_days,
            "customer_tenure_days": customer_tenure_days,
            "txn_amount": txn_amount.round(2),
            "fee_amount": fee_amount.round(2),
            "account_balance": account_balance.round(2),
            "num_txn_1h": num_txn_1h,
            "num_txn_24h": num_txn_24h,
            "sum_amount_24h": sum_amount_24h.round(2),
            "velocity_score": velocity_score.round(3),
            "ip_risk_score": ip_risk_score.round(3),
            "geo_lat": geo_lat.round(5),
            "geo_lon": geo_lon.round(5),
            "is_international": is_international,
            "is_new_device": is_new_device,
            "is_high_risk_country": is_high_risk_country,
            "is_crypto_related": is_crypto_related,
            "is_pep": is_pep,
            "sanctions_match": sanctions_match,
            "is_business_account": is_business_account,
            "sar_actual": sar_actual,
        }
    )

    _inject_numeric_missing(df, rng, rate=0.015)
    _inject_boolean_missing(df, rng, rate=0.01)
    _inject_string_missing(df, rng, rate=0.02)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed bank + fintech AML dataset.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: ./data)")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sar-rate", type=float, default=0.04, help="Base SAR rate")
    parser.add_argument("--label-noise", type=float, default=0.01, help="Label noise rate")
    parser.add_argument("--parquet", action="store_true", help="Also write parquet output")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        rows=args.rows,
        seed=args.seed,
        sar_rate=args.sar_rate,
        label_noise=args.label_noise,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"synthetic_aml_mixed_50k_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    if args.parquet:
        parquet_path = out_dir / f"synthetic_aml_mixed_50k_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)

    meta = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "generated_at": timestamp,
        "seed": args.seed,
        "sar_rate": args.sar_rate,
        "label_noise": args.label_noise,
        "target_col": "sar_actual",
    }
    meta_path = out_dir / f"synthetic_aml_mixed_50k_{timestamp}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    if args.parquet:
        print(f"Saved Parquet: {parquet_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
