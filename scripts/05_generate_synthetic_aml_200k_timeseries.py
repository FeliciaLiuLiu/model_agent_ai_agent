"""Generate a synthetic AML dataset with drift and time series patterns."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _make_timestamps(rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    mid = start + timedelta(days=90)
    end = start + timedelta(days=180)

    half = rows // 2
    seconds_a = rng.integers(0, int((mid - start).total_seconds()), size=half)
    seconds_b = rng.integers(0, int((end - mid).total_seconds()), size=rows - half)
    ts_a = np.array([start + timedelta(seconds=int(s)) for s in seconds_a], dtype="datetime64[ns]")
    ts_b = np.array([mid + timedelta(seconds=int(s)) for s in seconds_b], dtype="datetime64[ns]")
    period_flag = np.concatenate([np.zeros(half, dtype=int), np.ones(rows - half, dtype=int)])
    ts = np.concatenate([ts_a, ts_b])

    idx = rng.permutation(rows)
    return ts[idx], period_flag[idx]


def generate_dataset(
    rows: int = 200_000,
    seed: int = 42,
    suspicious_rate: float = 0.03,
    label_noise: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts, period_flag = _make_timestamps(rows, seed)

    channels = np.array(["card", "ach", "wire", "cash", "crypto"])
    channel_probs_a = np.array([0.45, 0.2, 0.1, 0.15, 0.1])
    channel_probs_b = np.array([0.35, 0.2, 0.2, 0.15, 0.1])
    channel = np.where(
        period_flag == 0,
        rng.choice(channels, size=rows, p=channel_probs_a),
        rng.choice(channels, size=rows, p=channel_probs_b),
    )

    origin_countries = np.array(["US", "DE", "GB", "SG", "HK", "AE", "RU", "NG", "PK", "BR"])
    origin_probs_a = np.array([0.28, 0.12, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.12])
    origin_probs_b = np.array([0.23, 0.11, 0.11, 0.07, 0.06, 0.06, 0.08, 0.07, 0.06, 0.15])
    origin_country = np.where(
        period_flag == 0,
        rng.choice(origin_countries, size=rows, p=origin_probs_a),
        rng.choice(origin_countries, size=rows, p=origin_probs_b),
    )

    dest_country = rng.choice(origin_countries, size=rows, replace=True)
    currency = np.where(origin_country == "US", "USD", np.where(origin_country == "DE", "EUR", "USD"))
    device_type = rng.choice(["ios", "android", "web"], size=rows, p=[0.4, 0.4, 0.2])
    txn_type = rng.choice(["purchase", "transfer", "cash_out", "refund"], size=rows, p=[0.6, 0.25, 0.1, 0.05])
    risk_segment = rng.choice(["low", "mid", "high"], size=rows, p=[0.65, 0.25, 0.1])

    txn_amount = rng.lognormal(mean=3.0 + 0.25 * period_flag, sigma=1.0, size=rows)
    account_balance = rng.lognormal(mean=8.0, sigma=0.7, size=rows)
    num_txn_1h = rng.poisson(lam=1.2 + 0.2 * period_flag, size=rows)
    num_txn_24h = rng.poisson(lam=4.0 + 0.5 * period_flag, size=rows)
    sum_amount_24h = txn_amount * num_txn_24h * rng.uniform(0.6, 1.4, size=rows)
    days_since_last_txn = rng.integers(0, 30, size=rows)
    velocity_score = np.clip((num_txn_24h * 0.3 + sum_amount_24h / 5000) + rng.normal(0, 0.3, size=rows), 0, None)

    is_international = origin_country != dest_country
    is_new_beneficiary = rng.random(size=rows) < 0.15

    base_rate = suspicious_rate + 0.01 * period_flag
    is_suspicious = (rng.random(size=rows) < base_rate).astype(int)
    if label_noise > 0:
        flip = rng.random(size=rows) < label_noise
        is_suspicious = np.where(flip, 1 - is_suspicious, is_suspicious)

    merchant_names = np.array(["Alpha Shop", "Bright Mart", "Cobalt LLC", "Delta Foods", "Echo Retail"])
    payment_memo = np.array(["invoice", "subscription", "refund", "cashout", "transfer", "fee"])

    df = pd.DataFrame(
        {
            "txn_id": [f"TXN{idx:09d}" for idx in range(rows)],
            "account_id": [f"ACC{idx % 50000:07d}" for idx in range(rows)],
            "customer_id": [f"CUST{idx % 60000:07d}" for idx in range(rows)],
            "merchant_id": [f"M{idx % 20000:06d}" for idx in range(rows)],
            "counterparty_id": [f"CP{idx % 40000:07d}" for idx in range(rows)],
            "txn_ts": ts,
            "txn_amount": txn_amount.round(2),
            "account_balance": account_balance.round(2),
            "num_txn_1h": num_txn_1h,
            "num_txn_24h": num_txn_24h,
            "sum_amount_24h": sum_amount_24h.round(2),
            "days_since_last_txn": days_since_last_txn,
            "velocity_score": velocity_score.round(3),
            "channel": channel,
            "txn_type": txn_type,
            "origin_country": origin_country,
            "dest_country": dest_country,
            "currency": currency,
            "device_type": device_type,
            "risk_segment": risk_segment,
            "is_international": is_international,
            "is_new_beneficiary": is_new_beneficiary,
            "merchant_name": rng.choice(merchant_names, size=rows),
            "payment_memo": rng.choice(payment_memo, size=rows),
            "is_suspicious": is_suspicious,
        }
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AML dataset with drift.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: ./data)")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--suspicious-rate", type=float, default=0.03, help="Base suspicious rate")
    parser.add_argument("--label-noise", type=float, default=0.0, help="Label noise rate")
    parser.add_argument("--parquet", action="store_true", help="Also write parquet output")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        rows=args.rows,
        seed=args.seed,
        suspicious_rate=args.suspicious_rate,
        label_noise=args.label_noise,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"synthetic_aml_200k_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    if args.parquet:
        parquet_path = out_dir / f"synthetic_aml_200k_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)

    meta = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "generated_at": timestamp,
        "seed": args.seed,
        "suspicious_rate": args.suspicious_rate,
        "label_noise": args.label_noise,
    }
    meta_path = out_dir / f"synthetic_aml_200k_{timestamp}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    if args.parquet:
        print(f"Saved Parquet: {parquet_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
