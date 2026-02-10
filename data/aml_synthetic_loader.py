"""Generate a synthetic AML dataset similar to data/aml_synthetic.sql.

This loader returns a pandas DataFrame so it can be used by both eda and eda_spark.
Environment overrides:
  AML_SYNTHETIC_SEED
  AML_SYNTHETIC_CUSTOMERS
  AML_SYNTHETIC_ACCOUNTS
  AML_SYNTHETIC_MERCHANTS
  AML_SYNTHETIC_TXNS
  AML_SYNTHETIC_AMOUNT_SCALE
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _random_dates(rng: np.random.Generator, rows: int, max_days: int) -> np.ndarray:
    base = datetime.now()
    days = rng.integers(0, max_days + 1, size=rows)
    out = [base - timedelta(days=int(d)) for d in days]
    return np.array([d.date().isoformat() for d in out], dtype=object)


def _random_datetimes(rng: np.random.Generator, rows: int, max_days: int) -> np.ndarray:
    base = datetime.now()
    days = rng.integers(0, max_days + 1, size=rows)
    seconds = rng.integers(0, 86400, size=rows)
    out = [base - timedelta(days=int(d), seconds=int(s)) for d, s in zip(days, seconds)]
    return np.array([d.strftime("%Y-%m-%d %H:%M:%S") for d in out], dtype=object)


def load() -> pd.DataFrame:
    """Return a pandas DataFrame that matches the aml_dataset view in aml_synthetic.sql."""
    seed = _env_int("AML_SYNTHETIC_SEED", 42)
    n_customers = _env_int("AML_SYNTHETIC_CUSTOMERS", 200)
    n_accounts = _env_int("AML_SYNTHETIC_ACCOUNTS", 300)
    n_merchants = _env_int("AML_SYNTHETIC_MERCHANTS", 80)
    n_txns = _env_int("AML_SYNTHETIC_TXNS", 2000)
    amount_scale = _env_float("AML_SYNTHETIC_AMOUNT_SCALE", 1.0)

    rng = np.random.default_rng(seed)

    countries = np.array(["US", "UK", "SG", "NG", "RU", "AE"], dtype=object)
    segments = np.array(["retail", "smb", "vip"], dtype=object)
    account_types = np.array(["checking", "savings", "wallet"], dtype=object)
    merchant_categories = np.array(
        ["groceries", "electronics", "travel", "gambling", "crypto_exchange", "money_service", "luxury", "utilities"],
        dtype=object,
    )
    risk_tiers = np.array(["low", "medium", "high"], dtype=object)
    currencies = np.array(["USD", "EUR", "GBP", "NGN", "RUB", "AED"], dtype=object)
    channels = np.array(["card", "bank_transfer", "wire", "cash", "crypto"], dtype=object)
    payment_memos = np.array(
        ["rent", "loan", "crypto", "invoice", "gift", "subscription", "refund", "UNKNOWN"],
        dtype=object,
    )
    device_types = np.array(["mobile", "web", "atm"], dtype=object)

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_customers + 1),
            "kyc_risk_score": np.round(rng.integers(0, 1000, size=n_customers) / 100.0, 2),
            "country": rng.choice(countries, size=n_customers),
            "is_pep": (rng.integers(0, 20, size=n_customers) == 0).astype(int),
            "segment": rng.choice(segments, size=n_customers),
            "onboarding_date": _random_dates(rng, n_customers, 2000),
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": np.arange(1, n_accounts + 1),
            "customer_id": rng.integers(1, n_customers + 1, size=n_accounts),
            "account_type": rng.choice(account_types, size=n_accounts),
            "balance": np.round(rng.integers(0, 100000, size=n_accounts) / 100.0, 2),
            "opened_date": _random_dates(rng, n_accounts, 1500),
        }
    )

    merchants = pd.DataFrame(
        {
            "merchant_id": np.arange(1, n_merchants + 1),
            "merchant_category": rng.choice(merchant_categories, size=n_merchants),
            "merchant_country": rng.choice(countries, size=n_merchants),
            "risk_tier": rng.choice(risk_tiers, size=n_merchants),
        }
    )

    transactions = pd.DataFrame(
        {
            "txn_id": np.arange(1, n_txns + 1),
            "account_id": rng.integers(1, n_accounts + 1, size=n_txns),
            "merchant_id": rng.integers(1, n_merchants + 1, size=n_txns),
            "txn_datetime": _random_datetimes(rng, n_txns, 365),
            "amount": np.round((rng.integers(0, 200000, size=n_txns) / 100.0) * amount_scale, 2),
            "currency": rng.choice(currencies, size=n_txns),
            "channel": rng.choice(channels, size=n_txns),
            "payment_memo": rng.choice(payment_memos, size=n_txns),
            "device_type": rng.choice(device_types, size=n_txns),
        }
    )

    df = (
        transactions.merge(accounts, on="account_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(merchants, on="merchant_id", how="left")
    )

    df["sar_actual"] = (
        (df["amount"] >= 10000)
        | ((df["channel"] == "crypto") & (df["amount"] >= 2000))
        | ((df["is_pep"] == 1) & (df["amount"] >= 3000))
        | ((df["risk_tier"] == "high") & (df["amount"] >= 5000))
        | ((df["kyc_risk_score"] >= 8.0) & (df["amount"] >= 4000))
    ).astype(int)

    df["is_suspicious"] = (
        (df["amount"] >= 8000)
        | ((df["channel"] == "wire") & (df["amount"] >= 4000))
        | ((df["risk_tier"] == "high") & (df["amount"] >= 3000))
    ).astype(int)

    columns = [
        "txn_id",
        "txn_datetime",
        "amount",
        "currency",
        "channel",
        "payment_memo",
        "device_type",
        "account_type",
        "balance",
        "customer_id",
        "kyc_risk_score",
        "customer_country",
        "is_pep",
        "segment",
        "merchant_category",
        "merchant_country",
        "risk_tier",
        "sar_actual",
        "is_suspicious",
    ]

    df = df.rename(columns={"country": "customer_country"})[columns]
    return df
