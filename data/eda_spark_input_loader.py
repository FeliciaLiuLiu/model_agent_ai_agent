"""Small synthetic dataset for EDA Spark input tests."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _random_datetimes(rng: np.random.Generator, rows: int, max_days: int) -> list[str]:
    base = datetime.now()
    days = rng.integers(0, max_days + 1, size=rows)
    minutes = rng.integers(0, 24 * 60, size=rows)
    out = [base - timedelta(days=int(d), minutes=int(m)) for d, m in zip(days, minutes)]
    return [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in out]


def load() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = 1500

    df = pd.DataFrame(
        {
            "txn_id": np.arange(1, rows + 1),
            "txn_datetime": _random_datetimes(rng, rows, 540),
            "amount": np.round(rng.gamma(2.0, 1500, size=rows), 2),
            "channel": rng.choice(["card", "bank_transfer", "wire", "cash", "crypto"], size=rows),
            "customer_country": rng.choice(["US", "UK", "SG", "NG", "AE", "RU"], size=rows),
            "segment": rng.choice(["retail", "smb", "vip"], size=rows),
            "risk_score": np.round(rng.normal(5.5, 2.1, size=rows).clip(0, 10), 2),
            "velocity_score": np.round(rng.normal(50.0, 15.0, size=rows).clip(0, 100), 2),
            "device_type": rng.choice(["mobile", "web", "atm"], size=rows),
        }
    )

    df["sar_actual"] = (
        (df["amount"] >= 8500)
        | ((df["channel"] == "crypto") & (df["amount"] >= 2200))
        | ((df["channel"] == "wire") & (df["risk_score"] >= 7.2))
        | ((df["velocity_score"] >= 85) & (df["amount"] >= 3000))
    ).astype(int)

    miss_idx = rng.choice(df.index, size=int(rows * 0.02), replace=False)
    df.loc[miss_idx, "device_type"] = None
    return df
