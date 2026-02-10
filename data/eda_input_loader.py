"""Small synthetic dataset for EDA input tests."""
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
    rng = np.random.default_rng(7)
    rows = 1200

    df = pd.DataFrame(
        {
            "txn_id": np.arange(1, rows + 1),
            "txn_datetime": _random_datetimes(rng, rows, 365),
            "amount": np.round(rng.gamma(2.2, 1800, size=rows), 2),
            "channel": rng.choice(["card", "bank_transfer", "wire", "cash", "crypto"], size=rows),
            "customer_country": rng.choice(["US", "UK", "SG", "NG", "AE"], size=rows),
            "segment": rng.choice(["retail", "smb", "vip"], size=rows),
            "risk_score": np.round(rng.normal(5.0, 2.0, size=rows).clip(0, 10), 2),
            "is_pep": (rng.random(rows) < 0.06).astype(int),
            "payment_memo": rng.choice(["rent", "invoice", "crypto", "refund", "gift", "UNKNOWN"], size=rows),
        }
    )

    df["sar_actual"] = (
        (df["amount"] >= 9000)
        | ((df["channel"] == "crypto") & (df["amount"] >= 2500))
        | ((df["channel"] == "wire") & (df["risk_score"] >= 7.5))
        | ((df["is_pep"] == 1) & (df["amount"] >= 3500))
    ).astype(int)

    miss_idx = rng.choice(df.index, size=int(rows * 0.03), replace=False)
    df.loc[miss_idx, "payment_memo"] = None
    return df
