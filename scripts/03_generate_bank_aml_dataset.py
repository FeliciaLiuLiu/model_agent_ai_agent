
"""Generate a bank-like AML dataset (200,000 rows) with different columns from SAML-D-like dataset.

Includes FinTech rails/providers; suspicious rate = 4%.
Adds numeric-encoded categorical columns for interpretability workflows.
Output: ./data/synthetic_bank_aml_200k.csv (override with OUT_DIR)
"""

import argparse
import os
import uuid
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def make_bank_aml(
    n_rows: int = 200_000,
    suspicious_rate: float = 0.04,
    seed: int = 7,
    label_noise: float = 0.02,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    start = datetime(2025, 6, 1)

    countries = ['US','GB','DE','FR','ES','IT','NL','CH','AE','TR','MA','MX','CN','IN','BR','NG','SG','HK']
    country_w = [20,6,6,5,4,4,3,2,2,2,2,2,3,3,3,1,2,2]

    ccy_map = {
        'US':'USD','GB':'GBP','DE':'EUR','FR':'EUR','ES':'EUR','IT':'EUR','NL':'EUR','CH':'CHF',
        'AE':'AED','TR':'TRY','MA':'MAD','MX':'MXN','CN':'CNY','IN':'INR','BR':'BRL','NG':'NGN','SG':'SGD','HK':'HKD'
    }

    channels = ['mobile_app','web','branch','call_center','api']
    channel_w = [45,30,10,5,10]

    rails = ['ACH','wire','rtp','card','swift','sepa','paypal','venmo','zelle','cashapp','apple_pay','google_pay','stripe','adyen']
    rail_w = [18,12,8,10,6,6,10,6,6,5,4,4,3,2]

    txn_types = ['p2p_transfer','bill_pay','merchant_pay','cash_withdrawal','cash_deposit','loan_payment','crypto_exchange','international_remit']
    txn_w = [25,15,20,8,8,10,6,8]

    mcc = ['grocery','electronics','luxury','travel','gaming','crypto','charity','utilities','restaurants','rideshare']
    mcc_w = [15,10,6,8,8,4,5,14,18,12]

    device_types = ['ios','android','web','api']
    device_w = [30,35,25,10]

    # Stable categorical encodings for numeric-only modeling.
    country_code = {c: i for i, c in enumerate(countries)}
    channel_code = {c: i for i, c in enumerate(channels)}
    rail_code = {r: i for i, r in enumerate(rails)}
    txn_type_code = {t: i for i, t in enumerate(txn_types)}
    mcc_code = {m: i for i, m in enumerate(mcc)}
    device_code = {d: i for i, d in enumerate(device_types)}
    currency_list = sorted(set(ccy_map.values()))
    currency_code = {c: i for i, c in enumerate(currency_list)}

    cust = [f"CUST_{random.randint(1, 50000):06d}" for _ in range(n_rows)]
    acct = [f"ACCT_{random.randint(1, 80000):06d}" for _ in range(n_rows)]
    cp_acct = [f"CP_{random.randint(1, 120000):06d}" for _ in range(n_rows)]

    origin_country = random.choices(countries, weights=country_w, k=n_rows)
    dest_country = random.choices(countries, weights=country_w, k=n_rows)
    currency = [ccy_map.get(c,'USD') for c in origin_country]

    txn_channel = random.choices(channels, weights=channel_w, k=n_rows)
    payment_rail = random.choices(rails, weights=rail_w, k=n_rows)
    txn_type = random.choices(txn_types, weights=txn_w, k=n_rows)
    merchant_category = random.choices(mcc, weights=mcc_w, k=n_rows)
    device_type = random.choices(device_types, weights=device_w, k=n_rows)

    tx_time = [start + timedelta(seconds=int(np.random.exponential(scale=3600))) for _ in range(n_rows)]

    base = np.random.lognormal(mean=4.0, sigma=1.1, size=n_rows)
    amount = base
    for i in range(n_rows):
        if payment_rail[i] in ('wire','swift') or txn_type[i] == 'international_remit':
            amount[i] *= np.random.uniform(2, 15)
        if merchant_category[i] in ('luxury','travel'):
            amount[i] *= np.random.uniform(1.5, 6)
        if payment_rail[i] in ('paypal','venmo','zelle','cashapp','apple_pay','google_pay') and txn_type[i] == 'p2p_transfer':
            amount[i] *= np.random.uniform(0.6, 2.0)

    outlier = np.random.rand(n_rows) < 0.01
    amount[outlier] *= np.random.uniform(20, 120, size=outlier.sum())
    amount = np.round(amount, 2)

    account_age_days = np.random.randint(1, 3650, size=n_rows)
    kyc_risk_score = np.clip(np.random.normal(loc=35, scale=15, size=n_rows), 0, 100)
    num_txn_24h = np.random.poisson(lam=2.2, size=n_rows)
    avg_amount_7d = np.round(np.random.lognormal(mean=3.5, sigma=0.8, size=n_rows), 2)

    is_pep = (np.random.rand(n_rows) < 0.01).astype(int)
    sanctions_match = (np.random.rand(n_rows) < 0.002).astype(int)

    is_suspicious = (np.random.rand(n_rows) < suspicious_rate).astype(int)

    high_risk = {'MX','TR','MA','AE','NG'}
    fintech_set = {'paypal','venmo','zelle','cashapp','apple_pay','google_pay','stripe','adyen'}

    for i in range(n_rows):
        if is_suspicious[i] == 1:
            if random.random() < 0.7:
                origin_country[i] = random.choice(list(high_risk))
            if random.random() < 0.7:
                dest_country[i] = random.choice(list(high_risk))
            if random.random() < 0.6:
                payment_rail[i] = random.choice(list(fintech_set))
                txn_channel[i] = random.choice(['mobile_app','api','web'])
            if random.random() < 0.8:
                amount[i] = float(np.round(amount[i] * random.uniform(3, 40), 2))
            if random.random() < 0.6:
                num_txn_24h[i] = int(num_txn_24h[i] + np.random.randint(10, 80))
            if random.random() < 0.4:
                is_pep[i] = 1
            if random.random() < 0.2:
                sanctions_match[i] = 1
            kyc_risk_score[i] = float(min(100, kyc_risk_score[i] + random.uniform(20, 60)))

    is_cross_border = (np.array(origin_country) != np.array(dest_country)).astype(int)
    is_fintech_rail = np.array([1 if r in fintech_set else 0 for r in payment_rail]).astype(int)

    ip_country = random.choices(countries, weights=country_w, k=n_rows)

    # Encoded categorical columns for numeric modeling and interpretability.
    origin_country_code = [country_code[c] for c in origin_country]
    destination_country_code = [country_code[c] for c in dest_country]
    ip_country_code = [country_code[c] for c in ip_country]
    currency_code_col = [currency_code.get(c, 0) for c in currency]
    txn_channel_code = [channel_code[c] for c in txn_channel]
    payment_rail_code = [rail_code[c] for c in payment_rail]
    txn_type_code_col = [txn_type_code[c] for c in txn_type]
    merchant_category_code = [mcc_code[c] for c in merchant_category]
    device_type_code = [device_code[c] for c in device_type]

    if label_noise and label_noise > 0:
        flip = np.random.rand(n_rows) < label_noise
        is_suspicious = np.where(flip, 1 - is_suspicious, is_suspicious).astype(int)

    df = pd.DataFrame({
        'txn_id': [str(uuid.uuid4()) for _ in range(n_rows)],
        'txn_datetime': [t.strftime('%Y-%m-%d %H:%M:%S') for t in tx_time],
        'customer_id': cust,
        'account_id': acct,
        'counterparty_account_id': cp_acct,
        'txn_amount': amount,
        'currency': currency,
        'origin_country': origin_country,
        'destination_country': dest_country,
        'txn_channel': txn_channel,
        'payment_rail': payment_rail,
        'txn_type': txn_type,
        'merchant_category': merchant_category,
        'device_type': device_type,
        'device_id': [f"DEV_{random.randint(1, 200000):06d}" for _ in range(n_rows)],
        'ip_country': ip_country,
        'account_age_days': account_age_days,
        'kyc_risk_score': np.round(kyc_risk_score, 2),
        'num_txn_24h': num_txn_24h,
        'avg_amount_7d': avg_amount_7d,
        'is_pep': is_pep,
        'sanctions_match': sanctions_match,
        'is_cross_border': is_cross_border,
        'is_fintech_rail': is_fintech_rail,
        'origin_country_code': origin_country_code,
        'destination_country_code': destination_country_code,
        'ip_country_code': ip_country_code,
        'currency_code': currency_code_col,
        'txn_channel_code': txn_channel_code,
        'payment_rail_code': payment_rail_code,
        'txn_type_code': txn_type_code_col,
        'merchant_category_code': merchant_category_code,
        'device_type_code': device_type_code,
        'is_suspicious': is_suspicious,
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic bank AML dataset.")
    parser.add_argument('--out-dir', default=os.environ.get('OUT_DIR', './data'))
    parser.add_argument('--rows', type=int, default=int(os.environ.get('N_ROWS', 200_000)))
    parser.add_argument('--suspicious-rate', type=float, default=float(os.environ.get('SUSPICIOUS_RATE', 0.04)))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 7)))
    parser.add_argument('--label-noise', type=float, default=float(os.environ.get('LABEL_NOISE', 0.02)))
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'synthetic_bank_aml_200k.csv')
    df = make_bank_aml(args.rows, suspicious_rate=args.suspicious_rate, seed=args.seed, label_noise=args.label_noise)
    df.to_csv(out_path, index=False)
    print('Saved:', out_path)
    print(f"Rows={len(df)} suspicious_rate={args.suspicious_rate} label_noise={args.label_noise}")
    print(df.head(3))
