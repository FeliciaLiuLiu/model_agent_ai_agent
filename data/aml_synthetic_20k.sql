PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS aml_synthetic;

CREATE TABLE aml_synthetic (
  row_id INTEGER PRIMARY KEY,
  txn_id INTEGER,
  customer_id INTEGER,
  account_id INTEGER,
  merchant_id INTEGER,
  txn_datetime TEXT,
  amount REAL,
  currency TEXT,
  channel TEXT,
  payment_memo TEXT,
  device_type TEXT,
  merchant_category TEXT,
  merchant_country TEXT,
  customer_country TEXT,
  risk_tier TEXT,
  segment TEXT,
  account_type TEXT,
  balance REAL,
  kyc_risk_score REAL,
  is_pep INTEGER,
  is_foreign INTEGER,
  is_weekend INTEGER,
  hour_of_day INTEGER,
  day_of_week INTEGER,
  txn_count_24h INTEGER,
  txn_count_7d INTEGER,
  avg_amount_7d REAL,
  max_amount_7d REAL,
  min_amount_7d REAL,
  velocity_score REAL,
  geo_distance_km REAL,
  ip_risk_score REAL,
  chargeback_flag INTEGER,
  num_chargebacks_90d INTEGER,
  num_failed_logins_30d INTEGER,
  kyc_reviewed INTEGER,
  customer_tenure_months INTEGER,
  occupation TEXT,
  income_band TEXT,
  employment_status TEXT,
  education_level TEXT,
  payment_rail TEXT,
  card_present INTEGER,
  mcc_code TEXT,
  bank_country TEXT,
  beneficiary_country TEXT,
  beneficiary_type TEXT,
  transaction_type TEXT,
  notes TEXT,
  device_age_days INTEGER,
  account_age_days INTEGER,
  merchant_age_days INTEGER,
  high_risk_country INTEGER,
  sar_actual INTEGER,
  is_suspicious INTEGER,
  predicted_label INTEGER,
  amount_zscore REAL,
  balance_to_amount_ratio REAL,
  usd_amount REAL,
  fx_rate REAL
);

WITH RECURSIVE seq(x) AS (
  SELECT 1
  UNION ALL SELECT x + 1 FROM seq WHERE x <= 19800
),
base AS (
  SELECT
    x AS txn_id,
    (ABS(RANDOM()) % 4000) + 1 AS customer_id,
    (ABS(RANDOM()) % 6000) + 1 AS account_id,
    (ABS(RANDOM()) % 800) + 1 AS merchant_id,
    DATETIME('now', '-' || (ABS(RANDOM()) % 365) || ' days', '-' || (ABS(RANDOM()) % 86400) || ' seconds') AS txn_datetime,
    CASE
      WHEN x % 997 = 0 THEN 80000 + (ABS(RANDOM()) % 20000)
      WHEN x % 499 = 0 THEN 50000 + (ABS(RANDOM()) % 15000)
      ELSE ROUND((ABS(RANDOM()) % 500000) / 100.0 + 5, 2)
    END AS amount,
    CASE ABS(RANDOM()) % 8
      WHEN 0 THEN 'USD'
      WHEN 1 THEN 'EUR'
      WHEN 2 THEN 'GBP'
      WHEN 3 THEN 'NGN'
      WHEN 4 THEN 'RUB'
      WHEN 5 THEN 'AED'
      WHEN 6 THEN 'SGD'
      ELSE 'CAD'
    END AS currency,
    CASE ABS(RANDOM()) % 6
      WHEN 0 THEN 'card'
      WHEN 1 THEN 'bank_transfer'
      WHEN 2 THEN 'wire'
      WHEN 3 THEN 'cash'
      WHEN 4 THEN 'crypto'
      ELSE 'mobile'
    END AS channel,
    CASE ABS(RANDOM()) % 9
      WHEN 0 THEN 'rent'
      WHEN 1 THEN 'loan'
      WHEN 2 THEN 'crypto'
      WHEN 3 THEN 'invoice'
      WHEN 4 THEN 'gift'
      WHEN 5 THEN 'subscription'
      WHEN 6 THEN 'refund'
      WHEN 7 THEN 'salary'
      ELSE 'UNKNOWN'
    END AS payment_memo,
    CASE ABS(RANDOM()) % 4
      WHEN 0 THEN 'mobile'
      WHEN 1 THEN 'web'
      WHEN 2 THEN 'pos'
      ELSE 'atm'
    END AS device_type,
    CASE ABS(RANDOM()) % 10
      WHEN 0 THEN 'groceries'
      WHEN 1 THEN 'electronics'
      WHEN 2 THEN 'travel'
      WHEN 3 THEN 'gambling'
      WHEN 4 THEN 'crypto_exchange'
      WHEN 5 THEN 'money_service'
      WHEN 6 THEN 'luxury'
      WHEN 7 THEN 'utilities'
      WHEN 8 THEN 'healthcare'
      ELSE 'education'
    END AS merchant_category,
    CASE ABS(RANDOM()) % 8
      WHEN 0 THEN 'US'
      WHEN 1 THEN 'UK'
      WHEN 2 THEN 'SG'
      WHEN 3 THEN 'NG'
      WHEN 4 THEN 'RU'
      WHEN 5 THEN 'AE'
      WHEN 6 THEN 'PK'
      ELSE 'HK'
    END AS merchant_country,
    CASE ABS(RANDOM()) % 8
      WHEN 0 THEN 'US'
      WHEN 1 THEN 'UK'
      WHEN 2 THEN 'SG'
      WHEN 3 THEN 'NG'
      WHEN 4 THEN 'RU'
      WHEN 5 THEN 'AE'
      WHEN 6 THEN 'PK'
      ELSE 'HK'
    END AS customer_country,
    CASE ABS(RANDOM()) % 3
      WHEN 0 THEN 'low'
      WHEN 1 THEN 'medium'
      ELSE 'high'
    END AS risk_tier,
    CASE ABS(RANDOM()) % 4
      WHEN 0 THEN 'retail'
      WHEN 1 THEN 'smb'
      WHEN 2 THEN 'vip'
      ELSE 'enterprise'
    END AS segment,
    CASE ABS(RANDOM()) % 4
      WHEN 0 THEN 'checking'
      WHEN 1 THEN 'savings'
      WHEN 2 THEN 'wallet'
      ELSE 'credit'
    END AS account_type,
    ROUND((ABS(RANDOM()) % 8000000) / 100.0, 2) AS balance,
    ROUND((ABS(RANDOM()) % 1000) / 100.0, 2) AS kyc_risk_score,
    CASE WHEN ABS(RANDOM()) % 20 = 0 THEN 1 ELSE 0 END AS is_pep,
    CASE WHEN ABS(RANDOM()) % 4 = 0 THEN 1 ELSE 0 END AS is_foreign,
    (ABS(RANDOM()) % 20) AS txn_count_24h,
    (ABS(RANDOM()) % 120) AS txn_count_7d,
    ROUND((ABS(RANDOM()) % 400000) / 100.0, 2) AS avg_amount_7d,
    ROUND((ABS(RANDOM()) % 900000) / 100.0, 2) AS max_amount_7d,
    ROUND((ABS(RANDOM()) % 200000) / 100.0, 2) AS min_amount_7d,
    ROUND((ABS(RANDOM()) % 1000) / 100.0, 2) AS velocity_score,
    ROUND((ABS(RANDOM()) % 200000) / 10.0, 1) AS geo_distance_km,
    ROUND((ABS(RANDOM()) % 1000) / 10.0, 1) AS ip_risk_score,
    CASE WHEN ABS(RANDOM()) % 50 = 0 THEN 1 ELSE 0 END AS chargeback_flag,
    (ABS(RANDOM()) % 10) AS num_chargebacks_90d,
    (ABS(RANDOM()) % 15) AS num_failed_logins_30d,
    CASE WHEN ABS(RANDOM()) % 3 = 0 THEN 1 ELSE 0 END AS kyc_reviewed,
    (ABS(RANDOM()) % 120) AS customer_tenure_months,
    CASE ABS(RANDOM()) % 8
      WHEN 0 THEN 'engineer'
      WHEN 1 THEN 'teacher'
      WHEN 2 THEN 'doctor'
      WHEN 3 THEN 'trader'
      WHEN 4 THEN 'designer'
      WHEN 5 THEN 'student'
      WHEN 6 THEN 'retired'
      ELSE 'other'
    END AS occupation,
    CASE
      WHEN ABS(RANDOM()) % 12 = 0 THEN NULL
      ELSE CASE ABS(RANDOM()) % 5
        WHEN 0 THEN '<25k'
        WHEN 1 THEN '25-50k'
        WHEN 2 THEN '50-100k'
        WHEN 3 THEN '100-200k'
        ELSE '200k+'
      END
    END AS income_band,
    CASE ABS(RANDOM()) % 4
      WHEN 0 THEN 'employed'
      WHEN 1 THEN 'self_employed'
      WHEN 2 THEN 'unemployed'
      ELSE 'student'
    END AS employment_status,
    CASE ABS(RANDOM()) % 5
      WHEN 0 THEN 'highschool'
      WHEN 1 THEN 'bachelor'
      WHEN 2 THEN 'master'
      WHEN 3 THEN 'phd'
      ELSE 'other'
    END AS education_level,
    CASE ABS(RANDOM()) % 5
      WHEN 0 THEN 'card'
      WHEN 1 THEN 'ach'
      WHEN 2 THEN 'swift'
      WHEN 3 THEN 'local'
      ELSE 'crypto'
    END AS payment_rail,
    CASE WHEN ABS(RANDOM()) % 2 = 0 THEN 1 ELSE 0 END AS card_present,
    CASE ABS(RANDOM()) % 6
      WHEN 0 THEN '5411'
      WHEN 1 THEN '4812'
      WHEN 2 THEN '6011'
      WHEN 3 THEN '7995'
      WHEN 4 THEN '5732'
      ELSE '4899'
    END AS mcc_code,
    CASE ABS(RANDOM()) % 6
      WHEN 0 THEN 'US'
      WHEN 1 THEN 'UK'
      WHEN 2 THEN 'SG'
      WHEN 3 THEN 'NG'
      WHEN 4 THEN 'RU'
      ELSE 'AE'
    END AS bank_country,
    CASE ABS(RANDOM()) % 6
      WHEN 0 THEN 'US'
      WHEN 1 THEN 'UK'
      WHEN 2 THEN 'SG'
      WHEN 3 THEN 'NG'
      WHEN 4 THEN 'RU'
      ELSE 'AE'
    END AS beneficiary_country,
    CASE ABS(RANDOM()) % 4
      WHEN 0 THEN 'individual'
      WHEN 1 THEN 'business'
      WHEN 2 THEN 'charity'
      ELSE 'unknown'
    END AS beneficiary_type,
    CASE ABS(RANDOM()) % 5
      WHEN 0 THEN 'p2p'
      WHEN 1 THEN 'billpay'
      WHEN 2 THEN 'merchant_pay'
      WHEN 3 THEN 'cash_withdrawal'
      ELSE 'refund'
    END AS transaction_type,
    CASE
      WHEN ABS(RANDOM()) % 10 = 0 THEN NULL
      ELSE 'note_' || (ABS(RANDOM()) % 1000)
    END AS notes,
    (ABS(RANDOM()) % 2000) AS device_age_days,
    (ABS(RANDOM()) % 4000) AS account_age_days,
    (ABS(RANDOM()) % 5000) AS merchant_age_days
  FROM seq
)
INSERT INTO aml_synthetic (
  txn_id,
  customer_id,
  account_id,
  merchant_id,
  txn_datetime,
  amount,
  currency,
  channel,
  payment_memo,
  device_type,
  merchant_category,
  merchant_country,
  customer_country,
  risk_tier,
  segment,
  account_type,
  balance,
  kyc_risk_score,
  is_pep,
  is_foreign,
  is_weekend,
  hour_of_day,
  day_of_week,
  txn_count_24h,
  txn_count_7d,
  avg_amount_7d,
  max_amount_7d,
  min_amount_7d,
  velocity_score,
  geo_distance_km,
  ip_risk_score,
  chargeback_flag,
  num_chargebacks_90d,
  num_failed_logins_30d,
  kyc_reviewed,
  customer_tenure_months,
  occupation,
  income_band,
  employment_status,
  education_level,
  payment_rail,
  card_present,
  mcc_code,
  bank_country,
  beneficiary_country,
  beneficiary_type,
  transaction_type,
  notes,
  device_age_days,
  account_age_days,
  merchant_age_days,
  high_risk_country,
  sar_actual,
  is_suspicious,
  predicted_label,
  amount_zscore,
  balance_to_amount_ratio,
  usd_amount,
  fx_rate
)
SELECT
  txn_id,
  customer_id,
  account_id,
  merchant_id,
  txn_datetime,
  amount,
  currency,
  channel,
  payment_memo,
  device_type,
  merchant_category,
  merchant_country,
  customer_country,
  risk_tier,
  segment,
  account_type,
  balance,
  kyc_risk_score,
  is_pep,
  is_foreign,
  CASE WHEN CAST(strftime('%w', txn_datetime) AS integer) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
  CAST(strftime('%H', txn_datetime) AS integer) AS hour_of_day,
  CAST(strftime('%w', txn_datetime) AS integer) AS day_of_week,
  txn_count_24h,
  txn_count_7d,
  avg_amount_7d,
  max_amount_7d,
  min_amount_7d,
  velocity_score,
  geo_distance_km,
  ip_risk_score,
  chargeback_flag,
  num_chargebacks_90d,
  num_failed_logins_30d,
  kyc_reviewed,
  customer_tenure_months,
  occupation,
  income_band,
  employment_status,
  education_level,
  payment_rail,
  card_present,
  mcc_code,
  bank_country,
  beneficiary_country,
  beneficiary_type,
  transaction_type,
  notes,
  device_age_days,
  account_age_days,
  merchant_age_days,
  CASE
    WHEN customer_country IN ('NG', 'RU') OR merchant_country IN ('NG', 'RU') THEN 1
    ELSE 0
  END AS high_risk_country,
  CASE
    WHEN amount >= 10000 THEN 1
    WHEN channel = 'crypto' AND amount >= 2000 THEN 1
    WHEN is_pep = 1 AND amount >= 3000 THEN 1
    WHEN risk_tier = 'high' AND amount >= 5000 THEN 1
    WHEN kyc_risk_score >= 8.0 AND amount >= 4000 THEN 1
    ELSE 0
  END AS sar_actual,
  CASE
    WHEN amount >= 8000 THEN 1
    WHEN channel = 'wire' AND amount >= 4000 THEN 1
    WHEN ip_risk_score >= 80 THEN 1
    ELSE 0
  END AS is_suspicious,
  CASE WHEN ABS(RANDOM()) % 5 = 0 THEN 1 ELSE 0 END AS predicted_label,
  ROUND((amount - 500.0) / 200.0, 3) AS amount_zscore,
  ROUND(balance / (amount + 1.0), 3) AS balance_to_amount_ratio,
  ROUND(
    amount * (CASE currency
      WHEN 'USD' THEN 1.0
      WHEN 'EUR' THEN 1.1
      WHEN 'GBP' THEN 1.25
      WHEN 'NGN' THEN 0.0013
      WHEN 'RUB' THEN 0.012
      WHEN 'AED' THEN 0.27
      WHEN 'SGD' THEN 0.74
      WHEN 'CAD' THEN 0.75
      ELSE 1.0
    END),
    2
  ) AS usd_amount,
  (CASE currency
    WHEN 'USD' THEN 1.0
    WHEN 'EUR' THEN 1.1
    WHEN 'GBP' THEN 1.25
    WHEN 'NGN' THEN 0.0013
    WHEN 'RUB' THEN 0.012
    WHEN 'AED' THEN 0.27
    WHEN 'SGD' THEN 0.74
    WHEN 'CAD' THEN 0.75
    ELSE 1.0
  END) AS fx_rate
FROM base;

INSERT INTO aml_synthetic (
  txn_id,
  customer_id,
  account_id,
  merchant_id,
  txn_datetime,
  amount,
  currency,
  channel,
  payment_memo,
  device_type,
  merchant_category,
  merchant_country,
  customer_country,
  risk_tier,
  segment,
  account_type,
  balance,
  kyc_risk_score,
  is_pep,
  is_foreign,
  is_weekend,
  hour_of_day,
  day_of_week,
  txn_count_24h,
  txn_count_7d,
  avg_amount_7d,
  max_amount_7d,
  min_amount_7d,
  velocity_score,
  geo_distance_km,
  ip_risk_score,
  chargeback_flag,
  num_chargebacks_90d,
  num_failed_logins_30d,
  kyc_reviewed,
  customer_tenure_months,
  occupation,
  income_band,
  employment_status,
  education_level,
  payment_rail,
  card_present,
  mcc_code,
  bank_country,
  beneficiary_country,
  beneficiary_type,
  transaction_type,
  notes,
  device_age_days,
  account_age_days,
  merchant_age_days,
  high_risk_country,
  sar_actual,
  is_suspicious,
  predicted_label,
  amount_zscore,
  balance_to_amount_ratio,
  usd_amount,
  fx_rate
)
SELECT
  txn_id,
  customer_id,
  account_id,
  merchant_id,
  txn_datetime,
  amount,
  currency,
  channel,
  payment_memo,
  device_type,
  merchant_category,
  merchant_country,
  customer_country,
  risk_tier,
  segment,
  account_type,
  balance,
  kyc_risk_score,
  is_pep,
  is_foreign,
  is_weekend,
  hour_of_day,
  day_of_week,
  txn_count_24h,
  txn_count_7d,
  avg_amount_7d,
  max_amount_7d,
  min_amount_7d,
  velocity_score,
  geo_distance_km,
  ip_risk_score,
  chargeback_flag,
  num_chargebacks_90d,
  num_failed_logins_30d,
  kyc_reviewed,
  customer_tenure_months,
  occupation,
  income_band,
  employment_status,
  education_level,
  payment_rail,
  card_present,
  mcc_code,
  bank_country,
  beneficiary_country,
  beneficiary_type,
  transaction_type,
  notes,
  device_age_days,
  account_age_days,
  merchant_age_days,
  high_risk_country,
  sar_actual,
  is_suspicious,
  predicted_label,
  amount_zscore,
  balance_to_amount_ratio,
  usd_amount,
  fx_rate
FROM aml_synthetic
WHERE row_id <= 200;
