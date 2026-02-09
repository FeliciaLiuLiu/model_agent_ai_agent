PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS merchants;
DROP TABLE IF EXISTS transactions;

CREATE TABLE customers (
  customer_id     INTEGER PRIMARY KEY,
  kyc_risk_score  REAL,
  country         TEXT,
  is_pep          INTEGER,
  segment         TEXT,
  onboarding_date TEXT
);

CREATE TABLE accounts (
  account_id   INTEGER PRIMARY KEY,
  customer_id  INTEGER,
  account_type TEXT,
  balance      REAL,
  opened_date  TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE merchants (
  merchant_id       INTEGER PRIMARY KEY,
  merchant_category TEXT,
  merchant_country  TEXT,
  risk_tier         TEXT
);

CREATE TABLE transactions (
  txn_id        INTEGER PRIMARY KEY,
  account_id    INTEGER,
  merchant_id   INTEGER,
  txn_datetime  TEXT,
  amount        REAL,
  currency      TEXT,
  channel       TEXT,
  payment_memo  TEXT,
  device_type   TEXT,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id),
  FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);

WITH RECURSIVE seq(x) AS (
  SELECT 1
  UNION ALL SELECT x + 1 FROM seq WHERE x <= 200
)
INSERT INTO customers (customer_id, kyc_risk_score, country, is_pep, segment, onboarding_date)
SELECT
  x,
  ROUND((ABS(RANDOM()) % 1000) / 100.0, 2),
  CASE ABS(RANDOM()) % 6
    WHEN 0 THEN 'US'
    WHEN 1 THEN 'UK'
    WHEN 2 THEN 'SG'
    WHEN 3 THEN 'NG'
    WHEN 4 THEN 'RU'
    ELSE 'AE'
  END,
  CASE WHEN ABS(RANDOM()) % 20 = 0 THEN 1 ELSE 0 END,
  CASE ABS(RANDOM()) % 3
    WHEN 0 THEN 'retail'
    WHEN 1 THEN 'smb'
    ELSE 'vip'
  END,
  DATE('now', '-' || (ABS(RANDOM()) % 2000) || ' days')
FROM seq;

WITH RECURSIVE seq(x) AS (
  SELECT 1
  UNION ALL SELECT x + 1 FROM seq WHERE x <= 300
)
INSERT INTO accounts (account_id, customer_id, account_type, balance, opened_date)
SELECT
  x,
  (ABS(RANDOM()) % 200) + 1,
  CASE ABS(RANDOM()) % 3
    WHEN 0 THEN 'checking'
    WHEN 1 THEN 'savings'
    ELSE 'wallet'
  END,
  ROUND((ABS(RANDOM()) % 100000) / 100.0, 2),
  DATE('now', '-' || (ABS(RANDOM()) % 1500) || ' days')
FROM seq;

WITH RECURSIVE seq(x) AS (
  SELECT 1
  UNION ALL SELECT x + 1 FROM seq WHERE x <= 80
)
INSERT INTO merchants (merchant_id, merchant_category, merchant_country, risk_tier)
SELECT
  x,
  CASE ABS(RANDOM()) % 8
    WHEN 0 THEN 'groceries'
    WHEN 1 THEN 'electronics'
    WHEN 2 THEN 'travel'
    WHEN 3 THEN 'gambling'
    WHEN 4 THEN 'crypto_exchange'
    WHEN 5 THEN 'money_service'
    WHEN 6 THEN 'luxury'
    ELSE 'utilities'
  END,
  CASE ABS(RANDOM()) % 6
    WHEN 0 THEN 'US'
    WHEN 1 THEN 'UK'
    WHEN 2 THEN 'SG'
    WHEN 3 THEN 'NG'
    WHEN 4 THEN 'RU'
    ELSE 'AE'
  END,
  CASE ABS(RANDOM()) % 3
    WHEN 0 THEN 'low'
    WHEN 1 THEN 'medium'
    ELSE 'high'
  END
FROM seq;

WITH RECURSIVE seq(x) AS (
  SELECT 1
  UNION ALL SELECT x + 1 FROM seq WHERE x <= 2000
)
INSERT INTO transactions (txn_id, account_id, merchant_id, txn_datetime, amount, currency, channel, payment_memo, device_type)
SELECT
  x,
  (ABS(RANDOM()) % 300) + 1,
  (ABS(RANDOM()) % 80) + 1,
  DATETIME('now', '-' || (ABS(RANDOM()) % 365) || ' days', '-' || (ABS(RANDOM()) % 86400) || ' seconds'),
  ROUND((ABS(RANDOM()) % 200000) / 100.0, 2),
  CASE ABS(RANDOM()) % 6
    WHEN 0 THEN 'USD'
    WHEN 1 THEN 'EUR'
    WHEN 2 THEN 'GBP'
    WHEN 3 THEN 'NGN'
    WHEN 4 THEN 'RUB'
    ELSE 'AED'
  END,
  CASE ABS(RANDOM()) % 5
    WHEN 0 THEN 'card'
    WHEN 1 THEN 'bank_transfer'
    WHEN 2 THEN 'wire'
    WHEN 3 THEN 'cash'
    ELSE 'crypto'
  END,
  CASE ABS(RANDOM()) % 8
    WHEN 0 THEN 'rent'
    WHEN 1 THEN 'loan'
    WHEN 2 THEN 'crypto'
    WHEN 3 THEN 'invoice'
    WHEN 4 THEN 'gift'
    WHEN 5 THEN 'subscription'
    WHEN 6 THEN 'refund'
    ELSE 'UNKNOWN'
  END,
  CASE ABS(RANDOM()) % 3
    WHEN 0 THEN 'mobile'
    WHEN 1 THEN 'web'
    ELSE 'atm'
  END
FROM seq;

DROP VIEW IF EXISTS aml_dataset;
CREATE VIEW aml_dataset AS
SELECT
  t.txn_id,
  t.txn_datetime,
  t.amount,
  t.currency,
  t.channel,
  t.payment_memo,
  t.device_type,
  a.account_type,
  a.balance,
  c.customer_id,
  c.kyc_risk_score,
  c.country AS customer_country,
  c.is_pep,
  c.segment,
  m.merchant_category,
  m.merchant_country,
  m.risk_tier,
  CASE
    WHEN t.amount >= 10000 THEN 1
    WHEN t.channel = 'crypto' AND t.amount >= 2000 THEN 1
    WHEN c.is_pep = 1 AND t.amount >= 3000 THEN 1
    WHEN m.risk_tier = 'high' AND t.amount >= 5000 THEN 1
    WHEN c.kyc_risk_score >= 8.0 AND t.amount >= 4000 THEN 1
    ELSE 0
  END AS sar_actual,
  CASE
    WHEN t.amount >= 8000 THEN 1
    WHEN t.channel = 'wire' AND t.amount >= 4000 THEN 1
    WHEN m.risk_tier = 'high' AND t.amount >= 3000 THEN 1
    ELSE 0
  END AS is_suspicious
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
JOIN customers c ON a.customer_id = c.customer_id
JOIN merchants m ON t.merchant_id = m.merchant_id;
