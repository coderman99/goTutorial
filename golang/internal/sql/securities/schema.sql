-- ==========================
-- SECTOR
-- ==========================
CREATE TABLE sectors (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE sector_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    sector_id INTEGER REFERENCES sectors(id) ON DELETE CASCADE,
    date TEXT,
    avg_price DOUBLE PRECISION,
    total_volume BIGINT,
    rsi DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    returns DOUBLE PRECISION
);

CREATE TABLE sector_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    sector_id INTEGER REFERENCES sectors(id) ON DELETE CASCADE,
    period TEXT,
    avg_pe DOUBLE PRECISION,
    avg_eps DOUBLE PRECISION,
    total_market_cap DOUBLE PRECISION,
    avg_debt_to_equity DOUBLE PRECISION,
    avg_roe DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION
);

-- ==========================
-- INDUSTRY
-- ==========================
CREATE TABLE industries (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT UNIQUE NOT NULL,
    sector_id INTEGER REFERENCES sectors(id) ON DELETE CASCADE
);

CREATE TABLE industry_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    industry_id INTEGER REFERENCES industries(id) ON DELETE CASCADE,
    date TEXT,
    avg_price DOUBLE PRECISION,
    total_volume BIGINT,
    rsi DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    returns DOUBLE PRECISION
);

CREATE TABLE industry_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    industry_id INTEGER REFERENCES industries(id) ON DELETE CASCADE,
    period TEXT,
    avg_pe DOUBLE PRECISION,
    avg_eps DOUBLE PRECISION,
    total_market_cap DOUBLE PRECISION,
    avg_debt_to_equity DOUBLE PRECISION,
    avg_roe DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION
);

-- ==========================
-- EQUITY
-- ==========================
CREATE TABLE equities (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    ticker TEXT UNIQUE NOT NULL,
    industry_id INTEGER REFERENCES industries(id) ON DELETE CASCADE
);

CREATE TABLE equity_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    equity_id INTEGER REFERENCES equities(id) ON DELETE CASCADE,
    closing_price DOUBLE PRECISION,
    volume BIGINT,
    stochastic_rsi DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    bollinger_bands TEXT,
    volatility DOUBLE PRECISION
);

CREATE TABLE equity_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    equity_id INTEGER REFERENCES equities(id) ON DELETE CASCADE,
    pe DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    cash_flow DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    roa DOUBLE PRECISION,
    book_value_ps DOUBLE PRECISION
);

-- ==========================
-- BONDS
-- ==========================
CREATE TABLE bonds (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    issuer TEXT,
    type TEXT
);

CREATE TABLE bond_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    bond_id INTEGER REFERENCES bonds(id) ON DELETE CASCADE,
    price DOUBLE PRECISION,
    yield_to_maturity DOUBLE PRECISION,
    current_yield DOUBLE PRECISION,
    duration DOUBLE PRECISION,
    convexity DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    volume BIGINT
);

CREATE TABLE bond_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    bond_id INTEGER REFERENCES bonds(id) ON DELETE CASCADE,
    coupon_rate DOUBLE PRECISION,
    issue_date TEXT,
    maturity_date TEXT,
    credit_rating TEXT,
    callable BOOLEAN,
    puttable BOOLEAN,
    default_probability DOUBLE PRECISION
);

-- ==========================
-- COMMODITIES
-- ==========================
CREATE TABLE commodities (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    ticker TEXT
);

CREATE TABLE commodity_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    commodity_id INTEGER REFERENCES commodities(id) ON DELETE CASCADE,
    spot_price DOUBLE PRECISION,
    futures_curve TEXT,
    volume BIGINT,
    open_interest BIGINT,
    rsi DOUBLE PRECISION,
    volatility DOUBLE PRECISION
);

CREATE TABLE commodity_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    commodity_id INTEGER REFERENCES commodities(id) ON DELETE CASCADE,
    supply DOUBLE PRECISION,
    demand DOUBLE PRECISION,
    inventories DOUBLE PRECISION,
    seasonality TEXT,
    production_cost DOUBLE PRECISION,
    geopolitical_risk TEXT
);

-- ==========================
-- CRYPTO
-- ==========================
CREATE TABLE cryptos (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    ticker TEXT
);

CREATE TABLE crypto_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    crypto_id INTEGER REFERENCES cryptos(id) ON DELETE CASCADE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    volatility DOUBLE PRECISION
);

CREATE TABLE crypto_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    crypto_id INTEGER REFERENCES cryptos(id) ON DELETE CASCADE,
    market_cap DOUBLE PRECISION,
    circulating_supply DOUBLE PRECISION,
    total_supply DOUBLE PRECISION,
    hash_rate DOUBLE PRECISION,
    transaction_volume DOUBLE PRECISION,
    active_addresses BIGINT,
    staking_yield DOUBLE PRECISION,
    developer_activity BIGINT
);

-- ==========================
-- CURRENCIES
-- ==========================
CREATE TABLE currencies (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    ticker TEXT
);

CREATE TABLE currency_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    currency_id INTEGER REFERENCES currencies(id) ON DELETE CASCADE,
    exchange_rate DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    ema DOUBLE PRECISION
);

CREATE TABLE currency_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    currency_id INTEGER REFERENCES currencies(id) ON DELETE CASCADE,
    interest_rate DOUBLE PRECISION,
    inflation DOUBLE PRECISION,
    trade_balance DOUBLE PRECISION,
    gdp_growth DOUBLE PRECISION,
    policy_rate DOUBLE PRECISION,
    sovereign_risk TEXT
);

-- ==========================
-- ETFs
-- ==========================
CREATE TABLE etfs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    name TEXT NOT NULL,
    ticker TEXT
);

CREATE TABLE etf_technicals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    etf_id INTEGER REFERENCES etfs(id) ON DELETE CASCADE,
    date TEXT,
    price DOUBLE PRECISION,
    volume BIGINT,
    rsi DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    volatility DOUBLE PRECISION
);

CREATE TABLE etf_fundamentals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    etf_id INTEGER REFERENCES etfs(id) ON DELETE CASCADE,
    period TEXT,
    expense_ratio DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION,
    aum DOUBLE PRECISION,
    tracking_error DOUBLE PRECISION,
    sector_exposure TEXT
);

CREATE TABLE etf_holdings (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP,
    etf_id INTEGER REFERENCES etfs(id) ON DELETE CASCADE,
    equity_id INTEGER REFERENCES equities(id) ON DELETE CASCADE,
    period TEXT,
    weight DOUBLE PRECISION,
    shares_held DOUBLE PRECISION,
    market_value DOUBLE PRECISION
);
