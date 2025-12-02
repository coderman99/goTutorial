create database indicators_db;

-- Main table for indicators
CREATE TABLE indicators (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    Category TEXT,
    FredSeriedID TEXT,
    Description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);

-- Table for indicator values (time series)
CREATE TABLE indicator_values (
    id SERIAL PRIMARY KEY,
    indicator_id INTEGER NOT NULL REFERENCES indicators(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);
