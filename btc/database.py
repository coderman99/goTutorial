import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "btc_data.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS btc_metrics (
    date TEXT PRIMARY KEY,
    close REAL,
    ema_20 REAL,
    bb_middle REAL,
    bb_upper REAL,
    bb_lower REAL,
    stoch_rsi REAL,
    fear_greed_index REAL
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def upsert_metrics(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Insert or replace metrics into SQLite."""

    records: Iterable[tuple] = df[
        [
            "date",
            "close",
            "ema_20",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "stoch_rsi",
            "fear_greed_index",
        ]
    ].itertuples(index=False, name=None)

    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            """
            INSERT INTO btc_metrics (
                date, close, ema_20, bb_middle, bb_upper, bb_lower, stoch_rsi, fear_greed_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                close=excluded.close,
                ema_20=excluded.ema_20,
                bb_middle=excluded.bb_middle,
                bb_upper=excluded.bb_upper,
                bb_lower=excluded.bb_lower,
                stoch_rsi=excluded.stoch_rsi,
                fear_greed_index=excluded.fear_greed_index;
            """,
            list(records),
        )
        conn.commit()


__all__ = ["init_db", "upsert_metrics", "DB_PATH"]
