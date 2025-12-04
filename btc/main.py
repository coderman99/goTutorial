import logging
from pathlib import Path

import pandas as pd

from .database import DB_PATH, init_db, upsert_metrics
from .fetch import merged_market_and_sentiment
from .indicators import compute_indicators

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def build_dataset(years: int = 7) -> pd.DataFrame:
    """Fetch raw data, compute indicators, and return merged dataset."""

    logging.info("Fetching Bitcoin market and sentiment data")
    base_df = merged_market_and_sentiment(years=years)

    logging.info("Computing indicators")
    enriched_df = compute_indicators(base_df)

    # ensure consistent column order
    columns = [
        "date",
        "close",
        "ema_20",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "stoch_rsi",
        "fear_greed_index",
    ]
    return enriched_df[columns]


def save_to_db(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    logging.info("Saving %d rows to %s", len(df), db_path)
    init_db(db_path)
    upsert_metrics(df, db_path)


def main() -> None:
    dataset = build_dataset()
    save_to_db(dataset)
    logging.info("Data pipeline completed")


if __name__ == "__main__":
    main()
