"""Data loaders for the business cycle model.

The original version expected a running Postgres database populated with indicator
data. To keep the training script runnable in lightweight environments, these
helpers now fall back to the bundled ``sp500_monthly.csv`` when a database URL is
not configured. The fallback fabricates a few simple indicators derived from the
S&P 500 level so downstream feature engineering still has data to work with.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from .config import DATABASE_URL


def _get_engine():
    if not DATABASE_URL:
        return None

    try:
        return create_engine(DATABASE_URL)
    except Exception:
        return None


def _load_local_sp500():
    """Load the packaged S&P 500 monthly CSV as a pandas DataFrame."""

    csv_path = Path(__file__).with_name("sp500_monthly.csv")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)
    df = df.rename(columns={"SPX_Close": "sp500"})
    return df[["timestamp", "sp500"]].sort_values("timestamp")


def _normalize_sp500_monthly(df: pd.DataFrame, value_col: str = "sp500") -> pd.DataFrame:
    """Align raw SP500 rows to month-end without overwriting earlier history."""

    if df.empty:
        return pd.DataFrame(columns=["sp500"])

    aligned = df.copy()
    # Normalize timestamps to UTC then drop timezone info to avoid tz-aware
    # conversion errors when different sources (DB vs CSV) are combined.
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True).dt.tz_convert(None)
    aligned = aligned.rename(columns={value_col: "sp500"})
    aligned = aligned.set_index("timestamp").sort_index()
    aligned = aligned.resample("ME").last()
    aligned.index = aligned.index.to_period("M").to_timestamp("M")

    # keep the first occurrence of a month so newly appended ranges never
    # clobber earlier history (e.g., 1995-2009 from CSV vs. 2010+ from DB)
    aligned = aligned[~aligned.index.duplicated(keep="first")]
    return aligned[["sp500"]]


def _build_sample_indicators(spx_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a small set of indicators derived from S&P 500 levels."""

    indicators = pd.DataFrame({
        "timestamp": spx_df["timestamp"],
        "StockMarketIndex": spx_df["sp500"],
        "StockMarketMomentum": spx_df["sp500"].pct_change() * 100,
        "StockMarketVolatility": spx_df["sp500"].pct_change().rolling(3).std(),
    })

    long_df = indicators.melt(
        id_vars=["timestamp"],
        var_name="name",
        value_name="value",
    )

    long_df["series_id"] = long_df["name"].str.upper()
    long_df["indicator_cat"] = long_df["name"].apply(
        lambda x: "Leading" if "Momentum" in x or "Index" in x else "Lagging"
    )
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], utc=True)
    long_df["id"] = range(1, len(long_df) + 1)

    return long_df.dropna(subset=["value"])


def load_indicator_data():
    """Load indicator data from the database or local CSV fallback."""

    engine = _get_engine()
    if engine:
        df = pd.read_sql("SELECT * FROM indicator_models", engine)

        # Remove SP500 because we will load it separately
        df = df[df["series_id"] != "SP500"]

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        # Keep timestamp as both column and index for consistent downstream
        # merges (e.g., YieldCurve alignment) without truncating history.
        return df.set_index("timestamp", drop=False)

    spx_df = _load_local_sp500()
    fallback = _build_sample_indicators(spx_df)
    return fallback.set_index("timestamp", drop=False)


def load_sp500_from_db():
    """
    Load SP500 from the database or the packaged CSV.
    Converts daily → monthly frequency like your script expected.
    """

    engine = _get_engine()
    sources = []

    if engine:
        spx_db = pd.read_sql(
            """
            SELECT *
            FROM indicator_models
            WHERE series_id = 'SP500'
            ORDER BY timestamp ASC
        """,
            engine,
        )

        if spx_db.empty:
            raise ValueError("ERROR: SP500 not found in DB. Check series_id or name.")

        sources.append(_normalize_sp500_monthly(spx_db.rename(columns={"value": "sp500"})))

    # Always include the packaged CSV so early history (1995-2009) is available
    # even when the database only contains recent daily observations.
    sources.append(_normalize_sp500_monthly(_load_local_sp500()))

    spx = pd.concat(sources).sort_index()
    spx = spx[~spx.index.duplicated(keep="first")]
    return spx[["sp500"]]
