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

from config import DATABASE_URL
from indicator_definitions import KEY_INDICATORS

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
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True).dt.tz_convert(
        None
    )
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
    # Base level and growth proxies that are non-leaking with respect to the
    # cycle labels (which are derived from future returns).
    pct_change = spx_df["sp500"].pct_change()
    rolling_change = pct_change.rolling(3).mean()
    rolling_vol = pct_change.rolling(3).std()

    synthetic_macro = {
        # Leading indicators should move with equity momentum without exposing
        # the exact return calculation used for labels.
        "Leading indicators index": rolling_change.mul(100),
        "PMI": rolling_change.mul(80).add(50),
        "NFIB sentiment": rolling_change.mul(60).add(95),
        "Yield curve": pct_change.rolling(6).mean().mul(10),
        "Credit spreads": rolling_vol.mul(150),
        # Coincident/lagging indicators are slower moving transformations.
        "Unemployment": rolling_vol.mul(30).add(4.5),
        "CPI": pct_change.rolling(6).mean().abs().mul(100),
        "IP": pct_change.rolling(4).mean().mul(120),
        "Housing starts": pct_change.rolling(5).mean().mul(140),
        "M2 YoY": pct_change.rolling(12).sum().mul(100),
        # Legacy stock features kept for backwards compatibility but filtered
        # out before training to avoid leakage.
        "StockMarketIndex": spx_df["sp500"],
        "StockMarketMomentum": pct_change.mul(100),
        "StockMarketVolatility": rolling_vol,
    }

    indicators = pd.DataFrame({"timestamp": spx_df["timestamp"]})
    for name, series in synthetic_macro.items():
        indicators[name] = series

    long_df = indicators.melt(
        id_vars=["timestamp"],
        var_name="name",
        value_name="value",
    )

    long_df["series_id"] = long_df["name"].str.upper()

    def _categorize(name: str) -> str:
        if name in {"Leading indicators index", "PMI", "Yield curve", "Credit spreads"}:
            return "Leading"
        if name in {"Unemployment", "CPI", "IP", "Housing starts"}:
            return "Lagging"
        if name in {"NFIB sentiment", "M2 YoY"}:
            return "Coincident"
        # Stock-derived proxies are treated as Leading but later filtered out
        # of the training feature set.
        return "Leading"

    long_df["indicator_cat"] = long_df["name"].apply(_categorize)
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], utc=True)
    long_df["id"] = range(1, len(long_df) + 1)

    return long_df.dropna(subset=["value"])


def _append_missing_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the returned indicator frame always includes the key macro set.

    Uses KEY_INDICATORS from indicator_definitions as the "wish list", but
    never overwrites real DB data. Any missing names that can be synthesized
    from the S&P history are appended; others are simply left missing.
    """
    if "name" not in df.columns:
        return df

    present = set(df["name"].unique())
    missing = KEY_INDICATORS - present
    if not missing:
        return df

    # Use the packaged SP500 history to synthesize proxies when the database
    # does not provide all macro indicators. Filter to only the missing set so
    # real DB columns are left untouched.
    spx_df = _load_local_sp500()
    synthetic = _build_sample_indicators(spx_df)
    synthetic = synthetic[synthetic["name"].isin(missing)]

    combined = pd.concat([df, synthetic], ignore_index=True)
    combined = combined.sort_values("timestamp")
    return combined


def load_indicator_data():
    """Load indicator data from the database or local CSV fallback."""
    engine = _get_engine()
    if engine:
        # Pull indicators from the dedicated table that already excludes S&P 500 rows
        df = pd.read_sql("SELECT * FROM econ_indicator_models", engine)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df = _append_missing_macro_indicators(df)
        # Keep timestamp as both column and index for consistent downstream
        # merges (e.g., YieldCurve alignment) without truncating history.
        return df.set_index("timestamp", drop=False)

    spx_df = _load_local_sp500()
    fallback = _build_sample_indicators(spx_df)
    fallback = _append_missing_macro_indicators(fallback)
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
            SELECT timestamp, close
            FROM sp500_models
            ORDER BY timestamp ASC
        """,
            engine,
        )

        if spx_db.empty:
            raise ValueError("ERROR: SP500 not found in DB. Check series_id or name.")

        sources.append(_normalize_sp500_monthly(spx_db.rename(columns={"close": "sp500"})))

    # Always include the packaged CSV so early history (1995-2009) is available
    # even when the database only contains recent daily observations.
    sources.append(_normalize_sp500_monthly(_load_local_sp500()))

    spx = pd.concat(sources).sort_index()
    spx = spx[~spx.index.duplicated(keep="first")]
    return spx[["sp500"]]
