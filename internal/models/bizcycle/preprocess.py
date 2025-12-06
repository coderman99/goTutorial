# preprocess.py
import pandas as pd

from .db_loader import synthesize_key_indicators

# Lazy import to avoid circular dependency at module import time
# (to_wide_monthly and KEY_INDICATORS live in enhance_model.py).


CANONICAL_INDICATOR_ALIASES = {
    "unemployment rate": "Unemployment",
    "unemployment": "Unemployment",
    "pmi": "PMI",
    "purchasing managers index": "PMI",
    "industrial production": "IP",
    "ip": "IP",
    "cpi": "CPI",
    "inflation": "CPI",
    "consumer price index": "CPI",
    "housing starts": "Housing starts",
    "yield curve": "Yield curve",
    "10y-2y spread": "Yield curve",
    "credit spreads": "Credit spreads",
    "credit spread": "Credit spreads",
    "leading indicators": "Leading indicators index",
    "leading index": "Leading indicators index",
    "nfib sentiment": "NFIB sentiment",
    "nfib": "NFIB sentiment",
    "m2 yoy": "M2 YoY",
    "m2": "M2 YoY",
}


def _canonicalize_indicator_name(name: str) -> str:
    if not isinstance(name, str):
        return name

    key = name.strip().lower()
    return CANONICAL_INDICATOR_ALIASES.get(key, name)

def preprocess_monthly(df):
    """
    Convert raw indicator rows into end-of-month monthly data.
    Preserves indicator 'name' so wide pivoting works.
    """

    return _resample_indicators(df, freq="ME")


def preprocess_weekly(df):
    """Convert raw indicator rows into end-of-week (Friday) data."""

    return _resample_indicators(df, freq="W-FRI")


def _resample_indicators(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Shared resampling helper for monthly or weekly aggregation."""

    df = df.copy()

    # If callers provided a timestamp index *and* a timestamp column, drop the
    # index to avoid ambiguous lookups when sorting. The timestamp column is the
    # single source of truth for downstream resampling.
    if "timestamp" in df.index.names:
        df = df.reset_index(drop=True)

    # Drop exact duplicate rows (common when appending new history) so they
    # are not double-counted during aggregation.
    df = df.drop_duplicates()

    # ensure tz-naive
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Normalize indicator naming so downstream pivots treat equivalent series
    # (e.g., "Unemployment Rate" vs. "Unemployment") as the same column.
    df["name"] = df["name"].apply(_canonicalize_indicator_name)

    # Sort before resampling to make "last" deterministic when multiple points
    # land in the same bucket (e.g., daily SP500 history vs. legacy monthly rows).
    sort_cols = [c for c in ["timestamp", "name", "id"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last")

    # Preserve per-indicator metadata (series_id + category) so we can
    # reconstruct a long-form table after resampling a wide pivot.
    meta_cols = [c for c in ["series_id", "indicator_cat"] if c in df.columns]
    metadata = df.groupby("name")[meta_cols].first() if meta_cols else None

    # Pivot to wide, resample once, then stack back to long to avoid losing
    # indicators due to per-group resample quirks.
    wide = df.pivot_table(
        index="timestamp",
        columns="name",
        values="value",
        aggfunc="last",
    ).sort_index()

    resampled = wide.resample(freq).last()
    resampled.index = resampled.index.to_period(freq).to_timestamp(how="end")
    resampled = resampled[~resampled.index.duplicated(keep="first")]

    long_df = resampled.stack(dropna=False).reset_index()
    long_df = long_df.rename(columns={"level_1": "name", 0: "value"})

    if metadata is not None:
        long_df = long_df.merge(
            metadata.reset_index(),
            on="name",
            how="left",
        )

    category_map = {
        "Leading": 1,
        "Lagging": 2,
        "Coincident": 3,
        "Coincidental": 3,
    }
    long_df["indicator_cat_code"] = (
        long_df.get("indicator_cat")
        .map(category_map)
        .fillna(0)
        .astype(int)
    )

    # Keep the latest observation for each (timestamp, name) pair without
    # discarding other indicators that share the same timestamp.
    long_df = long_df.drop_duplicates(subset=["timestamp", "name"], keep="last")
    long_df = long_df.set_index("timestamp")
    long_df = long_df.sort_index()

    return long_df


def backfill_missing_key_indicators(wide_df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Add synthetic key indicator columns when upstream data is incomplete.

    Keeping this helper in ``preprocess`` makes it reusable from both the training
    script and any data quality checks without depending on where it is invoked.
    """

    # Import lazily to avoid circular dependency at module import time.
    from .enhance_model import KEY_INDICATORS, to_wide_monthly

    present = {col for col in wide_df.columns if col in KEY_INDICATORS}
    missing = KEY_INDICATORS - present
    if not missing:
        return wide_df

    synthetic_long = synthesize_key_indicators(freq=freq)
    synthetic_long = synthetic_long[synthetic_long["name"].isin(missing)]

    value_wide = to_wide_monthly(synthetic_long[["name", "value"]], freq=freq)
    cat_wide = synthetic_long.pivot_table(
        index=synthetic_long.index,
        columns="name",
        values="indicator_cat_code",
        aggfunc="first",
    ).add_suffix("_catcode")

    wide_df = wide_df.join(value_wide, how="left")
    wide_df = wide_df.join(cat_wide, how="left")

    return wide_df
