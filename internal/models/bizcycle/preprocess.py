# preprocess.py
import pandas as pd

from internal.models.bizcycle.indicator_definitions import KEY_INDICATORS

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


def backfill_missing_key_indicators(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Ensure the wide feature table always includes every economic indicator.

    The database contains ~30+ macro/market series defined alongside the Go
    structs. When some series have sparse history, we still want the model to
    see a stable column set so lagged features and composite scores align
    correctly. This helper adds missing indicator columns (filled with NA) and
    reindexes to the requested frequency to keep weekly alignment.
    """

    wide = df.copy()
    wide = wide.loc[~wide.index.duplicated()].sort_index()

    if freq and not wide.empty:
        full_index = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
        wide = wide.reindex(full_index)

    missing = [col for col in KEY_INDICATORS if col not in wide.columns]
    for col in missing:
        wide[col] = pd.NA

    # Preserve original column order while appending the newly added indicators
    # in a deterministic way for downstream reproducibility.
    existing_cols = [c for c in df.columns if c in wide.columns]
    appended_cols = sorted(set(missing))
    ordered_cols = existing_cols + [c for c in appended_cols if c not in existing_cols]
    wide = wide[ordered_cols]

    return wide

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
    sort_cols = [c for c in ["name", "timestamp", "id"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last")

    # Normalize using a per-indicator resample so we never overwrite earlier
    # history when new data ranges are appended.
    def _resample_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("timestamp").sort_index()
        resampled = group.resample(freq).last()
        resampled.index = resampled.index.to_period(freq).to_timestamp(how="end")
        # Keep earliest observation when history is appended later
        resampled = resampled[~resampled.index.duplicated(keep="first")]
        resampled["name"] = group["name"].iloc[0]
        return resampled

    aggregated = df.groupby("name", group_keys=False).apply(_resample_group)
    aggregated = aggregated.reset_index().rename(columns={"index": "timestamp"})

    category_map = {
        "Leading": 1,
        "Lagging": 2,
        "Coincident": 3,
        "Coincidental": 3
    }
    aggregated["indicator_cat_code"] = (
        aggregated["indicator_cat"]
        .map(category_map)
        .fillna(0)
        .astype(int)
    )

    aggregated = aggregated.set_index("timestamp")
    aggregated = aggregated[~aggregated.index.duplicated(keep="first")]
    aggregated = aggregated.sort_index()

    return aggregated
