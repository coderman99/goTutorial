# preprocess.py
import pandas as pd

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
