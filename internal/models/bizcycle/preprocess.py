# preprocess.py
import pandas as pd

def preprocess_monthly(df):
    """
    Convert raw indicator rows into end-of-month monthly data.
    Preserves indicator 'name' so wide pivoting works.
    """

    df = df.copy()

    # If callers provided a timestamp index *and* a timestamp column, drop the
    # index to avoid ambiguous lookups when sorting. The timestamp column is the
    # single source of truth for downstream resampling.
    if "timestamp" in df.index.names:
        df = df.reset_index(drop=True)

    # Drop exact duplicate rows (common when appending new history) so they
    # are not double-counted during monthly aggregation.
    df = df.drop_duplicates()

    # ensure tz-naive
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Sort before resampling to make "last" deterministic when multiple points
    # land in the same month (e.g., daily SP500 history vs. legacy monthly rows).
    sort_cols = [c for c in ["name", "timestamp", "id"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last")

    # Normalize to month-end using a per-indicator resample so we never
    # overwrite earlier history when new data ranges are appended.
    def _resample_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("timestamp").sort_index()
        resampled = group.resample("ME").last()
        resampled.index = resampled.index.to_period("M").to_timestamp("M")
        # Keep earliest monthly observation when history is appended later
        resampled = resampled[~resampled.index.duplicated(keep="first")]
        resampled["name"] = group["name"].iloc[0]
        return resampled

    monthly = df.groupby("name", group_keys=False).apply(_resample_group)
    monthly = monthly.reset_index().rename(columns={"index": "timestamp"})

    category_map = {
        "Leading": 1,
        "Lagging": 2,
        "Coincident": 3,
        "Coincidental": 3
        
    }
    monthly["indicator_cat_code"] = (
        monthly["indicator_cat"]
        .map(category_map)
        .fillna(0)
        .astype(int)
    )

    # Ensure strict month-end ordering without clobbering earlier history
    monthly = monthly.set_index("timestamp")
    monthly = monthly[~monthly.index.duplicated(keep="first")]
    monthly = monthly.sort_index()

    return monthly
