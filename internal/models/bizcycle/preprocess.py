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
    df = df.set_index("timestamp")
    monthly = df.groupby("name").resample("ME").last()

    # Force month-end timestamps after resampling for consistent merging
    monthly.index = monthly.index.set_levels(
        monthly.index.levels[1].to_period("M").to_timestamp("M"), level=1
    )
    monthly.index = monthly.index.rename(["name", "timestamp"])
    monthly = monthly.reset_index("name")

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
    monthly = monthly[~monthly.index.duplicated(keep="last")]
    monthly = monthly.sort_index()

    return monthly
