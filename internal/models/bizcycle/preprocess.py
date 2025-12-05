# preprocess.py
import pandas as pd

def preprocess_monthly(df):
    """
    Convert raw indicator rows into end-of-month monthly data.
    Preserves indicator 'name' so wide pivoting works.
    """

    df = df.copy()

    # Drop exact duplicate rows (common when appending new history) so they
    # are not double-counted during monthly aggregation.
    df = df.drop_duplicates()

    # ensure tz-naive
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # convert to end-of-month
    df["timestamp"] = df["timestamp"].dt.to_period("M").dt.to_timestamp("M")

    # Sort so "keep='last'" is deterministic when removing duplicate months
    # from newly appended history.
    sort_cols = [c for c in ["timestamp", "name", "id"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last")

    # Remove duplicate monthly points per indicator, keeping the most recent
    # record for that month instead of averaging conflicting duplicates.
    df = df.drop_duplicates(subset=["timestamp", "name"], keep="last")

    # KEEP all information
    # (1) group by timestamp + name
    grouped = df.groupby(["timestamp", "name"])

    monthly = grouped.agg({
        "value": "mean",
        "indicator_cat": "first",
        "id": "first"
    }).reset_index()

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

    # reindex to timestamp
    monthly = monthly.set_index("timestamp")

    return monthly
