# preprocess.py
import pandas as pd

def preprocess_monthly(df):
    """
    Convert raw indicator rows into end-of-month monthly data.
    Preserves indicator 'name' so wide pivoting works.
    """

    df = df.copy()

    # ensure tz-naive
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # convert to end-of-month
    df["timestamp"] = df["timestamp"].dt.to_period("M").dt.to_timestamp("M")

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
        "Coincident": 2,
        "Coincidental": 2,
        "Lagging": 3
        
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
