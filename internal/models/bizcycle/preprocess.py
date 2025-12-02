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

    # reindex to timestamp
    monthly = monthly.set_index("timestamp")

    return monthly
