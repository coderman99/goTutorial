import pandas as pd
import numpy as np

# ------------------------------------------------------------
# WEEKLY RESAMPLING
# ------------------------------------------------------------

def _resample_group(g):
    """Resample a single indicator group to weekly."""
    g = g.sort_index()
    weekly = g.resample("W-FRI").last()
    weekly["value"] = weekly["value"].interpolate(method="linear")
    return weekly


def preprocess_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw indicator data → weekly format.
    """
    df = df.copy()
    df = df.sort_index()
    aggregated = df.groupby("name", group_keys=False).apply(_resample_group)
    aggregated["indicator_cat_code"] = aggregated["indicator_cat"].astype("category").cat.codes
    return aggregated


# ------------------------------------------------------------
# KEY INDICATOR BACKFILL
def backfill_missing_key_indicators(df: pd.DataFrame, key_ids: list) -> pd.DataFrame:
    """
    Ensures key indicators exist every week by backfilling missing rows.
    Handles:
      - duplicate (timestamp, series_id)
      - timestamp column collision on reset_index
      - non-unique MultiIndex errors
    """
    df = df.copy()

    # Ensure timestamp is index
    if df.index.name != "timestamp":
        df = df.set_index("timestamp")

    # --- FIX 1: Remove duplicate timestamp rows before grouping ---
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    after = len(df)
    if before != after:
        print(f"[Warning] Removed {before - after} duplicate timestamp rows before MultiIndex.")

    # --- FIX 2: If timestamp ALSO exists as a column, drop it BEFORE reset_index() ---
    if "timestamp" in df.columns:
        print("[Fix] Dropping duplicate 'timestamp' column before reset_index")
        df = df.drop(columns=["timestamp"])

    # --- Now safely convert to a flat table ---
    df = df.reset_index()

    # --- FIX 3: Drop duplicate (timestamp, series_id) pairs ---
    df = df.drop_duplicates(subset=["timestamp", "series_id"], keep="last")

    # --- Convert back into a MultiIndex ---
    df = df.set_index(["timestamp", "series_id"])

    # --- Build full MultiIndex grid (all timestamps × all key indicators) ---
    full_ts = df.index.get_level_values("timestamp").unique()
    full_series = key_ids

    idx = pd.MultiIndex.from_product(
        [full_ts, full_series],
        names=["timestamp", "series_id"]
    )

    # --- Reindex safely ---
    df = df.reindex(idx)

    # Interpolate indicator values
    if "value" in df.columns:
        df["value"] = df["value"].interpolate(method="linear")

    return df




# ------------------------------------------------------------
# MONTHLY WIDE FORMAT BUILDER
# ------------------------------------------------------------

def to_wide_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts MultiIndex (timestamp, series_id) weekly data into a monthly wide-format matrix.
    Steps:
      1. Flatten MultiIndex
      2. Resample monthly on timestamp
      3. Pivot to wide
    """
    df = df.copy()

    # --- FIX 1: If MultiIndex, flatten it ---
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- FIX 2: Set timestamp as only index ---
    df = df.set_index("timestamp")

    # --- FIX 3: Monthly resample (use 'ME' instead of deprecated 'M') ---
    df = df.resample("ME").last()

    # If missing values exist, forward-fill them
    df = df.ffill()

    # --- FIX 4: Pivot into wide format ---
    if "series_id" in df.columns and "value" in df.columns:
        wide = df.pivot_table(
            index=df.index,
            columns="series_id",
            values="value",
            aggfunc="last"
        )
    else:
        raise ValueError("Expected columns ['series_id','value'] before pivoting.")

    # Flatten pivot column hierarchy (series_id becomes column name)
    wide.columns = [str(c) for c in wide.columns]

    print(f"Monthly wide shape: {wide.shape}")
    return wide

