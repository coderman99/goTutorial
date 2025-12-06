import pandas as pd


def compute_smoothed_returns(sp500):
    return sp500.pct_change(periods=3)


def label_business_cycle(df, sp500, target_freq="M"):
    """
    Label business cycle phases.

    Parameters
    ----------
    df : DataFrame
        Indicator observations indexed by timestamp.
    sp500 : DataFrame
        S&P 500 levels indexed by timestamp (monthly or daily).
    target_freq : str
        Frequency to align labels to. Use "W-FRI" for end-of-week training.
    """

    # force tz-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    sp500.index = pd.to_datetime(sp500.index).tz_localize(None)

    # Normalize sp500 to month-end before applying business-cycle rules
    sp500 = sp500.sort_index().copy()
    sp500.index = sp500.index.to_period("M").to_timestamp("M")

    # compute returns
    sp500["returns"] = sp500["sp500"].pct_change()
    sp500["smoothed_returns"] = compute_smoothed_returns(sp500["returns"])

    # cycle label rules
    labels = []
    for r in sp500["returns"]:
        if pd.isna(r):
            labels.append(None)
        elif r > 0.005:
            labels.append("Expansion")
        elif r < -0.005:
            labels.append("Contraction")
        else:
            labels.append("Peak")

    sp500["cycle_phase"] = labels

    # Resample labels/returns to the requested target frequency so weekly
    # features can be aligned even when the underlying target is monthly-only.
    if target_freq != "M":
        sp500 = sp500.resample(target_freq).ffill()
        sp500.index = sp500.index.to_period(target_freq).to_timestamp(how="end")

    # Align df to the same frequency endpoints
    df.index = df.index.to_period(target_freq).to_timestamp(how="end")

    # MERGE — will now work because timestamps align
    merged = df.merge(
        sp500[["returns", "cycle_phase"]],
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Hard-truncate the merged dataset so feature engineering never sees
    # synthetic early rows or backward-filled pct_change data. Anchor the
    # cutoff to the first available cycle label but never before 2005 to match
    # the requested training window.
    min_cycle_label_date = merged.index.min()
    cutoff_date = max(pd.Timestamp("2005-01-01"), min_cycle_label_date)
    merged = merged[merged.index >= cutoff_date]

    return merged


def create_future_cycle_targets(labeled_df, target_freq="M"):
    """
    Adds 1m, 3m, and 6m future business cycle labels for supervised learning.
    """
    out = labeled_df.copy()

    if target_freq.startswith("W"):
        offsets = {"cycle_1m": 4, "cycle_3m": 12, "cycle_6m": 24}
    else:
        offsets = {"cycle_1m": 1, "cycle_3m": 3, "cycle_6m": 6}

    # Shift cycle labels backward so future labels appear on current timestamp
    for col, step in offsets.items():
        out[col] = out["cycle_phase"].shift(-step)

    # Drop rows where future labels are missing
    out = out.dropna(subset=["cycle_1m", "cycle_3m", "cycle_6m"], how="all")

    return out


# ============================================================
# Test block — run:  python labeling.py
# ============================================================

if __name__ == "__main__":
    print("=== Running labeling.py test ===")

    # Mock data for quick correctness check
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    
    df = pd.DataFrame({
        "name": ["Test"] * 12,
        "series_id": ["X"] * 12,
        "value": range(12),
        "indicator_cat": ["Leading"] * 12
    }, index=dates)

    spx = pd.DataFrame({
        "value": [100, 101, 103, 104, 103, 102, 101, 105, 108, 110, 109, 111]
    }, index=dates)

    labeled = label_business_cycle(df, spx)
    print("\n=== Labeled business cycle ===")
    print(labeled.head(10))

    final = create_future_cycle_targets(labeled)
    print("\n=== With future targets ===")
    print(final.head(10))
