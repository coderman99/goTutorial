import pandas as pd

def compute_smoothed_returns(sp500):
    return sp500.pct_change(periods=3)

def label_business_cycle(df, sp500):
    # force tz-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    sp500.index = pd.to_datetime(sp500.index).tz_localize(None)

    # ensure end-of-month for BOTH
    df.index = df.index.to_period("M").to_timestamp("M")
    sp500.index = sp500.index.to_period("M").to_timestamp("M")

    sp500 = sp500.sort_index().copy()

    # compute returns
    sp500["returns"] = sp500["sp500"].pct_change()
    
    sp500['smoothed_returns'] = compute_smoothed_returns(sp500['returns'])

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

    # MERGE — will now work because timestamps align
    merged = df.merge(
        sp500[["returns", "cycle_phase"]],
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Hard-truncate the merged dataset so feature engineering never sees
    # synthetic early rows or backward-filled pct_change data. Anchor the
    # cutoff to the first available cycle label but never before 2000.
    min_cycle_label_date = merged.index.min()
    cutoff_date = max(pd.Timestamp("2000-01-01"), min_cycle_label_date)
    merged = merged[merged.index >= cutoff_date]

    return merged


def create_future_cycle_targets(labeled_df):
    """
    Adds 1m, 3m, and 6m future business cycle labels for supervised learning.
    """
    out = labeled_df.copy()

    # Shift cycle labels backward so future labels appear on current month
    out["cycle_1m"] = out["cycle_phase"].shift(-1)
    out["cycle_3m"] = out["cycle_phase"].shift(-3)
    out["cycle_6m"] = out["cycle_phase"].shift(-6)

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
