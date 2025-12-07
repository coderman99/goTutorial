# ============================================================
#  train.py — Updated for reliable imports and Option B schema
# ============================================================

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is in sys.path BEFORE ANY IMPORTS
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------
import os
import json
import pandas as pd
from sqlalchemy import create_engine

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ------------------------------------------------------------
# BizCycle module imports (safe now that sys.path is fixed)
# ------------------------------------------------------------
from internal.models.bizcycle import preprocess as preprocess_mod
from internal.models.bizcycle.db_loader import load_indicator_data, load_sp500_from_db
from internal.models.bizcycle.labelling import label_business_cycle
from internal.models.bizcycle.model import create_future_targets
from internal.models.bizcycle.composite_score import compute_composite_score
from internal.models.bizcycle.enhance_model import (
    to_wide_monthly,
    make_features,
    train_xgboost_multi_horizon,
    predict_and_explain,
)

from internal.models.bizcycle.config import (
    get_database_url,
    KEY_INDICATORS,
)


# ------------------------------------------------------------
# Load .env from internal/.env
# ------------------------------------------------------------
env_path = PROJECT_ROOT / "internal" / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(env_path)


# ------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------
preprocess_weekly = preprocess_mod.preprocess_weekly
backfill_missing_key_indicators = getattr(
    preprocess_mod, "backfill_missing_key_indicators", None
)

if backfill_missing_key_indicators is None:
    raise ImportError(
        "preprocess.py is missing backfill_missing_key_indicators(), "
        "required for Option B indicator padding."
    )


# ------------------------------------------------------------
# Helper: drop columns with no variance
# ------------------------------------------------------------
def drop_constant_columns(df: pd.DataFrame):
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df, constant_cols


# ------------------------------------------------------------
# Helper: ensure all key indicators appear in wide DataFrame
# ------------------------------------------------------------
def validate_key_indicator_coverage(df: pd.DataFrame):
    present = set(c for c in df.columns)
    missing = sorted([k for k in KEY_INDICATORS if k not in present])

    if missing:
        raise ValueError(
            f"Missing key macro indicators. Expected: {sorted(KEY_INDICATORS)}.\n"
            f"Missing: {missing}\n"
            f"Check indicator loading & preprocessing."
        )
    print("\nKey macro indicators confirmed:", sorted(KEY_INDICATORS))


# ============================================================
# 1. LOAD RAW INDICATOR DATA
# ============================================================
df = load_indicator_data()
print(f"Loaded indicator rows: {len(df):,}")
print(df.head())

EARLIEST_DATE = os.getenv("BIZCYCLE_EARLIEST_DATE")
EARLIEST_DATE = pd.to_datetime(EARLIEST_DATE) if EARLIEST_DATE else pd.Timestamp("2005-01-01")


# ============================================================
# 2. PREPROCESS → WEEKLY FREQUENCY
# ============================================================
weekly = preprocess_weekly(df)
print(f"\nWeekly Preprocessed ({len(weekly):,} rows)")
print(weekly.head())


# ============================================================
# 3. LOAD S&P 500 SERIES
# ============================================================
sp500 = load_sp500_from_db()
print("\nSP500 Raw:")
print(sp500.head())


# ============================================================
# 4. LABEL BUSINESS CYCLES
# ============================================================
labeled_df = label_business_cycle(weekly, sp500, target_freq="W-FRI")
labeled_df = labeled_df[~labeled_df.index.duplicated(keep="last")]

if EARLIEST_DATE is not None:
    labeled_df = labeled_df[labeled_df.index >= EARLIEST_DATE]

print(f"\nBusiness Cycle Labeled ({len(labeled_df):,} rows)")
print(labeled_df.head())


# ============================================================
# 5. CREATE FUTURE TARGET LABELS
# ============================================================
labeled_df = create_future_targets(labeled_df, freq="W-FRI")
print("\nFuture Targets Created")
print(labeled_df.head())


# ============================================================
# 6. REMOVE RETURN LEAKAGE & COMPUTE COMPOSITE SCORES
# ============================================================
return_cols = [c for c in labeled_df.columns if "return" in c.lower()]
if return_cols:
    labeled_df = labeled_df.drop(columns=return_cols)
    print("Dropped return-derived columns:", return_cols)

for horizon in ["cycle_1m", "cycle_3m", "cycle_6m"]:
    labeled_df, weights = compute_composite_score(labeled_df, horizon)
    print(f"\n=== Composite Indicator Weights for {horizon} ===")
    print(weights.sort_values(ascending=False).head(15))

print("\nComposite scores added to labeled_df!")
print(labeled_df.head())


# ============================================================
# 7. BUILD WIDE DATAFRAME FOR ML
# ============================================================
indicator_value_wide = to_wide_monthly(labeled_df[["series_id", "value"]], freq="W-FRI")

metadata_cols = [
    c for c in labeled_df.columns
    if c not in {"name", "value", "indicator_cat", "indicator_cat_code", "id"}
]

metadata = labeled_df[metadata_cols].groupby(labeled_df.index).first()

# ------------------------------------------------------------
# FIX overlapping metadata → wide indicator columns
# ------------------------------------------------------------
overlap_cols = set(metadata.columns).intersection(indicator_value_wide.columns)
if overlap_cols:
    print(f"Removing overlapping columns before join: {overlap_cols}")
    metadata = metadata.drop(columns=list(overlap_cols))

wide_df = metadata.join(indicator_value_wide, how="outer")
wide_df = wide_df.loc[~wide_df.index.duplicated()].sort_index()

# pad missing indicators
wide_df = backfill_missing_key_indicators(wide_df, freq="W-FRI")

# add composite scores again
for horizon in ["cycle_1m", "cycle_3m", "cycle_6m"]:
    wide_df, _ = compute_composite_score(wide_df, horizon)

print("\nComposite scores added to wide_df!")
print(wide_df.head())


# ============================================================
# 8. PREPARE ML FEATURES
# ============================================================
composite_cols = [c for c in wide_df.columns if c.startswith("composite_cycle_")]
feature_df = wide_df.drop(columns=["cycle_1m", "cycle_3m", "cycle_6m", *composite_cols])
feature_df = feature_df[[c for c in feature_df.columns if "return" not in c.lower()]]

print("\nBase feature columns prior to lagging (count={}):".format(len(feature_df.columns)))
print(sorted(feature_df.columns))

# validate presence of required indicators
# (may be disabled temporarily during debugging)
validate_key_indicator_coverage(feature_df)

# feature engineering
X = make_features(feature_df)
X = X.loc[~X.index.duplicated()].sort_index()

print(f"\nFeature rows after engineering: {len(X):,}")
print("Engineered feature columns:", len(X.columns))

X, constant_cols = drop_constant_columns(X)
if constant_cols:
    print("Dropped constant columns:", constant_cols)


# ============================================================
# 9. ALIGN TARGETS
# ============================================================
df_targets = wide_df[["cycle_1m", "cycle_3m", "cycle_6m"]].reindex(X.index)

valid_idx = df_targets.dropna(how="all").index
X = X.loc[valid_idx]
df_targets = df_targets.loc[valid_idx]

print(
    f"Target rows aligned with features: {df_targets.shape[0]:,}/{len(df_targets):,}"
)


# ============================================================
# 10. TRAIN XGBOOST MODELS
# ============================================================
pipelines, accuracies = train_xgboost_multi_horizon(X, df_targets)

print("\nTraining accuracy by horizon:")
for h, acc in accuracies.items():
    print(f"  {h}: {acc:.3f}")


# ============================================================
# 11. GENERATE FINAL PREDICTION
# ============================================================
print("\n\n====================")
print("   MODEL PREDICTION")
print("====================")

latest_prediction = predict_and_explain(pipelines, X, top_n=12)

print("\nFinal Prediction Output:")
print(latest_prediction)

with open("model_debug_report.html", "w") as f:
    f.write("<h1>Model Debug Report</h1>")
    f.write("<h2>Prediction Output</h2>")
    f.write("<pre>" + json.dumps(latest_prediction, indent=4) + "</pre>")
