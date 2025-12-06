import os
from pathlib import Path
import sys

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from internal.models.bizcycle.db_loader import load_indicator_data, load_sp500_from_db
from internal.models.bizcycle.preprocess import preprocess_monthly
from internal.models.bizcycle.labelling import label_business_cycle
from internal.models.bizcycle.model import create_future_targets
from internal.models.bizcycle.composite_score import compute_composite_score
from internal.models.bizcycle.enhance_model import (
    to_wide_monthly,
    make_features,
    train_lightgbm_multi_horizon,
    predict_and_explain,
)
from internal.models.bizcycle.config import get_database_url

# ---------------------------------------------
# Load .env
# ---------------------------------------------
env_path = Path(__file__).resolve().parents[3] / "internal" / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(env_path)


def drop_constant_columns(df: pd.DataFrame):
    """Remove columns that carry no signal before model training."""

    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df, constant_cols

# ---------------------------------------------
# 1. Load indicator data
# ---------------------------------------------
# Allow callers to keep the full history by omitting the env var.
EARLIEST_DATE = os.getenv("BIZCYCLE_EARLIEST_DATE")
EARLIEST_DATE = pd.to_datetime(EARLIEST_DATE) if EARLIEST_DATE else None

df = load_indicator_data()
print(f"Loaded indicator rows: {len(df):,}")
print(df.head())

# ---------------------------------------------
# 2. Convert indicators → Monthly frequency
# ---------------------------------------------
monthly = preprocess_monthly(df)
print(f"\nMonthly Preprocessed ({len(monthly):,} rows)")
print(monthly.head())

# ---------------------------------------------
# 3. Load SP500 from database (or fallback CSV)
# ---------------------------------------------
db_url = get_database_url()
engine = create_engine(db_url) if db_url else None
sp500 = load_sp500_from_db()
print("\nSP500 Raw:")
print(sp500.head())

# ---------------------------------------------
# 4. Label business cycle phases
# ---------------------------------------------
labeled_df = label_business_cycle(monthly, sp500)
labeled_df = labeled_df[~labeled_df.index.duplicated(keep="last")]

if EARLIEST_DATE is not None:
    labeled_df = labeled_df[labeled_df.index >= EARLIEST_DATE]

print(f"\nBusiness Cycle Labeled ({len(labeled_df):,} rows)")
print(labeled_df.head())

# ---------------------------------------------
# 5. Create future prediction targets (1m, 3m, 6m)
# ---------------------------------------------
labeled_df = create_future_targets(labeled_df)
print("\nFuture Targets Created")
print(labeled_df.head())

# ---------------------------------------------
# 6. Compute Composite Scores for each horizon
# ---------------------------------------------
# Remove stock/return-derived columns before computing composite scores to ensure
# the weights only reflect macro indicators.
if "returns" in labeled_df.columns:
    labeled_df = labeled_df.drop(columns=["returns"])

for horizon in ["cycle_1m", "cycle_3m", "cycle_6m"]:
    labeled_df, weights = compute_composite_score(labeled_df, horizon)
    print(f"\n=== Composite Indicator Weights for {horizon} ===")
    print(weights.sort_values(ascending=False).head(15))

print("\nComposite scores added to labeled_df!")
print(labeled_df.head())

# ---------------------------------------------
# 7. Convert into WIDE format for ML
# ---------------------------------------------
# Build a single row per month for indicator values and metadata
indicator_value_wide = to_wide_monthly(labeled_df[["name", "value"]])
indicator_cat_wide = labeled_df.pivot_table(
    index=labeled_df.index.to_period("M").to_timestamp("M"),
    columns="name",
    values="indicator_cat_code",
    aggfunc="first",
).add_suffix("_catcode")

metadata_cols = [
    c for c in labeled_df.columns
    if c not in {"name", "value", "indicator_cat", "indicator_cat_code", "id"}
]
metadata = labeled_df[metadata_cols].groupby(labeled_df.index).first()

# Outer joins keep months even when some indicators are missing so we don't
# silently discard history because of sparse series.
wide_df = metadata.join(indicator_value_wide, how="outer")
wide_df = wide_df.join(indicator_cat_wide, how="left")
wide_df = wide_df.loc[~wide_df.index.duplicated()].sort_index()

if EARLIEST_DATE is not None:
    wide_df = wide_df[wide_df.index >= EARLIEST_DATE]

# 8. Compute composite scores for each horizon using the wide feature set
for horizon in ["cycle_1m", "cycle_3m", "cycle_6m"]:
    wide_df, weights = compute_composite_score(wide_df, horizon)
    print(f"\n=== Composite Indicator Weights for {horizon} ===")
    print(weights.sort_values(ascending=False).head(15))

print("\nComposite scores added to wide_df!")
print(wide_df.head())

# 9. Build model-ready features (drop target columns and composite scores to avoid leakage)
composite_cols = [c for c in wide_df.columns if c.startswith("composite_cycle_")]
feature_df = wide_df.drop(columns=["cycle_1m", "cycle_3m", "cycle_6m", *composite_cols])
# Strip any stock-market return-derived columns before feature engineering
feature_df = feature_df[[c for c in feature_df.columns if "return" not in c.lower()]]
X = make_features(feature_df)
X = X.loc[~X.index.duplicated()].sort_index()
print(
    f"\nFeature rows after engineering: {len(X):,} "
    f"(from labeled rows: {len(labeled_df):,})"
)

# Remove constant columns that provide no predictive signal
X, constant_cols = drop_constant_columns(X)
if constant_cols:
    print(f"Dropped constant columns: {constant_cols}")

# 10. Build targets aligned with X
df_targets = wide_df[["cycle_1m", "cycle_3m", "cycle_6m"]].reindex(X.index)
available_targets = df_targets.dropna(how="all").shape[0]
print(
    f"Target rows aligned with features: {available_targets:,}/{len(df_targets):,}"
)

# Trim rows where all horizons are missing to keep features/labels in sync.
valid_idx = df_targets.dropna(how="all").index
X = X.loc[valid_idx]
df_targets = df_targets.loc[valid_idx]

available_targets = df_targets.shape[0]
print(
    f"Target rows aligned with features: {available_targets:,}/{len(df_targets):,}"
)

# 11. Train multi-horizon models with LightGBM classifiers
pipelines, accuracies = train_lightgbm_multi_horizon(X, df_targets)

print("\nTraining accuracy by horizon:")
for horizon, acc in accuracies.items():
    if acc is None:
        print(f"  {horizon}: no labels available")
    else:
        print(f"  {horizon}: {acc:.3f}")

# ---------------------------------------------
# 12. Make prediction on the latest data & explain it
# ---------------------------------------------
print("\n\n====================")
print("   MODEL PREDICTION ")
print("====================")

# Generate probabilities + top indicators for each horizon
latest_prediction = predict_and_explain(pipelines, X, top_n=12)

print("\n\nFinal Prediction Output:")
print(latest_prediction)
import json
output_path = r"C:\Users\harte\Documents\goTutorial\internal\model_debug_report.html"

with open("model_debug_report.html", "w") as f:
    f.write("<h1>Model Debug Report</h1>")
    f.write("<h2>Prediction Output</h2>")
    f.write("<pre>" + json.dumps(latest_prediction, indent=4) + "</pre>")
