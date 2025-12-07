import pandas as pd
import numpy as np

from db_loader import (
    load_indicator_data,
    load_sp500_from_db
)


from preprocess import (
    preprocess_weekly,
    backfill_missing_key_indicators,
    to_wide_monthly
)



from enhance_model import (
    make_features,
    train_models,
    predict_and_explain
)


from config import (
    get_database_url,
    KEY_INDICATORS
)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

print("Loading indicators...")
df = load_indicator_data()
print(df.head())

print("\nResampling weekly...")
weekly = preprocess_weekly(df)
print("Weekly:", weekly.shape)

print("\nBackfilling key indicators...")
weekly = backfill_missing_key_indicators(weekly, KEY_INDICATORS)

print("\nConverting to wide monthly...")
wide = to_wide_monthly(weekly)
print("Wide monthly:", wide.shape)

print("\nLoading SP500...")
sp500 = load_sp500_from_db()
print(sp500.head())


# ------------------------------------------------------------
# Build targets
# ------------------------------------------------------------

weekly["returns"] = weekly["value"].pct_change()

conditions = [
    weekly["returns"] > 0.02,
    weekly["returns"] < -0.02
]
choices = ["Expansion", "Contraction"]
weekly["cycle_phase"] = np.select(conditions, choices, default="Peak")

# Future horizons
weekly["cycle_1m"] = weekly["cycle_phase"].shift(-4)
weekly["cycle_3m"] = weekly["cycle_phase"].shift(-12)
weekly["cycle_6m"] = weekly["cycle_phase"].shift(-24)

weekly = weekly.dropna(subset=["cycle_1m", "cycle_3m", "cycle_6m"])


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

print("\nBuilding features...")
X = make_features(weekly, wide)

y_dict = {
    "cycle_1m": weekly["cycle_1m"].reindex(X.index),
    "cycle_3m": weekly["cycle_3m"].reindex(X.index),
    "cycle_6m": weekly["cycle_6m"].reindex(X.index),
}

print(f"Feature rows: {len(X)}  Target rows: {len(y_dict['cycle_1m'])}")

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

models = train_models(X, y_dict)

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

print("\n\n====================")
print("   MODEL PREDICTION")
print("====================\n")

results = predict_and_explain(models, X)
print(results)
